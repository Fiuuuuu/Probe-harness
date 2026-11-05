"""DeepSeek OpenAI-compatible Chat Completions adapter (stream + non-stream, C9 retry)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from probe_harness.providers.base import HttpCallRecord, ModelTurn, ProviderCallError, ToolCallSpec
from probe_harness.providers.guards import merge_system_messages_openai, strip_reasoning_for_wire
from probe_harness.providers.ledger import TokenCapReached, UsageLedger
from probe_harness.providers.usage import estimate_chat_input_tokens, normalise_openai_usage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_S = 60.0
ERROR_BODY_MAX = 800

_SENSITIVE_KEYS = frozenset({"authorization", "x-api-key", "api-key", "x-auth-token", "bearer"})
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def scrub_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy and redact secret-like keys anywhere in the JSON tree."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                if str(key).lower() in _SENSITIVE_KEYS:
                    cleaned[key] = "[REDACTED]"
                else:
                    cleaned[key] = scrub(item)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(body)


def request_sha256(body: dict[str, Any]) -> str:
    scrubbed = scrub_request_body(body)
    raw = json.dumps(scrubbed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _request_id(resp: httpx.Response, data: dict[str, Any] | None = None) -> str | None:
    rid = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
    if rid:
        return rid
    if data and data.get("id"):
        return str(data["id"])
    return None


def error_body_from_text(text: str) -> str:
    scrubbed = _BEARER_RE.sub("Bearer [REDACTED]", text)
    return scrubbed[:ERROR_BODY_MAX]


def error_body_from_response(resp: httpx.Response) -> str:
    try:
        return error_body_from_text(resp.text)
    except Exception:
        return ""


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


class RetryAfterTooLarge(Exception):
    """Retry-After header exceeds the allowed wait cap."""

    def __init__(self, seconds: float, cap: float) -> None:
        self.seconds = seconds
        self.cap = cap
        super().__init__(f"Retry-After {seconds}s exceeds maximum {cap}s")


def _retry_sleep_seconds(resp: httpx.Response | None, attempt: int) -> float:
    if resp is not None:
        raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        if raw:
            try:
                val = max(0.0, float(raw))
                if val > MAX_RETRY_AFTER_S:
                    raise RetryAfterTooLarge(val, MAX_RETRY_AFTER_S)
                return val
            except ValueError:
                pass
    idx = min(attempt - 1, len(RETRY_BACKOFF_S) - 1)
    return RETRY_BACKOFF_S[idx]


def assemble_stream_tool_calls(
    deltas: list[dict[str, Any]],
) -> list[ToolCallSpec]:
    """Concatenate streamed tool-call argument fragments (CONSTRAINTS C8)."""
    by_index: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        idx = int(delta.get("index", 0))
        slot = by_index.setdefault(
            idx,
            {"id": "", "name": "", "arguments": ""},
        )
        if delta.get("id"):
            slot["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]
    ordered = [by_index[i] for i in sorted(by_index)]
    return [
        ToolCallSpec(
            id=item["id"],
            name=item["name"],
            arguments=item["arguments"],
        )
        for item in ordered
        if item["id"] or item["name"] or item["arguments"]
    ]


def parse_sse_stream_payload(lines: list[str]) -> tuple[ModelTurn, dict[str, Any]]:
    """Parse OpenAI-style SSE lines (offline tests + live stream)."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[dict[str, Any]] = []
    usage_raw: dict[str, Any] = {}
    finish_reason = "stop"
    http_status = 200
    response_model: str | None = None
    request_id: str | None = None

    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("model"):
            response_model = str(chunk["model"])
        if chunk.get("id") and request_id is None:
            request_id = str(chunk["id"])
        if chunk.get("usage"):
            usage_raw = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            tool_deltas.extend(delta.get("tool_calls") or [])

    tool_calls = assemble_stream_tool_calls(tool_deltas)
    content = "".join(content_parts) or None
    reasoning = "".join(reasoning_parts) or None
    turn = ModelTurn(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning,
        finish_reason=finish_reason,
        usage=usage_raw,
        http_status=http_status,
        response_model=response_model,
        request_id=request_id,
    )
    return turn, usage_raw


@dataclass
class _StreamReadResult:
    lines: list[str]
    resp: httpx.Response
    latency_ms: int
    received_stream_bytes: bool
    stream_error: BaseException | None = None


class OpenAIChatProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        ledger: UsageLedger | None = None,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        test_run_id: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ledger = ledger or UsageLedger()
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.test_run_id = test_run_id
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0, trust_env=False)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def build_request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        thinking_enabled: bool,
        stream: bool,
    ) -> dict[str, Any]:
        thinking_param = {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
        wire_messages = strip_reasoning_for_wire(
            merge_system_messages_openai(messages),
            thinking_enabled=thinking_enabled,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "thinking": thinking_param,
            "stream": stream,
        }
        if thinking_enabled:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _usage_has_tokens(self, usage_raw: dict[str, Any]) -> bool:
        norm = normalise_openai_usage(usage_raw)
        return norm["input_total"] + norm["output"] > 0

    def _record_response_usage(
        self,
        usage_raw: dict[str, Any],
        *,
        est_in: int,
        max_tokens: int,
        stream: bool,
    ) -> None:
        if stream and not self._usage_has_tokens(usage_raw):
            self.ledger.record_usage(est_in, max_tokens, test_run_id=self.test_run_id)
            return
        norm = normalise_openai_usage(usage_raw)
        self.ledger.record_usage(
            norm["input_total"],
            norm["output"],
            test_run_id=self.test_run_id,
        )

    def _http_call_record(
        self,
        attempt: int,
        usage_raw: dict[str, Any],
        resp: httpx.Response,
        latency_ms: int,
        finish_reason: str,
        *,
        error_body: str = "",
        usage_estimated: bool = False,
        response_model: str | None = None,
        request_id: str | None = None,
    ) -> HttpCallRecord:
        return HttpCallRecord(
            attempt=attempt,
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=request_id if request_id is not None else _request_id(resp, None),
            latency_ms=latency_ms,
            stop_reason=finish_reason,
            error_body=error_body,
            usage_estimated=usage_estimated,
            response_model=response_model,
        )

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> httpx.Response:
        return await client.post(url, headers=headers, json=body)

    async def _post_stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> _StreamReadResult:
        t0 = time.perf_counter()
        lines: list[str] = []
        received_stream_bytes = False
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return _StreamReadResult(lines, resp, latency_ms, False)

            try:
                async for line in resp.aiter_lines():
                    if line:
                        lines.append(line)
                        if line.strip().startswith("data:") and line.strip() not in (
                            "data: [DONE]",
                            "data:[DONE]",
                        ):
                            received_stream_bytes = True
            except httpx.HTTPError as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return _StreamReadResult(
                    lines,
                    resp,
                    latency_ms,
                    received_stream_bytes,
                    stream_error=exc,
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return _StreamReadResult(lines, resp, latency_ms, received_stream_bytes)

    async def _one_json_attempt(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        attempt: int,
        *,
        est_in: int,
        max_tokens: int,
    ) -> tuple[ModelTurn, HttpCallRecord, bool, httpx.Response | None]:
        """Returns (turn, record, should_retry, response_for_backoff)."""
        t0 = time.perf_counter()
        resp = await self._post_json(client, url, headers, body)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_raw: dict[str, Any] = {}
        data: dict[str, Any] = {}
        err_body = ""
        try:
            data = resp.json()
            usage_raw = data.get("usage") or {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        if resp.status_code >= 400:
            err_body = error_body_from_response(resp)

        finish = "error"
        if resp.status_code < 400:
            choice = (data.get("choices") or [{}])[0]
            finish = choice.get("finish_reason") or "stop"

        resp_model = str(data["model"]) if resp.status_code < 400 and data.get("model") else None
        rid = _request_id(resp, data if data else None)
        record = self._http_call_record(
            attempt,
            usage_raw,
            resp,
            latency_ms,
            finish,
            error_body=err_body,
            response_model=resp_model,
            request_id=rid,
        )
        try:
            self._record_response_usage(
                usage_raw, est_in=est_in, max_tokens=max_tokens, stream=False
            )
        except TokenCapReached as cap:
            cap.pending_record = record  # type: ignore[attr-defined]
            raise

        if resp.status_code < 400:
            turn = self._turn_from_json(data, resp, latency_ms)
            record.response_model = turn.response_model
            record.request_id = turn.request_id
            return turn, record, False, resp

        should_retry = _is_retryable_status(resp.status_code)
        turn = ModelTurn(
            content=None,
            finish_reason="error",
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=_request_id(resp, data),
            latency_ms=latency_ms,
            error_body=err_body,
            response_model=str(data["model"]) if data.get("model") else None,
        )
        return turn, record, should_retry, resp

    async def _one_stream_attempt(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        attempt: int,
        *,
        est_in: int,
        max_tokens: int,
    ) -> tuple[ModelTurn, HttpCallRecord, bool, httpx.Response | None]:
        stream_result = await self._post_stream(client, url, headers, body)
        resp = stream_result.resp
        latency_ms = stream_result.latency_ms
        received = stream_result.received_stream_bytes
        usage_raw: dict[str, Any] = {}
        finish = "error"
        err_body = ""

        if stream_result.stream_error is not None:
            err_body = error_body_from_text(str(stream_result.stream_error))
            usage_raw = {}
            if stream_result.lines and received:
                parsed_turn, usage_raw = parse_sse_stream_payload(stream_result.lines)
                stream_rid = parsed_turn.request_id
            else:
                stream_rid = None
            finish = "stream_interrupted"
            estimated_usage = usage_raw
            usage_estimated = False
            if not self._usage_has_tokens(usage_raw):
                estimated_usage = {
                    "prompt_tokens": est_in,
                    "completion_tokens": max_tokens,
                }
                usage_estimated = True
            try:
                self._record_response_usage(
                    estimated_usage, est_in=est_in, max_tokens=max_tokens, stream=True
                )
            except TokenCapReached as cap:
                record = self._http_call_record(
                    attempt,
                    estimated_usage,
                    resp,
                    latency_ms,
                    finish,
                    error_body=err_body,
                    usage_estimated=usage_estimated,
                    request_id=stream_rid or _request_id(resp),
                )
                cap.pending_record = record  # type: ignore[attr-defined]
                raise
            record = self._http_call_record(
                attempt,
                estimated_usage,
                resp,
                latency_ms,
                finish,
                error_body=err_body,
                usage_estimated=usage_estimated,
                request_id=stream_rid or _request_id(resp),
            )
            turn = ModelTurn(
                content=None,
                tool_calls=[],
                finish_reason="stream_interrupted",
                usage=estimated_usage,
                http_status=resp.status_code,
                request_id=stream_rid or _request_id(resp),
                latency_ms=latency_ms,
                error_body=err_body,
            )
            should_retry = not received
            return turn, record, should_retry, resp

        if resp.status_code < 400:
            turn, usage_raw = parse_sse_stream_payload(stream_result.lines)
            turn.latency_ms = latency_ms
            turn.request_id = turn.request_id or _request_id(resp)
            turn.http_status = resp.status_code
            finish = turn.finish_reason
            usage_estimated = False
            if not self._usage_has_tokens(usage_raw):
                usage_raw = {
                    "prompt_tokens": est_in,
                    "completion_tokens": max_tokens,
                }
                usage_estimated = True
                turn.usage = usage_raw
            record = self._http_call_record(
                attempt,
                usage_raw,
                resp,
                latency_ms,
                finish,
                response_model=turn.response_model,
                usage_estimated=usage_estimated,
                request_id=turn.request_id,
            )
            try:
                self._record_response_usage(
                    usage_raw, est_in=est_in, max_tokens=max_tokens, stream=True
                )
            except TokenCapReached as cap:
                cap.pending_record = record  # type: ignore[attr-defined]
                raise
            return turn, record, False, resp

        err_body = error_body_from_response(resp)
        err_data: dict[str, Any] = {}
        try:
            if resp.content:
                err_data = json.loads(resp.content)
                usage_raw = err_data.get("usage") or {}
        except (json.JSONDecodeError, ValueError):
            err_data = {}
        if usage_raw:
            self._record_response_usage(
                usage_raw, est_in=est_in, max_tokens=max_tokens, stream=False
            )
        record = self._http_call_record(
            attempt,
            usage_raw,
            resp,
            latency_ms,
            finish,
            error_body=err_body,
            response_model=str(err_data["model"]) if err_data.get("model") else None,
            request_id=_request_id(resp, err_data or None),
        )
        turn = ModelTurn(
            content=None,
            finish_reason="error",
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=_request_id(resp, err_data or None),
            latency_ms=latency_ms,
            error_body=err_body,
        )
        should_retry = _is_retryable_status(resp.status_code) and not received
        return turn, record, should_retry, resp

    def _raise_token_cap(self, exc: TokenCapReached, http_calls: list[HttpCallRecord]) -> None:
        pending = getattr(exc, "pending_record", None)
        if pending is not None and pending not in http_calls:
            http_calls.append(pending)
        exc.http_calls = http_calls  # type: ignore[attr-defined]
        raise exc

    async def _execute_with_retry(
        self,
        body: dict[str, Any],
        *,
        est_in: int,
        max_tokens: int,
        stream: bool,
    ) -> tuple[ModelTurn, list[HttpCallRecord]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        client = await self._get_client()
        http_calls: list[HttpCallRecord] = []
        turn = ModelTurn(content=None, finish_reason="error", http_status=0)

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                self.ledger.check_reservation(est_in, max_tokens)
            except TokenCapReached as cap:
                self._raise_token_cap(cap, http_calls)

            should_retry = False
            retry_resp: httpx.Response | None = None
            record: HttpCallRecord | None = None
            try:
                if stream:
                    turn, record, should_retry, retry_resp = await self._one_stream_attempt(
                        client,
                        url,
                        headers,
                        body,
                        attempt,
                        est_in=est_in,
                        max_tokens=max_tokens,
                    )
                else:
                    turn, record, should_retry, retry_resp = await self._one_json_attempt(
                        client,
                        url,
                        headers,
                        body,
                        attempt,
                        est_in=est_in,
                        max_tokens=max_tokens,
                    )
            except TokenCapReached as cap:
                if record is not None:
                    http_calls.append(record)
                self._raise_token_cap(cap, http_calls)
            except httpx.HTTPError as exc:
                err_body = error_body_from_text(str(exc))
                record = HttpCallRecord(
                    attempt=attempt,
                    usage={},
                    http_status=0,
                    request_id=None,
                    latency_ms=0,
                    stop_reason="network_error",
                    error_body=err_body,
                )
                http_calls.append(record)
                if attempt >= MAX_RETRY_ATTEMPTS:
                    err_turn = ModelTurn(
                        content=None,
                        finish_reason="error",
                        http_status=record.http_status,
                        request_id=record.request_id,
                        error_body=record.error_body,
                    )
                    raise ProviderCallError(str(exc), http_calls=http_calls, turn=err_turn) from exc
                should_retry = True
                retry_resp = None
            else:
                assert record is not None
                http_calls.append(record)
                if not should_retry:
                    turn.http_calls = http_calls
                    return turn, http_calls

            if attempt < MAX_RETRY_ATTEMPTS and should_retry:
                try:
                    sleep_s = _retry_sleep_seconds(retry_resp, attempt)
                except RetryAfterTooLarge as exc:
                    err_turn = ModelTurn(
                        content=None,
                        finish_reason="error",
                        http_status=0,
                        error_body=str(exc)[:ERROR_BODY_MAX],
                    )
                    raise ProviderCallError(str(exc), http_calls=http_calls, turn=err_turn) from exc
                await asyncio.sleep(sleep_s)
                continue

            turn.http_calls = http_calls
            return turn, http_calls

        turn.http_calls = http_calls
        return turn, http_calls

    def _turn_from_json(
        self,
        data: dict[str, Any],
        resp: httpx.Response,
        latency_ms: int,
    ) -> ModelTurn:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason") or "stop"
        tool_calls: list[ToolCallSpec] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if raw_args is None:
                arg_str = ""
            else:
                arg_str = raw_args
            tool_calls.append(
                ToolCallSpec(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=arg_str,
                )
            )
        return ModelTurn(
            content=message.get("content"),
            tool_calls=tool_calls,
            reasoning_content=message.get("reasoning_content"),
            finish_reason=finish,
            usage=data.get("usage") or {},
            http_status=resp.status_code,
            request_id=_request_id(resp, data),
            latency_ms=latency_ms,
            response_model=str(data["model"]) if data.get("model") else None,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        estimated_input_tokens: int | None = None,
        stream: bool = False,
    ) -> ModelTurn:
        mtoks = max_tokens if max_tokens is not None else self.max_tokens
        think = self.thinking_enabled if thinking_enabled is None else thinking_enabled
        body = self.build_request_body(
            messages,
            tools,
            max_tokens=mtoks,
            thinking_enabled=think,
            stream=stream,
        )
        est_in = (
            estimated_input_tokens
            if estimated_input_tokens is not None
            else estimate_chat_input_tokens(messages, tools)
        )
        turn, _ = await self._execute_with_retry(
            body,
            est_in=est_in,
            max_tokens=mtoks,
            stream=stream,
        )
        return turn
