"""DeepSeek OpenAI-compatible Chat Completions adapter (stream + non-stream, C9 retry)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any

import httpx

from probe_harness.providers.base import HttpCallRecord, ModelTurn, ToolCallSpec
from probe_harness.providers.ledger import TokenCapReached, UsageLedger
from probe_harness.providers.usage import estimate_chat_input_tokens, normalise_openai_usage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = (1.0, 2.0, 4.0)

_SENSITIVE_KEYS = frozenset({"authorization", "x-api-key", "api-key", "x-auth-token", "bearer"})


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


def _request_id(resp: httpx.Response) -> str | None:
    return resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


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
            arguments=item["arguments"] or "{}",
        )
        for item in ordered
    ]


def parse_sse_stream_payload(lines: list[str]) -> tuple[ModelTurn, dict[str, Any]]:
    """Parse OpenAI-style SSE lines (offline tests + live stream)."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[dict[str, Any]] = []
    usage_raw: dict[str, Any] = {}
    finish_reason = "stop"
    http_status = 200

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
    )
    return turn, usage_raw


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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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

    def _record_response_usage(self, usage_raw: dict[str, Any]) -> None:
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
    ) -> HttpCallRecord:
        return HttpCallRecord(
            attempt=attempt,
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=_request_id(resp),
            latency_ms=latency_ms,
            stop_reason=finish_reason,
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
    ) -> tuple[list[str], httpx.Response, int]:
        t0 = time.perf_counter()
        lines: list[str] = []
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            await resp.aread()
            return lines, resp, latency_ms

    async def _one_json_attempt(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        attempt: int,
    ) -> tuple[ModelTurn, HttpCallRecord, bool]:
        """Returns (turn, record, should_retry)."""
        t0 = time.perf_counter()
        resp = await self._post_json(client, url, headers, body)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_raw: dict[str, Any] = {}
        data: dict[str, Any] = {}
        try:
            data = resp.json()
            usage_raw = data.get("usage") or {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        self._record_response_usage(usage_raw)

        finish = "error"
        if resp.status_code < 400:
            choice = (data.get("choices") or [{}])[0]
            finish = choice.get("finish_reason") or "stop"

        record = self._http_call_record(attempt, usage_raw, resp, latency_ms, finish)
        if resp.status_code < 400:
            turn = self._turn_from_json(data, resp, latency_ms)
            return turn, record, False

        should_retry = _is_retryable_status(resp.status_code)
        turn = ModelTurn(
            content=None,
            finish_reason="error",
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=_request_id(resp),
            latency_ms=latency_ms,
        )
        return turn, record, should_retry

    async def _one_stream_attempt(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        attempt: int,
    ) -> tuple[ModelTurn, HttpCallRecord, bool]:
        lines, resp, latency_ms = await self._post_stream(client, url, headers, body)
        streamed_payload = any(
            ln.strip().startswith("data:") and ln.strip() not in ("data: [DONE]", "data:[DONE]")
            for ln in lines
        )
        usage_raw: dict[str, Any] = {}
        finish = "error"

        if resp.status_code < 400:
            turn, usage_raw = parse_sse_stream_payload(lines)
            turn.latency_ms = latency_ms
            turn.request_id = _request_id(resp)
            turn.http_status = resp.status_code
            finish = turn.finish_reason
            self._record_response_usage(usage_raw)
            record = self._http_call_record(attempt, usage_raw, resp, latency_ms, finish)
            return turn, record, False

        try:
            if resp.content:
                err_data = json.loads(resp.content)
                usage_raw = err_data.get("usage") or {}
        except (json.JSONDecodeError, ValueError):
            pass
        if usage_raw:
            self._record_response_usage(usage_raw)
        record = self._http_call_record(attempt, usage_raw, resp, latency_ms, finish)
        turn = ModelTurn(
            content=None,
            finish_reason="error",
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=_request_id(resp),
            latency_ms=latency_ms,
        )
        should_retry = _is_retryable_status(resp.status_code) and not streamed_payload
        return turn, record, should_retry

    async def _execute_with_retry(
        self,
        body: dict[str, Any],
        *,
        est_in: int,
        max_tokens: int,
        stream: bool,
    ) -> tuple[ModelTurn, list[HttpCallRecord]]:
        self.ledger.check_reservation(est_in, max_tokens)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        client = await self._get_client()
        http_calls: list[HttpCallRecord] = []

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            should_retry = False
            try:
                if stream:
                    turn, record, should_retry = await self._one_stream_attempt(
                        client, url, headers, body, attempt
                    )
                else:
                    turn, record, should_retry = await self._one_json_attempt(
                        client, url, headers, body, attempt
                    )
            except TokenCapReached:
                raise
            except httpx.HTTPError:
                if attempt >= MAX_RETRY_ATTEMPTS:
                    raise
                should_retry = True
                turn = ModelTurn(content=None, finish_reason="error", http_status=0)
                record = HttpCallRecord(
                    attempt=attempt,
                    usage={},
                    http_status=0,
                    request_id=None,
                    latency_ms=0,
                    stop_reason="network_error",
                )

            http_calls.append(record)
            if not should_retry:
                turn.http_calls = http_calls
                return turn, http_calls
            if attempt < MAX_RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF_S[attempt - 1])
                continue

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
            tool_calls.append(
                ToolCallSpec(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or "{}",
                )
            )
        return ModelTurn(
            content=message.get("content"),
            tool_calls=tool_calls,
            reasoning_content=message.get("reasoning_content"),
            finish_reason=finish,
            usage=data.get("usage") or {},
            http_status=resp.status_code,
            request_id=_request_id(resp),
            latency_ms=latency_ms,
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
