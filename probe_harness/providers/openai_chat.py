"""Minimal DeepSeek OpenAI-compatible client (live calls via UsageLedger)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx

from probe_harness.providers.base import ModelTurn, ToolCallSpec
from probe_harness.providers.ledger import TokenCapReached, UsageLedger
from probe_harness.providers.usage import estimate_chat_input_tokens, normalise_openai_usage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

_SENSITIVE_KEYS = frozenset(
    {"authorization", "x-api-key", "api-key", "x-auth-token", "bearer"}
)


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

    def _record_response_usage(self, usage_raw: dict[str, Any]) -> None:
        norm = normalise_openai_usage(usage_raw)
        self.ledger.record_usage(
            norm["input_total"],
            norm["output"],
            test_run_id=self.test_run_id,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        estimated_input_tokens: int | None = None,
    ) -> ModelTurn:
        mtoks = max_tokens if max_tokens is not None else self.max_tokens
        think = self.thinking_enabled if thinking_enabled is None else thinking_enabled
        thinking_param = {"type": "enabled"} if think else {"type": "disabled"}

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": mtoks,
            "thinking": thinking_param,
            "stream": False,
        }
        if think:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools

        est_in = (
            estimated_input_tokens
            if estimated_input_tokens is not None
            else estimate_chat_input_tokens(messages, tools)
        )
        self.ledger.check_reservation(est_in, mtoks)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        client = await self._get_client()
        resp = await client.post(url, headers=headers, json=body)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")

        usage_raw: dict[str, Any] = {}
        try:
            data = resp.json()
            usage_raw = data.get("usage") or {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        try:
            self._record_response_usage(usage_raw)
        except TokenCapReached:
            raise

        if resp.status_code >= 400:
            return ModelTurn(
                content=None,
                finish_reason="error",
                usage=usage_raw,
                http_status=resp.status_code,
                request_id=req_id,
                latency_ms=latency_ms,
            )

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
            usage=usage_raw,
            http_status=resp.status_code,
            request_id=req_id,
            latency_ms=latency_ms,
        )
