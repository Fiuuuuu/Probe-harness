"""Minimal DeepSeek OpenAI-compatible client (live calls via UsageLedger)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx

from probe_harness.providers.base import ModelTurn, ToolCallSpec
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.usage import normalise_openai_usage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def scrub_request_body(body: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(body))


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
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ledger = ledger or UsageLedger()
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.test_run_id = test_run_id

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        estimated_input_tokens: int = 4096,
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

        self.ledger.check_reservation(estimated_input_tokens, mtoks)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=body)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")

        if resp.status_code >= 400:
            return ModelTurn(
                content=None,
                finish_reason="error",
                usage={},
                http_status=resp.status_code,
                request_id=req_id,
                latency_ms=latency_ms,
            )

        data = resp.json()
        usage_raw = data.get("usage") or {}
        norm = normalise_openai_usage(usage_raw)
        self.ledger.record_usage(
            norm["input_total"],
            norm["output"],
            test_run_id=self.test_run_id,
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
