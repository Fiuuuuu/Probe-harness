"""Scripted provider for offline runs and unit tests."""

from __future__ import annotations

import uuid
from typing import Any

from probe_harness.providers.base import ModelTurn, ToolCallSpec


class FakeProvider:
    """Replays a list of scripted turn dicts from a task file."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self._index = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> ModelTurn:
        if self._index >= len(self._script):
            return ModelTurn(content="done", finish_reason="stop")

        step = self._script[self._index]
        self._index += 1

        tool_calls: list[ToolCallSpec] = []
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function") or tc
            tool_calls.append(
                ToolCallSpec(
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", "bash"),
                    arguments=fn.get("arguments", "{}"),
                )
            )

        usage = step.get("usage") or {}
        return ModelTurn(
            content=step.get("content"),
            tool_calls=tool_calls,
            reasoning_content=step.get("reasoning_content"),
            finish_reason=step.get("finish_reason", "stop"),
            usage=usage,
            http_status=int(step.get("http_status", 200)),
            request_id=step.get("request_id"),
            latency_ms=int(step.get("latency_ms", 1)),
        )
