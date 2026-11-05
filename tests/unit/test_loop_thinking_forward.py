"""Verify loop forwards task thinking_enabled to provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.store.db import Store


@pytest.mark.asyncio
async def test_loop_forwards_thinking_enabled(tmp_path: Path) -> None:
    seen: dict[str, bool | None] = {"thinking_enabled": None}

    class CaptureProvider:
        async def complete(self, *args, **kwargs):
            seen["thinking_enabled"] = kwargs.get("thinking_enabled")
            return type(
                "T",
                (),
                {
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "tool_calls": [],
                    "content": "ok",
                    "finish_reason": "stop",
                    "http_status": 200,
                    "request_id": None,
                    "latency_ms": 0,
                    "reasoning_content": None,
                    "response_model": None,
                    "http_calls": [],
                    "error_body": "",
                },
            )()

    store = Store(tmp_path / "think.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "thinking_enabled": True}))
    await run_agent_loop(
        store,
        CaptureProvider(),
        task,
        provider_name="deepseek",
        model="deepseek-flash",
    )
    await store.close()
    assert seen["thinking_enabled"] is True
