from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.store.db import Store


@pytest.mark.asyncio
async def test_loop_protocol_repair_after_cancel_during_tool_exec(tmp_path: Path) -> None:
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_exec",
                    "function": {"name": "bash", "arguments": '{"command": "echo x"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "content": "done",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "user_message": "go",
                "fake_script": script,
                "simulate_interrupt_before_tool_results": True,
                "pairing_repair_reason": "cancel_during_tool_exec",
            }
        )
    )
    store = Store(tmp_path / "loop.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(script),
        task,
        provider_name="fake",
        model="fake",
    )
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    await store.close()
    assert repairs
    payload = json.loads(repairs[0]["data_json"])
    assert payload.get("reason") == "cancel_during_tool_exec"
    assert payload.get("kind") == "pairing"


@pytest.mark.asyncio
async def test_loop_protocol_repair_after_cancel_during_approval(tmp_path: Path) -> None:
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_appr",
                    "function": {"name": "bash", "arguments": '{"command": "echo y"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "content": "done",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "user_message": "go",
                "fake_script": script,
                "simulate_interrupt_before_approval": True,
                "pairing_repair_reason": "cancel_during_approval",
            }
        )
    )
    store = Store(tmp_path / "appr.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(script),
        task,
        provider_name="fake",
        model="fake",
    )
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    await store.close()
    assert repairs
    payload = json.loads(repairs[0]["data_json"])
    assert payload.get("reason") == "cancel_during_approval"
