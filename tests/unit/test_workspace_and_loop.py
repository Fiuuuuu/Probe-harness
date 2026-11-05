from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.providers.ledger import UsageLedger
from probe_harness.store.db import Store
from probe_harness.tools.builtin import edit_file, glob_paths, read_file


@pytest.mark.asyncio
async def test_read_file_rejects_escape_to_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws2 = tmp_path / "ws2"
    ws.mkdir()
    ws2.mkdir()
    secret = ws2 / "secret.txt"
    secret.write_text("leak", encoding="utf-8")
    monkeypatch.setenv("HARNESS_WORKSPACE", str(ws))
    rel = os.path.relpath(secret, ws)
    result = await read_file(rel)
    assert "error" in result
    assert result.get("content") == ""


@pytest.mark.asyncio
async def test_edit_file_rejects_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HARNESS_WORKSPACE", str(ws))
    rel = os.path.relpath(outside, ws)
    result = await edit_file(rel, "x", "y")
    assert result.get("replaced") is False
    assert "error" in result


@pytest.mark.asyncio
async def test_glob_filters_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws2 = tmp_path / "ws2"
    ws.mkdir()
    ws2.mkdir()
    (ws2 / "outside.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("HARNESS_WORKSPACE", str(ws))
    result = await glob_paths("../*")
    assert "outside.txt" not in "".join(result.get("matches", []))


@pytest.mark.asyncio
async def test_bad_read_file_args_do_not_stuck_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "f.txt"
    target.write_text("line", encoding="utf-8")
    monkeypatch.setenv("HARNESS_WORKSPACE", str(ws))

    db = tmp_path / "run.db"
    store = Store(db)
    await store.connect()
    script = [
        {
            "tool_calls": [
                {
                    "id": "badread",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "f.txt", "offset": "x"}),
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "content": "ok",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "go"}), encoding="utf-8")
    run_id = await run_agent_loop(
        store,
        FakeProvider(script),
        task,
        provider_name="fake",
        model="fake",
    )
    run = await store.fetchone("SELECT status FROM runs WHERE id = ?", (run_id,))
    assert run["status"] == "completed"
    await store.close()


@pytest.mark.asyncio
async def test_token_cap_marks_run_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TASK_TOKENS", "100")
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = UsageLedger(path=ledger_path, task_id="cap-test")
    ledger.record_usage(95, 0, test_run_id="seed")

    class CapProvider:
        async def complete(self, *args, **kwargs):
            ledger.record_usage(10, 10, test_run_id="boom")
            return type(
                "T",
                (),
                {
                    "usage": {},
                    "tool_calls": [],
                    "content": "x",
                    "finish_reason": "stop",
                    "http_status": 200,
                    "request_id": None,
                    "latency_ms": 0,
                    "reasoning_content": None,
                },
            )()

    db = tmp_path / "cap.db"
    store = Store(db)
    await store.connect()
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"user_message": "hi", "fake_script": []}), encoding="utf-8")
    run_id = await run_agent_loop(
        store, CapProvider(), task, provider_name="deepseek", model="deepseek-flash"
    )
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    assert run["status"] == "failed"
    assert run["stop_reason"] == "token_cap"
    await store.close()


@pytest.mark.asyncio
async def test_token_cap_live_guard_skips_not_asserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_TASK_TOKENS", "100")
    ledger = UsageLedger(path=tmp_path / "ledger.jsonl", task_id="cap-test")
    ledger.record_usage(95, 0, test_run_id="seed")

    class CapProvider:
        async def complete(self, *args, **kwargs):
            ledger.record_usage(10, 10, test_run_id="boom")
            return type(
                "T",
                (),
                {
                    "usage": {},
                    "tool_calls": [],
                    "content": "x",
                    "finish_reason": "stop",
                    "http_status": 200,
                    "request_id": None,
                    "latency_ms": 0,
                    "reasoning_content": None,
                },
            )()

    db = tmp_path / "cap.db"
    store = Store(db)
    await store.connect()
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"user_message": "hi", "fake_script": []}), encoding="utf-8")
    run_id = await run_agent_loop(
        store, CapProvider(), task, provider_name="deepseek", model="deepseek-flash"
    )
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    assert run["stop_reason"] == "token_cap"
    with pytest.raises(pytest.skip.Exception, match="token cap reached"):
        if run["stop_reason"] == "token_cap":
            pytest.skip("token cap reached")
        assert run["status"] == "completed"
    await store.close()


@pytest.mark.asyncio
async def test_provider_exception_finalizes_then_reraises(tmp_path: Path) -> None:
    class FailProvider:
        async def complete(self, *args, **kwargs):
            raise ConnectionError("network down")

    db = tmp_path / "exc.db"
    store = Store(db)
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi"}), encoding="utf-8")

    with pytest.raises(ConnectionError, match="network down"):
        await run_agent_loop(store, FailProvider(), task, provider_name="deepseek", model="x")

    run = await store.fetchone(
        "SELECT id, status, stop_reason FROM runs ORDER BY started_at DESC LIMIT 1"
    )
    assert run["status"] == "failed"
    assert run["stop_reason"] == "exception"
    events = await store.fetchall(
        "SELECT type FROM events WHERE run_id = ? ORDER BY seq",
        (run["id"],),
    )
    assert events[-1]["type"] == "run_finished"
    await store.close()


@pytest.mark.asyncio
async def test_tool_store_write_failure_finalizes_and_reraises(tmp_path: Path) -> None:
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_bash",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo hi"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    db = tmp_path / "fail.db"
    store = Store(db)
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "go"}), encoding="utf-8")

    real_execute = store.execute

    async def execute_fail_on_tool_insert(sql: str, params=()):
        if "INSERT INTO tool_calls" in sql:
            raise RuntimeError("simulated store write failure")
        return await real_execute(sql, params)

    store.execute = execute_fail_on_tool_insert  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated store write failure"):
        await run_agent_loop(
            store,
            FakeProvider(script),
            task,
            provider_name="fake",
            model="fake",
        )

    run = await store.fetchone(
        "SELECT id, status, stop_reason FROM runs ORDER BY started_at DESC LIMIT 1"
    )
    assert run["status"] == "failed"
    assert run["stop_reason"] == "exception"
    events = await store.fetchall(
        "SELECT type FROM events WHERE run_id = ? ORDER BY seq",
        (run["id"],),
    )
    assert events[-1]["type"] == "run_finished"
    await store.close()
