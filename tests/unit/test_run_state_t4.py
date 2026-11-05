"""T4: run state machine, checkpoints, recovery, startup sweep, journal (offline)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.runs.journal import JOURNAL_STARTED
from probe_harness.runs.service import reset_startup_sweep_cache_for_tests, sweep_running_runs
from probe_harness.runs.state import STATUS_INTERRUPTED, STATUS_RUNNING
from probe_harness.store.db import Store
from probe_harness.tools import builtin as builtin_mod


def _counter_script(counter_path: Path, call_id: str = "call_inc") -> list[dict]:
    cmd = (
        "python3 -c "
        f"\"from pathlib import Path; p=Path('{counter_path}'); "
        'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1))"'
    )
    return [
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        },
        {
            "content": "done",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        },
    ]


@pytest.mark.asyncio
async def test_startup_sweep_marks_running_interrupted(tmp_path: Path) -> None:
    reset_startup_sweep_cache_for_tests()
    db = tmp_path / "sweep.db"
    store = Store(db)
    await store.connect()
    run_id = str(uuid.uuid4())
    await store.execute(
        """
        INSERT INTO runs (id, task_ref, status, stop_reason, provider, model, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, "t", STATUS_RUNNING, None, "fake", "fake", 1.0),
    )
    n = await sweep_running_runs(store)
    assert n == 1
    row = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    assert row["status"] == STATUS_INTERRUPTED
    assert row["stop_reason"] == "host_restart"
    await store.close()


@pytest.mark.parametrize(
    "kill_at",
    ["pre_model", "during_tool", "post_tools"],
)
@pytest.mark.asyncio
async def test_sigkill_recovery_no_double_side_effect(tmp_path: Path, kill_at: str) -> None:
    reset_startup_sweep_cache_for_tests()
    work = tmp_path / "work"
    work.mkdir()
    counter = work / "counter.txt"
    counter.write_text("0")
    db = tmp_path / "h.db"
    task_path = work / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "inc",
                "fake_script": _counter_script(counter),
                "max_steps": 5,
                "simulate_hard_kill_at": kill_at,
            }
        )
    )
    os.environ["HARNESS_WORKSPACE"] = str(work)
    os.environ["HARNESS_ALLOW_TEST_CONTROLS"] = "1"

    store = Store(db)
    await store.connect()
    source_id = await run_agent_loop(
        store,
        FakeProvider(_counter_script(counter)),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=5,
    )
    row = await store.fetchone("SELECT status FROM runs WHERE id = ?", (source_id,))
    assert row["status"] == STATUS_RUNNING
    await sweep_running_runs(store)

    task_no_kill = json.loads(task_path.read_text())
    task_no_kill.pop("simulate_hard_kill_at", None)
    task_path.write_text(json.dumps(task_no_kill))

    recovered_id = await run_agent_loop(
        store,
        FakeProvider(_counter_script(counter)),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=5,
        recover_from=source_id,
    )
    new_row = await store.fetchone(
        "SELECT recovered_from, status FROM runs WHERE id = ?", (recovered_id,)
    )
    calls = await store.fetchall("SELECT run_id FROM model_calls ORDER BY id")
    await store.close()
    assert new_row["recovered_from"] == source_id
    expected_counter = 0 if kill_at == "during_tool" else 1
    assert int(counter.read_text()) == expected_counter
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_recover_creates_new_run_with_link(tmp_path: Path) -> None:
    reset_startup_sweep_cache_for_tests()
    db = tmp_path / "r.db"
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "user_message": "hi",
                "fake_script": [
                    {
                        "content": "ok",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ],
            }
        )
    )
    store = Store(db)
    await store.connect()
    run_id = await run_agent_loop(
        store, FakeProvider([]), task, provider_name="fake", model="fake", max_steps=1
    )
    await store.execute(
        "UPDATE runs SET status = ?, stop_reason = ? WHERE id = ?",
        (STATUS_INTERRUPTED, "host_restart", run_id),
    )
    ck = await store.fetchall("SELECT boundary FROM checkpoints WHERE run_id = ?", (run_id,))
    assert ck
    new_id = await run_agent_loop(
        store,
        FakeProvider(
            [
                {
                    "content": "ok",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
        task,
        provider_name="fake",
        model="fake",
        max_steps=1,
        recover_from=run_id,
    )
    row = await store.fetchone("SELECT recovered_from FROM runs WHERE id = ?", (new_id,))
    assert row["recovered_from"] == run_id
    old = await store.fetchone("SELECT status FROM runs WHERE id = ?", (run_id,))
    assert old["status"] == STATUS_INTERRUPTED
    await store.close()


@pytest.mark.asyncio
async def test_journal_started_not_re_executed_on_recovery(tmp_path: Path) -> None:
    reset_startup_sweep_cache_for_tests()
    work = tmp_path / "work"
    work.mkdir()
    counter = work / "counter.txt"
    counter.write_text("0")
    db = tmp_path / "j.db"
    task_path = work / "task.json"
    os.environ["HARNESS_WORKSPACE"] = str(work)
    cmd = (
        "python3 -c "
        f"\"from pathlib import Path; p=Path('{counter}'); "
        'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1))"'
    )
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_inc",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
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
    task_path.write_text(json.dumps({"user_message": "x", "fake_script": script, "max_steps": 5}))

    store = Store(db)
    await store.connect()
    source_id = str(uuid.uuid4())
    await store.execute(
        """
        INSERT INTO runs (id, task_ref, status, stop_reason, provider, model, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, str(task_path), STATUS_INTERRUPTED, "host_restart", "fake", "fake", 1.0),
    )
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_inc",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                }
            ],
        },
    ]
    meta = json.dumps({"fake_script_index": 1, "model_response_done": True})
    await store.execute(
        """
        INSERT INTO checkpoints (run_id, step, boundary, messages_json, usage_json, budget_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_id, 0, "pre_model", json.dumps(messages), meta, "{}"),
    )
    counter.write_text("1")
    await store.execute(
        """
        INSERT INTO tool_journal (
            lineage_run_id, run_id, call_id, norm_name, args_sha256, status, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, source_id, "call_inc", "bash", "abc", JOURNAL_STARTED, 1.0),
    )

    recovered_id = await run_agent_loop(
        store,
        FakeProvider(script, start_index=1),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=5,
        recover_from=source_id,
    )
    assert counter.read_text() == "1"
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (recovered_id,),
    )
    assert any("journal_no_replay" in r["data_json"] for r in repairs)
    row = await store.fetchone("SELECT status FROM tool_journal WHERE call_id = ?", ("call_inc",))
    from probe_harness.runs.journal import JOURNAL_UNCERTAIN

    assert row["status"] == JOURNAL_UNCERTAIN
    await store.close()


def _ps_sleep_lines(fragment: str = "sleep 300") -> list[str]:
    out = subprocess.check_output(["ps", "-eo", "pid,ppid,pgid,sid,args"], text=True)
    lines: list[str] = []
    for line in out.splitlines():
        if fragment in line and "pytest" not in line:
            lines.append(line.strip())
    return lines


@pytest.mark.asyncio
async def test_cancel_kills_bash_process_group(tmp_path: Path) -> None:
    reset_startup_sweep_cache_for_tests()
    work = tmp_path / "work"
    work.mkdir()
    db = tmp_path / "pg.db"
    task_path = work / "task.json"
    cmd = "sleep 300 & sleep 300 & wait"
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_sleeps",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
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
    task_path.write_text(
        json.dumps({"user_message": "sleep", "fake_script": script, "max_steps": 3})
    )
    os.environ["HARNESS_WORKSPACE"] = str(work)

    store = Store(db)
    await store.connect()
    loop_task = asyncio.create_task(
        run_agent_loop(
            store, FakeProvider(script), task_path, provider_name="fake", model="fake", max_steps=3
        )
    )
    for _ in range(150):
        if len(_ps_sleep_lines()) >= 2:
            break
        await asyncio.sleep(0.05)
    assert len(_ps_sleep_lines()) >= 2
    assert len(builtin_mod._active_bash_procs) >= 1

    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

    await asyncio.sleep(0.4)
    survivors = _ps_sleep_lines()
    for line in survivors:
        try:
            os.kill(int(line.split(None, 1)[0]), signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert survivors == []

    run_row = await store.fetchone(
        "SELECT id, status, stop_reason FROM runs ORDER BY started_at DESC LIMIT 1"
    )
    assert run_row is not None
    assert run_row["status"] == "cancelled"
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_row["id"],),
    )
    assert repairs
    await store.close()


@pytest.mark.asyncio
async def test_cancel_repairs_unpaired_tool_call(tmp_path: Path) -> None:
    reset_startup_sweep_cache_for_tests()
    db = tmp_path / "c.db"
    task = tmp_path / "task.json"
    script = [
        {
            "tool_calls": [
                {
                    "id": "call_cancel",
                    "function": {"name": "bash", "arguments": '{"command": "sleep 60"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    task.write_text(
        json.dumps(
            {
                "user_message": "go",
                "fake_script": script,
                "simulate_interrupt_before_tool_results": True,
            }
        )
    )
    store = Store(db)
    await store.connect()
    run_id = await run_agent_loop(
        store, FakeProvider(script), task, provider_name="fake", model="fake", max_steps=3
    )
    repairs = await store.fetchall(
        "SELECT type FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    assert repairs
    await store.close()
