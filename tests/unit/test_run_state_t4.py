"""T4: run state machine, checkpoints, recovery, startup sweep, journal (offline)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.runs.service import reset_startup_sweep_cache_for_tests, sweep_running_runs
from probe_harness.runs.state import STATUS_INTERRUPTED, STATUS_RUNNING
from probe_harness.store.db import Store


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
    assert int(counter.read_text()) == 1
    if kill_at == "post_tools":
        assert len(calls) == 2
    else:
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
