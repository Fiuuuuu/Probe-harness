"""Real SIGKILL subprocess tests (ACCEPTANCE P2-1)."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.runs.service import reset_startup_sweep_cache_for_tests, sweep_running_runs
from probe_harness.runs.state import STATUS_INTERRUPTED, STATUS_RUNNING
from probe_harness.store.db import Store

REPO_ROOT = Path(__file__).resolve().parents[2]
VICTIM = REPO_ROOT / "tests" / "fixtures" / "sigkill_victim.py"


def _counter_script(counter: Path, command: str, call_id: str = "call_inc") -> list[dict]:
    return [
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
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


def _fast_inc_cmd(counter: Path) -> str:
    return (
        "python3 -c "
        f"\"from pathlib import Path; p=Path('{counter}'); "
        'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1))"'
    )


def _wait_ready(ready: Path, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready.exists() and ready.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"ready file missing: {ready}")


def _sleep_pids_matching(fragment: str) -> list[int]:
    out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    pids: list[int] = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid_s, args = parts
        if fragment in args and "pytest" not in args:
            pids.append(int(pid_s))
    return pids


def _kill_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def _recover_from_interrupted(
    db: Path,
    task_path: Path,
    counter: Path,
    source_id: str,
) -> str:
    reset_startup_sweep_cache_for_tests()
    store = Store(db)
    await store.connect()
    await sweep_running_runs(store)
    fast_cmd = _fast_inc_cmd(counter)
    script = _counter_script(counter, fast_cmd)
    task_path.write_text(json.dumps({"user_message": "inc", "fake_script": script, "max_steps": 5}))
    recovered = await run_agent_loop(
        store,
        FakeProvider(script),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=5,
        recover_from=source_id,
    )
    await store.close()
    return recovered


@pytest.mark.parametrize("mode", ["mid_tool", "mid_model"])
def test_real_sigkill_then_recover_no_double_side_effect(tmp_path: Path, mode: str) -> None:
    case = tmp_path / mode
    case.mkdir()
    work = case / "work"
    work.mkdir()
    counter = work / "counter.txt"
    counter.write_text("0")
    db = case / "h.db"
    ready = case / "ready.txt"
    task_path = work / "task.json"
    out_id = case / "run_id.txt"

    if mode == "mid_tool":
        cmd = f"{_fast_inc_cmd(counter)} && sleep 120"
    else:
        cmd = _fast_inc_cmd(counter)

    task_path.write_text(
        json.dumps(
            {
                "user_message": "inc",
                "fake_script": _counter_script(counter, cmd),
                "max_steps": 5,
            }
        )
    )

    env = os.environ.copy()
    env.update(
        {
            "REPRO_WORK": str(work),
            "REPRO_DB": str(db),
            "REPRO_TASK": str(task_path),
            "REPRO_READY": str(ready),
            "REPRO_OUT": str(out_id),
            "REPRO_MODE": mode,
            "HARNESS_WORKSPACE": str(work),
            "HARNESS_ALLOW_TEST_CONTROLS": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("RUN_LIVE", None)

    proc = subprocess.Popen(
        [sys.executable, str(VICTIM)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_ready(ready)
        if mode == "mid_tool":
            sleep_pids: list[int] = []
            for _ in range(200):
                sleep_pids = _sleep_pids_matching("sleep 120")
                if sleep_pids:
                    break
                time.sleep(0.05)
            time.sleep(0.15)
        else:
            time.sleep(0.25)

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        _kill_pids(_sleep_pids_matching("sleep 120"))

    assert counter.read_text() == ("1" if mode == "mid_tool" else "0")

    con = sqlite3.connect(db)
    row = con.execute("SELECT id, status FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    con.close()
    assert row is not None
    assert row[1] == STATUS_RUNNING
    source_id = row[0]

    import asyncio

    recovered_id = asyncio.run(_recover_from_interrupted(db, task_path, counter, source_id))

    async def _assert_db() -> None:
        store = Store(db)
        await store.connect()
        src = await store.fetchone("SELECT status FROM runs WHERE id = ?", (source_id,))
        assert src["status"] == STATUS_INTERRUPTED
        rec = await store.fetchone(
            "SELECT recovered_from, status FROM runs WHERE id = ?", (recovered_id,)
        )
        assert rec["recovered_from"] == source_id
        assert rec["status"] == "completed"
        calls = await store.fetchall("SELECT id FROM model_calls")
        await store.close()
        assert len(calls) == 2

    asyncio.run(_assert_db())
    assert counter.read_text() == "1"
