from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.store.db import Store
from probe_harness.store.metrics import run_metrics


@pytest.mark.asyncio
async def test_fake_echo_task_writes_rows(tmp_path: Path) -> None:
    db = tmp_path / "h.db"
    store = Store(db)
    await store.connect()
    task = Path("tests/fixtures/echo_task.json")
    script = json.loads(task.read_text())["fake_script"]
    provider = FakeProvider(script)
    run_id = await run_agent_loop(store, provider, task, provider_name="fake", model="fake")

    run = await store.fetchone("SELECT status FROM runs WHERE id = ?", (run_id,))
    assert run["status"] == "completed"

    events = await store.fetchall("SELECT seq FROM events WHERE run_id = ? ORDER BY seq", (run_id,))
    assert len(events) >= 4

    calls = await store.fetchall("SELECT * FROM model_calls WHERE run_id = ?", (run_id,))
    assert len(calls) == 3

    tools = await store.fetchall("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,))
    assert len(tools) == 2

    metrics = await run_metrics(store, run_id)
    assert metrics is not None
    assert metrics["calls"] == 3
    assert metrics["input_total"] == 120 + 200 + 350
    await store.close()
