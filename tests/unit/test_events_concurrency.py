from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from probe_harness.store.db import EventWriter, Store


@pytest.mark.asyncio
async def test_event_seq_unique_under_concurrency(tmp_path) -> None:
    db = tmp_path / "test.db"
    store = Store(db)
    await store.connect()
    run_id = str(uuid.uuid4())
    await store.execute(
        "INSERT INTO runs (id, task_ref, status, provider, started_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, "t", "running", "fake", 0.0),
    )

    writer = EventWriter(store)
    seen: list[int] = []

    async def worker(n: int) -> None:
        for i in range(n):
            seq = await writer.append(run_id, 0, "tool_call", 0.0, json.dumps({"i": i}))
            seen.append(seq)

    await asyncio.gather(worker(500), worker(500))
    assert len(seen) == 1000
    assert len(set(seen)) == 1000
    assert min(seen) == 0
    assert max(seen) == 999
    await store.close()
