"""Subprocess entrypoint for real SIGKILL integration tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.runs.service import reset_startup_sweep_cache_for_tests
from probe_harness.store.db import Store


async def main() -> None:
    work = Path(os.environ["REPRO_WORK"])
    db = Path(os.environ["REPRO_DB"])
    task_path = Path(os.environ["REPRO_TASK"])
    mode = os.environ.get("REPRO_MODE", "mid_tool")
    ready = Path(os.environ["REPRO_READY"])

    os.environ["HARNESS_WORKSPACE"] = str(work)
    os.environ["HARNESS_ALLOW_TEST_CONTROLS"] = "1"
    reset_startup_sweep_cache_for_tests()

    task = json.loads(task_path.read_text())
    script = task["fake_script"]

    if mode == "mid_model":
        orig_complete = FakeProvider.complete

        async def slow_complete(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            ready.write_text(f"mid_model_enter pid={os.getpid()} t={time.time()}")
            await asyncio.sleep(60)
            return await orig_complete(self, *args, **kwargs)

        FakeProvider.complete = slow_complete  # type: ignore[method-assign]

    store = Store(db)
    await store.connect()
    if mode == "mid_tool":
        ready.write_text(f"store_ready pid={os.getpid()} t={time.time()}")

    try:
        run_id = await run_agent_loop(
            store,
            FakeProvider(script),
            task_path,
            provider_name="fake",
            model="fake",
            max_steps=5,
        )
        Path(os.environ["REPRO_OUT"]).write_text(run_id)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
