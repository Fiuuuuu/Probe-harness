"""Run lifecycle: startup sweep and run row helpers."""

from __future__ import annotations

import time

from probe_harness.runs.state import STATUS_INTERRUPTED, STATUS_RUNNING
from probe_harness.store.db import Store

_sweep_done_for: set[str] = set()


async def sweep_running_runs(store: Store, *, reason: str = "host_restart") -> int:
    """Mark every in-flight run interrupted (never silently resume)."""
    ended = time.time()
    cur = await store.execute(
        """
        UPDATE runs
        SET status = ?, stop_reason = ?, ended_at = COALESCE(ended_at, ?)
        WHERE status = ?
        """,
        (STATUS_INTERRUPTED, reason, ended, STATUS_RUNNING),
    )
    return cur.rowcount or 0


async def ensure_startup_sweep(store: Store) -> None:
    key = str(store.db_path.resolve())
    if key in _sweep_done_for:
        return
    await sweep_running_runs(store)
    _sweep_done_for.add(key)


def reset_startup_sweep_cache_for_tests() -> None:
    _sweep_done_for.clear()
