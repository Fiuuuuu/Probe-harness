"""Per-call journal so side-effecting tools are never replayed after crash/recovery."""

from __future__ import annotations

import json
import time
from typing import Any

from probe_harness.runs.state import is_side_effecting_tool
from probe_harness.store.db import Store

JOURNAL_STARTED = "started"
JOURNAL_COMPLETED = "completed"
JOURNAL_UNCERTAIN = "uncertain"


async def resolve_lineage_run_id(store: Store, run_id: str) -> str:
    row = await store.fetchone("SELECT recovered_from FROM runs WHERE id = ?", (run_id,))
    if row is None:
        return run_id
    recovered_from = row["recovered_from"]
    if not recovered_from:
        return run_id
    return await resolve_lineage_run_id(store, recovered_from)


async def journal_mark_started(
    store: Store,
    *,
    lineage_run_id: str,
    run_id: str,
    call_id: str,
    norm_name: str,
    args_sha256: str,
) -> None:
    if not is_side_effecting_tool(norm_name):
        return
    existing = await journal_lookup(store, lineage_run_id=lineage_run_id, call_id=call_id)
    if existing is not None:
        return
    await store.execute(
        """
        INSERT INTO tool_journal (
            lineage_run_id, run_id, call_id, norm_name, args_sha256, status, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lineage_run_id,
            run_id,
            call_id,
            norm_name,
            args_sha256,
            JOURNAL_STARTED,
            time.time(),
        ),
    )


async def journal_mark_completed(
    store: Store,
    *,
    lineage_run_id: str,
    run_id: str,
    call_id: str,
    norm_name: str,
    args_sha256: str,
    result: dict[str, Any],
) -> None:
    if not is_side_effecting_tool(norm_name):
        return
    await store.execute(
        """
        INSERT INTO tool_journal (
            lineage_run_id, run_id, call_id, norm_name, args_sha256, status,
            result_json, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lineage_run_id, call_id) DO UPDATE SET
            status = excluded.status,
            result_json = excluded.result_json,
            run_id = excluded.run_id,
            ts = excluded.ts
        """,
        (
            lineage_run_id,
            run_id,
            call_id,
            norm_name,
            args_sha256,
            JOURNAL_COMPLETED,
            json.dumps(result, ensure_ascii=False),
            time.time(),
        ),
    )


async def journal_mark_uncertain(
    store: Store,
    *,
    lineage_run_id: str,
    run_id: str,
    call_id: str,
) -> None:
    await store.execute(
        """
        UPDATE tool_journal
        SET status = ?, run_id = ?, ts = ?
        WHERE lineage_run_id = ? AND call_id = ?
        """,
        (JOURNAL_UNCERTAIN, run_id, time.time(), lineage_run_id, call_id),
    )


async def journal_lookup(
    store: Store,
    *,
    lineage_run_id: str,
    call_id: str,
) -> dict[str, Any] | None:
    row = await store.fetchone(
        """
        SELECT status, result_json, norm_name, args_sha256
        FROM tool_journal
        WHERE lineage_run_id = ? AND call_id = ?
        """,
        (lineage_run_id, call_id),
    )
    if row is None:
        return None
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return {
        "status": row["status"],
        "result": result,
        "norm_name": row["norm_name"],
        "args_sha256": row["args_sha256"],
    }


def uncertain_tool_result(
    call_id: str,
    *,
    reason: str = "outcome uncertain after interrupt",
) -> dict[str, Any]:
    return {
        "error": f"tool call not replayed: {reason}",
        "is_error": True,
        "outcome_uncertain": True,
        "call_id": call_id,
    }
