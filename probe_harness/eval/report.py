"""Eval report derived from store + eval_results."""

from __future__ import annotations

import json
from typing import Any

from probe_harness.store.db import Store
from probe_harness.store.metrics import run_metrics


async def fetch_eval_rows(store: Store, agent_run_id: str) -> list[dict[str, Any]]:
    rows = await store.fetchall(
        """
        SELECT instance_id, patch_sha256, patch_files, resolved, harness_run_id, eval_log_path
        FROM eval_results WHERE run_id = ?
        ORDER BY instance_id
        """,
        (agent_run_id,),
    )
    return [dict(r) for r in rows]


async def eval_report_data(store: Store, agent_run_id: str) -> dict[str, Any] | None:
    run = await store.fetchone("SELECT * FROM runs WHERE id = ?", (agent_run_id,))
    if run is None:
        return None

    metrics = await run_metrics(store, agent_run_id)
    eval_rows = await fetch_eval_rows(store, agent_run_id)

    protocol_repairs = await store.fetchone(
        """
        SELECT COUNT(*) AS n FROM events
        WHERE run_id = ? AND type = 'protocol_repair'
        """,
        (agent_run_id,),
    )
    tool_errors = await store.fetchone(
        """
        SELECT COUNT(*) AS n FROM tool_calls
        WHERE run_id = ? AND status = 'error'
        """,
        (agent_run_id,),
    )
    first_pass_step = await _steps_after_first_passing_repro(store, agent_run_id)

    empty_patches = sum(
        1 for row in eval_rows if not row.get("patch_sha256") and row.get("resolved") in (0, None)
    )
    resolved_count = sum(1 for row in eval_rows if row.get("resolved") == 1)

    return {
        "agent_run_id": agent_run_id,
        "status": run["status"],
        "stop_reason": run["stop_reason"] or (metrics or {}).get("stop_reason", ""),
        "metrics": metrics,
        "eval_instances": eval_rows,
        "resolved_count": resolved_count,
        "instance_count": len(eval_rows),
        "empty_patch_count": empty_patches,
        "protocol_repairs": int(protocol_repairs["n"]) if protocol_repairs else 0,
        "tool_errors": int(tool_errors["n"]) if tool_errors else 0,
        "steps_after_first_passing_repro": first_pass_step,
    }


async def _steps_after_first_passing_repro(store: Store, run_id: str) -> int | None:
    rows = await store.fetchall(
        """
        SELECT step, data_json FROM events
        WHERE run_id = ? AND type = 'goal_check'
        ORDER BY seq
        """,
        (run_id,),
    )
    first_pass: int | None = None
    for row in rows:
        try:
            data = json.loads(row["data_json"])
        except json.JSONDecodeError:
            continue
        if data.get("passed"):
            first_pass = int(row["step"])
            break
    if first_pass is None:
        return None
    last_step_row = await store.fetchone(
        "SELECT MAX(step) AS mx FROM events WHERE run_id = ?",
        (run_id,),
    )
    if last_step_row is None or last_step_row["mx"] is None:
        return None
    return max(0, int(last_step_row["mx"]) - first_pass)


def format_eval_report(data: dict[str, Any]) -> str:
    metrics = data.get("metrics") or {}
    lines = [
        "# eval report",
        "",
        f"agent_run_id: {data['agent_run_id']}",
        f"status: {data.get('status', '')}",
        f"stop_reason: {data.get('stop_reason', '')}",
        f"resolved: {data.get('resolved_count', 0)}/{data.get('instance_count', 0)}",
        f"empty_patch_count: {data.get('empty_patch_count', 0)}",
        f"protocol_repairs: {data.get('protocol_repairs', 0)}",
        f"tool_errors: {data.get('tool_errors', 0)}",
        f"steps_after_first_passing_repro: {data.get('steps_after_first_passing_repro')}",
        "",
        "## agent metrics",
        "| metric | value |",
        "|---|---|",
        f"| model_calls | {metrics.get('calls', 0)} |",
        f"| input_total | {metrics.get('input_total', 0)} |",
        f"| cached_read | {metrics.get('cached_read', 0)} |",
        f"| uncached | {metrics.get('uncached', 0)} |",
        f"| output | {metrics.get('output', 0)} |",
        f"| cache_ratio | {metrics.get('cache_ratio_pct', 0.0):.1f}% |",
        f"| agent_time_s | {metrics.get('agent_time_s', 0.0):.3f} |",
        "",
        "## instances",
    ]
    for row in data.get("eval_instances") or []:
        patch_files = row.get("patch_files") or "[]"
        lines.append(
            f"- {row['instance_id']}: resolved={row.get('resolved')} "
            f"harness_run_id={row.get('harness_run_id')} patch_files={patch_files}"
        )
    return "\n".join(lines)
