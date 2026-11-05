"""Run metrics derived from the store."""

from __future__ import annotations

from probe_harness.store.db import Store


async def run_metrics(store: Store, run_id: str) -> dict | None:
    run = await store.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if run is None:
        return None

    rows = await store.fetchall(
        """
        SELECT input_total, cached_read, uncached, output, stop_reason
        FROM model_calls WHERE run_id = ? ORDER BY step, attempt
        """,
        (run_id,),
    )
    calls = len(rows)
    input_total = sum(r["input_total"] for r in rows)
    cached = sum(r["cached_read"] for r in rows)
    uncached = sum(r["uncached"] for r in rows)
    output = sum(r["output"] for r in rows)
    cache_ratio = (cached / input_total * 100.0) if input_total else 0.0
    stop_reason = run["stop_reason"] or ""
    if not stop_reason and rows:
        stop_reason = rows[-1]["stop_reason"] or ""

    started = run["started_at"]
    ended = run["ended_at"] or started
    agent_time_s = max(0.0, ended - started)

    return {
        "run_id": run_id,
        "status": run["status"],
        "calls": calls,
        "input_total": input_total,
        "cached_read": cached,
        "uncached": uncached,
        "output": output,
        "cache_ratio_pct": cache_ratio,
        "stop_reason": stop_reason,
        "agent_time_s": agent_time_s,
    }


def format_metrics_table(metrics: dict) -> str:
    lines = [
        "| metric | value |",
        "|---|---|",
        f"| model_calls | {metrics['calls']} |",
        f"| input_total | {metrics['input_total']} |",
        f"| cached_read | {metrics['cached_read']} |",
        f"| uncached | {metrics['uncached']} |",
        f"| output | {metrics['output']} |",
        f"| cache_ratio | {metrics['cache_ratio_pct']:.1f}% |",
        f"| stop_reason | {metrics['stop_reason']} |",
        f"| agent_time_s | {metrics['agent_time_s']:.3f} |",
    ]
    return "\n".join(lines)
