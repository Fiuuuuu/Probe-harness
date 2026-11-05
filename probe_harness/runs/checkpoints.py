"""Checkpoint persistence at model and tool boundaries."""

from __future__ import annotations

import json
import time
from typing import Any

from probe_harness.runs.state import CHECKPOINT_POST_TOOLS, CHECKPOINT_PRE_MODEL
from probe_harness.store.db import EventWriter, Store


def checkpoint_meta(
    *,
    fake_script_index: int = 0,
    hook_metrics: dict[str, int] | None = None,
    model_response_done: bool = False,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fake_script_index": fake_script_index,
        "model_response_done": model_response_done,
    }
    if hook_metrics:
        payload["hook_metrics"] = hook_metrics
    if budget is not None:
        payload["budget"] = budget
    return payload


async def save_checkpoint(
    store: Store,
    events: EventWriter,
    *,
    run_id: str,
    step: int,
    boundary: str,
    messages: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    meta_dict = meta or {}
    usage_json = json.dumps(meta_dict, ensure_ascii=False)
    budget_payload = meta_dict.get("budget")
    if budget_payload is None:
        budget_payload = meta_dict.get("hook_metrics") or {}
    budget_json = json.dumps(budget_payload, ensure_ascii=False)
    await store.execute(
        """
        INSERT INTO checkpoints (run_id, step, boundary, messages_json, usage_json, budget_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, step, boundary) DO UPDATE SET
            messages_json = excluded.messages_json,
            usage_json = excluded.usage_json,
            budget_json = excluded.budget_json
        """,
        (
            run_id,
            step,
            boundary,
            json.dumps(messages, ensure_ascii=False),
            usage_json,
            budget_json,
        ),
    )
    await events.append(
        run_id,
        step,
        "checkpoint_saved",
        time.time(),
        json.dumps({"boundary": boundary, "step": step}, ensure_ascii=False),
    )


async def load_latest_checkpoint(store: Store, run_id: str) -> dict[str, Any] | None:
    rows = await store.fetchall(
        """
        SELECT step, boundary, messages_json, usage_json
        FROM checkpoints
        WHERE run_id = ?
        ORDER BY step DESC,
            CASE boundary WHEN ? THEN 1 WHEN ? THEN 0 ELSE -1 END DESC
        LIMIT 1
        """,
        (run_id, CHECKPOINT_POST_TOOLS, CHECKPOINT_PRE_MODEL),
    )
    if not rows:
        return None
    row = rows[0]
    meta = json.loads(row["usage_json"] or "{}")
    return {
        "step": int(row["step"]),
        "boundary": row["boundary"],
        "messages": json.loads(row["messages_json"]),
        "meta": meta,
    }
