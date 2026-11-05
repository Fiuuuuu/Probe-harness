"""Recovery: new run from latest checkpoint, linked by recovered_from."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probe_harness.runs.checkpoints import load_latest_checkpoint
from probe_harness.runs.journal import resolve_lineage_run_id
from probe_harness.runs.state import CHECKPOINT_POST_TOOLS, CHECKPOINT_PRE_MODEL, is_recoverable
from probe_harness.store.db import EventWriter, Store


def _config_hash(provider: str, model: str, task_path: str) -> str:
    import hashlib

    payload = json.dumps({"provider": provider, "model": model, "task": task_path}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class RecoveryContext:
    run_id: str
    source_run_id: str
    lineage_run_id: str
    task_path: Path
    messages: list[dict[str, Any]]
    step: int
    boundary: str
    fake_script_index: int
    provider_name: str
    model: str
    config_hash_value: str
    resume_at_tools: bool = False


async def recover_run(
    store: Store,
    source_run_id: str,
    *,
    provider_name: str,
    model: str,
) -> RecoveryContext:
    source = await store.fetchone("SELECT * FROM runs WHERE id = ?", (source_run_id,))
    if source is None:
        raise ValueError(f"run not found: {source_run_id}")
    if not is_recoverable(source["status"]):
        raise ValueError(f"run not recoverable (status={source['status']!r}): {source_run_id}")

    ckpt = await load_latest_checkpoint(store, source_run_id)
    if ckpt is None:
        raise ValueError(f"no checkpoint for run: {source_run_id}")

    task_path = Path(source["task_ref"])
    meta = ckpt.get("meta") or {}
    fake_script_index = int(meta.get("fake_script_index", 0))
    model_response_done = bool(meta.get("model_response_done"))
    step = int(ckpt["step"])
    boundary = str(ckpt["boundary"])
    resume_at_tools = model_response_done and boundary == CHECKPOINT_PRE_MODEL
    if boundary == CHECKPOINT_POST_TOOLS or resume_at_tools:
        if boundary == CHECKPOINT_POST_TOOLS:
            step += 1

    new_run_id = str(uuid.uuid4())
    lineage = await resolve_lineage_run_id(store, source_run_id)
    started_at = time.time()
    cfg = source["config_hash"] or _config_hash(provider_name, model, str(task_path))

    await store.execute(
        """
        INSERT INTO runs (
            id, task_ref, status, stop_reason, provider, model, config_hash,
            recovered_from, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_run_id,
            str(task_path),
            "running",
            None,
            provider_name,
            model,
            cfg,
            source_run_id,
            started_at,
        ),
    )

    events = EventWriter(store)
    await events.append(
        new_run_id,
        step,
        "run_started",
        time.time(),
        json.dumps(
            {
                "task": str(task_path),
                "recovered_from": source_run_id,
                "checkpoint_boundary": boundary,
                "checkpoint_step": ckpt["step"],
            },
            ensure_ascii=False,
        ),
    )

    return RecoveryContext(
        run_id=new_run_id,
        source_run_id=source_run_id,
        lineage_run_id=lineage,
        task_path=task_path,
        messages=list(ckpt["messages"]),
        step=step,
        boundary=boundary,
        fake_script_index=fake_script_index,
        provider_name=provider_name,
        model=model,
        config_hash_value=cfg,
        resume_at_tools=resume_at_tools,
    )
