"""Wrapper around the official SWE-bench harness (always fresh run_id)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from probe_harness.eval.docker_sandbox import install_secure_swebench_container_factory


def new_harness_run_id(prefix: str = "ph") -> str:
    """Unique run id so harness does not reuse cached instance results (C11)."""
    return f"{prefix}-{uuid.uuid4().hex}"


def run_official_harness(
    *,
    predictions_path: Path,
    instance_ids: list[str],
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
    max_workers: int = 1,
    timeout: int = 1800,
    harness_run_id: str | None = None,
    apply_d10_hardening: bool = True,
) -> dict[str, Any]:
    """Run swebench.harness.run_evaluation.main and return summary metadata."""
    from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR
    from swebench.harness.run_evaluation import main as swebench_main

    run_id = harness_run_id or new_harness_run_id()
    if apply_d10_hardening:
        install_secure_swebench_container_factory()

    swebench_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        rewrite_reports=False,
        modal=False,
        report_dir=".",
        task_repo=None,
    )

    resolved: dict[str, bool | None] = {}
    empty_patch: dict[str, bool] = {}
    for iid in instance_ids:
        pred_path = predictions_path
        preds = _load_predictions_map(pred_path)
        pred = preds.get(iid)
        if pred is None:
            resolved[iid] = None
            continue
        patch_text = pred.get("model_patch") or ""
        if not str(patch_text).strip():
            resolved[iid] = False
            empty_patch[iid] = True
            continue
        model_key = pred["model_name_or_path"].replace("/", "__")
        report_file = RUN_EVALUATION_LOG_DIR / run_id / model_key / iid / LOG_REPORT
        resolved[iid] = _read_resolved_flag(report_file, iid)
        empty_patch[iid] = False

    return {
        "harness_run_id": run_id,
        "resolved": resolved,
        "empty_patch": empty_patch,
        "log_dir": str(RUN_EVALUATION_LOG_DIR / run_id),
    }


def _load_predictions_map(path: Path) -> dict[str, dict[str, Any]]:
    from probe_harness.eval.predictions import indexed_predictions

    return indexed_predictions(path)


def _read_resolved_flag(report_file: Path, instance_id: str) -> bool | None:
    if not report_file.is_file():
        return None
    try:
        payload = json.loads(report_file.read_text(encoding="utf-8"))
        entry = payload.get(instance_id) or payload
        if isinstance(entry, dict) and "resolved" in entry:
            return bool(entry["resolved"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    return None
