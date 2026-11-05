"""SWE-bench predictions JSONL (C11): empty patches, resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def indexed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_predictions(path):
        iid = row.get("instance_id")
        if iid:
            out[str(iid)] = row
    return out


def prediction_line(
    *,
    instance_id: str,
    model_name_or_path: str,
    model_patch: str,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
    }


def append_prediction(
    path: Path,
    record: dict[str, Any],
    *,
    resume: bool = True,
) -> bool:
    """Append one JSONL line. Returns False if skipped due to resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    instance_id = str(record["instance_id"])
    if resume and path.is_file():
        existing = indexed_predictions(path)
        if instance_id in existing:
            return False
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return True


def write_empty_prediction(
    path: Path,
    *,
    instance_id: str,
    model_name_or_path: str,
    resume: bool = True,
) -> bool:
    return append_prediction(
        path,
        prediction_line(
            instance_id=instance_id,
            model_name_or_path=model_name_or_path,
            model_patch="",
        ),
        resume=resume,
    )
