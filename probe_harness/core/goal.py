"""Minimal task goal checks (learn-claude-code s10/s17 mechanics)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoalCheckResult:
    passed: bool
    feedback: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "feedback": self.feedback, **self.details}


def _workspace() -> Path:
    return Path(os.environ.get("HARNESS_WORKSPACE", os.getcwd())).resolve()


def _resolve_in_workspace(rel: str) -> Path | None:
    ws = _workspace()
    try:
        candidate = (ws / rel).resolve()
    except OSError:
        return None
    if not candidate.is_relative_to(ws):
        return None
    return candidate


def _check_one(spec: dict[str, Any]) -> tuple[bool, str]:
    kind = spec.get("type") or spec.get("kind")
    if kind == "always_pass":
        return True, "goal check passed (always_pass)"
    if kind == "always_fail":
        return False, str(spec.get("message", "goal check failed (always_fail)"))
    if kind == "file_exists":
        path = str(spec.get("path", ""))
        p = _resolve_in_workspace(path)
        if p is None:
            return False, f"path outside workspace: {path}"
        if not p.is_file():
            return False, f"required file missing: {path}"
        return True, f"file exists: {path}"
    if kind == "file_contains":
        path = str(spec.get("path", ""))
        needle = spec.get("text") or spec.get("substring") or ""
        p = _resolve_in_workspace(path)
        if p is None or not p.is_file():
            return False, f"cannot read file for contains check: {path}"
        body = p.read_text(encoding="utf-8", errors="replace")
        if str(needle) not in body:
            return False, f"file {path!r} does not contain required text"
        return True, f"file contains required text: {path}"
    if kind == "assistant_contains":
        needle = str(spec.get("text", ""))
        content = str(spec.get("assistant_content") or "")
        if needle and needle not in content:
            return False, "assistant answer missing required text"
        return True, "assistant answer contains required text"
    return False, f"unknown goal check type: {kind!r}"


def evaluate_goal(task: dict[str, Any], *, assistant_content: str | None = None) -> GoalCheckResult:
    goal = task.get("goal") or task.get("success_criteria")
    if goal is None:
        return GoalCheckResult(
            passed=True,
            feedback="no goal configured",
            details={"checks": []},
        )

    checks: list[dict[str, Any]]
    if isinstance(goal, list):
        checks = goal
    elif isinstance(goal, dict) and "checks" in goal:
        checks = list(goal["checks"])
    elif isinstance(goal, dict) and goal.get("type"):
        checks = [goal]
    else:
        return GoalCheckResult(
            passed=False,
            feedback="invalid goal configuration",
            details={"goal": goal},
        )

    results: list[dict[str, Any]] = []
    for spec in checks:
        enriched = dict(spec)
        if enriched.get("type") == "assistant_contains" and assistant_content is not None:
            enriched["assistant_content"] = assistant_content
        ok, msg = _check_one(enriched)
        results.append({"spec": spec, "passed": ok, "message": msg})
        if not ok:
            return GoalCheckResult(
                passed=False,
                feedback=msg,
                details={"checks": results},
            )

    return GoalCheckResult(
        passed=True,
        feedback="all goal checks passed",
        details={"checks": results},
    )


def goal_feedback_message(result: GoalCheckResult) -> str:
    payload = result.as_dict()
    return "Goal check failed:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
