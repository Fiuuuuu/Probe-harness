"""SWE-bench eval runner: agent → predictions JSONL → official harness."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probe_harness.core.loop import run_agent_loop
from probe_harness.eval.docker_sandbox import is_docker_available
from probe_harness.eval.harness_wrapper import new_harness_run_id, run_official_harness
from probe_harness.eval.patches import collect_workspace_patch
from probe_harness.eval.predictions import (
    append_prediction,
    indexed_predictions,
    prediction_line,
    write_empty_prediction,
)
from probe_harness.providers.ledger import TokenCapReached, UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.store.db import Store

DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL_PATH = "probe-harness/deepseek-flash"
_FIXTURE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_FIXTURE = _FIXTURE_ROOT / "tests/fixtures/swebench_eval_task.json"


@dataclass
class InstanceEvalResult:
    instance_id: str
    agent_run_id: str
    patch: str
    patch_files: list[str]
    patch_sha256: str
    harness_run_id: str | None
    resolved: bool | None
    eval_log_path: str | None
    skipped_agent: bool = False
    agent_error: str | None = None


@dataclass
class EvalBatchResult:
    batch_run_id: str
    predictions_path: Path
    instances: list[InstanceEvalResult]
    harness_skipped_reason: str | None = None


def require_run_eval() -> None:
    if os.environ.get("RUN_EVAL") != "1":
        raise SystemExit("RUN_EVAL=1 is required for harness eval commands")


def default_predictions_path(batch_run_id: str) -> Path:
    return Path(".harness") / "eval" / batch_run_id / "predictions.jsonl"


def load_swebench_instance(
    instance_id: str,
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    from swebench.harness.utils import load_swebench_dataset

    rows = load_swebench_dataset(dataset_name, split, [instance_id])
    if not rows:
        raise ValueError(f"instance not found in dataset: {instance_id}")
    return rows[0]


def build_eval_task(
    instance: dict[str, Any],
    *,
    task_fixture: Path | None = None,
) -> dict[str, Any]:
    path = task_fixture or DEFAULT_TASK_FIXTURE
    task = json.loads(path.read_text(encoding="utf-8"))
    template = task.pop("user_message_template", None)
    problem = instance.get("problem_statement", "")
    if template:
        task["user_message"] = str(template).format(problem_statement=problem)
    elif "user_message" not in task:
        task["user_message"] = problem
    task["instance_id"] = instance["instance_id"]
    task["repo"] = instance.get("repo")
    return task


def prepare_instance_workspace(instance: dict[str, Any], workspace: Path) -> None:
    repo = str(instance["repo"])
    base_commit = str(instance["base_commit"])
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        clone_url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(workspace)],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(["git", "fetch", "origin", base_commit], cwd=workspace, check=False)
    subprocess.run(["git", "checkout", "-f", base_commit], cwd=workspace, check=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace, check=True)


async def _run_agent_on_instance(
    store: Store,
    *,
    task: dict[str, Any],
    task_path: Path,
    agent_run_id: str,
    instance_id: str,
) -> tuple[str, str | None]:
    ledger = UsageLedger(task_id="harness-eval")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=bool(task.get("thinking_enabled", True)),
        max_tokens=int(task.get("max_tokens", 8192)),
        test_run_id=f"eval-{agent_run_id}-{instance_id}",
    )
    try:
        run_id = await run_agent_loop(
            store,
            provider,
            task_path,
            provider_name="deepseek",
            model="deepseek-flash",
            max_steps=int(task.get("max_steps", 50)),
            run_id=agent_run_id,
        )
        row = await store.fetchone(
            "SELECT stop_reason FROM runs WHERE id = ?",
            (run_id,),
        )
        return run_id, row["stop_reason"] if row else None
    except TokenCapReached:
        raise
    finally:
        await provider.aclose()


async def _ensure_eval_run_stub(
    store: Store,
    agent_run_id: str,
    instance_id: str,
    agent_error: str | None,
) -> None:
    row = await store.fetchone("SELECT id FROM runs WHERE id = ?", (agent_run_id,))
    if row:
        return
    now = time.time()
    await store.execute(
        """
        INSERT INTO runs (
            id, task_ref, status, stop_reason, provider, model, config_hash, started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_run_id,
            instance_id,
            "failed",
            agent_error or "eval_setup_failed",
            "deepseek",
            "deepseek-flash",
            "",
            now,
            now,
        ),
    )


async def record_eval_result(
    store: Store,
    *,
    agent_run_id: str,
    instance_id: str,
    patch: str,
    patch_files: list[str],
    resolved: bool | None,
    harness_run_id: str | None,
    eval_log_path: str | None,
) -> None:
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else ""
    resolved_int = None if resolved is None else int(bool(resolved))
    await store.execute(
        """
        INSERT INTO eval_results (
            run_id, instance_id, patch_sha256, patch_files, resolved,
            harness_run_id, eval_log_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, instance_id) DO UPDATE SET
            patch_sha256=excluded.patch_sha256,
            patch_files=excluded.patch_files,
            resolved=excluded.resolved,
            harness_run_id=excluded.harness_run_id,
            eval_log_path=excluded.eval_log_path
        """,
        (
            agent_run_id,
            instance_id,
            digest,
            json.dumps(patch_files),
            resolved_int,
            harness_run_id,
            eval_log_path,
        ),
    )


async def run_eval_batch(
    *,
    instance_ids: list[str],
    batch_run_id: str,
    store: Store,
    predictions_path: Path | None = None,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    run_harness: bool = True,
    task_fixture: Path | None = None,
    model_name_or_path: str = DEFAULT_MODEL_PATH,
    resume: bool = True,
) -> EvalBatchResult:
    pred_path = predictions_path or default_predictions_path(batch_run_id)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    harness_skip: str | None = None
    if run_harness and not is_docker_available():
        run_harness = False
        harness_skip = "docker not available"

    results: list[InstanceEvalResult] = []
    completed = indexed_predictions(pred_path) if resume else {}

    for instance_id in instance_ids:
        if resume and instance_id in completed:
            continue
        agent_run_id = batch_run_id if len(instance_ids) == 1 else f"{batch_run_id}-{instance_id}"
        agent_error: str | None = None
        patch = ""
        patch_files: list[str] = []
        harness_run_id: str | None = None
        eval_log_path: str | None = None
        resolved: bool | None = None

        try:
            instance = load_swebench_instance(instance_id, dataset_name=dataset_name, split=split)
            workspace = pred_path.parent / "workspaces" / instance_id
            task_path = pred_path.parent / f"task_{instance_id}.json"
            prepare_instance_workspace(instance, workspace)
            task = build_eval_task(instance, task_fixture=task_fixture)
            task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
            prev_ws = os.environ.get("HARNESS_WORKSPACE")
            os.environ["HARNESS_WORKSPACE"] = str(workspace)
            try:
                agent_run_id, _stop_reason = await _run_agent_on_instance(
                    store,
                    task=task,
                    task_path=task_path,
                    agent_run_id=agent_run_id,
                    instance_id=instance_id,
                )
            finally:
                if prev_ws is None:
                    os.environ.pop("HARNESS_WORKSPACE", None)
                else:
                    os.environ["HARNESS_WORKSPACE"] = prev_ws

            patch, patch_files = collect_workspace_patch(workspace)
            append_prediction(
                pred_path,
                prediction_line(
                    instance_id=instance_id,
                    model_name_or_path=model_name_or_path,
                    model_patch=patch,
                ),
                resume=True,
            )
        except TokenCapReached as exc:
            agent_error = str(exc)
            write_empty_prediction(
                pred_path,
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                resume=True,
            )
        except Exception as exc:
            agent_error = str(exc)
            write_empty_prediction(
                pred_path,
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                resume=True,
            )

        if run_harness and agent_error is None:
            harness_run_id = new_harness_run_id(batch_run_id[:8])
            summary = run_official_harness(
                predictions_path=pred_path,
                instance_ids=[instance_id],
                dataset_name=dataset_name,
                split=split,
                harness_run_id=harness_run_id,
            )
            resolved = summary["resolved"].get(instance_id)
            eval_log_path = summary["log_dir"]

        await _ensure_eval_run_stub(store, agent_run_id, instance_id, agent_error)
        await record_eval_result(
            store,
            agent_run_id=agent_run_id,
            instance_id=instance_id,
            patch=patch,
            patch_files=patch_files,
            resolved=resolved,
            harness_run_id=harness_run_id,
            eval_log_path=eval_log_path,
        )

        results.append(
            InstanceEvalResult(
                instance_id=instance_id,
                agent_run_id=agent_run_id,
                patch=patch,
                patch_files=patch_files,
                patch_sha256=hashlib.sha256(patch.encode()).hexdigest() if patch else "",
                harness_run_id=harness_run_id,
                resolved=resolved,
                eval_log_path=eval_log_path,
                agent_error=agent_error,
            )
        )

        if agent_error and "token cap" in agent_error.lower():
            break

    return EvalBatchResult(
        batch_run_id=batch_run_id,
        predictions_path=pred_path,
        instances=results,
        harness_skipped_reason=harness_skip,
    )


def new_batch_run_id(prefix: str = "eval") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
