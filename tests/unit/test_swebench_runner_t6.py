"""T6 SWE-bench runner unit tests (offline, Docker mocked)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from probe_harness.eval.cost import deepseek_flash_cost_usd
from probe_harness.eval.docker_sandbox import (
    DOCKER_HARDENING_CREATE_KWARGS,
    DOCKER_HARDENING_RUN_ARGS,
    is_docker_available,
)
from probe_harness.eval.harness_wrapper import new_harness_run_id, run_official_harness
from probe_harness.eval.patches import collect_workspace_patch, is_hygienic_path
from probe_harness.eval.predictions import (
    append_prediction,
    indexed_predictions,
    write_empty_prediction,
)
from probe_harness.eval.report import eval_report_data, format_eval_report
from probe_harness.eval.swebench_runner import (
    build_eval_task,
    record_eval_result,
    run_eval_batch,
)
from probe_harness.store.db import Store


def test_hygiene_filter_excludes_harness_artifacts() -> None:
    assert is_hygienic_path("src/foo.py") is True
    assert is_hygienic_path(".harness/usage_ledger.jsonl") is False
    assert is_hygienic_path("tmp/scratch.txt") is False


def test_collect_workspace_patch_filters_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@test.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "good.py").write_text("a\n", encoding="utf-8")
    (repo / ".harness").mkdir()
    (repo / ".harness" / "x.txt").write_text("secret\n", encoding="utf-8")
    subprocess.run(["git", "add", "good.py", ".harness/x.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    (repo / "good.py").write_text("b\n", encoding="utf-8")

    patch, files = collect_workspace_patch(repo)
    assert files == ["good.py"]
    assert "good.py" in patch
    assert ".harness" not in patch


def test_predictions_jsonl_resume(tmp_path: Path) -> None:
    path = tmp_path / "pred.jsonl"
    ok = append_prediction(
        path,
        {
            "instance_id": "i1",
            "model_name_or_path": "m",
            "model_patch": "p",
        },
    )
    assert ok is True
    ok2 = append_prediction(
        path,
        {
            "instance_id": "i1",
            "model_name_or_path": "m",
            "model_patch": "other",
        },
        resume=True,
    )
    assert ok2 is False
    assert indexed_predictions(path)["i1"]["model_patch"] == "p"

    write_empty_prediction(path, instance_id="i2", model_name_or_path="m")
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[1])["model_patch"] == ""


def test_new_harness_run_id_unique() -> None:
    a = new_harness_run_id()
    b = new_harness_run_id()
    assert a != b


def test_docker_hardening_constants() -> None:
    assert "cap_drop" in DOCKER_HARDENING_CREATE_KWARGS
    assert "--cap-drop" in DOCKER_HARDENING_RUN_ARGS
    assert "--pids-limit" in DOCKER_HARDENING_RUN_ARGS


@patch("probe_harness.eval.docker_sandbox.subprocess.run")
@patch("probe_harness.eval.docker_sandbox.shutil.which")
def test_is_docker_available(mock_which, mock_run) -> None:
    mock_which.return_value = None
    assert is_docker_available() is False
    mock_which.return_value = "/usr/bin/docker"
    mock_run.return_value = subprocess.CompletedProcess([], 1, "", "")
    assert is_docker_available() is False
    mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
    assert is_docker_available() is True


def test_build_eval_task(tmp_path: Path) -> None:
    fixture = tmp_path / "task.json"
    fixture.write_text(
        json.dumps(
            {
                "system": "s",
                "user_message_template": "Problem:\n{problem_statement}",
            }
        ),
        encoding="utf-8",
    )
    task = build_eval_task(
        {"instance_id": "x", "problem_statement": "bug", "repo": "r/r"},
        task_fixture=fixture,
    )
    assert "bug" in task["user_message"]
    assert task["instance_id"] == "x"


@patch("swebench.harness.run_evaluation.main")
@patch("probe_harness.eval.harness_wrapper.install_secure_swebench_container_factory")
def test_run_official_harness_invokes_swebench(mock_install, mock_main, tmp_path: Path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "instance_id": "pytest-dev__pytest-7432",
                "model_name_or_path": "probe-harness/deepseek-flash",
                "model_patch": "diff",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_id = "ph-test123"
    with patch(
        "probe_harness.eval.harness_wrapper._read_resolved_flag",
        return_value=True,
    ):
        summary = run_official_harness(
            predictions_path=pred,
            instance_ids=["pytest-dev__pytest-7432"],
            harness_run_id=run_id,
        )
    assert summary["harness_run_id"] == run_id
    mock_install.assert_called_once()
    mock_main.assert_called_once()
    assert mock_main.call_args.kwargs["run_id"] == run_id


@pytest.mark.asyncio
async def test_eval_report(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite")
    await store.connect()
    now = 1.0
    await store.execute(
        """
        INSERT INTO runs (
            id, task_ref, status, stop_reason, provider, model, config_hash, started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run1",
            "task",
            "completed",
            "submitted",
            "deepseek",
            "deepseek-flash",
            "abc",
            now,
            now + 1,
        ),
    )
    await record_eval_result(
        store,
        agent_run_id="run1",
        instance_id="pytest-dev__pytest-7432",
        patch="diff",
        patch_files=["a.py"],
        resolved=True,
        harness_run_id="ph-1",
        eval_log_path="/tmp/log",
    )
    data = await eval_report_data(store, "run1")
    assert data is not None
    text = format_eval_report(data)
    assert "resolved: 1/1" in text
    await store.close()


@pytest.mark.asyncio
@patch("probe_harness.eval.swebench_runner.is_docker_available", return_value=False)
@patch("probe_harness.eval.swebench_runner.prepare_instance_workspace")
@patch("probe_harness.eval.swebench_runner.load_swebench_instance")
@patch("probe_harness.eval.swebench_runner._run_agent_on_instance")
async def test_run_eval_batch_skips_harness_without_docker(
    mock_agent,
    mock_load,
    mock_prepare,
    _mock_docker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN_EVAL", "1")
    mock_load.return_value = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "base_commit": "abc",
        "problem_statement": "fix",
    }
    mock_agent.return_value = ("run-eval-1", None)

    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, check=True)
    (ws / "f.py").write_text("1\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.py"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-m", "m"], cwd=ws, check=True)
    (ws / "f.py").write_text("2\n", encoding="utf-8")

    def _prep(instance, workspace):
        import shutil

        workspace.mkdir(parents=True, exist_ok=True)
        for item in ws.iterdir():
            dest = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    mock_prepare.side_effect = _prep

    store = Store(tmp_path / "db.sqlite")
    await store.connect()
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "swebench_eval_task.json"
    result = await run_eval_batch(
        instance_ids=["pytest-dev__pytest-7432"],
        batch_run_id="run-eval-1",
        store=store,
        predictions_path=tmp_path / "pred.jsonl",
        task_fixture=fixture,
    )
    await store.close()

    assert result.harness_skipped_reason == "docker not available"
    assert result.instances[0].patch_files == ["f.py"]
    assert result.predictions_path.is_file()


@patch("probe_harness.eval.docker_sandbox.subprocess.run")
def test_install_secure_container_factory_sets_create_kwargs(mock_run) -> None:
    import swebench.harness.run_evaluation as run_eval

    from probe_harness.eval.docker_sandbox import install_secure_swebench_container_factory

    mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
    original = run_eval.create_container
    install_secure_swebench_container_factory()
    assert run_eval.create_container is not original
    client = MagicMock()
    client.images.get.return_value = object()
    container = MagicMock()
    container.id = "cid"
    client.containers.create.return_value = container

    import docker.errors

    client.containers.get.side_effect = docker.errors.NotFound("missing")
    logger = MagicMock()
    spec = MagicMock()
    spec.instance_id = "Org__repo-1"
    spec.image = "img"
    run_eval.create_container(spec, client, "run123", logger)
    kwargs = client.containers.create.call_args.kwargs
    assert kwargs["cap_drop"] == DOCKER_HARDENING_CREATE_KWARGS["cap_drop"]
    assert kwargs["pids_limit"] == 512
    assert kwargs.get("network_mode") == "none"


def test_deepseek_cost_off_peak() -> None:
    cost = deepseek_flash_cost_usd(cached_read=1792, uncached=180, output=50)
    assert cost > 0


@patch("swebench.harness.run_evaluation.main")
@patch("probe_harness.eval.harness_wrapper.install_secure_swebench_container_factory")
def test_empty_patch_resolved_false(mock_install, mock_main, tmp_path: Path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "instance_id": "pytest-dev__pytest-7432",
                "model_name_or_path": "probe-harness/deepseek-flash",
                "model_patch": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = run_official_harness(
        predictions_path=pred,
        instance_ids=["pytest-dev__pytest-7432"],
        harness_run_id="empty-test",
    )
    assert summary["resolved"]["pytest-dev__pytest-7432"] is False
    assert summary["empty_patch"]["pytest-dev__pytest-7432"] is True
    mock_main.assert_called_once()


@pytest.mark.asyncio
@patch("probe_harness.eval.swebench_runner.run_official_harness")
@patch("probe_harness.eval.swebench_runner.is_docker_available", return_value=True)
@patch("probe_harness.eval.swebench_runner._run_agent_on_instance")
async def test_resume_runs_harness_only_after_prediction(
    mock_agent,
    _mock_docker,
    mock_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN_EVAL", "1")
    pred_path = tmp_path / "pred.jsonl"
    append_prediction(
        pred_path,
        {
            "instance_id": "pytest-dev__pytest-7432",
            "model_name_or_path": "probe-harness/fake",
            "model_patch": "diff --git a/x b/x",
        },
    )
    mock_harness.return_value = {
        "harness_run_id": "ph-resume",
        "resolved": {"pytest-dev__pytest-7432": True},
        "empty_patch": {"pytest-dev__pytest-7432": False},
        "log_dir": "/tmp/log",
    }

    store = Store(tmp_path / "db.sqlite")
    await store.connect()
    result = await run_eval_batch(
        instance_ids=["pytest-dev__pytest-7432"],
        batch_run_id="batch-resume",
        store=store,
        predictions_path=pred_path,
        run_harness=True,
        resume=True,
    )
    await store.close()

    mock_agent.assert_not_called()
    mock_harness.assert_called_once()
    assert len(result.instances) == 1
    assert result.instances[0].skipped_agent is True
    assert result.instances[0].resolved is True


@pytest.mark.asyncio
@patch("probe_harness.eval.swebench_runner.run_official_harness")
@patch("probe_harness.eval.swebench_runner.is_docker_available", return_value=False)
@patch("probe_harness.eval.swebench_runner.prepare_instance_workspace")
@patch("probe_harness.eval.swebench_runner.load_swebench_instance")
async def test_fake_provider_batch_offline(
    mock_load,
    mock_prepare,
    _mock_docker,
    mock_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN_EVAL", "1")
    mock_load.return_value = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "base_commit": "abc",
        "problem_statement": "fix",
    }

    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, check=True)
    (ws / "f.py").write_text("1\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.py"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-m", "m"], cwd=ws, check=True)
    (ws / "f.py").write_text("2\n", encoding="utf-8")

    def _prep(_instance, workspace):
        import shutil

        workspace.mkdir(parents=True, exist_ok=True)
        for item in ws.iterdir():
            dest = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    mock_prepare.side_effect = _prep

    fixture = tmp_path / "task.json"
    fixture.write_text(
        json.dumps(
            {
                "system": "s",
                "user_message": "edit f.py",
                "fake_script": [
                    {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({"path": "f.py", "content": "2\n"}),
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                    {
                        "content": "done",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    store = Store(tmp_path / "db.sqlite")
    await store.connect()
    result = await run_eval_batch(
        instance_ids=["pytest-dev__pytest-7432"],
        batch_run_id="fake-batch",
        store=store,
        predictions_path=tmp_path / "pred.jsonl",
        task_fixture=fixture,
        provider="fake",
        run_harness=False,
    )
    await store.close()

    mock_harness.assert_not_called()
    assert result.instances[0].patch_files == ["f.py"]
    line = json.loads(result.predictions_path.read_text(encoding="utf-8").strip())
    assert line["model_name_or_path"] == "probe-harness/fake"


@pytest.mark.asyncio
async def test_eval_report_includes_cost_and_stop_reason(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite")
    await store.connect()
    now = 1.0
    await store.execute(
        """
        INSERT INTO runs (
            id, task_ref, status, stop_reason, provider, model, config_hash, started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run1",
            "task",
            "submitted",
            "goal_met",
            "fake",
            "fake",
            "abc",
            now,
            now + 1,
        ),
    )
    await store.execute(
        """
        INSERT INTO model_calls (
            run_id, step, attempt, input_total, cached_read, uncached, output, stop_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run1", 1, 1, 100, 80, 20, 10, "stop"),
    )
    await record_eval_result(
        store,
        agent_run_id="run1",
        instance_id="pytest-dev__pytest-7432",
        patch="diff",
        patch_files=["a.py"],
        resolved=True,
        harness_run_id="ph-1",
        eval_log_path="/tmp/log",
    )
    data = await eval_report_data(store, "run1")
    assert data is not None
    text = format_eval_report(data)
    assert "total_cost_usd" in text
    assert "goal_met" in text
    assert data["summary"]["total_tokens"] == 110
    await store.close()
