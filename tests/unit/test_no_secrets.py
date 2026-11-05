from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import request_sha256, scrub_request_body
from probe_harness.store.db import Store

FORBIDDEN = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|DEEPSEEK_API_KEY\s*=|Authorization:\s*Bearer|x-api-key\s*:)",
    re.IGNORECASE,
)

REPO_SCAN_ROOTS = (
    Path("probe_harness"),
    Path("tests"),
    Path("pyproject.toml"),
    Path("README.md"),
)


def _iter_scan_files(extra_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in list(REPO_SCAN_ROOTS) + extra_roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append(path)
    for root in (Path(".harness"), Path("logs")):
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file())
    return files


def test_scrub_request_body_redacts_auth() -> None:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "Authorization": "Bearer secret-value",
        "headers": {"x-api-key": "abc"},
    }
    scrubbed = scrub_request_body(body)
    dumped = json.dumps(scrubbed)
    assert "secret-value" not in dumped
    assert "abc" not in dumped
    assert scrubbed["Authorization"] == "[REDACTED]"
    assert scrubbed["headers"]["x-api-key"] == "[REDACTED]"
    assert len(request_sha256(body)) == 64


@pytest.mark.asyncio
async def test_no_secrets_in_repo_and_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    ledger = UsageLedger(path=artifact_root / "usage_ledger.jsonl", task_id="scan-test")
    ledger.record_usage(10, 5, test_run_id="scan")

    db = artifact_root / "harness.db"
    store = Store(db)
    await store.connect()
    task = Path("tests/fixtures/echo_task.json")
    script = json.loads(task.read_text(encoding="utf-8"))["fake_script"]
    await run_agent_loop(store, FakeProvider(script), task, provider_name="fake", model="fake")
    await store.close()

    offenders: list[str] = []
    for path in _iter_scan_files([artifact_root]):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if FORBIDDEN.search(text):
            offenders.append(str(path))
    assert not offenders, f"secret-like content in: {offenders}"
