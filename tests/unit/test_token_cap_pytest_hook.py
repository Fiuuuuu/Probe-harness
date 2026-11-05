from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_token_cap_exception_becomes_skipped(tmp_path: Path) -> None:
    case_dir = tmp_path / "cap_case"
    case_dir.mkdir()
    (case_dir / "conftest.py").write_text(
        textwrap.dedent(
            """
            import pytest
            from probe_harness.providers.ledger import TokenCapReached

            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_makereport(item, call):
                outcome = yield
                report = outcome.get_result()
                if call.when == "call" and call.excinfo is not None:
                    if call.excinfo.errisinstance(TokenCapReached):
                        report.outcome = "skipped"
                        lineno = (item.location[1] + 1) if item.location else 1
                        report.longrepr = (
                            str(item.path),
                            lineno,
                            "Skipped: token cap reached",
                        )
            """
        ),
        encoding="utf-8",
    )
    (case_dir / "test_cap.py").write_text(
        textwrap.dedent(
            """
            from probe_harness.providers.ledger import TokenCapReached

            def test_hits_cap():
                raise TokenCapReached("over cap")
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", str(case_dir)],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "1 skipped" in combined
    assert "token cap reached" in combined
