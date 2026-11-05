from __future__ import annotations

import pytest

from probe_harness.providers.ledger import TokenCapReached


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        fspath = str(getattr(item, "path", ""))
        if "/tests/live/" in fspath.replace("\\", "/"):
            item.add_marker(pytest.mark.live)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "call" and call.excinfo is not None:
        if call.excinfo.errisinstance(TokenCapReached):
            report.outcome = "skipped"
            lineno = item.location[1] if item.location else 0
            report.longrepr = (
                str(item.path),
                lineno or 0,
                "Skipped: token cap reached",
            )
