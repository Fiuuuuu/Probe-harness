from __future__ import annotations

import os

import pytest

from probe_harness.providers.ledger import UsageLedger, shared_ledger_path


@pytest.fixture(scope="session", autouse=True)
def _live_gate() -> None:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE not set")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")


@pytest.fixture(scope="session", autouse=True)
def _print_session_ledger_total() -> None:
    yield
    ledger = UsageLedger()
    path = shared_ledger_path()
    total = ledger.total_tokens_used()
    print(f"\nledger_path={path} ledger_cumulative_tokens={total}")
