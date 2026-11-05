from __future__ import annotations

import pytest

from probe_harness.providers.ledger import UsageLedger, shared_ledger_path


@pytest.fixture(scope="session", autouse=True)
def _print_session_ledger_total() -> None:
    yield
    ledger = UsageLedger()
    path = shared_ledger_path()
    total = ledger.total_tokens_used()
    print(f"\nledger_path={path} ledger_cumulative_tokens={total}")
