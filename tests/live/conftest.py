from __future__ import annotations

import pytest

from probe_harness.providers.ledger import UsageLedger


@pytest.fixture(scope="session", autouse=True)
def _print_session_ledger_total() -> None:
    yield
    ledger = UsageLedger()
    print(f"\nledger_cumulative_tokens={ledger.total_tokens_used()}")
