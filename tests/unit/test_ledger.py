from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.providers.ledger import (
    HARD_MAX_TASK_TOKENS,
    InvalidTokenCapConfig,
    TokenCapReached,
    UsageLedger,
    effective_token_cap,
)


def test_cap_clamped_to_hard_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TASK_TOKENS", "999999")
    assert effective_token_cap() == HARD_MAX_TASK_TOKENS


def test_non_positive_cap_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TASK_TOKENS", "0")
    with pytest.raises(InvalidTokenCapConfig):
        effective_token_cap()


def test_record_over_cap_writes_then_raises(tmp_path: Path) -> None:
    ledger = UsageLedger(path=tmp_path / "ledger.jsonl", task_id="t")
    ledger.record_usage(199_000, 0, test_run_id="seed")
    with pytest.raises(TokenCapReached):
        ledger.record_usage(1_000, 900, test_run_id="over")
    assert ledger.read_cumulative() == 200_900
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    last = json.loads(lines[-1])
    assert last["cumulative"] == 200_900


def test_reservation_raises_when_would_exceed(tmp_path: Path) -> None:
    ledger = UsageLedger(path=tmp_path / "ledger.jsonl", task_id="t")
    ledger.record_usage(199_990, 5, test_run_id="seed")
    with pytest.raises(TokenCapReached):
        ledger.check_reservation(estimated_input=10, max_tokens=10)


def test_record_logs_cumulative(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = UsageLedger(path=path, task_id="t")
    cum = ledger.record_usage(100, 50, test_run_id="a")
    assert cum == 150
    assert ledger.read_cumulative() == 150
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["cumulative"] == 150
    assert rec["input_total"] == 100
    assert rec["output"] == 50
