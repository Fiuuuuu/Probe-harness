"""Token usage ledger and per-task cap enforcement."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

HARD_MAX_TASK_TOKENS = 200_000
DEFAULT_LEDGER_PATH = Path(".harness/usage_ledger.jsonl")


def shared_ledger_path() -> Path:
    override = os.environ.get("HARNESS_LEDGER_PATH", "").strip()
    if override:
        return Path(override)
    return DEFAULT_LEDGER_PATH


class TokenCapReached(Exception):
    """Raised when a live API call would exceed the task token cap."""


class InvalidTokenCapConfig(ValueError):
    """MAX_TASK_TOKENS is missing or not a positive integer."""


def effective_token_cap() -> int:
    raw = os.environ.get("MAX_TASK_TOKENS", str(HARD_MAX_TASK_TOKENS))
    try:
        val = int(raw)
    except ValueError as exc:
        raise InvalidTokenCapConfig(f"MAX_TASK_TOKENS must be an integer, got {raw!r}") from exc
    if val <= 0:
        raise InvalidTokenCapConfig(f"MAX_TASK_TOKENS must be positive, got {val}")
    return min(val, HARD_MAX_TASK_TOKENS)


@dataclass
class LedgerEntry:
    timestamp: float
    task_id: str
    test_run_id: str
    input_total: int
    output: int
    cumulative: int


class UsageLedger:
    """Append-only ledger with file lock; shared across processes on the VM.

    Reservation (check_reservation) and record_usage both take an exclusive file lock
    while reading the cumulative total. Multi-process over-admission is still possible
    between reservation and the HTTP call because the lock is not held across the network.
    """

    def __init__(
        self,
        path: Path | None = None,
        task_id: str = "default",
    ) -> None:
        self.path = path or shared_ledger_path()
        self.task_id = task_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    @staticmethod
    def _read_cumulative_from_handle(fh) -> int:
        fh.seek(0)
        cumulative = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cumulative = int(rec.get("cumulative", cumulative))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return cumulative

    def read_cumulative(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_cumulative_from_handle(fh)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def check_reservation(self, estimated_input: int, max_tokens: int) -> None:
        cap = effective_token_cap()
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                cumulative = self._read_cumulative_from_handle(fh)
                projected = cumulative + estimated_input + max_tokens
                if projected > cap:
                    raise TokenCapReached(
                        f"token cap {cap}: cumulative={cumulative}, "
                        f"estimated_input={estimated_input}, max_tokens={max_tokens}"
                    )
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def record_usage(
        self,
        input_total: int,
        output: int,
        test_run_id: str = "",
    ) -> int:
        cap = effective_token_cap()
        over_cap = False
        new_cumulative = 0
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                cumulative = self._read_cumulative_from_handle(fh)
                new_cumulative = cumulative + input_total + output
                over_cap = new_cumulative > cap
                entry = {
                    "timestamp": time.time(),
                    "task_id": self.task_id,
                    "test_run_id": test_run_id,
                    "input_total": input_total,
                    "output": output,
                    "cumulative": new_cumulative,
                }
                fh.seek(0, 2)
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        if over_cap:
            raise TokenCapReached(
                f"token cap {cap} exceeded after call: cumulative={new_cumulative}"
            )
        return new_cumulative

    def total_tokens_used(self) -> int:
        return self.read_cumulative()
