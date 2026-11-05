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


class TokenCapReached(Exception):
    """Raised when a live API call would exceed the task token cap."""


def effective_token_cap() -> int:
    raw = os.environ.get("MAX_TASK_TOKENS", str(HARD_MAX_TASK_TOKENS))
    try:
        val = int(raw)
    except ValueError:
        val = HARD_MAX_TASK_TOKENS
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
    """Append-only ledger with file lock; shared across processes on the VM."""

    def __init__(
        self,
        path: Path | None = None,
        task_id: str = "default",
    ) -> None:
        self.path = path or DEFAULT_LEDGER_PATH
        self.task_id = task_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def read_cumulative(self) -> int:
        total = 0
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        total = int(rec.get("cumulative", total))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return total

    def check_reservation(self, estimated_input: int, max_tokens: int) -> None:
        cap = effective_token_cap()
        cumulative = self.read_cumulative()
        projected = cumulative + estimated_input + max_tokens
        if projected > cap:
            raise TokenCapReached(
                f"token cap {cap}: cumulative={cumulative}, "
                f"estimated_input={estimated_input}, max_tokens={max_tokens}"
            )

    def record_usage(
        self,
        input_total: int,
        output: int,
        test_run_id: str = "",
    ) -> int:
        cap = effective_token_cap()
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
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
                new_cumulative = cumulative + input_total + output
                if new_cumulative > cap:
                    raise TokenCapReached(
                        f"token cap {cap} exceeded after call: would be {new_cumulative}"
                    )
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
                return new_cumulative
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def total_tokens_used(self) -> int:
        return self.read_cumulative()
