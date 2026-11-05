"""Run-level token/call budgets (warning → finalizing → hard), separate from API token cap."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from probe_harness.providers.usage import normalise_openai_usage

DEFAULT_FINALIZATION_USER_TEXT = (
    "Run budget is in finalization: answer in plain text only. Do not call tools."
)

Level = str  # normal | warning | finalizing | hard


def _level_rank(level: Level) -> int:
    return {"normal": 0, "warning": 1, "finalizing": 2, "hard": 3}.get(level, 0)


def _max_level(a: Level, b: Level) -> Level:
    return a if _level_rank(a) >= _level_rank(b) else b


@dataclass(frozen=True)
class BudgetLimits:
    tokens_warning: int
    tokens_finalizing: int
    tokens_hard: int
    calls_warning: int
    calls_finalizing: int
    calls_hard: int

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> BudgetLimits | None:
        raw = task.get("budget")
        if raw is None:
            return None
        if isinstance(raw, dict) and "tokens" in raw:
            tw, tf, th = raw["tokens"]
            cw, cf, ch = raw["calls"]
            return cls(
                tokens_warning=int(tw),
                tokens_finalizing=int(tf),
                tokens_hard=int(th),
                calls_warning=int(cw),
                calls_finalizing=int(cf),
                calls_hard=int(ch),
            )
        return cls(
            tokens_warning=int(raw.get("tokens_warning", 80_000)),
            tokens_finalizing=int(raw.get("tokens_finalizing", 100_000)),
            tokens_hard=int(raw.get("tokens_hard", 120_000)),
            calls_warning=int(raw.get("calls_warning", 48)),
            calls_finalizing=int(raw.get("calls_finalizing", 56)),
            calls_hard=int(raw.get("calls_hard", 64)),
        )


@dataclass
class BudgetTransition:
    dimension: str
    from_level: Level
    to_level: Level


@dataclass
class RepeatedCallGuard:
    _streak: list[str] = field(default_factory=list)

    def note_tool_call(self, norm_name: str, arguments: str) -> str | None:
        fp = hashlib.sha256(f"{norm_name}\0{arguments}".encode()).hexdigest()
        if self._streak and self._streak[-1] == fp:
            self._streak.append(fp)
        else:
            self._streak = [fp]
        if len(self._streak) >= 3:
            return "repeated_identical_call"
        return None


@dataclass
class RunBudget:
    limits: BudgetLimits
    chargeable_tokens: int = 0
    model_calls: int = 0
    tokens_level: Level = "normal"
    calls_level: Level = "normal"
    finalization_appended: bool = False
    finalization_wrap_up_pending: bool = False

    @property
    def combined_level(self) -> Level:
        return _max_level(self.tokens_level, self.calls_level)

    @property
    def in_finalizing(self) -> bool:
        return self.combined_level in {"finalizing", "hard"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "chargeable_tokens": self.chargeable_tokens,
            "model_calls": self.model_calls,
            "tokens_level": self.tokens_level,
            "calls_level": self.calls_level,
            "combined_level": self.combined_level,
            "finalization_appended": self.finalization_appended,
            "finalization_wrap_up_pending": self.finalization_wrap_up_pending,
        }

    @classmethod
    def from_dict(cls, limits: BudgetLimits, data: dict[str, Any]) -> RunBudget:
        rb = cls(limits=limits)
        rb.chargeable_tokens = int(data.get("chargeable_tokens", 0))
        rb.model_calls = int(data.get("model_calls", 0))
        rb.tokens_level = str(data.get("tokens_level", "normal"))
        rb.calls_level = str(data.get("calls_level", "normal"))
        rb.finalization_appended = bool(data.get("finalization_appended", False))
        rb.finalization_wrap_up_pending = bool(data.get("finalization_wrap_up_pending", False))
        return rb

    def _tokens_level_for(self, chargeable: int) -> Level:
        lim = self.limits
        if chargeable >= lim.tokens_hard:
            return "hard"
        if chargeable >= lim.tokens_finalizing:
            return "finalizing"
        if chargeable >= lim.tokens_warning:
            return "warning"
        return "normal"

    def _calls_level_for(self, calls: int) -> Level:
        lim = self.limits
        if calls >= lim.calls_hard:
            return "hard"
        if calls >= lim.calls_finalizing:
            return "finalizing"
        if calls >= lim.calls_warning:
            return "warning"
        return "normal"

    def _refresh_levels(self) -> list[BudgetTransition]:
        transitions: list[BudgetTransition] = []
        new_t = self._tokens_level_for(self.chargeable_tokens)
        if new_t != self.tokens_level:
            transitions.append(BudgetTransition("tokens", self.tokens_level, new_t))
            self.tokens_level = new_t
            if new_t == "finalizing":
                self.finalization_wrap_up_pending = True
        new_c = self._calls_level_for(self.model_calls)
        if new_c != self.calls_level:
            transitions.append(BudgetTransition("calls", self.calls_level, new_c))
            self.calls_level = new_c
            if new_c == "finalizing":
                self.finalization_wrap_up_pending = True
        return transitions

    def check_before_model_call(self) -> tuple[bool, str | None]:
        """Return (may_call_model, hard_stop_reason)."""
        self._refresh_levels()
        if self.combined_level == "hard":
            return False, "run_budget"
        return True, None

    def record_model_usage(
        self, usage: dict[str, Any], *, provider_name: str
    ) -> list[BudgetTransition]:
        norm = normalise_openai_usage(usage)
        charge = norm["input_total"] + norm["output"]
        self.chargeable_tokens += charge
        self.model_calls += 1
        return self._refresh_levels()

    def check_after_model_call(self) -> tuple[bool, str | None]:
        self._refresh_levels()
        if self.combined_level == "hard":
            return False, "run_budget"
        return True, None

    def finalization_user_text(self, task: dict[str, Any]) -> str:
        raw = task.get("budget") or {}
        if isinstance(raw, dict) and raw.get("finalization_user_message"):
            return str(raw["finalization_user_message"])
        return DEFAULT_FINALIZATION_USER_TEXT

    def ensure_finalization_user_message(
        self, messages: list[dict[str, Any]], task: dict[str, Any]
    ) -> bool:
        if (
            self.finalization_appended
            or not self.finalization_wrap_up_pending
            or self.combined_level != "finalizing"
        ):
            return False
        messages.append({"role": "user", "content": self.finalization_user_text(task)})
        self.finalization_appended = True
        return True

    def snapshot_event_payload(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "limits": {
                "tokens": [
                    self.limits.tokens_warning,
                    self.limits.tokens_finalizing,
                    self.limits.tokens_hard,
                ],
                "calls": [
                    self.limits.calls_warning,
                    self.limits.calls_finalizing,
                    self.limits.calls_hard,
                ],
            },
        }


def cache_prefix_fingerprint(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Stable hash of tool schemas + system messages (cache prefix per C7)."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    payload = {"tools": tools, "system": system_msgs}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
