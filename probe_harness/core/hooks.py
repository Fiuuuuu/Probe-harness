"""In-process hook / event bus (learn-claude-code s04: PreToolUse / PostToolUse / Stop)."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from probe_harness.store.db import EventWriter, Store

HookHandler = Callable[["HookEvent"], Awaitable[None]]


class HookName(StrEnum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"


@dataclass
class HookEvent:
    name: HookName
    run_id: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetricsSnapshot:
    """In-run counters maintained by metric hooks (tool-side; model usage stays in model_calls)."""

    tool_calls: int = 0
    tool_errors: int = 0
    tool_duration_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_duration_ms": self.tool_duration_ms,
        }


class HookBus:
    def __init__(self) -> None:
        self._handlers: dict[HookName, list[HookHandler]] = {h: [] for h in HookName}
        self.metrics = RunMetricsSnapshot()
        self.fired: list[HookName] = []

    def on(self, name: HookName, handler: HookHandler) -> None:
        self._handlers[name].append(handler)

    async def emit(self, event: HookEvent) -> None:
        self.fired.append(event.name)
        for handler in self._handlers[event.name]:
            await handler(event)


async def _persist_tool_call(event: HookEvent, *, store: Store, events: EventWriter) -> None:
    p = event.payload
    out_text = json.dumps(p["result"], ensure_ascii=False)
    await store.execute(
        """
        INSERT INTO tool_calls (
            run_id, step, call_id, raw_name, norm_name, args_sha256, status,
            exit_code, duration_ms, output_tokens, truncated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.run_id,
            event.step,
            p["call_id"],
            p["raw_name"],
            p["norm_name"],
            p["args_sha256"],
            "ok" if p["exit_code"] == 0 else "error",
            p["exit_code"],
            p["duration_ms"],
            max(1, len(out_text) // 4),
            0,
        ),
    )
    await events.append(
        event.run_id,
        event.step,
        "tool_result",
        time.time(),
        json.dumps({"call_id": p["call_id"], "exit_code": p["exit_code"]}),
    )


def register_default_metric_hooks(
    bus: HookBus,
    *,
    store: Store,
    events: EventWriter,
) -> None:
    """Wire tool persistence and run-level tool metrics onto the hook bus."""

    async def on_post_tool_use(event: HookEvent) -> None:
        bus.metrics.tool_calls += 1
        exit_code = int(event.payload.get("exit_code", 0))
        if exit_code != 0:
            bus.metrics.tool_errors += 1
        bus.metrics.tool_duration_ms += int(event.payload.get("duration_ms", 0))
        await _persist_tool_call(event, store=store, events=events)

    async def on_stop(event: HookEvent) -> None:
        snap = bus.metrics.as_dict()
        await events.append(
            event.run_id,
            event.step,
            "budget_state",
            time.time(),
            json.dumps({"metrics_from_hooks": snap, **snap}),
        )

    bus.on(HookName.POST_TOOL_USE, on_post_tool_use)
    bus.on(HookName.STOP, on_stop)


def default_hook_bus(*, store: Store, events: EventWriter) -> HookBus:
    bus = HookBus()
    register_default_metric_hooks(bus, store=store, events=events)
    return bus
