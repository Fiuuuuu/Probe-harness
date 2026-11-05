"""In-process hook / event bus (learn-claude-code s04: PreToolUse / PostToolUse / Stop)."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from probe_harness.providers.openai_chat import request_sha256
from probe_harness.providers.usage import normalise_anthropic_usage, normalise_openai_usage
from probe_harness.store.db import EventWriter, Store

HookHandler = Callable[["HookEvent"], Awaitable[None]]


class HookHandlerError(Exception):
    """One or more hook handlers failed during emit."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        super().__init__(errors)


class HookName(StrEnum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_MODEL_CALL = "PostModelCall"
    STOP = "Stop"


@dataclass
class HookEvent:
    name: HookName
    run_id: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetricsSnapshot:
    """In-run counters maintained by metric hooks."""

    tool_calls: int = 0
    tool_errors: int = 0
    tool_duration_ms: int = 0
    model_calls: int = 0
    input_total: int = 0
    cached_read: int = 0
    uncached: int = 0
    output: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_duration_ms": self.tool_duration_ms,
            "model_calls": self.model_calls,
            "input_total": self.input_total,
            "cached_read": self.cached_read,
            "uncached": self.uncached,
            "output": self.output,
        }


class HookBus:
    def __init__(self) -> None:
        self._handlers: dict[HookName, list[HookHandler]] = {h: [] for h in HookName}
        self.metrics = RunMetricsSnapshot()
        self.fired: list[HookName] = []
        self._metric_hooks_registered = False

    def on(self, name: HookName, handler: HookHandler) -> None:
        self._handlers[name].append(handler)

    async def emit(self, event: HookEvent) -> None:
        self.fired.append(event.name)
        errors: list[BaseException] = []
        for handler in self._handlers[event.name]:
            try:
                await handler(event)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise HookHandlerError(errors)


def _normalise_usage(provider_name: str, usage: dict[str, Any]) -> dict[str, int]:
    if provider_name == "fake" or provider_name == "deepseek":
        return normalise_openai_usage(usage)
    if provider_name == "anthropic" or provider_name.endswith("-anthropic"):
        return normalise_anthropic_usage(usage)
    raise ValueError(f"unknown provider for usage normalisation: {provider_name!r}")


async def _persist_model_call_rows(event: HookEvent, *, store: Store) -> None:
    p = event.payload
    provider_name = p["provider_name"]
    turn = p["turn"]
    call_rows = p["call_rows"]
    req_hash = request_sha256(p["req_body"])
    for call_row in call_rows:
        row_usage = _normalise_usage(provider_name, call_row.usage)
        stop = call_row.stop_reason or turn.finish_reason
        if call_row.usage_estimated:
            stop = f"{stop}|usage_estimated"
        await store.execute(
            """
            INSERT INTO model_calls (
                run_id, step, attempt, input_total, cached_read, cache_write,
                uncached, output, reasoning, stop_reason, latency_ms,
                request_sha256, http_status, request_id, response_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.step,
                call_row.attempt,
                row_usage["input_total"],
                row_usage["cached_read"],
                row_usage["cache_write"],
                row_usage["uncached"],
                row_usage["output"],
                row_usage["reasoning"],
                stop,
                call_row.latency_ms,
                req_hash,
                call_row.http_status,
                call_row.request_id,
                call_row.response_model or turn.response_model,
            ),
        )


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
    """Wire tool + model persistence and run-level metrics onto the hook bus."""
    if bus._metric_hooks_registered:
        return
    bus._metric_hooks_registered = True

    async def on_post_model_call(event: HookEvent) -> None:
        await _persist_model_call_rows(event, store=store)
        for call_row in event.payload["call_rows"]:
            row_usage = _normalise_usage(event.payload["provider_name"], call_row.usage)
            bus.metrics.model_calls += 1
            bus.metrics.input_total += row_usage["input_total"]
            bus.metrics.cached_read += row_usage["cached_read"]
            bus.metrics.uncached += row_usage["uncached"]
            bus.metrics.output += row_usage["output"]

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

    bus.on(HookName.POST_MODEL_CALL, on_post_model_call)
    bus.on(HookName.POST_TOOL_USE, on_post_tool_use)
    bus.on(HookName.STOP, on_stop)


def default_hook_bus(*, store: Store, events: EventWriter) -> HookBus:
    bus = HookBus()
    register_default_metric_hooks(bus, store=store, events=events)
    return bus


def ensure_metric_hooks(bus: HookBus, *, store: Store, events: EventWriter) -> HookBus:
    register_default_metric_hooks(bus, store=store, events=events)
    return bus
