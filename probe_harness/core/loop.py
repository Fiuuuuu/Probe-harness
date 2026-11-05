"""Minimal async agent loop (s01/s02 mechanics)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from probe_harness.core.context import prepare_context
from probe_harness.core.hooks import (
    HookBus,
    HookEvent,
    HookHandlerError,
    HookName,
    ensure_metric_hooks,
)
from probe_harness.providers.base import HttpCallRecord, ModelTurn, ProviderCallError, ToolCallSpec
from probe_harness.providers.guards import (
    is_truncated_turn,
    merge_system_messages_openai,
    repair_tool_pairing,
)
from probe_harness.providers.ledger import TokenCapReached
from probe_harness.providers.usage import estimate_chat_input_tokens
from probe_harness.runs.checkpoints import checkpoint_meta, save_checkpoint
from probe_harness.runs.journal import (
    JOURNAL_COMPLETED,
    JOURNAL_STARTED,
    JOURNAL_UNCERTAIN,
    journal_lookup,
    journal_mark_completed,
    journal_mark_started,
    journal_mark_uncertain,
    uncertain_tool_result,
)
from probe_harness.runs.recovery import RecoveryContext, recover_run
from probe_harness.runs.state import CHECKPOINT_POST_TOOLS, CHECKPOINT_PRE_MODEL
from probe_harness.store.db import EventWriter, Store
from probe_harness.tools.builtin import cancel_all_bash_processes
from probe_harness.tools.names import (
    ToolNameMap,
    denormalise_tool_name,
    normalise_tool_name,
    normalise_tool_schemas,
)
from probe_harness.tools.registry import TOOL_SCHEMAS, execute_tool


class ChatProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any: ...


def load_task(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_hash(provider: str, model: str, task_path: str) -> str:
    payload = json.dumps({"provider": provider, "model": model, "task": task_path}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def _emit_model_call_metrics(
    bus: HookBus,
    store: Store,
    *,
    run_id: str,
    step: int,
    model: str,
    messages: list[dict[str, Any]],
    max_tok: int,
    turn: Any,
    call_rows: list[HttpCallRecord],
    provider_name: str,
) -> None:
    req_body = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "max_tokens": max_tok,
    }
    try:
        await bus.emit(
            HookEvent(
                HookName.POST_MODEL_CALL,
                run_id,
                step,
                {
                    "provider_name": provider_name,
                    "turn": turn,
                    "call_rows": call_rows,
                    "req_body": req_body,
                },
            )
        )
    except HookHandlerError as err:
        raise err.errors[0] from err


def _provider_error_payload(turn: Any, call_rows: list[HttpCallRecord]) -> dict[str, Any]:
    last = call_rows[-1] if call_rows else None
    status = last.http_status if last is not None else turn.http_status
    request_id = turn.request_id or (last.request_id if last else None)
    error_body = turn.error_body or (last.error_body if last else "")
    return {
        "http_status": status,
        "request_id": request_id,
        "error_body": error_body,
    }


async def _finalize_run(
    store: Store,
    events: EventWriter,
    run_id: str,
    step: int,
    status: str,
    stop_reason: str,
    started_at: float,
    *,
    hook_bus: HookBus | None = None,
    error_detail: dict[str, Any] | None = None,
) -> None:
    hook_errors: list[str] = []
    if hook_bus is not None:
        try:
            await hook_bus.emit(
                HookEvent(
                    HookName.STOP,
                    run_id,
                    step,
                    {"status": status, "stop_reason": stop_reason},
                )
            )
        except HookHandlerError as err:
            hook_errors.extend(str(e) for e in err.errors)
        except Exception as exc:
            hook_errors.append(str(exc))

    ended = time.time()
    final_status = status
    final_reason = stop_reason
    if hook_errors and final_status == "running":
        final_status = "failed"
        final_reason = "hook_error"
    await store.execute(
        "UPDATE runs SET status = ?, stop_reason = ?, ended_at = ? WHERE id = ?",
        (final_status, final_reason, ended, run_id),
    )
    payload: dict[str, Any] = {
        "status": final_status,
        "stop_reason": final_reason,
        "agent_time_s": ended - started_at,
    }
    if error_detail:
        payload["provider_error"] = error_detail
    if hook_errors:
        payload["hook_errors"] = hook_errors
    await events.append(
        run_id,
        step,
        "run_finished",
        time.time(),
        json.dumps(payload),
    )


def _record_loop_exception(exc: BaseException) -> tuple[str, str, BaseException]:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled", "cancelled", exc
    return "failed", "exception", exc


async def _repair_unpaired_on_cancel(
    messages: list[dict[str, Any]],
    events: EventWriter,
    run_id: str,
    step: int,
    *,
    reason: str = "cancelled",
) -> None:
    repaired, repairs = repair_tool_pairing(messages, reason=reason)
    if not repairs:
        return
    messages.clear()
    messages.extend(repaired)
    for repair in repairs:
        await events.append(
            run_id,
            step,
            "protocol_repair",
            time.time(),
            json.dumps(repair, ensure_ascii=False),
        )
        call_id = repair.get("tool_call_id")
        if call_id:
            await events.append(
                run_id,
                step,
                "tool_result",
                time.time(),
                json.dumps({"call_id": call_id, "synthetic": True, "reason": reason}),
            )


class SimulatedHardKill(Exception):
    """Test-only stand-in for SIGKILL (skips ``finally`` only when using ``os._exit``)."""


def _maybe_hard_kill(task: dict[str, Any], task_path: Path, when: str) -> None:
    if not _allow_task_test_controls(task_path):
        return
    target = task.get("simulate_hard_kill_at")
    if target == when:
        if task.get("simulate_hard_kill_use_exit"):
            os._exit(137)
        raise SimulatedHardKill(when)


def _allow_task_test_controls(task_path: Path) -> bool:
    """Task JSON knobs (tool_choice, reject_length_*) are for tests/live probes only."""
    if os.environ.get("HARNESS_ALLOW_TEST_CONTROLS") == "1":
        return True
    resolved = str(task_path.resolve())
    return "/tests/" in resolved or "/tmp/" in resolved or "pytest-" in resolved


async def run_agent_loop(
    store: Store,
    provider: ChatProvider,
    task_path: Path,
    *,
    provider_name: str = "fake",
    model: str = "fake",
    max_steps: int = 20,
    hook_bus: HookBus | None = None,
    recover_from: str | None = None,
) -> str:
    task = load_task(task_path)
    events = EventWriter(store)
    bus = hook_bus if hook_bus is not None else HookBus()
    ensure_metric_hooks(bus, store=store, events=events)
    started_at = time.time()

    recovery_ctx: RecoveryContext | None = None
    skip_model_once = False
    if recover_from:
        recovery_ctx = await recover_run(
            store,
            recover_from,
            provider_name=provider_name,
            model=model,
        )
        run_id = recovery_ctx.run_id
        messages = recovery_ctx.messages
        step = recovery_ctx.step
        lineage_run_id = recovery_ctx.lineage_run_id
        fake_script_index = recovery_ctx.fake_script_index
        skip_model_once = recovery_ctx.resume_at_tools
    else:
        run_id = str(uuid.uuid4())
        lineage_run_id = run_id
        fake_script_index = 0
        step = 0
        await store.execute(
            """
            INSERT INTO runs (
                id, task_ref, status, stop_reason, provider, model, config_hash, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(task_path),
                "running",
                None,
                provider_name,
                model,
                config_hash(provider_name, model, str(task_path)),
                started_at,
            ),
        )
        await events.append(
            run_id, 0, "run_started", time.time(), json.dumps({"task": str(task_path)})
        )
        user_content = task.get("user_message", task.get("instructions", "Complete the task."))
        messages = [
            {"role": "system", "content": task.get("system", "You are a helpful coding agent.")},
            {"role": "user", "content": user_content},
        ]

    script = task.get("fake_script") or []
    name_map = ToolNameMap()
    thinking_enabled = task.get("thinking_enabled")
    if provider_name == "fake" and script:
        from probe_harness.providers.fake import FakeProvider

        if not isinstance(provider, FakeProvider):
            provider = FakeProvider(script, start_index=fake_script_index)
        else:
            provider = FakeProvider(script, start_index=fake_script_index)

    stop_reason = "max_steps"
    status = "completed"
    pending_exc: BaseException | None = None
    error_detail: dict[str, Any] | None = None

    hard_killed = False
    try:
        while step < max_steps:
            try:
                if skip_model_once:
                    provider_messages = merge_system_messages_openai(messages)
                    api_tools = normalise_tool_schemas(TOOL_SCHEMAS, name_map=name_map)
                    repairs: list[dict[str, Any]] = []
                else:
                    provider_messages, repairs, api_tools = prepare_context(
                        messages,
                        TOOL_SCHEMAS,
                        repair_reason=task.get("pairing_repair_reason", "interrupted"),
                        name_map=name_map,
                    )
                for repair in repairs:
                    await events.append(
                        run_id,
                        step,
                        "protocol_repair",
                        time.time(),
                        json.dumps(repair, ensure_ascii=False),
                    )

                await events.append(
                    run_id,
                    step,
                    "context_prepared",
                    time.time(),
                    json.dumps(
                        {
                            "step": step,
                            "message_count": len(provider_messages),
                            "repairs": len(repairs),
                        }
                    ),
                )

                max_tok = task.get("max_tokens", 8192)
                turn: ModelTurn | None = None
                call_rows: list[HttpCallRecord] = []

                if skip_model_once:
                    skip_model_once = False
                    last = messages[-1]
                    if last.get("role") != "assistant" or not last.get("tool_calls"):
                        status = "failed"
                        stop_reason = "recovery_error"
                        break
                    resumed_calls: list[ToolCallSpec] = []
                    for tc in last["tool_calls"]:
                        fn = tc.get("function") or {}
                        resumed_calls.append(
                            ToolCallSpec(
                                id=str(tc.get("id")),
                                name=str(fn.get("name", "")),
                                arguments=str(fn.get("arguments", "{}")),
                            )
                        )
                    turn = ModelTurn(
                        content=last.get("content"),
                        tool_calls=resumed_calls,
                        finish_reason="tool_calls",
                    )
                else:
                    await events.append(
                        run_id,
                        step,
                        "model_request",
                        time.time(),
                        json.dumps({"step": step, "message_count": len(provider_messages)}),
                    )

                    fake_idx = getattr(provider, "_index", fake_script_index)
                    await save_checkpoint(
                        store,
                        events,
                        run_id=run_id,
                        step=step,
                        boundary=CHECKPOINT_PRE_MODEL,
                        messages=messages,
                        meta=checkpoint_meta(fake_script_index=fake_idx),
                    )
                    _maybe_hard_kill(task, task_path, CHECKPOINT_PRE_MODEL)

                    est_in = estimate_chat_input_tokens(provider_messages, api_tools)
                    use_stream = bool(task.get("stream", False)) and provider_name == "deepseek"
                    complete_kwargs: dict[str, Any] = {
                        "max_tokens": max_tok,
                        "estimated_input_tokens": est_in,
                        "stream": use_stream,
                    }
                    if thinking_enabled is not None:
                        complete_kwargs["thinking_enabled"] = bool(thinking_enabled)
                    test_controls = _allow_task_test_controls(task_path)
                    if test_controls and task.get("tool_choice") is not None:
                        complete_kwargs["tool_choice"] = task.get("tool_choice")

                    empty_retries = 0
                    max_empty = int(task.get("max_length_empty_retries", 3)) if test_controls else 0
                    while True:
                        turn = await provider.complete(
                            provider_messages,
                            tools=api_tools,
                            **complete_kwargs,
                        )
                        call_rows = turn.http_calls or [
                            HttpCallRecord(
                                attempt=1,
                                usage=turn.usage,
                                http_status=turn.http_status,
                                request_id=turn.request_id,
                                latency_ms=turn.latency_ms,
                                stop_reason=turn.finish_reason,
                                error_body=turn.error_body,
                                response_model=turn.response_model,
                            )
                        ]
                        reject_empty = test_controls and bool(
                            task.get("reject_length_without_tool_calls")
                        )
                        if reject_empty and turn.finish_reason == "length" and not turn.tool_calls:
                            empty_retries += 1
                            if empty_retries >= max_empty:
                                stop_reason = "provider_error"
                                status = "failed"
                                error_detail = {
                                    "error": (
                                        "length finish_reason without tool_calls after "
                                        f"{max_empty} attempts"
                                    ),
                                    "finish_reason": turn.finish_reason,
                                }
                                await events.append(
                                    run_id,
                                    step,
                                    "provider_error",
                                    time.time(),
                                    json.dumps(error_detail),
                                )
                                break
                            continue
                        break

                    assert turn is not None
                    await _emit_model_call_metrics(
                        bus,
                        store,
                        run_id=run_id,
                        step=step,
                        model=model,
                        messages=provider_messages,
                        max_tok=max_tok,
                        turn=turn,
                        call_rows=call_rows,
                        provider_name=provider_name,
                    )
                    if provider_name == "fake":
                        fake_script_index = getattr(provider, "_index", fake_script_index)

                    if status == "failed" and stop_reason == "provider_error":
                        break

                    if turn.finish_reason == "stream_interrupted":
                        stop_reason = "provider_error"
                        status = "failed"
                        error_detail = _provider_error_payload(turn, call_rows)
                        await events.append(
                            run_id,
                            step,
                            "provider_error",
                            time.time(),
                            json.dumps(error_detail),
                        )
                        break

                    assistant_msg: dict[str, Any] = {"role": "assistant", "content": turn.content}
                    think_on = thinking_enabled if thinking_enabled is not None else False
                    if turn.reasoning_content:
                        assistant_msg["reasoning_content"] = turn.reasoning_content
                    elif think_on and turn.tool_calls:
                        assistant_msg["reasoning_content"] = ""
                    if turn.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in turn.tool_calls
                        ]
                    messages.append(assistant_msg)

                    resp_payload = {
                        "finish_reason": turn.finish_reason,
                        "tool_calls": len(turn.tool_calls),
                        "response_model": turn.response_model,
                        "request_id": turn.request_id,
                    }
                    await events.append(
                        run_id,
                        step,
                        "model_response",
                        time.time(),
                        json.dumps(resp_payload),
                    )

                    if turn.tool_calls:
                        await save_checkpoint(
                            store,
                            events,
                            run_id=run_id,
                            step=step,
                            boundary=CHECKPOINT_PRE_MODEL,
                            messages=messages,
                            meta=checkpoint_meta(
                                fake_script_index=fake_script_index,
                                model_response_done=True,
                            ),
                        )

                    if turn.http_status >= 400:
                        stop_reason = "provider_error"
                        status = "failed"
                        error_detail = _provider_error_payload(turn, call_rows)
                        await events.append(
                            run_id,
                            step,
                            "provider_error",
                            time.time(),
                            json.dumps(error_detail),
                        )
                        break

                assert turn is not None
                if not turn.tool_calls:
                    stop_reason = turn.finish_reason or "stop"
                    break

                truncated = bool(turn.tool_calls) and (
                    turn.finish_reason in {"length", "max_tokens"}
                    or is_truncated_turn(
                        turn.tool_calls,
                        finish_reason=turn.finish_reason,
                        max_tokens=max_tok,
                        usage=turn.usage,
                        provider_name=provider_name,
                    )
                )
                if truncated:
                    for tc in turn.tool_calls:
                        payload = json.dumps(
                            {
                                "error": "tool call truncated: incomplete arguments",
                                "is_error": True,
                                "truncated": True,
                            },
                            ensure_ascii=False,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": payload,
                            }
                        )
                        await events.append(
                            run_id,
                            step,
                            "protocol_repair",
                            time.time(),
                            json.dumps(
                                {
                                    "kind": "truncation",
                                    "tool_call_id": tc.id,
                                    "finish_reason": turn.finish_reason,
                                }
                            ),
                        )
                    step += 1
                    continue

                if provider_name == "fake" and task.get("simulate_interrupt_before_tool_results"):
                    step += 1
                    continue

                if provider_name == "fake" and task.get("simulate_interrupt_before_approval"):
                    await events.append(
                        run_id,
                        step,
                        "approval_requested",
                        time.time(),
                        json.dumps(
                            {"call_id": turn.tool_calls[0].id, "simulated": True},
                            ensure_ascii=False,
                        ),
                    )
                    step += 1
                    continue

                for tc in turn.tool_calls:
                    raw_name = denormalise_tool_name(tc.name, name_map=name_map)
                    norm = normalise_tool_name(raw_name, name_map=name_map)
                    args_sha = hashlib.sha256(tc.arguments.encode()).hexdigest()
                    await events.append(
                        run_id,
                        step,
                        "tool_call",
                        time.time(),
                        json.dumps({"call_id": tc.id, "name": norm}),
                    )
                    prior = await journal_lookup(
                        store, lineage_run_id=lineage_run_id, call_id=tc.id
                    )
                    if prior is not None and prior["status"] == JOURNAL_COMPLETED:
                        if prior.get("result"):
                            result = prior["result"]
                        else:
                            result = uncertain_tool_result(
                                tc.id, reason="side effect not replayed after interrupt"
                            )
                        exit_code = 0 if not result.get("is_error") else 1
                        duration_ms = 0
                        out_text = json.dumps(result, ensure_ascii=False)
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": out_text}
                        )
                        await events.append(
                            run_id,
                            step,
                            "tool_result",
                            time.time(),
                            json.dumps(
                                {"call_id": tc.id, "replayed_from_journal": True},
                                ensure_ascii=False,
                            ),
                        )
                        continue
                    if prior is not None and prior["status"] in {
                        JOURNAL_STARTED,
                        JOURNAL_UNCERTAIN,
                    }:
                        result = uncertain_tool_result(
                            tc.id,
                            reason="interrupted before completion; side effect not replayed",
                        )
                        await journal_mark_uncertain(
                            store,
                            lineage_run_id=lineage_run_id,
                            run_id=run_id,
                            call_id=tc.id,
                        )
                        await events.append(
                            run_id,
                            step,
                            "protocol_repair",
                            time.time(),
                            json.dumps(
                                {
                                    "kind": "journal_no_replay",
                                    "tool_call_id": tc.id,
                                    "journal_status": prior["status"],
                                },
                                ensure_ascii=False,
                            ),
                        )
                        out_text = json.dumps(result, ensure_ascii=False)
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": out_text}
                        )
                        await events.append(
                            run_id,
                            step,
                            "tool_result",
                            time.time(),
                            json.dumps(
                                {
                                    "call_id": tc.id,
                                    "synthetic": True,
                                    "outcome_uncertain": True,
                                },
                                ensure_ascii=False,
                            ),
                        )
                        continue
                    try:
                        await bus.emit(
                            HookEvent(
                                HookName.PRE_TOOL_USE,
                                run_id,
                                step,
                                {
                                    "call_id": tc.id,
                                    "raw_name": raw_name,
                                    "norm_name": norm,
                                    "arguments": tc.arguments,
                                },
                            )
                        )
                    except HookHandlerError as pre_err:
                        interrupt = {
                            "error": "tool call interrupted: pre_tool_use hook failed",
                            "is_error": True,
                        }
                        await events.append(
                            run_id,
                            step,
                            "protocol_repair",
                            time.time(),
                            json.dumps(
                                {
                                    "kind": "pre_tool_use_hook_failed",
                                    "tool_call_id": tc.id,
                                    "reason": str(pre_err.errors[0]),
                                },
                                ensure_ascii=False,
                            ),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(interrupt, ensure_ascii=False),
                            }
                        )
                        raise pre_err.errors[0] from pre_err
                    await journal_mark_started(
                        store,
                        lineage_run_id=lineage_run_id,
                        run_id=run_id,
                        call_id=tc.id,
                        norm_name=norm,
                        args_sha256=args_sha,
                    )
                    _maybe_hard_kill(task, task_path, "during_tool")
                    try:
                        result, exit_code, duration_ms = await execute_tool(
                            raw_name,
                            tc.arguments,
                        )
                    except Exception as exc:
                        result = {"error": f"tool execution failed: {exc}", "is_error": True}
                        exit_code = 1
                        duration_ms = 0

                    await journal_mark_completed(
                        store,
                        lineage_run_id=lineage_run_id,
                        run_id=run_id,
                        call_id=tc.id,
                        norm_name=norm,
                        args_sha256=args_sha,
                        result=result,
                    )

                    out_text = json.dumps(result, ensure_ascii=False)
                    try:
                        await bus.emit(
                            HookEvent(
                                HookName.POST_TOOL_USE,
                                run_id,
                                step,
                                {
                                    "call_id": tc.id,
                                    "raw_name": raw_name,
                                    "norm_name": norm,
                                    "args_sha256": args_sha,
                                    "result": result,
                                    "exit_code": exit_code,
                                    "duration_ms": duration_ms,
                                },
                            )
                        )
                    except HookHandlerError as post_err:
                        raise post_err.errors[0] from post_err
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": out_text,
                        }
                    )

                await save_checkpoint(
                    store,
                    events,
                    run_id=run_id,
                    step=step,
                    boundary=CHECKPOINT_POST_TOOLS,
                    messages=messages,
                    meta=checkpoint_meta(fake_script_index=fake_script_index),
                )
                _maybe_hard_kill(task, task_path, CHECKPOINT_POST_TOOLS)

                step += 1
            except TokenCapReached as cap:
                stop_reason = "token_cap"
                status = "failed"
                extra_calls = getattr(cap, "http_calls", None) or []
                if extra_calls:
                    stub_turn = ModelTurn(content=None, finish_reason="token_cap", http_status=0)
                    await _emit_model_call_metrics(
                        bus,
                        store,
                        run_id=run_id,
                        step=step,
                        model=model,
                        messages=messages,
                        max_tok=max_tok,
                        turn=stub_turn,
                        call_rows=extra_calls,
                        provider_name=provider_name,
                    )
                break
            except ProviderCallError as pce:
                stop_reason = "provider_error"
                status = "failed"
                turn_err = pce.turn or ModelTurn(
                    content=None,
                    finish_reason="error",
                    http_status=0,
                    error_body=str(pce)[:800],
                )
                if pce.http_calls:
                    await _emit_model_call_metrics(
                        bus,
                        store,
                        run_id=run_id,
                        step=step,
                        model=model,
                        messages=messages,
                        max_tok=max_tok,
                        turn=turn_err,
                        call_rows=pce.http_calls,
                        provider_name=provider_name,
                    )
                    error_detail = _provider_error_payload(turn_err, pce.http_calls)
                    await events.append(
                        run_id,
                        step,
                        "provider_error",
                        time.time(),
                        json.dumps(error_detail),
                    )
                pending_exc = pce
                break
            except asyncio.CancelledError as exc:
                cancel_all_bash_processes()
                await _repair_unpaired_on_cancel(messages, events, run_id, step, reason="cancelled")
                status, stop_reason, pending_exc = _record_loop_exception(exc)
                break
            except SimulatedHardKill:
                raise
            except Exception as exc:
                status, stop_reason, pending_exc = _record_loop_exception(exc)
                break
        else:
            status = "failed"
    except SimulatedHardKill:
        hard_killed = True
    finally:
        if not hard_killed:
            await _finalize_run(
                store,
                events,
                run_id,
                step,
                status,
                stop_reason,
                started_at,
                hook_bus=bus,
                error_detail=error_detail,
            )

    if pending_exc is not None:
        raise pending_exc

    return run_id
