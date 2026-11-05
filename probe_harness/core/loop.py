"""Minimal async agent loop (s01/s02 mechanics)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from probe_harness.core.context import prepare_context
from probe_harness.providers.base import HttpCallRecord, ModelTurn, ProviderCallError
from probe_harness.providers.guards import is_truncated_turn
from probe_harness.providers.ledger import TokenCapReached
from probe_harness.providers.openai_chat import request_sha256
from probe_harness.providers.usage import (
    estimate_chat_input_tokens,
    normalise_anthropic_usage,
    normalise_openai_usage,
)
from probe_harness.store.db import EventWriter, Store
from probe_harness.tools.names import ToolNameMap, denormalise_tool_name, normalise_tool_name
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


def _normalise_usage(provider_name: str, usage: dict[str, Any]) -> dict[str, int]:
    if provider_name == "fake" or provider_name == "deepseek":
        return normalise_openai_usage(usage)
    if provider_name == "anthropic" or provider_name.endswith("-anthropic"):
        return normalise_anthropic_usage(usage)
    raise ValueError(f"unknown provider for usage normalisation: {provider_name!r}")


async def _persist_model_calls(
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
    req_hash = request_sha256(req_body)
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
                run_id,
                step,
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
    error_detail: dict[str, Any] | None = None,
) -> None:
    ended = time.time()
    await store.execute(
        "UPDATE runs SET status = ?, stop_reason = ?, ended_at = ? WHERE id = ?",
        (status, stop_reason, ended, run_id),
    )
    payload: dict[str, Any] = {
        "status": status,
        "stop_reason": stop_reason,
        "agent_time_s": ended - started_at,
    }
    if error_detail:
        payload["provider_error"] = error_detail
    await events.append(
        run_id,
        step,
        "run_finished",
        time.time(),
        json.dumps(payload),
    )


def _record_loop_exception(exc: BaseException) -> tuple[str, str, BaseException]:
    if isinstance(exc, asyncio.CancelledError):
        return "failed", "cancelled", exc
    return "failed", "exception", exc


async def run_agent_loop(
    store: Store,
    provider: ChatProvider,
    task_path: Path,
    *,
    provider_name: str = "fake",
    model: str = "fake",
    max_steps: int = 20,
) -> str:
    task = load_task(task_path)
    run_id = str(uuid.uuid4())
    events = EventWriter(store)
    started_at = time.time()

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

    await events.append(run_id, 0, "run_started", time.time(), json.dumps({"task": str(task_path)}))

    user_content = task.get("user_message", task.get("instructions", "Complete the task."))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": task.get("system", "You are a helpful coding agent.")},
        {"role": "user", "content": user_content},
    ]

    script = task.get("fake_script") or []
    name_map = ToolNameMap()
    thinking_enabled = task.get("thinking_enabled")
    if provider_name == "fake" and script:
        from probe_harness.providers.fake import FakeProvider

        if not isinstance(provider, FakeProvider):
            provider = FakeProvider(script)

    step = 0
    stop_reason = "max_steps"
    status = "completed"
    pending_exc: BaseException | None = None
    error_detail: dict[str, Any] | None = None

    try:
        while step < max_steps:
            try:
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

                await events.append(
                    run_id,
                    step,
                    "model_request",
                    time.time(),
                    json.dumps({"step": step, "message_count": len(provider_messages)}),
                )

                max_tok = task.get("max_tokens", 8192)
                est_in = estimate_chat_input_tokens(provider_messages, api_tools)
                use_stream = bool(task.get("stream", False)) and provider_name == "deepseek"
                complete_kwargs: dict[str, Any] = {
                    "max_tokens": max_tok,
                    "estimated_input_tokens": est_in,
                    "stream": use_stream,
                }
                if thinking_enabled is not None:
                    complete_kwargs["thinking_enabled"] = bool(thinking_enabled)
                if task.get("tool_choice") is not None:
                    complete_kwargs["tool_choice"] = task.get("tool_choice")

                empty_retries = 0
                max_empty = int(task.get("max_length_empty_retries", 3))
                turn = None
                call_rows: list[HttpCallRecord] = []
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
                    reject_empty = bool(task.get("reject_length_without_tool_calls"))
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
                await _persist_model_calls(
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
                    try:
                        result, exit_code, duration_ms = await execute_tool(
                            raw_name,
                            tc.arguments,
                        )
                    except Exception as exc:
                        result = {"error": f"tool execution failed: {exc}", "is_error": True}
                        exit_code = 1
                        duration_ms = 0

                    out_text = json.dumps(result, ensure_ascii=False)
                    await store.execute(
                        """
                        INSERT INTO tool_calls (
                            run_id, step, call_id, raw_name, norm_name, args_sha256, status,
                            exit_code, duration_ms, output_tokens, truncated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            step,
                            tc.id,
                            raw_name,
                            norm,
                            args_sha,
                            "ok" if exit_code == 0 else "error",
                            exit_code,
                            duration_ms,
                            max(1, len(out_text) // 4),
                            0,
                        ),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": out_text,
                        }
                    )
                    await events.append(
                        run_id,
                        step,
                        "tool_result",
                        time.time(),
                        json.dumps({"call_id": tc.id, "exit_code": exit_code}),
                    )

                step += 1
            except TokenCapReached as cap:
                stop_reason = "token_cap"
                status = "failed"
                extra_calls = getattr(cap, "http_calls", None) or []
                if extra_calls:
                    stub_turn = ModelTurn(content=None, finish_reason="token_cap", http_status=0)
                    await _persist_model_calls(
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
                    await _persist_model_calls(
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
                status, stop_reason, pending_exc = _record_loop_exception(exc)
                break
            except Exception as exc:
                status, stop_reason, pending_exc = _record_loop_exception(exc)
                break
        else:
            status = "failed"
    finally:
        await _finalize_run(
            store,
            events,
            run_id,
            step,
            status,
            stop_reason,
            started_at,
            error_detail=error_detail,
        )

    if pending_exc is not None:
        raise pending_exc

    return run_id
