"""Minimal async agent loop (s01/s02 mechanics)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from probe_harness.providers.base import HttpCallRecord, ModelTurn, ProviderCallError
from probe_harness.providers.ledger import TokenCapReached
from probe_harness.providers.openai_chat import request_sha256
from probe_harness.providers.usage import (
    estimate_chat_input_tokens,
    normalise_anthropic_usage,
    normalise_openai_usage,
)
from probe_harness.store.db import EventWriter, Store
from probe_harness.tools.names import normalise_tool_name
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
    if provider_name == "anthropic":
        return normalise_anthropic_usage(usage)
    return normalise_openai_usage(usage)


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
        await store.execute(
            """
            INSERT INTO model_calls (
                run_id, step, attempt, input_total, cached_read, cache_write,
                uncached, output, reasoning, stop_reason, latency_ms,
                request_sha256, http_status, request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                call_row.stop_reason or turn.finish_reason,
                call_row.latency_ms,
                req_hash,
                call_row.http_status,
                call_row.request_id,
            ),
        )


def _provider_error_payload(turn: Any, call_rows: list[HttpCallRecord]) -> dict[str, Any]:
    last = call_rows[-1] if call_rows else None
    return {
        "http_status": turn.http_status,
        "request_id": turn.request_id or (last.request_id if last else None),
        "error_body": turn.error_body or (last.error_body if last else ""),
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
                await events.append(
                    run_id,
                    step,
                    "model_request",
                    time.time(),
                    json.dumps({"step": step, "message_count": len(messages)}),
                )

                max_tok = task.get("max_tokens", 8192)
                est_in = estimate_chat_input_tokens(messages, TOOL_SCHEMAS)
                use_stream = bool(task.get("stream", False)) and provider_name == "deepseek"
                turn = await provider.complete(
                    messages,
                    tools=TOOL_SCHEMAS,
                    max_tokens=max_tok,
                    estimated_input_tokens=est_in,
                    stream=use_stream,
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
                    )
                ]
                await _persist_model_calls(
                    store,
                    run_id=run_id,
                    step=step,
                    model=model,
                    messages=messages,
                    max_tok=max_tok,
                    turn=turn,
                    call_rows=call_rows,
                    provider_name=provider_name,
                )

                assistant_msg: dict[str, Any] = {"role": "assistant", "content": turn.content}
                if turn.reasoning_content:
                    assistant_msg["reasoning_content"] = turn.reasoning_content
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

                for tc in turn.tool_calls:
                    norm = normalise_tool_name(tc.name)
                    args_sha = hashlib.sha256(tc.arguments.encode()).hexdigest()
                    await events.append(
                        run_id,
                        step,
                        "tool_call",
                        time.time(),
                        json.dumps({"call_id": tc.id, "name": norm}),
                    )
                    try:
                        result, exit_code, duration_ms = await execute_tool(norm, tc.arguments)
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
                            tc.name,
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
