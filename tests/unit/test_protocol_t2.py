from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from probe_harness.core.context import prepare_context
from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.base import ToolCallSpec
from probe_harness.providers.guards import (
    is_truncated_tool_call,
    merge_system_messages_openai,
    repair_tool_pairing,
)
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider, assemble_stream_tool_calls
from probe_harness.store.db import EventWriter, Store
from probe_harness.tools.names import ToolNameMap, denormalise_tool_name, normalise_tool_name
from probe_harness.tools.registry import TOOL_SCHEMAS


def test_c8_assemble_stream_tool_calls() -> None:
    deltas = [
        {"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":'}},
        {"index": 0, "function": {"arguments": " 1}"}},
    ]
    out = assemble_stream_tool_calls(deltas)
    assert len(out) == 1
    assert out[0].id == "c1"
    assert out[0].arguments == '{"a": 1}'


def test_c3_merge_system_never_overwrites() -> None:
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "extra"},
    ]
    merged = merge_system_messages_openai(messages)
    assert merged[0]["role"] == "system"
    assert "base" in merged[0]["content"] and "extra" in merged[0]["content"]
    assert all(m.get("role") != "system" for m in merged[1:])


@pytest.mark.parametrize(
    "reason",
    ["policy_stop", "loop_guard", "cancel_during_tool_exec", "cancel_during_approval"],
)
def test_c1_pairing_repair_orphans(reason: str) -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
    ]
    repaired, repairs = repair_tool_pairing(messages, reason=reason)
    assert repairs and repairs[0]["tool_call_id"] == "call_orphan"
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_orphan" for m in repaired)


@pytest.mark.asyncio
async def test_loop_emits_protocol_repair_event(tmp_path: Path) -> None:
    script = [
        {
            "tool_calls": [
                {
                    "id": "call1",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "content": "done",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    store = Store(tmp_path / "pair.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "fake_script": script}))
    from probe_harness.providers.fake import FakeProvider

    run_id = await run_agent_loop(
        store,
        FakeProvider(script),
        task,
        provider_name="fake",
        model="fake",
    )
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "orph",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
    ]
    _, repairs, _ = prepare_context(messages, TOOL_SCHEMAS, repair_reason="cancel_during_tool_exec")
    events = EventWriter(store)
    for repair in repairs:
        await events.append(run_id, 1, "protocol_repair", 0, json.dumps(repair))
    rows = await store.fetchall(
        "SELECT type FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    await store.close()
    assert rows


def test_c4_mcp_name_never_split_on_double_underscore() -> None:
    mapper = ToolNameMap()
    raw = "mcp__my__server__tool__with__parts"
    norm = normalise_tool_name(raw, name_map=mapper)
    assert "/" not in norm and "." not in norm
    assert denormalise_tool_name(norm, name_map=mapper) == raw


def test_c4_dot_slash_and_collision() -> None:
    mapper = ToolNameMap()
    a = normalise_tool_name("mcp.server/tool", name_map=mapper)
    b = normalise_tool_name("mcp_server_tool", name_map=mapper)
    assert a != b or "mcp.server/tool" == "mcp_server_tool"
    long70 = "x" * 70
    norm70 = normalise_tool_name(long70, name_map=mapper)
    assert len(norm70) <= 64


def test_c6_truncation_from_fixture_shape() -> None:
    tc = ToolCallSpec(id="c1", name="get_weather", arguments='{"city": "Par')
    assert is_truncated_tool_call(
        tc,
        finish_reason="length",
        max_tokens=40,
        usage={"prompt_tokens": 295, "completion_tokens": 40},
    )


def test_c6_empty_args_not_executed(tmp_path: Path) -> None:
    tc = ToolCallSpec(id="c1", name="bash", arguments="")
    assert is_truncated_tool_call(
        tc,
        finish_reason="tool_calls",
        max_tokens=8192,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


@pytest.mark.asyncio
async def test_c6_truncated_tool_not_executed(tmp_path: Path) -> None:
    script = [
        {
            "tool_calls": [
                {
                    "id": "trunc1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo incomplete',
                    },
                }
            ],
            "finish_reason": "length",
            "usage": {"prompt_tokens": 10, "completion_tokens": 40},
        },
        {
            "content": "ok",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    store = Store(tmp_path / "trunc.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "go", "max_tokens": 40}))
    from probe_harness.providers.fake import FakeProvider

    run_id = await run_agent_loop(
        store,
        FakeProvider(script),
        task,
        provider_name="fake",
        model="fake",
    )
    tool_rows = await store.fetchall("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,))
    trunc_events = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    await store.close()
    assert len(tool_rows) == 0
    assert any("truncation" in e["data_json"] for e in trunc_events)


def test_c5_reasoning_content_preserved_in_prepare() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think step",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
    ]
    provider_messages, repairs, _ = prepare_context(messages, TOOL_SCHEMAS)
    assistant = next(m for m in provider_messages if m.get("role") == "assistant")
    assert assistant.get("reasoning_content") == "think step"
    assert not repairs


@pytest.mark.asyncio
async def test_request_id_falls_back_to_body_id(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-body-id",
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = OpenAIChatProvider(
        api_key="k",
        ledger=UsageLedger(path=tmp_path / "rid.jsonl", task_id="rid"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    turn = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=4)
    await provider.aclose()
    assert turn.request_id == "chatcmpl-body-id"
    assert turn.response_model == "deepseek-flash"


@pytest.mark.asyncio
async def test_stream_interrupt_model_calls_usage_estimated(tmp_path: Path) -> None:
    from tests.unit.test_openai_m_fixes import _deepseek_stream_chunks, _sse

    body = _sse(_deepseek_stream_chunks()).encode()

    async def handle(reader, writer):
        while b"\r\n\r\n" not in (await reader.read(4096)):
            pass
        part = body[:200]
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
        )
        writer.write(part)
        await writer.drain()
        writer.close()

    import asyncio

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    store = Store(tmp_path / "est.db")
    await store.connect()
    ledger = UsageLedger(path=tmp_path / "est.jsonl", task_id="est")
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        base_url=f"http://127.0.0.1:{port}",
    )
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "max_tokens": 64, "stream": True}))
    try:
        run_id = await run_agent_loop(
            store,
            provider,
            task,
            provider_name="deepseek",
            model="deepseek-flash",
        )
    finally:
        await provider.aclose()
        server.close()
        await server.wait_closed()
    row = await store.fetchone(
        "SELECT input_total, output, stop_reason FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    await store.close()
    assert row["input_total"] > 0
    assert row["output"] > 0
    assert "usage_estimated" in row["stop_reason"]
