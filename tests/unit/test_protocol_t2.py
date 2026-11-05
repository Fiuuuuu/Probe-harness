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
from probe_harness.providers.openai_chat import (
    OpenAIChatProvider,
    assemble_stream_tool_calls,
    parse_sse_stream_payload,
)
from probe_harness.store.db import Store
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
    assert merged[0]["content"] == "base"
    assert merged[1]["content"] == "hi\n\nextra"
    assert all(m.get("role") != "system" for m in merged[1:])


def test_c3_list_system_content_not_dropped() -> None:
    messages = [
        {"role": "system", "content": "BASE"},
        {"role": "system", "content": [{"type": "text", "text": "LIST-SYS"}]},
        {"role": "user", "content": "u"},
    ]
    merged = merge_system_messages_openai(messages)
    assert merged[0]["content"] == "BASE\nLIST-SYS"


def test_c1_pairing_repair_orphan() -> None:
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
    repaired, repairs = repair_tool_pairing(messages, reason="interrupted")
    assert repairs and repairs[0]["tool_call_id"] == "call_orphan"
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_orphan" for m in repaired)


def test_c6_all_fixture_i_shapes_truncated() -> None:
    fixtures = json.loads(Path("fixtures/probe_results.json").read_text(encoding="utf-8"))[
        "results"
    ]
    for row in fixtures:
        if row["probe"] != "i_maxtokens_in_toolcall":
            continue
        fr = str(row.get("finish_reason") or row.get("stop_reason") or "")
        args = '{"city": "Paris"}' if row["provider"] != "deepseek" else '{"city": "Pa'
        tc = ToolCallSpec(id="c1", name="get_weather", arguments=args)
        assert is_truncated_tool_call(
            tc,
            finish_reason=fr,
            max_tokens=40,
            usage=row["usage"],
            provider_name=row["provider"],
        )


@pytest.mark.asyncio
async def test_request_id_and_response_model_persisted(tmp_path: Path) -> None:
    posts: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        posts.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-B1",
                "model": "deepseek-v4-flash-0105",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = OpenAIChatProvider(
        api_key="k",
        ledger=UsageLedger(path=tmp_path / "persist.jsonl", task_id="persist"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "persist.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "max_tokens": 8}))
    run_id = await run_agent_loop(
        store,
        provider,
        task,
        provider_name="deepseek",
        model="deepseek-flash",
    )
    row = await store.fetchone(
        "SELECT request_id, response_model, request_sha256 FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    ev = await store.fetchone(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'model_response'",
        (run_id,),
    )
    await store.close()
    await provider.aclose()
    assert row["request_id"] == "chatcmpl-B1"
    assert row["response_model"] == "deepseek-v4-flash-0105"
    assert posts[0]["model"] == "deepseek-flash"
    payload = json.loads(ev["data_json"])
    assert payload["response_model"] == "deepseek-v4-flash-0105"


@pytest.mark.asyncio
async def test_thinking_disabled_strips_reasoning_on_wire(tmp_path: Path) -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        n = len(bodies)
        if n == 1:
            msg = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "secret chain",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "ok"}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{n}",
                "model": "deepseek-flash",
                "choices": [{"message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    provider = OpenAIChatProvider(
        api_key="k",
        thinking_enabled=False,
        ledger=UsageLedger(path=tmp_path / "think.jsonl", task_id="think"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "think.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "go", "max_tokens": 32, "thinking_enabled": False}))

    await run_agent_loop(
        store,
        provider,
        task,
        provider_name="deepseek",
        model="deepseek-flash",
    )
    await store.close()
    await provider.aclose()
    assert len(bodies) >= 2
    second_assistant = [m for m in bodies[1]["messages"] if m.get("role") == "assistant"]
    assert second_assistant
    assert "reasoning_content" not in second_assistant[0]


def test_sse_request_id_from_first_chunk() -> None:
    lines = [
        'data: {"id": "chatcmpl-S1", "model": "deepseek-v4-flash", "choices": []}',
        "data: [DONE]",
    ]
    turn, _ = parse_sse_stream_payload(lines)
    assert turn.request_id == "chatcmpl-S1"


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
