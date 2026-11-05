from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.base import ProviderCallError
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import (
    ERROR_BODY_MAX,
    OpenAIChatProvider,
    error_body_from_text,
)
from probe_harness.providers.usage import normalise_openai_usage
from probe_harness.store.db import Store


@pytest.fixture(autouse=True)
def _instant_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("probe_harness.providers.openai_chat.asyncio.sleep", _instant_sleep)


def _fixtures() -> list[dict]:
    return json.loads(Path("fixtures/probe_results.json").read_text())["results"]


def _probe(pid: str, provider: str = "deepseek") -> dict:
    return next(r for r in _fixtures() if r["probe"] == pid and r["provider"] == provider)


def _sse(chunks: list[dict]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _deepseek_stream_chunks(*, usage_on_last: bool = True) -> list[dict]:
    j = _probe("j_stream_include_usage")
    frags = j["arg_fragments"]
    usage = j["usage_seen"][0][1]
    chunks: list[dict] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_w",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": frags[0]},
                            }
                        ],
                    },
                }
            ]
        }
    ]
    for frag in frags[1:]:
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": frag}},
                            ]
                        },
                    }
                ]
            }
        )
    last: dict = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    if usage_on_last:
        last["usage"] = usage
        chunks.append(last)
    else:
        chunks.append(last)
        chunks.append({"choices": [], "usage": usage})
    return chunks


def test_error_body_truncated_to_800() -> None:
    long = "x" * 2000
    assert len(error_body_from_text(long)) == ERROR_BODY_MAX


@pytest.mark.asyncio
async def test_stream_mocktransport_j_include_usage_shape(tmp_path: Path) -> None:
    posts: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        posts.append(req)
        body = json.loads(req.content)
        assert body.get("stream_options") == {"include_usage": True}
        return httpx.Response(
            200,
            text=_sse(_deepseek_stream_chunks()),
            headers={"content-type": "text/event-stream", "x-request-id": "rid-j"},
        )

    ledger = UsageLedger(path=tmp_path / "j.jsonl", task_id="j")
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    turn = await provider.complete(
        [{"role": "user", "content": "weather?"}],
        max_tokens=64,
        stream=True,
    )
    await provider.aclose()
    assert len(posts) == 1
    assert turn.tool_calls[0].arguments == '{"city": "Paris"}'
    norm = normalise_openai_usage(turn.usage)
    assert norm["input_total"] == 272
    assert norm["cached_read"] == 128
    assert len(turn.http_calls) == 1
    assert ledger.read_cumulative() > 0


@pytest.mark.asyncio
async def test_real_localhost_sse_stream_200(tmp_path: Path) -> None:
    chunks = _deepseek_stream_chunks()
    body = _sse(chunks).encode()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            data += await reader.read(4096)
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"x-request-id: rid-real\r\n"
            b"Connection: close\r\n\r\n"
        )
        for i in range(0, len(body), 40):
            writer.write(body[i : i + 40])
            await writer.drain()
            await asyncio.sleep(0.001)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ledger = UsageLedger(path=tmp_path / "real.jsonl", task_id="real")
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        base_url=f"http://127.0.0.1:{port}",
    )
    try:
        turn = await provider.complete(
            [{"role": "user", "content": "x"}],
            max_tokens=64,
            stream=True,
        )
    finally:
        await provider.aclose()
        server.close()
        await server.wait_closed()

    assert turn.http_status == 200
    assert turn.tool_calls
    assert len(turn.http_calls) == 1
    assert ledger.read_cumulative() > 0


@pytest.mark.asyncio
async def test_stream_500_retries_without_stream_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def track_sleep(sec: float) -> None:
        sleeps.append(sec)

    monkeypatch.setattr("probe_harness.providers.openai_chat.asyncio.sleep", track_sleep)
    n = {"v": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["v"] += 1
        if n["v"] == 1:
            return httpx.Response(
                500,
                text='{"error":"boom"}',
                headers={"retry-after": "7", "x-request-id": "e1"},
            )
        return httpx.Response(
            200,
            text=_sse(_deepseek_stream_chunks()),
            headers={"content-type": "text/event-stream"},
        )

    ledger = UsageLedger(path=tmp_path / "s500.jsonl", task_id="s500")
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    turn = await provider.complete([{"role": "user", "content": "x"}], max_tokens=8, stream=True)
    await provider.aclose()
    assert n["v"] == 2
    assert sleeps == [7.0]
    assert turn.http_status == 200
    assert len(turn.http_calls) == 2
    assert turn.http_calls[0].error_body


@pytest.mark.asyncio
async def test_mid_stream_disconnect_no_retry(tmp_path: Path) -> None:
    body = _sse(_deepseek_stream_chunks()).encode()
    attempts = {"n": 0}
    real_retry = OpenAIChatProvider._execute_with_retry

    async def counting_retry(self, *args, **kwargs):
        attempts["n"] += 1
        return await real_retry(self, *args, **kwargs)

    OpenAIChatProvider._execute_with_retry = counting_retry  # type: ignore[method-assign]

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while b"\r\n\r\n" not in (await reader.read(4096)):
            pass
        part = body[:200]
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
        )
        writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ledger = UsageLedger(path=tmp_path / "cut.jsonl", task_id="cut")
    provider = OpenAIChatProvider(api_key="k", ledger=ledger, base_url=f"http://127.0.0.1:{port}")
    try:
        turn = await provider.complete(
            [{"role": "user", "content": "x"}], max_tokens=64, stream=True
        )
    finally:
        OpenAIChatProvider._execute_with_retry = real_retry  # type: ignore[method-assign]
        await provider.aclose()
        server.close()
        await server.wait_closed()
    assert attempts["n"] == 1
    assert len(turn.http_calls) == 1
    assert turn.finish_reason == "stream_interrupted"
    assert turn.tool_calls == []
    assert ledger.read_cumulative() > 0


@pytest.mark.asyncio
async def test_m7_stream_interrupt_via_loop_no_tool_rows(tmp_path: Path) -> None:
    body = _sse(_deepseek_stream_chunks()).encode()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while b"\r\n\r\n" not in (await reader.read(4096)):
            pass
        part = body[:200]
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
        )
        writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ledger = UsageLedger(path=tmp_path / "m7led.jsonl", task_id="m7")
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        base_url=f"http://127.0.0.1:{port}",
    )
    store = Store(tmp_path / "m7.db")
    await store.connect()
    task = tmp_path / "stream_task.json"
    task.write_text(
        json.dumps({"user_message": "hi", "max_tokens": 64, "stream": True, "max_steps": 5})
    )
    try:
        run_id = await run_agent_loop(
            store,
            provider,
            task,
            provider_name="deepseek",
            model="deepseek-flash",
            max_steps=5,
        )
    finally:
        await provider.aclose()
        server.close()
        await server.wait_closed()

    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    tool_rows = await store.fetchall("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,))
    mc = await store.fetchall("SELECT * FROM model_calls WHERE run_id = ?", (run_id,))
    ev = await store.fetchone(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'provider_error'",
        (run_id,),
    )
    await store.close()
    assert ledger.read_cumulative() > 0
    assert len(mc) == 1
    assert len(tool_rows) == 0
    assert run["status"] == "failed"
    assert run["stop_reason"] == "provider_error"
    assert ev is not None
    payload = json.loads(ev["data_json"])
    assert payload["error_body"]


@pytest.mark.asyncio
async def test_m8_provider_error_event_uses_last_http_status(tmp_path: Path) -> None:
    posts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        raise httpx.ConnectError("boom", request=req)

    provider = OpenAIChatProvider(
        api_key="k",
        ledger=UsageLedger(path=tmp_path / "m8.jsonl", task_id="m8"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "m8.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "max_tokens": 8}))
    with pytest.raises(ProviderCallError):
        await run_agent_loop(
            store, provider, task, provider_name="deepseek", model="deepseek-flash"
        )
    run = await store.fetchone("SELECT id FROM runs ORDER BY started_at DESC LIMIT 1")
    ev = await store.fetchone(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'provider_error'",
        (run["id"],),
    )
    await store.close()
    await provider.aclose()
    payload = json.loads(ev["data_json"])
    assert payload["http_status"] == 0
    assert payload["error_body"] == "boom"


@pytest.mark.asyncio
async def test_cap_mid_retry_blocks_second_post(tmp_path: Path) -> None:
    posts = {"n": 0}
    holder: dict[str, UsageLedger] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        if posts["n"] == 1:
            led = holder["led"]
            led.record_usage(199_990 - led.read_cumulative(), 0, test_run_id="other")
            return httpx.Response(503, text="busy", headers={"x-request-id": "r1"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
            },
        )

    ledger = UsageLedger(path=tmp_path / "cap.jsonl", task_id="cap")
    holder["led"] = ledger
    provider = OpenAIChatProvider(
        api_key="k",
        ledger=ledger,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "cap.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "max_tokens": 16}))
    run_id = await run_agent_loop(
        store, provider, task, provider_name="deepseek", model="deepseek-flash"
    )
    rows = await store.fetchall(
        "SELECT attempt, http_status FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    await store.close()
    await provider.aclose()
    assert posts["n"] == 1
    assert run["stop_reason"] == "token_cap"
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_network_errors_persist_model_calls(tmp_path: Path) -> None:
    posts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        raise httpx.ConnectError("boom", request=req)

    provider = OpenAIChatProvider(
        api_key="k",
        ledger=UsageLedger(path=tmp_path / "net.jsonl", task_id="net"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "net.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "hi", "max_tokens": 8}))
    with pytest.raises(ProviderCallError):
        await run_agent_loop(
            store, provider, task, provider_name="deepseek", model="deepseek-flash"
        )
    run = await store.fetchone(
        "SELECT id, status, stop_reason FROM runs ORDER BY started_at DESC LIMIT 1"
    )
    rows = await store.fetchall(
        "SELECT attempt, http_status, stop_reason FROM model_calls WHERE run_id = ?",
        (run["id"],),
    )
    await store.close()
    await provider.aclose()
    assert posts["n"] == 3
    assert len(rows) == 3
    assert run["stop_reason"] == "provider_error"


@pytest.mark.asyncio
async def test_503_error_body_on_record_and_event(tmp_path: Path) -> None:
    body = "BODY-" + "y" * 2000

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=body, headers={"x-request-id": "rid-e"})

    provider = OpenAIChatProvider(
        api_key="k",
        ledger=UsageLedger(path=tmp_path / "e.jsonl", task_id="e"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    store = Store(tmp_path / "e.db")
    await store.connect()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"user_message": "x", "max_tokens": 8}))
    run_id = await run_agent_loop(
        store, provider, task, provider_name="deepseek", model="deepseek-flash"
    )
    rows = await store.fetchall("SELECT attempt FROM model_calls WHERE run_id = ?", (run_id,))
    ev = await store.fetchone(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'provider_error'",
        (run_id,),
    )
    await store.close()
    await provider.aclose()
    assert len(rows) == 3
    payload = json.loads(ev["data_json"])
    assert len(payload["error_body"]) <= ERROR_BODY_MAX
    assert payload["error_body"].startswith("BODY-")
    assert payload["http_status"] == 503
