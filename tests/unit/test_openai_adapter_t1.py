from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import (
    OpenAIChatProvider,
    assemble_stream_tool_calls,
    parse_sse_stream_payload,
)
from probe_harness.providers.usage import normalise_openai_usage


def _load_fixtures() -> dict:
    return json.loads(Path("fixtures/probe_results.json").read_text(encoding="utf-8"))


def _probe(probe_id: str, provider: str = "deepseek") -> dict:
    for item in _load_fixtures()["results"]:
        if item.get("probe") == probe_id and item.get("provider") == provider:
            return item
    raise KeyError(probe_id)


def test_build_request_includes_thinking_and_stream_options() -> None:
    provider = OpenAIChatProvider(api_key="k")
    body = provider.build_request_body(
        [{"role": "user", "content": "hi"}],
        None,
        max_tokens=64,
        thinking_enabled=True,
        stream=True,
    )
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 64
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}

    body_off = provider.build_request_body(
        [{"role": "user", "content": "hi"}],
        None,
        max_tokens=1,
        thinking_enabled=False,
        stream=False,
    )
    assert body_off["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body_off
    assert "stream_options" not in body_off


def test_stream_tool_call_assembly_matches_fixture_fragments() -> None:
    fragments = _probe("j_stream_no_include_usage")["arg_fragments"]
    deltas = [
        {
            "index": 0,
            "id": "call_weather",
            "function": {"name": "get_weather", "arguments": frag},
        }
        for frag in fragments
    ]
    specs = assemble_stream_tool_calls(deltas)
    assert len(specs) == 1
    assert specs[0].name == "get_weather"
    assert json.loads(specs[0].arguments) == {"city": "Paris"}


def test_parse_sse_usage_from_fixture_shape() -> None:
    usage = _probe("j_stream_no_include_usage")["usage_seen"][0][1]
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}',
        f"data: {json.dumps({'usage': usage})}",
        "data: [DONE]",
    ]
    turn, usage_raw = parse_sse_stream_payload(lines)
    norm = normalise_openai_usage(usage_raw)
    assert norm["input_total"] == 272
    assert norm["output"] == 38
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_retry_on_503_then_success(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "probe_harness.providers.openai_chat.asyncio.sleep",
        AsyncMock(),
    )
    ledger_path = tmp_path / "ledger.jsonl"
    provider = OpenAIChatProvider(
        api_key="test-key",
        ledger=UsageLedger(path=ledger_path, task_id="retry"),
        test_run_id="retry-test",
    )

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.headers = {"x-request-id": "ok-id"}
    ok_resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }

    err_resp = MagicMock()
    err_resp.status_code = 503
    err_resp.headers = {"x-request-id": "err-id"}
    err_resp.json.return_value = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        "error": "unavailable",
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[err_resp, ok_resp])
    provider._client = mock_client

    turn = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=8)
    assert turn.http_status == 200
    assert len(turn.http_calls) == 2
    assert turn.http_calls[0].http_status == 503
    assert turn.http_calls[1].http_status == 200
    assert mock_client.post.await_count == 2
    await provider.aclose()


@pytest.mark.asyncio
async def test_no_retry_after_401(tmp_path) -> None:
    from probe_harness.providers.ledger import UsageLedger

    ledger = UsageLedger(path=tmp_path / "l.jsonl", task_id="no-retry")
    provider = OpenAIChatProvider(api_key="test-key", ledger=ledger)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.headers = {}
    mock_resp.json.return_value = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 0},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    provider._client = mock_client

    turn = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=1)
    assert turn.http_status == 401
    assert len(turn.http_calls) == 1
    assert mock_client.post.await_count == 1
    await provider.aclose()
