from __future__ import annotations

import json

import pytest

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.providers.usage import estimate_chat_input_tokens, normalise_openai_usage

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_stream_one_completion_live() -> None:
    ledger = UsageLedger(task_id="live-stream")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        max_tokens=64,
        test_run_id="live-stream",
    )
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    try:
        turn = await provider.complete(
            messages,
            max_tokens=16,
            thinking_enabled=False,
            estimated_input_tokens=estimate_chat_input_tokens(messages),
            stream=True,
        )
    finally:
        await provider.aclose()

    if turn.http_status == 0:
        pytest.fail(f"stream network failure: {turn.error_body}")
    if turn.finish_reason == "error":
        pytest.fail(f"stream error: status={turn.http_status} body={turn.error_body!r}")
    assert turn.http_status == 200
    assert len(turn.http_calls) == 1
    assert ledger.total_tokens_used() > 0
    print(
        f"LIVE_METRIC stream_one_completion "
        f"http_status={turn.http_status} finish_reason={turn.finish_reason} "
        f"response_model={turn.response_model} usage={turn.usage!r}"
    )
    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")


@pytest.mark.asyncio
async def test_stream_toolcall_without_include_usage_probe_j() -> None:
    ledger = UsageLedger(task_id="live-j-no-include")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-j-no-include",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Call get_weather for Paris now."}]
    try:
        turn = await provider.complete(
            messages,
            tools=tools,
            max_tokens=64,
            thinking_enabled=False,
            stream=True,
            stream_include_usage=False,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
    finally:
        await provider.aclose()

    assert turn.http_status == 200, turn.error_body
    assert "stream_options" not in provider.last_request_bodies[-1]
    norm = normalise_openai_usage(turn.usage or {})
    assert norm["input_total"] + norm["output"] > 0
    print(
        f"LIVE_METRIC probe_j_stream_no_include_usage "
        f"usage={json.dumps(turn.usage, sort_keys=True)} tool_calls={len(turn.tool_calls)}"
    )
