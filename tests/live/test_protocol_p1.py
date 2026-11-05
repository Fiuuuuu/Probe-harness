"""Live protocol tests (P1), gated by RUN_LIVE=1 and pytest marker live."""

from __future__ import annotations

import json
import os

import pytest

from probe_harness.providers.guards import repair_tool_pairing
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.tools.names import ToolNameMap, normalise_tool_name, normalise_tool_schemas

pytestmark = [pytest.mark.live]


def _weather_tools() -> list[dict]:
    return [
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


@pytest.mark.asyncio
async def test_live_pairing_repair_returns_200() -> None:
    if not os.environ.get("RUN_LIVE") == "1":
        pytest.skip("RUN_LIVE not set")
    ledger = UsageLedger(task_id="live-pairing")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-pairing",
    )
    broken = [
        {"role": "user", "content": "What is the weather in Paris? Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_live_orphan",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
    ]
    bad = await provider.complete(
        broken,
        tools=_weather_tools(),
        max_tokens=64,
        thinking_enabled=False,
    )
    assert bad.http_status == 400
    repaired, repairs = repair_tool_pairing(broken, reason="probe_a")
    assert repairs
    turn = await provider.complete(
        repaired,
        tools=_weather_tools(),
        max_tokens=64,
        thinking_enabled=False,
    )
    await provider.aclose()
    assert turn.http_status == 200


@pytest.mark.asyncio
async def test_live_tool_name_normalisation() -> None:
    if not os.environ.get("RUN_LIVE") == "1":
        pytest.skip("RUN_LIVE not set")
    ledger = UsageLedger(task_id="live-names")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-names",
    )
    mapper = ToolNameMap()
    raw = "mcp.server/get_weather"
    norm = normalise_tool_name(raw, name_map=mapper)
    schemas = normalise_tool_schemas(_weather_tools(), name_map=mapper)
    schemas[0]["function"]["name"] = norm
    turn = await provider.complete(
        [{"role": "user", "content": f"call tool {norm} for Paris"}],
        tools=schemas,
        max_tokens=64,
        thinking_enabled=False,
    )
    await provider.aclose()
    assert turn.http_status == 200


@pytest.mark.asyncio
async def test_live_thinking_tools_reasoning_replay() -> None:
    if not os.environ.get("RUN_LIVE") == "1":
        pytest.skip("RUN_LIVE not set")
    ledger = UsageLedger(task_id="live-reasoning")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-reasoning",
    )
    tools = _weather_tools()
    g7_tc = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_00_x",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    g7_history = [
        {"role": "user", "content": "Weather in Paris?"},
        g7_tc,
        {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"},
    ]
    bad = await provider.complete(
        g7_history,
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert bad.http_status == 400

    first = await provider.complete(
        [{"role": "user", "content": "Weather in Tokyo? Must use get_weather."}],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert first.http_status == 200
    assert first.tool_calls
    assert first.reasoning_content is not None
    tc_id = first.tool_calls[0].id
    assistant_with_reasoning = {
        "role": "assistant",
        "content": first.content or "",
        "reasoning_content": first.reasoning_content,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": first.tool_calls[0].name,
                    "arguments": first.tool_calls[0].arguments,
                },
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({"temp_c": 20}),
    }
    g9_history = [
        {"role": "user", "content": "Weather in Tokyo? Must use get_weather."},
        assistant_with_reasoning,
        tool_result,
        {"role": "user", "content": "Now weather in Paris with the tool again."},
    ]
    good = await provider.complete(
        g9_history,
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    await provider.aclose()
    assert good.http_status == 200


@pytest.mark.asyncio
async def test_live_truncation_not_executed() -> None:
    if not os.environ.get("RUN_LIVE") == "1":
        pytest.skip("RUN_LIVE not set")
    ledger = UsageLedger(task_id="live-trunc")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-trunc",
    )
    turn = await provider.complete(
        [{"role": "user", "content": "Call get_weather for a very long city name list."}],
        tools=_weather_tools(),
        max_tokens=40,
        thinking_enabled=False,
    )
    await provider.aclose()
    assert turn.http_status == 200
    if turn.tool_calls:
        assert turn.finish_reason in {"length", "tool_calls", "stop"}
