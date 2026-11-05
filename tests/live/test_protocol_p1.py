"""Live protocol tests (P1), gated by RUN_LIVE=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.guards import repair_tool_pairing
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.store.db import Store
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


def _broken_histories() -> dict[str, list[dict]]:
    return {
        "a_orphan": [
            {"role": "user", "content": "Weather in Paris? Use get_weather."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
        ],
        "b_mismatch": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "WRONG", "content": "18C"},
        ],
        "c_non_adjacent": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_c",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "interject"},
            {"role": "tool", "tool_call_id": "call_c", "content": "18C"},
        ],
    }


@pytest.mark.asyncio
async def test_live_pairing_repair_cases() -> None:
    ledger = UsageLedger(task_id="live-pairing")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-pairing",
    )
    tools = _weather_tools()
    histories = _broken_histories()
    for name, broken in histories.items():
        bad = await provider.complete(
            broken,
            tools=tools,
            max_tokens=64,
            thinking_enabled=False,
        )
        assert bad.http_status == 400, name
        repaired, repairs = repair_tool_pairing(broken, reason=f"probe_{name}")
        assert repairs, name
        ok = await provider.complete(
            repaired,
            tools=tools,
            max_tokens=64,
            thinking_enabled=False,
        )
        assert ok.http_status == 200, name
    await provider.aclose()


@pytest.mark.asyncio
async def test_live_tool_name_normalisation() -> None:
    ledger = UsageLedger(task_id="live-names")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-names",
    )
    mapper = ToolNameMap()
    raw_bad = "mcp.server/tool"
    bad_schema = [
        {
            "type": "function",
            "function": {
                "name": raw_bad,
                "description": "x",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    bad = await provider.complete(
        [{"role": "user", "content": "call tool"}],
        tools=bad_schema,
        max_tokens=32,
        thinking_enabled=False,
    )
    assert bad.http_status == 400

    norm = normalise_tool_name(raw_bad, name_map=mapper)
    norm_schema = normalise_tool_schemas(_weather_tools(), name_map=mapper)
    ok_norm = await provider.complete(
        [{"role": "user", "content": f"use {norm} for Paris"}],
        tools=norm_schema,
        max_tokens=64,
        thinking_enabled=False,
    )
    assert ok_norm.http_status == 200

    long70 = "w" * 70
    norm70 = normalise_tool_name(long70, name_map=mapper)
    schema70 = normalise_tool_schemas(_weather_tools(), name_map=mapper)
    schema70[0]["function"]["name"] = norm70
    ok70 = await provider.complete(
        [{"role": "user", "content": "weather"}],
        tools=schema70,
        max_tokens=64,
        thinking_enabled=False,
    )
    await provider.aclose()
    assert ok70.http_status == 200


@pytest.mark.asyncio
async def test_live_thinking_tools_reasoning_replay_two_rounds() -> None:
    ledger = UsageLedger(task_id="live-reasoning")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-reasoning",
    )
    tools = _weather_tools()
    first = await provider.complete(
        [{"role": "user", "content": "Weather in Tokyo? Must use get_weather."}],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert first.http_status == 200 and first.tool_calls
    if not first.reasoning_content:
        pytest.skip("live model returned empty reasoning_content on tool turn")

    history = [
        {"role": "user", "content": "Weather in Tokyo? Must use get_weather."},
        {
            "role": "assistant",
            "content": first.content or "",
            "reasoning_content": first.reasoning_content,
            "tool_calls": [
                {
                    "id": first.tool_calls[0].id,
                    "type": "function",
                    "function": {
                        "name": first.tool_calls[0].name,
                        "arguments": first.tool_calls[0].arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": first.tool_calls[0].id,
            "content": json.dumps({"temp_c": 20}),
        },
    ]
    second = await provider.complete(
        history + [{"role": "user", "content": "Now Paris with get_weather."}],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert second.http_status == 200
    if not second.tool_calls:
        pytest.skip("model did not issue second tool call")
    history.append(
        {
            "role": "assistant",
            "content": second.content or "",
            "reasoning_content": second.reasoning_content or "",
            "tool_calls": [
                {
                    "id": second.tool_calls[0].id,
                    "type": "function",
                    "function": {
                        "name": second.tool_calls[0].name,
                        "arguments": second.tool_calls[0].arguments,
                    },
                }
            ],
        }
    )
    history.append(
        {
            "role": "tool",
            "tool_call_id": second.tool_calls[0].id,
            "content": json.dumps({"temp_c": 18}),
        }
    )

    stripped = json.loads(json.dumps(history))
    for msg in stripped:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
    bad = await provider.complete(
        stripped + [{"role": "user", "content": "Summarize both cities."}],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert bad.http_status == 400

    good = await provider.complete(
        history + [{"role": "user", "content": "Summarize both cities."}],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    await provider.aclose()
    assert good.http_status == 200


@pytest.mark.asyncio
async def test_live_truncation_through_loop(tmp_path: Path) -> None:
    ledger = UsageLedger(task_id="live-trunc-loop")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-trunc-loop",
    )
    task = tmp_path / "trunc.json"
    task.write_text(
        json.dumps(
            {
                "user_message": "Call get_weather; answer must use the tool.",
                "max_tokens": 8,
                "max_steps": 3,
                "thinking_enabled": False,
            }
        )
    )
    store = Store(tmp_path / "trunc.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        provider,
        task,
        provider_name="deepseek",
        model="deepseek-flash",
        max_steps=3,
    )
    tool_rows = await store.fetchall("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,))
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    model_responses = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'model_response'",
        (run_id,),
    )
    await store.close()
    await provider.aclose()
    assert len(tool_rows) == 0
    trunc_repairs = any('"kind": "truncation"' in r["data_json"] for r in repairs)
    length_finish = any(
        json.loads(r["data_json"]).get("finish_reason") == "length" for r in model_responses
    )
    if not trunc_repairs and not length_finish:
        pytest.skip("live model did not truncate tool call at max_tokens=8")
    assert length_finish
    assert len(tool_rows) == 0
    if any(json.loads(r["data_json"]).get("tool_calls", 0) > 0 for r in model_responses):
        assert trunc_repairs
