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
from probe_harness.tools.names import ToolNameMap, normalise_tool_name

pytestmark = [pytest.mark.live]


def _weather_tool_schema(name: str = "get_weather") -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _tool_choice(name: str = "get_weather") -> dict:
    return {"type": "function", "function": {"name": name}}


def _reasoning_diag(provider: OpenAIChatProvider, turn) -> str:
    usage = turn.usage or {}
    details = usage.get("completion_tokens_details") or {}
    msg = provider.last_assistant_message or {}
    return (
        f"reasoning_tokens={details.get('reasoning_tokens')} "
        f"message_keys={sorted(msg.keys())} "
        f"finish_reason={turn.finish_reason} "
        f"response_model={turn.response_model} "
        f"reasoning_content_present={bool(turn.reasoning_content)}"
    )


def _assert_turn_from_deepseek(turn, *, status: int, label: str = "") -> None:
    """Prove the result came from a real HTTP call, not a harness-local short-circuit."""
    assert turn.http_calls, f"{label}: expected HTTP call to DeepSeek"
    rec = turn.http_calls[-1]
    assert turn.http_status == status, f"{label}: status {turn.http_status} != {status}"
    assert rec.http_status == status
    assert rec.latency_ms >= 0
    if status == 400:
        assert turn.error_body, f"{label}: missing error_body from server"
        assert "request_id" in turn.error_body, f"{label}: server error_body has no request_id"


def _live_metric_line(test_name: str, turn) -> str:
    usage = turn.usage or {}
    return (
        f"LIVE_METRIC {test_name} "
        f"http_status={turn.http_status} "
        f"finish_reason={turn.finish_reason} "
        f"response_model={turn.response_model} "
        f"usage={json.dumps(usage, sort_keys=True)}"
    )


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
    tools = _weather_tool_schema()
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
    bad_schema = _weather_tool_schema(raw_bad)
    bad = await provider.complete(
        [{"role": "user", "content": "call tool"}],
        tools=bad_schema,
        max_tokens=32,
        thinking_enabled=False,
    )
    assert bad.http_status == 400

    norm = normalise_tool_name(raw_bad, name_map=mapper)
    norm_schema = _weather_tool_schema(norm)
    ok_norm = await provider.complete(
        [{"role": "user", "content": "Weather in Paris."}],
        tools=norm_schema,
        max_tokens=64,
        thinking_enabled=False,
        tool_choice=_tool_choice(norm),
    )
    assert ok_norm.http_status == 200
    assert provider.last_request_bodies[-1]["tools"][0]["function"]["name"] == norm

    long70 = "w" * 70
    norm70 = normalise_tool_name(long70, name_map=mapper)
    schema70 = _weather_tool_schema(norm70)
    ok70 = await provider.complete(
        [{"role": "user", "content": "Weather in Paris."}],
        tools=schema70,
        max_tokens=64,
        thinking_enabled=False,
        tool_choice=_tool_choice(norm70),
    )
    await provider.aclose()
    assert ok70.http_status == 200
    assert provider.last_request_bodies[-1]["tools"][0]["function"]["name"] == norm70


@pytest.mark.asyncio
async def test_live_thinking_tools_reasoning_replay_two_rounds() -> None:
    ledger = UsageLedger(task_id="live-reasoning")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-reasoning",
    )
    tools = _weather_tool_schema()
    q = (
        "First compute 37*43 and 29*53 and decide which is larger "
        "(think it through briefly), then call get_weather for Paris."
    )
    first = await provider.complete(
        [{"role": "user", "content": q}],
        tools=tools,
        max_tokens=512,
        thinking_enabled=True,
    )
    for attempt in range(3):
        if first.http_status == 200 and first.tool_calls and first.reasoning_content:
            break
        if attempt >= 2:
            break
        first = await provider.complete(
            [{"role": "user", "content": q}],
            tools=tools,
            max_tokens=512,
            thinking_enabled=True,
        )
    assert first.http_status == 200 and first.tool_calls, _reasoning_diag(provider, first)
    assert first.reasoning_content, _reasoning_diag(provider, first)
    _assert_turn_from_deepseek(first, status=200, label="reasoning_r1")
    print(_live_metric_line("reasoning_r1", first))

    history: list[dict] = [
        {"role": "user", "content": q},
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
        history + [{"role": "user", "content": "Now call get_weather for Tokyo."}],
        tools=tools,
        max_tokens=512,
        thinking_enabled=True,
    )
    for attempt in range(3):
        if second.http_status == 200 and second.tool_calls and second.reasoning_content:
            break
        if attempt >= 2:
            break
        second = await provider.complete(
            history + [{"role": "user", "content": "Now call get_weather for Tokyo."}],
            tools=tools,
            max_tokens=512,
            thinking_enabled=True,
        )
    assert second.http_status == 200 and second.tool_calls, _reasoning_diag(provider, second)
    assert second.reasoning_content, _reasoning_diag(provider, second)
    _assert_turn_from_deepseek(second, status=200, label="reasoning_r2")
    print(_live_metric_line("reasoning_r2", second))
    history.extend(
        [
            {"role": "user", "content": "Now call get_weather for Tokyo."},
            {
                "role": "assistant",
                "content": second.content or "",
                "reasoning_content": second.reasoning_content,
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
            },
            {
                "role": "tool",
                "tool_call_id": second.tool_calls[0].id,
                "content": json.dumps({"temp_c": 18}),
            },
        ]
    )

    summarize = [{"role": "user", "content": "Summarize both cities in one sentence."}]
    good = await provider.complete(
        history + summarize,
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert good.http_status == 200, _reasoning_diag(provider, good)
    _assert_turn_from_deepseek(good, status=200, label="reasoning_forward_summarize")
    print(_live_metric_line("reasoning_forward_summarize", good))

    stripped = json.loads(json.dumps(history))
    for msg in stripped:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
    bodies_before = len(provider.last_request_bodies)
    strip_turn = await provider.complete(
        stripped + summarize,
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    assert len(provider.last_request_bodies) > bodies_before
    _assert_turn_from_deepseek(strip_turn, status=strip_turn.http_status, label="reasoning_strip")
    print(_live_metric_line("reasoning_strip_passback", strip_turn))
    if strip_turn.http_status == 400:
        assert "reasoning_content" in strip_turn.error_body.lower()
    else:
        print(
            "LIVE_NOTE stripped multi-turn passback without reasoning_content: "
            f"DeepSeek returned HTTP {strip_turn.http_status} (not 400); "
            f"finish_reason={strip_turn.finish_reason!r} "
            f"body={(strip_turn.error_body or strip_turn.content or '')[:200]!r}"
        )

    # fixtures g7/g9: server-measured thinking passback requirement (CONSTRAINTS g7/g9)
    g7_tc = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_00_x",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }
        ],
    }
    g7_msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        g7_tc,
        {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"},
    ]
    g7_bad = await provider.complete(
        g7_msgs,
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    _assert_turn_from_deepseek(g7_bad, status=400, label="reasoning_g7_reverse")
    assert "reasoning_content" in g7_bad.error_body.lower()
    print(_live_metric_line("reasoning_g7_reverse", g7_bad))

    g9_tc = dict(g7_tc)
    g9_tc["reasoning_content"] = "I should call get_weather for Paris."
    g9_good = await provider.complete(
        [
            {"role": "user", "content": "Weather in Paris?"},
            g9_tc,
            {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"},
        ],
        tools=tools,
        max_tokens=256,
        thinking_enabled=True,
    )
    _assert_turn_from_deepseek(g9_good, status=200, label="reasoning_g9_control")
    print(_live_metric_line("reasoning_g9_control", g9_good))
    await provider.aclose()
    assert g9_good.http_status == 200


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
                "user_message": (
                    "Use write_file to create poem.txt containing a 20-line poem about the sea. "
                    "Call write_file immediately with a long content string."
                ),
                "max_tokens": 40,
                "max_steps": 5,
                "thinking_enabled": False,
                "tool_choice": _tool_choice("write_file"),
                "reject_length_without_tool_calls": True,
                "max_length_empty_retries": 3,
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
        max_steps=5,
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
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    await store.close()
    await provider.aclose()

    assert len(tool_rows) == 0
    trunc_events = [json.loads(r["data_json"]) for r in repairs if "truncation" in r["data_json"]]
    assert trunc_events, f"run={run} responses={model_responses}"
    length_rows = [
        json.loads(r["data_json"])
        for r in model_responses
        if json.loads(r["data_json"]).get("finish_reason") == "length"
    ]
    assert length_rows, f"run={run} responses={model_responses}"
    assert any(r.get("tool_calls", 0) > 0 for r in length_rows), length_rows
    assert trunc_events[0].get("finish_reason") == "length"
