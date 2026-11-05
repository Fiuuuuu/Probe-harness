"""Live DeepSeek probes ported from fixtures/probe_scripts (C12 LIVE TEST rows)."""

from __future__ import annotations

import json

import pytest

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.providers.usage import normalise_openai_usage

pytestmark = [pytest.mark.live]

PREFIX = (
    "You are a meticulous code-review assistant. Follow these numbered house rules exactly.\n"
    + "\n".join(
        f"Rule {i}: When reviewing module m{i:03d}, check that function f{i}_{(i * 7) % 13} "
        f"validates argument x{i % 5} and logs at level {['debug', 'info', 'warn'][i % 3]} "
        f"before returning code {1000 + i}."
        for i in range(1, 60)
    )
)


@pytest.mark.asyncio
async def test_live_two_consecutive_user_messages_probe_d() -> None:
    ledger = UsageLedger(task_id="live-d")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-d",
    )
    messages = [
        {"role": "user", "content": "Say A."},
        {"role": "user", "content": "Now say B."},
    ]
    turn = await provider.complete(
        messages,
        max_tokens=32,
        thinking_enabled=False,
    )
    await provider.aclose()
    assert turn.http_status == 200, turn.error_body
    assert turn.http_calls
    print(
        f"LIVE_METRIC probe_d response_model={turn.response_model} "
        f"finish_reason={turn.finish_reason} "
        f"http_status={turn.http_status} "
        f"usage={json.dumps(turn.usage, sort_keys=True)}"
    )


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
async def test_live_g2_g6_reasoning_strip_regression() -> None:
    """C12 regression: API may return 200 without reasoning_content replay (harness must still store it)."""
    ledger = UsageLedger(task_id="live-g2-g6")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-g2-g6",
    )
    tools = _weather_tools()
    prompt = "Call get_weather for Paris now. Do not deliberate."
    first = None
    for attempt in range(4):
        first = await provider.complete(
            [{"role": "user", "content": prompt}],
            tools=tools,
            max_tokens=256,
            include_thinking_param=False,
        )
        if first.http_status == 200 and first.tool_calls:
            break
        if attempt >= 3:
            break
    assert first is not None
    assert first.http_status == 200 and first.tool_calls, first.error_body
    tc = first.tool_calls[0]
    stripped_assistant = {
        "role": "assistant",
        "content": first.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
        ],
    }
    g2_history = [
        {"role": "user", "content": prompt},
        stripped_assistant,
        {"role": "tool", "tool_call_id": tc.id, "content": "18C sunny"},
    ]
    g2 = await provider.complete(
        g2_history,
        tools=tools,
        max_tokens=128,
        include_thinking_param=False,
    )
    assert g2.http_status == 200, g2.error_body
    assert "thinking" not in provider.last_request_bodies[-1]
    print(
        f"LIVE_METRIC probe_g2_drop_reasoning response_model={g2.response_model} "
        f"finish_reason={g2.finish_reason} usage={json.dumps(g2.usage, sort_keys=True)}"
    )
    print(
        "LIVE_NOTE g2: DeepSeek returned HTTP 200 after assistant tool turn without "
        "reasoning_content; harness must still replay reasoning on real runs."
    )

    tc_static = {
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
    g6_messages = [
        {"role": "user", "content": "Weather in Paris?"},
        tc_static,
        {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"},
        {"role": "assistant", "content": "Paris: 18C sunny."},
        {"role": "user", "content": "Say OK."},
    ]
    g6 = await provider.complete(
        g6_messages,
        tools=tools,
        max_tokens=64,
        include_thinking_param=False,
    )
    await provider.aclose()
    assert g6.http_status == 200, g6.error_body
    print(
        f"LIVE_METRIC probe_g6_next_user_turn response_model={g6.response_model} "
        f"finish_reason={g6.finish_reason} usage={json.dumps(g6.usage, sort_keys=True)}"
    )
    print("LIVE_NOTE g6: thinking default with no reasoning_content in history still HTTP 200.")


@pytest.mark.asyncio
async def test_live_cache_repeat_probe_h() -> None:
    ledger = UsageLedger(task_id="live-h")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        test_run_id="live-h",
    )
    messages = [
        {"role": "system", "content": PREFIX},
        {"role": "user", "content": "Reply with just OK."},
    ]
    first = await provider.complete(messages, max_tokens=8, thinking_enabled=False)
    assert first.http_status == 200, first.error_body
    norm1 = normalise_openai_usage(first.usage or {})
    second = await provider.complete(messages, max_tokens=8, thinking_enabled=False)
    await provider.aclose()
    assert second.http_status == 200, second.error_body
    norm2 = normalise_openai_usage(second.usage or {})
    print(
        f"LIVE_METRIC probe_h1 input_total={norm1['input_total']} cached={norm1['cached_read']} "
        f"uncached={norm1['uncached']}"
    )
    print(
        f"LIVE_METRIC probe_h2 input_total={norm2['input_total']} cached={norm2['cached_read']} "
        f"uncached={norm2['uncached']}"
    )
    assert norm1["input_total"] > 500
    assert norm2["cached_read"] > 0, f"expected cache hit on repeat; h1={norm1} h2={norm2}"


@pytest.mark.asyncio
async def test_live_thinking_modes_explicit() -> None:
    ledger = UsageLedger(task_id="live-thinking-modes")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-thinking-modes",
    )
    prompt = [
        {
            "role": "user",
            "content": (
                "A shop sells notebooks for 7 yuan each and pens for 3 yuan each. "
                "Mei buys 4 notebooks and 5 pens, then returns 2 pens. "
                "How much did she spend in yuan? Show your reasoning, then give the final "
                "number on the last line only."
            ),
        }
    ]
    off = await provider.complete(prompt, max_tokens=128, thinking_enabled=False)
    assert off.http_status == 200
    body_off = provider.last_request_bodies[-1]
    assert body_off.get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in body_off or body_off.get("reasoning_effort") is None

    on = await provider.complete(prompt, max_tokens=512, thinking_enabled=True)
    await provider.aclose()
    assert on.http_status == 200
    body_on = provider.last_request_bodies[-1]
    assert body_on.get("thinking") == {"type": "enabled"}
    assert body_on.get("reasoning_effort") == "high"
    off_details = (off.usage or {}).get("completion_tokens_details") or {}
    on_details = (on.usage or {}).get("completion_tokens_details") or {}
    off_reasoning = int(off_details.get("reasoning_tokens") or 0)
    on_reasoning = int(on_details.get("reasoning_tokens") or 0)
    off_rc = (off.reasoning_content or "").strip()
    on_rc = (on.reasoning_content or "").strip()
    assert off_reasoning == 0, json.dumps(off.usage, sort_keys=True)
    assert not off_rc, f"unexpected reasoning_content with thinking off: {off_rc[:200]!r}"
    assert on_reasoning > 0 or bool(on_rc), (
        f"thinking on must produce reasoning_tokens or reasoning_content; "
        f"usage={json.dumps(on.usage, sort_keys=True)} reasoning_content_len={len(on_rc)}"
    )
    print(
        f"LIVE_METRIC thinking_off response_model={off.response_model} "
        f"finish_reason={off.finish_reason} reasoning_tokens={off_reasoning} "
        f"reasoning_content_len={len(off_rc)} "
        f"usage={json.dumps(off.usage, sort_keys=True)}"
    )
    print(
        f"LIVE_METRIC thinking_on response_model={on.response_model} "
        f"finish_reason={on.finish_reason} reasoning_tokens={on_reasoning} "
        f"reasoning_content_len={len(on_rc)} "
        f"usage={json.dumps(on.usage, sort_keys=True)}"
    )
