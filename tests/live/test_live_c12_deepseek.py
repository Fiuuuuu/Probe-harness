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
        f"LIVE_METRIC probe_d http_status={turn.http_status} "
        f"finish_reason={turn.finish_reason} usage={json.dumps(turn.usage, sort_keys=True)}"
    )


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
    assert norm2["input_total"] > 0
    if norm1["input_total"] > 100:
        assert norm2["cached_read"] >= norm1["cached_read"] or norm2["cached_read"] > 0


@pytest.mark.asyncio
async def test_live_thinking_modes_explicit() -> None:
    ledger = UsageLedger(task_id="live-thinking-modes")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        test_run_id="live-thinking-modes",
    )
    prompt = [{"role": "user", "content": "Reply with exactly: mode-check"}]
    off = await provider.complete(prompt, max_tokens=32, thinking_enabled=False)
    assert off.http_status == 200
    body_off = provider.last_request_bodies[-1]
    assert body_off.get("thinking") == {"type": "disabled"}

    on = await provider.complete(prompt, max_tokens=64, thinking_enabled=True)
    await provider.aclose()
    assert on.http_status == 200
    body_on = provider.last_request_bodies[-1]
    assert body_on.get("thinking") == {"type": "enabled"}
    assert body_on.get("reasoning_effort") == "high"
    details = (on.usage or {}).get("completion_tokens_details") or {}
    print(
        f"LIVE_METRIC thinking_off usage={json.dumps(off.usage, sort_keys=True)} "
        f"thinking_on reasoning_tokens={details.get('reasoning_tokens')}"
    )
