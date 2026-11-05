from __future__ import annotations

import os

import pytest

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.providers.usage import estimate_chat_input_tokens

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_stream_one_completion_live() -> None:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE not set")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

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
    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")
