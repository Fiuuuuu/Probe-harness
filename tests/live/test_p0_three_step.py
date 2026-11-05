from __future__ import annotations

import os
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.store.db import Store


@pytest.mark.asyncio
async def test_three_step_tool_task_live(tmp_path: Path) -> None:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE not set")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

    db = tmp_path / "live.db"
    store = Store(db)
    await store.connect()
    task = Path("tests/fixtures/live_three_step_tools.json")
    ledger = UsageLedger(path=tmp_path / "ledger.jsonl", task_id="live-three-step")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=True,
        max_tokens=512,
        test_run_id="live-three-step",
    )
    run_id = await run_agent_loop(
        store,
        provider,
        task,
        provider_name="deepseek",
        model="deepseek-flash",
        max_steps=8,
    )

    calls = await store.fetchall(
        "SELECT input_total, output, uncached FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    assert len(calls) >= 1
    total_in = sum(c["input_total"] for c in calls)
    total_out = sum(c["output"] for c in calls)
    assert total_in + total_out > 0

    assert calls[0]["input_total"] > 0

    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")
    await store.close()
