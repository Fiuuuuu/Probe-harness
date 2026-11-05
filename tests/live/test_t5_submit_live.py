from __future__ import annotations

import os
from pathlib import Path

import pytest

from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.store.db import Store

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_submit_goal_passes(tmp_path: Path) -> None:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE not set")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

    os.environ["HARNESS_WORKSPACE"] = str(tmp_path)
    db = tmp_path / "live_t5_submit.db"
    store = Store(db)
    await store.connect()
    task = Path("tests/fixtures/live_t5_submit_goal.json")
    ledger = UsageLedger(task_id="live-t5-submit")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        max_tokens=512,
        test_run_id="live-t5-submit",
    )
    try:
        run_id = await run_agent_loop(
            store,
            provider,
            task,
            provider_name="deepseek",
            model="deepseek-flash",
            max_steps=6,
        )
    finally:
        await provider.aclose()

    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    assert run is not None
    if run["stop_reason"] == "token_cap":
        pytest.fail("token cap reached during live T5 submit test")
    assert run["status"] == "submitted", run
    assert run["stop_reason"] == "goal_met"

    model_calls = await store.fetchall(
        "SELECT step, input_total, output, stop_reason FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    assert 1 <= len(model_calls) <= 6
    total_tokens = sum(r["input_total"] + r["output"] for r in model_calls)
    assert total_tokens <= 8000

    last_mc = model_calls[-1]
    print(
        f"LIVE_METRIC t5_submit_goal status={run['status']} stop_reason={run['stop_reason']} "
        f"model_calls={len(model_calls)} step_tokens={last_mc['input_total']}+{last_mc['output']} "
        f"finish_reason={last_mc['stop_reason']}"
    )
    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")
    await store.close()
