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
async def test_three_step_tool_task_live(tmp_path: Path) -> None:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE not set")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

    db = tmp_path / "live.db"
    store = Store(db)
    await store.connect()
    task = Path("tests/fixtures/live_three_step_tools.json")
    ledger = UsageLedger(task_id="live-three-step")
    provider = OpenAIChatProvider(
        ledger=ledger,
        thinking_enabled=False,
        max_tokens=1024,
        test_run_id="live-three-step",
    )
    try:
        run_id = await run_agent_loop(
            store,
            provider,
            task,
            provider_name="deepseek",
            model="deepseek-flash",
            max_steps=8,
        )
    finally:
        await provider.aclose()

    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    assert run is not None
    if run["stop_reason"] == "token_cap":
        pytest.skip("token cap reached")
    assert run["status"] == "completed"

    model_calls = await store.fetchall(
        "SELECT input_total, output FROM model_calls WHERE run_id = ?",
        (run_id,),
    )
    assert len(model_calls) >= 1
    total_in = sum(c["input_total"] for c in model_calls)
    total_out = sum(c["output"] for c in model_calls)
    assert total_in + total_out > 0

    tool_rows = await store.fetchall("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,))
    assert len(tool_rows) >= 3

    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")
    await store.close()
