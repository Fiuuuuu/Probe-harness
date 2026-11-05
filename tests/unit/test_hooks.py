from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.hooks import HookBus, HookEvent, HookName, default_hook_bus
from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.store.db import EventWriter, Store


@pytest.mark.asyncio
async def test_hook_order_pre_post_stop(tmp_path: Path) -> None:
    db = tmp_path / "hooks.db"
    store = Store(db)
    await store.connect()
    events = EventWriter(store)
    bus = default_hook_bus(store=store, events=events)
    order: list[str] = []

    async def pre(_: HookEvent) -> None:
        order.append("PreToolUse")

    async def post(_: HookEvent) -> None:
        order.append("PostToolUse")

    async def stop(_: HookEvent) -> None:
        order.append("Stop")

    bus.on(HookName.PRE_TOOL_USE, pre)
    bus.on(HookName.POST_TOOL_USE, post)
    bus.on(HookName.STOP, stop)

    task = Path("tests/fixtures/echo_task.json")
    provider = FakeProvider(json.loads(task.read_text())["fake_script"])
    run_id = await run_agent_loop(
        store,
        provider,
        task,
        provider_name="fake",
        model="fake",
        hook_bus=bus,
    )

    assert "PreToolUse" in order
    assert "PostToolUse" in order
    assert order.index("PreToolUse") < order.index("PostToolUse")
    assert order[-1] == "Stop"
    assert bus.metrics.tool_calls == 2

    budget = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'budget_state'",
        (run_id,),
    )
    assert budget
    payload = json.loads(budget[-1]["data_json"])
    assert payload["metrics_from_hooks"]["tool_calls"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_custom_hook_bus_without_default_metrics(tmp_path: Path) -> None:
    db = tmp_path / "bare.db"
    store = Store(db)
    await store.connect()
    bus = HookBus()
    seen: list[HookName] = []

    async def capture(ev: HookEvent) -> None:
        seen.append(ev.name)

    for name in HookName:
        bus.on(name, capture)

    task = Path("tests/fixtures/echo_task.json")
    provider = FakeProvider(json.loads(task.read_text())["fake_script"])
    await run_agent_loop(
        store,
        provider,
        task,
        provider_name="fake",
        model="fake",
        hook_bus=bus,
    )
    assert HookName.PRE_TOOL_USE in seen
    assert HookName.POST_TOOL_USE in seen
    assert seen.count(HookName.STOP) == 1
    tools = await store.fetchall("SELECT * FROM tool_calls")
    assert len(tools) == 0
    await store.close()
