from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_harness.core.budget import (
    DEFAULT_FINALIZATION_USER_TEXT,
    cache_prefix_fingerprint,
)
from probe_harness.core.loop import run_agent_loop
from probe_harness.providers.fake import FakeProvider
from probe_harness.providers.guards import repair_tool_pairing
from probe_harness.store.db import Store
from probe_harness.tools.registry import TOOL_SCHEMAS


def _usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


class RecordingFakeProvider(FakeProvider):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.requests: list[list[dict]] = []

    async def complete(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(list(messages))
        return await super().complete(messages, tools=tools, **kwargs)


@pytest.mark.asyncio
async def test_budget_levels_and_finalizing_tool_rejected(tmp_path: Path) -> None:
    task_path = tmp_path / "budget_task.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "work",
                "max_steps": 10,
                "budget": {"tokens": [50, 80, 120], "calls": [99, 99, 99]},
                "fake_script": [
                    {
                        "tool_calls": [
                            {
                                "id": "c0",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo s0"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(40, 10),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo s1"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(20, 10),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "c_fin",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo x"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(5, 5),
                    },
                ],
            }
        )
    )
    store = Store(tmp_path / "b.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(json.loads(task_path.read_text())["fake_script"]),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=10,
    )
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    budget_events = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'budget_state' ORDER BY seq",
        (run_id,),
    )
    repairs = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'protocol_repair'",
        (run_id,),
    )
    ckpts = await store.fetchall(
        "SELECT budget_json FROM checkpoints WHERE run_id = ? ORDER BY step",
        (run_id,),
    )
    await store.close()
    assert run["status"] == "failed"
    assert run["stop_reason"] == "run_budget"
    levels = [json.loads(r["data_json"]).get("combined_level") for r in budget_events]
    assert "warning" in levels
    assert "finalizing" in levels
    assert any("budget_finalization" in r["data_json"] for r in repairs)
    assert any(json.loads(c["budget_json"]).get("finalization_appended") is True for c in ckpts)


@pytest.mark.asyncio
async def test_finalization_appends_wrap_up_and_extra_model_call(tmp_path: Path) -> None:
    fin_text = "FINALIZE_WRAP_UP_TEST_MESSAGE"
    task_path = tmp_path / "fin_round.json"
    script = [
        {
            "tool_calls": [
                {
                    "id": "c0",
                    "function": {"name": "bash", "arguments": '{"command":"echo s0"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": _usage(40, 10),
        },
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "bash", "arguments": '{"command":"echo s1"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": _usage(20, 10),
        },
        {
            "tool_calls": [
                {
                    "id": "c_fin",
                    "function": {"name": "bash", "arguments": '{"command":"echo x"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": _usage(5, 5),
        },
    ]
    task_path.write_text(
        json.dumps(
            {
                "user_message": "work",
                "max_steps": 10,
                "budget": {
                    "tokens": [50, 80, 120],
                    "calls": [99, 99, 99],
                    "finalization_user_message": fin_text,
                },
                "fake_script": script,
            }
        )
    )
    store = Store(tmp_path / "fin.db")
    await store.connect()
    provider = RecordingFakeProvider(script)
    await run_agent_loop(
        store, provider, task_path, provider_name="fake", model="fake", max_steps=10
    )
    await store.close()
    assert len(provider.requests) >= 3
    sent_wrap = any(
        any(m.get("role") == "user" and fin_text in str(m.get("content", "")) for m in req)
        for req in provider.requests
    )
    assert sent_wrap
    post_wrap_calls = 0
    for req in provider.requests:
        if any(fin_text in str(m.get("content", "")) for m in req if m.get("role") == "user"):
            post_wrap_calls += 1
    assert post_wrap_calls >= 1


@pytest.mark.asyncio
async def test_finalization_keeps_cache_prefix_stable(tmp_path: Path) -> None:
    task_path = tmp_path / "cache_task.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "work",
                "max_steps": 5,
                "budget": {"tokens": [10, 20, 100], "calls": [99, 99, 99]},
                "fake_script": [
                    {
                        "tool_calls": [
                            {
                                "id": "t0",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo a"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(5, 5),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "t1",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo b"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(8, 7),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "t2",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo c"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(1, 1),
                    },
                ],
            }
        )
    )
    store = Store(tmp_path / "c.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(json.loads(task_path.read_text())["fake_script"]),
        task_path,
        provider_name="fake",
        model="fake",
    )
    ctx = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'context_prepared'",
        (run_id,),
    )
    fin_text = DEFAULT_FINALIZATION_USER_TEXT
    msgs_rows = await store.fetchall(
        "SELECT messages_json FROM checkpoints WHERE run_id = ? ORDER BY step, boundary",
        (run_id,),
    )
    await store.close()
    payloads = [json.loads(row["data_json"]) for row in ctx]
    assert payloads
    first_fp = payloads[0]["cache_prefix_fingerprint"]
    assert all(p.get("cache_prefix_stable") is True for p in payloads)
    assert all(p["cache_prefix_fingerprint"] == first_fp for p in payloads)
    assert any(fin_text in json.dumps(json.loads(r["messages_json"])) for r in msgs_rows)


@pytest.mark.asyncio
async def test_finalizing_history_still_pairs(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
    ]
    repaired, repairs = repair_tool_pairing(messages, reason="run budget finalization")
    assert repairs
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "x1" for m in repaired)


@pytest.mark.asyncio
async def test_submit_goal_fail_then_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_WORKSPACE", str(tmp_path))
    out = tmp_path / "answer.txt"
    task_path = tmp_path / "submit_task.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "finish",
                "max_steps": 8,
                "goal": {"type": "file_contains", "path": "answer.txt", "text": "ok"},
                "fake_script": [
                    {
                        "tool_calls": [
                            {
                                "id": "s1",
                                "function": {
                                    "name": "submit",
                                    "arguments": '{"summary":"try1"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(10, 5),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "w1",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": "answer.txt", "content": "ok"}
                                    ),
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(10, 5),
                    },
                    {
                        "tool_calls": [
                            {
                                "id": "s2",
                                "function": {
                                    "name": "submit",
                                    "arguments": '{"summary":"try2"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                        "usage": _usage(10, 5),
                    },
                ],
            }
        )
    )
    store = Store(tmp_path / "s.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(json.loads(task_path.read_text())["fake_script"]),
        task_path,
        provider_name="fake",
        model="fake",
    )
    run = await store.fetchone("SELECT status, stop_reason FROM runs WHERE id = ?", (run_id,))
    goal_events = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'goal_check'",
        (run_id,),
    )
    submitted = await store.fetchall(
        "SELECT 1 FROM events WHERE run_id = ? AND type = 'submitted'",
        (run_id,),
    )
    await store.close()
    assert out.read_text() == "ok"
    assert run["status"] == "submitted"
    assert run["stop_reason"] == "goal_met"
    assert len(goal_events) == 2
    first = json.loads(goal_events[0]["data_json"])
    second = json.loads(goal_events[1]["data_json"])
    assert first["passed"] is False
    assert second["passed"] is True
    assert submitted


@pytest.mark.asyncio
async def test_repeated_identical_call_guard(tmp_path: Path) -> None:
    same = {
        "tool_calls": [
            {
                "id": "b1",
                "function": {"name": "bash", "arguments": '{"command":"echo same"}'},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": _usage(5, 5),
    }
    task_path = tmp_path / "repeat.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "loop",
                "max_steps": 10,
                "fake_script": [same, same, same],
            }
        )
    )
    store = Store(tmp_path / "r.db")
    await store.connect()
    run_id = await run_agent_loop(
        store,
        FakeProvider(json.loads(task_path.read_text())["fake_script"]),
        task_path,
        provider_name="fake",
        model="fake",
    )
    run = await store.fetchone("SELECT stop_reason FROM runs WHERE id = ?", (run_id,))
    await store.close()
    assert run["stop_reason"] == "repeated_identical_call"


@pytest.mark.asyncio
async def test_recover_restores_run_budget_counters(tmp_path: Path) -> None:
    import uuid

    from probe_harness.runs.state import STATUS_INTERRUPTED

    task_path = tmp_path / "recover_budget_task.json"
    task_path.write_text(
        json.dumps(
            {
                "user_message": "x",
                "max_steps": 5,
                "budget": {"tokens": [100, 200, 300], "calls": [99, 99, 99]},
                "fake_script": [
                    {
                        "content": "done",
                        "finish_reason": "stop",
                        "usage": _usage(3, 2),
                    }
                ],
            }
        )
    )
    source_id = str(uuid.uuid4())
    store = Store(tmp_path / "recover_budget.db")
    await store.connect()
    await store.execute(
        """
        INSERT INTO runs (id, task_ref, status, stop_reason, provider, model, config_hash, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, str(task_path), STATUS_INTERRUPTED, "host_restart", "fake", "fake", "abc", 1.0),
    )
    budget_state = {
        "chargeable_tokens": 12,
        "model_calls": 2,
        "tokens_level": "warning",
        "calls_level": "normal",
        "combined_level": "warning",
        "finalization_appended": False,
        "finalization_wrap_up_pending": False,
    }
    meta = json.dumps(
        {"fake_script_index": 0, "budget": budget_state},
        ensure_ascii=False,
    )
    await store.execute(
        """
        INSERT INTO checkpoints (run_id, step, boundary, messages_json, usage_json, budget_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            0,
            "post_tools",
            json.dumps(
                [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "x"},
                ]
            ),
            meta,
            json.dumps(budget_state),
        ),
    )
    run_id = await run_agent_loop(
        store,
        FakeProvider(json.loads(task_path.read_text())["fake_script"]),
        task_path,
        provider_name="fake",
        model="fake",
        max_steps=5,
        recover_from=source_id,
    )
    budget_events = await store.fetchall(
        "SELECT data_json FROM events WHERE run_id = ? AND type = 'budget_state' ORDER BY seq",
        (run_id,),
    )
    await store.close()
    assert budget_events
    snapshots = [
        json.loads(r["data_json"])
        for r in budget_events
        if "chargeable_tokens" in json.loads(r["data_json"])
    ]
    assert snapshots
    last = snapshots[-1]
    assert last["chargeable_tokens"] == 17
    assert last["model_calls"] == 3


def test_cache_prefix_fingerprint_ignores_user_messages() -> None:
    tools = TOOL_SCHEMAS[:2]
    sys_msg = [{"role": "system", "content": "same"}]
    fp1 = cache_prefix_fingerprint([*sys_msg, {"role": "user", "content": "a"}], tools)
    fp2 = cache_prefix_fingerprint(
        [*sys_msg, {"role": "user", "content": "finalization text"}], tools
    )
    assert fp1 == fp2
