from __future__ import annotations

from probe_harness.providers.anthropic import (
    build_anthropic_request_body,
    convert_openai_messages_to_anthropic,
    merge_system_messages,
)


def test_two_system_messages_merge() -> None:
    messages = [
        {"role": "system", "content": "You are A."},
        {"role": "system", "content": "Always be concise."},
        {"role": "user", "content": "hello"},
    ]
    system, converted = convert_openai_messages_to_anthropic(messages)
    assert system == "You are A.\nAlways be concise."
    assert converted == [{"role": "user", "content": "hello"}]
    body = build_anthropic_request_body(messages, model="MiniMax-M3", max_tokens=64)
    assert body["system"] == system
    assert "messages" in body


def test_tool_results_single_user_message() -> None:
    messages = [
        {"role": "user", "content": "call weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp":20}'},
        {"role": "tool", "tool_call_id": "call_2", "content": "ignored orphan in test"},
    ]
    _, converted = convert_openai_messages_to_anthropic(messages)
    user_blocks = [m for m in converted if m["role"] == "user"]
    assert len(user_blocks) == 2
    tool_turn = user_blocks[1]["content"]
    assert isinstance(tool_turn, list)
    assert len(tool_turn) == 2
    assert tool_turn[0]["type"] == "tool_result"
    assert tool_turn[0]["tool_use_id"] == "call_1"


def test_merge_system_messages_helper() -> None:
    assert merge_system_messages([{"role": "user", "content": "x"}]) is None
    assert merge_system_messages([{"role": "system", "content": "a"}]) == "a"
