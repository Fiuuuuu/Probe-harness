"""Protocol guards C1, C3 (OpenAI wire), C6."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from probe_harness.providers.base import ToolCallSpec
from probe_harness.providers.usage import normalise_anthropic_usage, normalise_openai_usage

TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "tool_use"})

THINKING_REASONING_PASSBACK_ERROR = (
    "The `reasoning_content` in the thinking mode must be passed back to the API."
)


def validate_thinking_reasoning_passback(
    messages: list[dict[str, Any]],
    *,
    thinking_enabled: bool,
) -> str | None:
    """C5/C12: with thinking enabled, every assistant tool turn needs reasoning_content."""
    if not thinking_enabled:
        return None
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        if "reasoning_content" not in msg:
            return THINKING_REASONING_PASSBACK_ERROR
    return None


def thinking_passback_error_body(message: str) -> str:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request_error",
            }
        }
    )


def _system_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return ""


def merge_system_messages_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """C3: keep initial system at top; late system text appended to last user message."""
    prefix_system: list[str] = []
    rest: list[dict[str, Any]] = []
    late_system: list[str] = []
    started = False

    for msg in messages:
        if msg.get("role") == "system":
            text = _system_text(msg.get("content"))
            if text:
                if started:
                    late_system.append(text)
                else:
                    prefix_system.append(text)
            continue
        started = True
        rest.append(dict(msg))

    if late_system and rest:
        for idx in range(len(rest) - 1, -1, -1):
            if rest[idx].get("role") == "user":
                user = dict(rest[idx])
                base = user.get("content")
                suffix = "\n".join(late_system)
                if isinstance(base, str) and base:
                    user["content"] = f"{base}\n\n{suffix}"
                elif isinstance(base, str):
                    user["content"] = suffix
                else:
                    user["content"] = suffix
                rest[idx] = user
                break

    if prefix_system:
        return [{"role": "system", "content": "\n".join(prefix_system)}] + rest
    return rest


def _pairing_repair_event(
    action: str,
    *,
    tool_call_id: str | None,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "pairing",
        "action": action,
        "reason": reason,
    }
    if tool_call_id is not None:
        payload["tool_call_id"] = tool_call_id
    payload.update(extra)
    return payload


def repair_tool_pairing(
    messages: list[dict[str, Any]],
    *,
    reason: str = "interrupted",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """C1: dedupe, drop orphans/mismatches, relocate results, synthesise missing."""
    tagged = [(i, deepcopy(m)) for i, m in enumerate(messages)]
    repairs: list[dict[str, Any]] = []

    valid_ids: set[str] = set()
    for _, m in tagged:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tid = tc.get("id")
                if tid:
                    valid_ids.add(tid)

    kept: list[tuple[int, dict[str, Any]]] = []
    for oi, m in tagged:
        if m.get("role") == "tool":
            tid = m.get("tool_call_id")
            if tid not in valid_ids:
                repairs.append(
                    _pairing_repair_event(
                        "drop_orphan_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )
                continue
        kept.append((oi, m))
    tagged = kept

    consumed: set[int] = set()
    rebuilt: list[dict[str, Any]] = []
    ti = 0
    while ti < len(tagged):
        oi, m = tagged[ti]
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            if m.get("role") == "tool":
                repairs.append(
                    _pairing_repair_event(
                        "drop_non_adjacent_tool_result",
                        tool_call_id=m.get("tool_call_id"),
                        reason=reason,
                    )
                )
                consumed.add(oi)
                ti += 1
                continue
            rebuilt.append(m)
            ti += 1
            continue

        wanted = [tc["id"] for tc in m["tool_calls"] if tc.get("id")]
        rebuilt.append(m)
        ti += 1
        pool: dict[str, dict[str, Any]] = {}

        while ti < len(tagged) and tagged[ti][1].get("role") == "tool":
            t_oi, tm = tagged[ti]
            tid = tm.get("tool_call_id")
            if tid not in wanted:
                repairs.append(
                    _pairing_repair_event(
                        "drop_mismatched_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )
            elif tid in pool:
                repairs.append(
                    _pairing_repair_event(
                        "drop_duplicate_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )
            else:
                pool[tid] = tm
            consumed.add(t_oi)
            ti += 1

        for j in range(len(tagged)):
            t_oi, tm = tagged[j]
            if t_oi in consumed or tm.get("role") != "tool":
                continue
            tid = tm.get("tool_call_id")
            if tid not in wanted:
                continue
            if tid in pool:
                repairs.append(
                    _pairing_repair_event(
                        "drop_duplicate_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )
            else:
                repairs.append(
                    _pairing_repair_event(
                        "relocate_non_adjacent_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )
                pool[tid] = tm
            consumed.add(t_oi)

        for tid in wanted:
            if tid in pool:
                rebuilt.append(pool[tid])
            else:
                payload = json.dumps(
                    {"error": f"tool call interrupted: {reason}", "is_error": True},
                    ensure_ascii=False,
                )
                rebuilt.append({"role": "tool", "tool_call_id": tid, "content": payload})
                repairs.append(
                    _pairing_repair_event(
                        "synthetic_tool_result",
                        tool_call_id=tid,
                        reason=reason,
                    )
                )

    for oi, tm in tagged:
        if oi in consumed or tm.get("role") != "tool":
            continue
        repairs.append(
            _pairing_repair_event(
                "drop_orphan_tool_result",
                tool_call_id=tm.get("tool_call_id"),
                reason=reason,
            )
        )

    return rebuilt, repairs


def arguments_json_valid(raw: str | None) -> bool:
    if raw is None:
        return False
    if raw == "":
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def _normalised_output_tokens(usage: dict[str, Any], provider_name: str) -> int:
    if provider_name.endswith("-anthropic") or provider_name == "anthropic":
        return normalise_anthropic_usage(usage or {})["output"]
    return normalise_openai_usage(usage or {})["output"]


def is_truncated_tool_call(
    tc: ToolCallSpec,
    *,
    finish_reason: str,
    max_tokens: int,
    usage: dict[str, Any],
    provider_name: str = "deepseek",
) -> bool:
    """C6: never execute truncated / incomplete tool arguments."""
    output = _normalised_output_tokens(usage, provider_name)
    if max_tokens > 0 and output >= max_tokens:
        return True
    if finish_reason in TRUNCATION_FINISH_REASONS:
        return True
    if not arguments_json_valid(tc.arguments):
        return True
    return False


def is_truncated_turn(
    turn_tool_calls: list[ToolCallSpec],
    *,
    finish_reason: str,
    max_tokens: int,
    usage: dict[str, Any],
    provider_name: str = "deepseek",
) -> bool:
    if not turn_tool_calls:
        return False
    return any(
        is_truncated_tool_call(
            tc,
            finish_reason=finish_reason,
            max_tokens=max_tokens,
            usage=usage,
            provider_name=provider_name,
        )
        for tc in turn_tool_calls
    )


def strip_reasoning_for_wire(
    messages: list[dict[str, Any]],
    *,
    thinking_enabled: bool,
) -> list[dict[str, Any]]:
    if thinking_enabled:
        return messages
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            copy = {k: v for k, v in msg.items() if k != "reasoning_content"}
            stripped.append(copy)
        else:
            stripped.append(msg)
    return stripped


def scrub_message_for_storage(msg: dict[str, Any]) -> dict[str, Any]:
    """Remove secret-like patterns from message content before persistence."""
    cleaned = dict(msg)
    content = cleaned.get("content")
    if isinstance(content, str):
        cleaned["content"] = re.sub(
            r"(?i)(authorization|x-api-key)\s*[:=]\s*\S+",
            r"\1: [REDACTED]",
            content,
        )
    return cleaned
