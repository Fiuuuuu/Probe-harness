"""Protocol guards C1, C3 (OpenAI wire), C6."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from probe_harness.providers.base import ToolCallSpec
from probe_harness.providers.usage import normalise_openai_usage

TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


def merge_system_messages_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """C3: concatenate system messages; never overwrite with a later lone system."""
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        rest.append(msg)
    if not system_parts:
        return list(messages)
    merged = [{"role": "system", "content": "\n".join(system_parts)}]
    merged.extend(rest)
    return merged


def repair_tool_pairing(
    messages: list[dict[str, Any]],
    *,
    reason: str = "interrupted",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """C1: synthetic error tool results for unanswered tool_call ids."""
    out = deepcopy(messages)
    repairs: list[dict[str, Any]] = []
    idx = 0
    while idx < len(out):
        msg = out[idx]
        if msg.get("role") != "assistant":
            idx += 1
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            idx += 1
            continue

        j = idx + 1
        seen: dict[str, int] = {}
        while j < len(out) and out[j].get("role") == "tool":
            tid = out[j].get("tool_call_id")
            if tid in seen:
                repairs.append(
                    {
                        "kind": "pairing",
                        "action": "duplicate_tool_result",
                        "tool_call_id": tid,
                        "reason": reason,
                    }
                )
            seen[tid] = j
            j += 1

        insert_at = j
        for tc in tool_calls:
            tid = tc.get("id")
            if not tid or tid in seen:
                continue
            payload = json.dumps(
                {"error": f"tool call interrupted: {reason}", "is_error": True},
                ensure_ascii=False,
            )
            synthetic = {"role": "tool", "tool_call_id": tid, "content": payload}
            out.insert(insert_at, synthetic)
            repairs.append(
                {
                    "kind": "pairing",
                    "action": "synthetic_tool_result",
                    "tool_call_id": tid,
                    "reason": reason,
                }
            )
            insert_at += 1
        idx = insert_at
    return out, repairs


def arguments_json_valid(raw: str) -> bool:
    if raw is None:
        return False
    if raw == "":
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def is_truncated_tool_call(
    tc: ToolCallSpec,
    *,
    finish_reason: str,
    max_tokens: int,
    usage: dict[str, Any],
) -> bool:
    """C6: never execute truncated / incomplete tool arguments."""
    norm = normalise_openai_usage(usage or {})
    if max_tokens > 0 and norm["output"] >= max_tokens:
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
) -> bool:
    if not turn_tool_calls:
        return False
    return any(
        is_truncated_tool_call(
            tc,
            finish_reason=finish_reason,
            max_tokens=max_tokens,
            usage=usage,
        )
        for tc in turn_tool_calls
    )


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
