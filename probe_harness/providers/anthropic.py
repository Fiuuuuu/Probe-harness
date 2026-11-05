"""Anthropic Messages adapter (offline / MiniMax-M3 shapes; no live calls in T1)."""

from __future__ import annotations

import json
from typing import Any


def merge_system_messages(messages: list[dict[str, Any]]) -> str | None:
    """Two (or more) system messages → one top-level system string (CONSTRAINTS C3)."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    if not parts:
        return None
    return "\n".join(parts)


def convert_openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map internal OpenAI-style history to Anthropic Messages API shape."""
    system = merge_system_messages(messages)
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results.copy()})
            pending_tool_results.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
            )
            continue
        flush_tool_results()
        if role == "user":
            out.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    tool_input: Any = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_input = raw_args
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": tool_input,
                    }
                )
            out.append({"role": "assistant", "content": blocks or ""})

    flush_tool_results()
    return system, out


def build_anthropic_request_body(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    system, anthropic_messages = convert_openai_messages_to_anthropic(messages)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    return body
