"""Normalise provider usage to store fields (CONSTRAINTS C7 / C12)."""

from __future__ import annotations

import json
from typing import Any


def normalise_openai_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """DeepSeek / OpenAI chat completions usage → model_calls columns."""
    if not usage:
        return {
            "input_total": 0,
            "cached_read": 0,
            "cache_write": 0,
            "uncached": 0,
            "output": 0,
            "reasoning": 0,
        }

    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)

    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    if cached == 0 and "prompt_cache_hit_tokens" in usage:
        cached = int(usage.get("prompt_cache_hit_tokens") or 0)

    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None:
        uncached = int(miss)
    else:
        uncached = max(0, prompt - cached)

    comp_details = usage.get("completion_tokens_details") or {}
    reasoning = int(comp_details.get("reasoning_tokens") or 0)

    return {
        "input_total": prompt,
        "cached_read": cached,
        "cache_write": 0,
        "uncached": uncached,
        "output": completion,
        "reasoning": reasoning,
    }


def normalise_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Anthropic Messages usage → model_calls columns."""
    if not usage:
        return {
            "input_total": 0,
            "cached_read": 0,
            "cache_write": 0,
            "uncached": 0,
            "output": 0,
            "reasoning": 0,
        }

    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    input_total = input_tokens + cache_read + cache_write
    uncached = input_tokens

    return {
        "input_total": input_total,
        "cached_read": cache_read,
        "cache_write": cache_write,
        "uncached": uncached,
        "output": output,
        "reasoning": 0,
    }


def merge_stream_usage_chunks(chunks: list[dict[str, Any]]) -> dict[str, int]:
    """Merge streaming usage chunks (Anthropic: take last non-zero per field, do not sum)."""
    if not chunks:
        return normalise_openai_usage(None)

    if any("input_tokens" in c for c in chunks):
        merged: dict[str, Any] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            last_nonzero = 0
            for chunk in chunks:
                val = int(chunk.get(key) or 0)
                if val != 0:
                    last_nonzero = val
            merged[key] = last_nonzero
        return normalise_anthropic_usage(merged)

    last = chunks[-1]
    return normalise_openai_usage(last)


def estimate_chat_input_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Conservative pre-call input estimate for token-cap reservation."""
    payload = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
    # ~3 chars/token plus headroom for provider-side formatting
    return max(256, len(payload) // 3 + 512)
