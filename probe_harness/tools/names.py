"""Tool name normalisation (T2)."""

from __future__ import annotations


def normalise_tool_name(raw: str) -> str:
    return raw[:64]
