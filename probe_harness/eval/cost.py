"""DeepSeek flash USD estimates (DOC pricing from docs/CONSTRAINTS.md C12)."""

from __future__ import annotations

PRICING_SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"

# USD per 1M tokens (off-peak / peak), read 2026-11-05 per CONSTRAINTS.md
_OFF_PEAK = {"cached": 0.003, "uncached": 0.15, "output": 0.60}
_PEAK = {"cached": 0.006, "uncached": 0.30, "output": 1.20}


def deepseek_flash_cost_usd(
    *,
    cached_read: int,
    uncached: int,
    output: int,
    peak: bool = False,
) -> float:
    rates = _PEAK if peak else _OFF_PEAK
    return (
        cached_read * rates["cached"] / 1_000_000
        + uncached * rates["uncached"] / 1_000_000
        + output * rates["output"] / 1_000_000
    )
