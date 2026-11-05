from __future__ import annotations

import json
from pathlib import Path

from probe_harness.providers.usage import (
    merge_stream_usage_chunks,
    normalise_anthropic_usage,
    normalise_openai_usage,
)


def _load_fixtures() -> dict:
    return json.loads(Path("fixtures/probe_results.json").read_text(encoding="utf-8"))


def _probe(probe_id: str, provider: str | None = "deepseek") -> dict:
    for item in _load_fixtures()["results"]:
        if item.get("probe") != probe_id:
            continue
        if provider is not None and item.get("provider") != provider:
            continue
        return item
    raise KeyError(probe_id)


def test_deepseek_h2_cache_normalisation() -> None:
    usage = _probe("h2_cache")["usage"]
    norm = normalise_openai_usage(usage)
    assert norm["input_total"] == 1972
    assert norm["cached_read"] == 1792
    assert norm["uncached"] == 180
    assert norm["output"] == 1


def test_reasoning_inside_output() -> None:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 12},
    }
    norm = normalise_openai_usage(usage)
    assert norm["output"] == 15
    assert norm["reasoning"] == 12


def test_anthropic_cache_read() -> None:
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 2136,
        "cache_creation_input_tokens": 0,
    }
    norm = normalise_anthropic_usage(usage)
    assert norm["input_total"] == 2137
    assert norm["cached_read"] == 2136
    assert norm["uncached"] == 1


def test_stream_j_no_include_usage() -> None:
    item = _probe("j_stream_no_include_usage")
    chunk = item["usage_seen"][0][1]
    norm = normalise_openai_usage(chunk)
    assert norm["input_total"] == 272
    assert norm["output"] == 38


def test_anthropic_stream_merge_start_zero_delta() -> None:
    start = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    delta = {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 2136}
    norm = merge_stream_usage_chunks([start, delta])
    assert norm["input_total"] == 2137
    assert norm["cached_read"] == 2136


def test_anthropic_stream_merge_last_nonzero_not_sum() -> None:
    start = {
        "input_tokens": 99,
        "output_tokens": 1,
        "cache_read_input_tokens": 500,
        "cache_creation_input_tokens": 0,
    }
    delta = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 2136,
        "cache_creation_input_tokens": 0,
    }
    norm = merge_stream_usage_chunks([start, delta])
    assert norm["input_total"] == 2137
    assert norm["cached_read"] == 2136
    assert norm["uncached"] == 1
