"""Context preparation before each model call (pairing repair, wire-format guards)."""

from __future__ import annotations

from typing import Any

from probe_harness.providers.guards import merge_system_messages_openai, repair_tool_pairing
from probe_harness.tools.names import ToolNameMap, normalise_tool_schemas


def prepare_context(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    *,
    repair_reason: str = "interrupted",
    name_map: ToolNameMap | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Mutate ``messages`` with C1 pairing repair when needed.

    Returns (provider_messages, protocol_repairs, api_tool_schemas).
    """
    repaired, repairs = repair_tool_pairing(messages, reason=repair_reason)
    if repairs:
        messages.clear()
        messages.extend(repaired)

    provider_messages = merge_system_messages_openai(messages)
    api_tools = normalise_tool_schemas(tool_schemas, name_map=name_map)
    return provider_messages, repairs, api_tools
