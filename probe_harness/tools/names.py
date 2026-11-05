"""Tool name normalisation (CONSTRAINTS C4)."""

from __future__ import annotations

import hashlib
import re

_MAX_LEN = 64
_VALID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_CHAR = re.compile(r"[^a-zA-Z0-9_-]+")


def _mcp_core(raw: str) -> str:
    """Parse mcp__{server}__{tool} without splitting the tool segment on ``__``."""
    if not raw.startswith("mcp__"):
        return raw
    rest = raw[5:]
    sep = rest.find("__")
    if sep < 0:
        return f"mcp_{rest}"
    server = rest[:sep]
    tool = rest[sep + 2 :]
    tool_part = tool.replace("__", "_")
    return f"mcp_{server}_{tool_part}"


def _base_slug(raw: str) -> str:
    core = _mcp_core(raw)
    slug = _INVALID_CHAR.sub("_", core).strip("_")
    return slug or "tool"


class ToolNameMap:
    """Bidirectional raw ↔ normalised name map."""

    def __init__(self) -> None:
        self._raw_to_norm: dict[str, str] = {}
        self._norm_to_raw: dict[str, str] = {}

    def register(self, raw: str) -> str:
        if raw in self._raw_to_norm:
            return self._raw_to_norm[raw]
        norm = self._allocate(raw)
        self._raw_to_norm[raw] = norm
        self._norm_to_raw[norm] = raw
        return norm

    def normalised(self, raw: str) -> str:
        return self.register(raw)

    def raw_name(self, norm: str) -> str:
        return self._norm_to_raw.get(norm, norm)

    def _allocate(self, raw: str) -> str:
        base = _base_slug(raw)[:_MAX_LEN]
        if _VALID.match(base) and base not in self._norm_to_raw:
            return base
        digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
        trimmed = base[: _MAX_LEN - 9] if len(base) > _MAX_LEN - 9 else base
        candidate = f"{trimmed}_{digest}"[:_MAX_LEN]
        while candidate in self._norm_to_raw and self._norm_to_raw[candidate] != raw:
            digest = hashlib.sha256(f"{raw}:{digest}".encode()).hexdigest()[:8]
            candidate = f"{trimmed}_{digest}"[:_MAX_LEN]
        return candidate


_GLOBAL_MAP = ToolNameMap()


def normalise_tool_name(raw: str, *, name_map: ToolNameMap | None = None) -> str:
    mapper = name_map or _GLOBAL_MAP
    return mapper.normalised(raw)


def denormalise_tool_name(norm: str, *, name_map: ToolNameMap | None = None) -> str:
    mapper = name_map or _GLOBAL_MAP
    return mapper.raw_name(norm)


def normalise_tool_schemas(
    schemas: list[dict],
    *,
    name_map: ToolNameMap | None = None,
) -> list[dict]:
    mapper = name_map or _GLOBAL_MAP
    out: list[dict] = []
    for schema in schemas:
        fn = schema.get("function") or {}
        raw = str(fn.get("name", ""))
        norm = mapper.register(raw)
        copied = dict(schema)
        copied_fn = dict(fn)
        copied_fn["name"] = norm
        copied["function"] = copied_fn
        out.append(copied)
    return out
