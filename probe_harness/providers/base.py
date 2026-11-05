"""Provider response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallSpec:
    id: str
    name: str
    arguments: str


@dataclass
class HttpCallRecord:
    """One row in model_calls — one HTTP request to the provider."""

    attempt: int
    usage: dict[str, Any]
    http_status: int
    request_id: str | None
    latency_ms: int
    stop_reason: str = ""
    error_body: str = ""
    usage_estimated: bool = False
    response_model: str | None = None


@dataclass
class ModelTurn:
    content: str | None
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    reasoning_content: str | None = None
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
    http_status: int = 200
    request_id: str | None = None
    latency_ms: int = 0
    http_calls: list[HttpCallRecord] = field(default_factory=list)
    error_body: str = ""
    response_model: str | None = None


class ProviderCallError(Exception):
    """Provider call failed after one or more HTTP attempts; carries partial call records."""

    def __init__(
        self,
        message: str = "",
        *,
        http_calls: list[HttpCallRecord] | None = None,
        turn: ModelTurn | None = None,
    ) -> None:
        super().__init__(message)
        self.http_calls = http_calls or []
        self.turn = turn
