"""Run status constants and transition helpers."""

from __future__ import annotations

from typing import Final

STATUS_QUEUED: Final = "queued"
STATUS_RUNNING: Final = "running"
STATUS_COMPLETED: Final = "completed"
STATUS_SUBMITTED: Final = "submitted"
STATUS_FAILED: Final = "failed"
STATUS_CANCELLED: Final = "cancelled"
STATUS_INTERRUPTED: Final = "interrupted"

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_COMPLETED,
        STATUS_SUBMITTED,
        STATUS_FAILED,
        STATUS_CANCELLED,
        STATUS_INTERRUPTED,
    }
)

RECOVERABLE_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_FAILED,
        STATUS_CANCELLED,
        STATUS_INTERRUPTED,
    }
)

CHECKPOINT_PRE_MODEL: Final = "pre_model"
CHECKPOINT_POST_TOOLS: Final = "post_tools"


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def is_recoverable(status: str) -> bool:
    return status in RECOVERABLE_STATUSES


SIDE_EFFECTING_TOOLS: frozenset[str] = frozenset({"bash", "write_file", "edit_file"})


def is_side_effecting_tool(name: str) -> bool:
    return name in SIDE_EFFECTING_TOOLS
