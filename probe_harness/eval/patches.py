"""Patch collection and hygiene filtering (CONSTRAINTS C11)."""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

# Paths excluded from SWE-bench predictions (scratch / harness artifacts).
HYGIENE_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".harness/**",
    "**/.harness/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.swp",
    "**/*.swo",
    "**/.DS_Store",
    "**/tmp/**",
    "**/*.tmp",
    "**/.probe_scratch/**",
    "**/probe_scratch/**",
)


def is_hygienic_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith(".harness/") or normalized == ".harness":
        return False
    if normalized.startswith("tmp/") or "/tmp/" in normalized:
        return False
    for pattern in HYGIENE_EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(Path(normalized).name, pattern):
            return False
    return bool(normalized)


def list_changed_files(workspace: Path, base_ref: str = "HEAD") -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_workspace_patch(
    workspace: Path,
    *,
    base_ref: str = "HEAD",
) -> tuple[str, list[str]]:
    """Return (unified diff, hygienic file list). Empty patch if nothing clean changed."""
    changed = list_changed_files(workspace, base_ref=base_ref)
    clean_files = sorted(f for f in changed if is_hygienic_path(f))
    if not clean_files:
        return "", []
    diff = subprocess.run(
        ["git", "diff", base_ref, "--", *clean_files],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return diff.stdout, clean_files
