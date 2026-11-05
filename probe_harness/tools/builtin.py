"""Built-in tools (learn-claude-code s02 subset)."""

from __future__ import annotations

import asyncio
import glob as glob_module
import json
import os
import signal
import time
from pathlib import Path
from typing import Any


def _workspace() -> Path:
    return Path(os.environ.get("HARNESS_WORKSPACE", os.getcwd())).resolve()


def _resolve_in_workspace(rel: str) -> Path | None:
    ws = _workspace()
    try:
        candidate = (ws / rel).resolve()
    except OSError:
        return None
    if not candidate.is_relative_to(ws):
        return None
    return candidate


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_active_bash_procs: set[asyncio.subprocess.Process] = set()


def cancel_all_bash_processes() -> None:
    """Kill every in-flight bash process group (e.g. on run cancel)."""
    for proc in list(_active_bash_procs):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    _active_bash_procs.clear()


async def run_bash(command: str, timeout: float = 120.0) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_workspace(),
        start_new_session=True,
    )
    _active_bash_procs.add(proc)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        await proc.wait()
        return {"exit_code": -1, "stdout": "", "stderr": "timeout", "truncated": False}
    finally:
        _active_bash_procs.discard(proc)
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "truncated": False,
    }


async def read_file(path: str, offset: int = 0, limit: int = 2000) -> dict[str, Any]:
    p = _resolve_in_workspace(path)
    if p is None:
        return {"error": "path outside workspace", "content": ""}
    if not p.is_file():
        return {"error": f"not found: {path}", "content": ""}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    slice_lines = lines[offset : offset + limit]
    return {"content": "\n".join(slice_lines), "truncated": len(lines) > offset + limit}


async def write_file(path: str, content: str) -> dict[str, Any]:
    p = _resolve_in_workspace(path)
    if p is None:
        return {"error": "path outside workspace", "written": False}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    ws = _workspace()
    return {"written": True, "path": str(p.relative_to(ws))}


async def edit_file(path: str, old_string: str, new_string: str) -> dict[str, Any]:
    p = _resolve_in_workspace(path)
    if p is None:
        return {"error": "path outside workspace", "replaced": False}
    if not p.is_file():
        return {"error": f"not found: {path}", "replaced": False}
    text = p.read_text(encoding="utf-8")
    if old_string not in text:
        return {"error": "old_string not found", "replaced": False}
    p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return {"replaced": True}


async def glob_paths(pattern: str) -> dict[str, Any]:
    ws = _workspace()
    matches: list[str] = []
    for raw in glob_module.glob(str(ws / pattern), recursive=True):
        p = Path(raw).resolve()
        if p.is_relative_to(ws):
            matches.append(str(p.relative_to(ws)))
    return {"matches": sorted(matches)[:500]}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one occurrence of old_string in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Glob files under the workspace",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


async def execute_tool(name: str, arguments_json: str) -> tuple[dict[str, Any], int, int]:
    t0 = time.monotonic()
    try:
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            args = {}

        if name == "bash":
            result = await run_bash(str(args.get("command", "")))
        elif name == "read_file":
            result = await read_file(
                str(args.get("path", "")),
                _as_int(args.get("offset", 0), 0),
                _as_int(args.get("limit", 2000), 2000),
            )
        elif name == "write_file":
            result = await write_file(str(args.get("path", "")), str(args.get("content", "")))
        elif name == "edit_file":
            result = await edit_file(
                str(args.get("path", "")),
                str(args.get("old_string", "")),
                str(args.get("new_string", "")),
            )
        elif name == "glob":
            result = await glob_paths(str(args.get("pattern", "*")))
        else:
            result = {"error": f"unknown tool: {name}", "is_error": True}
    except Exception as exc:
        result = {"error": f"tool execution failed: {exc}", "is_error": True}

    duration_ms = int((time.monotonic() - t0) * 1000)
    if "exit_code" in result:
        exit_code = _as_int(result.get("exit_code", 0), 0)
    elif result.get("is_error") or "error" in result:
        exit_code = 1
    else:
        exit_code = 0
    return result, exit_code, duration_ms
