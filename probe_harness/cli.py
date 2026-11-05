"""CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from probe_harness.core.loop import run_agent_loop
from probe_harness.net_check import print_net_check_table, run_net_check
from probe_harness.providers.fake import FakeProvider
from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.store.db import Store
from probe_harness.store.metrics import format_metrics_table, run_metrics


def _cmd_run(args: argparse.Namespace) -> int:
    task_path = Path(args.task)
    if not task_path.is_file():
        print(f"task file not found: {task_path}", file=sys.stderr)
        return 1

    task = json.loads(task_path.read_text(encoding="utf-8"))
    store = Store(Path(args.db) if args.db else None)

    async def _run() -> str:
        await store.connect()
        try:
            if args.provider == "fake":
                script = task.get("fake_script") or []
                provider: object = FakeProvider(script)
                model = "fake"
            elif args.provider == "deepseek":
                provider = OpenAIChatProvider(
                    ledger=UsageLedger(task_id="harness-run"),
                    thinking_enabled=task.get("thinking_enabled", True),
                    max_tokens=int(task.get("max_tokens", 8192)),
                    test_run_id=f"run-{task_path.name}",
                )
                model = "deepseek-flash"
            else:
                raise SystemExit(f"unknown provider: {args.provider}")

            run_id = await run_agent_loop(
                store,
                provider,
                task_path,
                provider_name=args.provider,
                model=model,
                max_steps=int(task.get("max_steps", 20)),
            )
            return run_id
        finally:
            await store.close()

    run_id = asyncio.run(_run())
    print(run_id)
    ledger = UsageLedger()
    print(f"ledger_cumulative_tokens={ledger.total_tokens_used()}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    store = Store(Path(args.db) if args.db else None)

    async def _report() -> int:
        await store.connect()
        try:
            metrics = await run_metrics(store, args.run_id)
            if metrics is None:
                print(f"run not found: {args.run_id}", file=sys.stderr)
                return 1
            print(format_metrics_table(metrics))
            return 0
        finally:
            await store.close()

    return asyncio.run(_report())


def _cmd_net_check(_: argparse.Namespace) -> int:
    rows, reachable, cumulative = asyncio.run(run_net_check())
    print_net_check_table(rows, reachable, cumulative)
    return 0 if reachable else 2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a task")
    p_run.add_argument("--provider", default="fake", choices=["fake", "deepseek"])
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--db", default=None, help="SQLite path (default .harness/harness.db)")
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Print run metrics")
    p_report.add_argument("run_id")
    p_report.add_argument("--db", default=None)
    p_report.set_defaults(func=_cmd_report)

    p_nc = sub.add_parser("net-check", help="Check connectivity to api.deepseek.com")
    p_nc.set_defaults(func=_cmd_net_check)

    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
