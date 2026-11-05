"""CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from probe_harness.core.loop import run_agent_loop
from probe_harness.eval.report import eval_report_data, format_eval_report
from probe_harness.eval.swebench_runner import (
    new_batch_run_id,
    require_run_eval,
    run_eval_batch,
)
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
    stop_reason: str | None = None
    run_status: str | None = None
    provider: object | None = None

    async def _run() -> str:
        nonlocal stop_reason, run_status, provider
        await store.connect()
        try:
            if args.provider == "fake":
                script = task.get("fake_script") or []
                provider = FakeProvider(script)
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
                recover_from=args.recover_from,
            )
            row = await store.fetchone(
                "SELECT stop_reason, status FROM runs WHERE id = ?",
                (run_id,),
            )
            if row:
                stop_reason = row["stop_reason"]
                run_status = row["status"]
            return run_id
        finally:
            if isinstance(provider, OpenAIChatProvider):
                await provider.aclose()
            await store.close()

    run_id = asyncio.run(_run())
    print(run_id)
    ledger = UsageLedger()
    cumulative = ledger.total_tokens_used()
    print(f"ledger_cumulative_tokens={cumulative}")
    if stop_reason == "token_cap":
        print("stop_reason=token_cap", file=sys.stderr)
        return 2
    if run_status == "failed":
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    store = Store(Path(args.db) if args.db else None)

    async def _report() -> int:
        await store.connect()
        try:
            if args.eval:
                data = await eval_report_data(store, args.run_id)
                if data is None:
                    print(f"run not found: {args.run_id}", file=sys.stderr)
                    return 1
                print(format_eval_report(data))
                return 0
            metrics = await run_metrics(store, args.run_id)
            if metrics is None:
                print(f"run not found: {args.run_id}", file=sys.stderr)
                return 1
            print(format_metrics_table(metrics))
            return 0
        finally:
            await store.close()

    return asyncio.run(_report())


def _cmd_eval_run(args: argparse.Namespace) -> int:
    require_run_eval()
    batch_id = args.run_id or new_batch_run_id()
    instance_ids = args.instances
    store = Store(Path(args.db) if args.db else None)

    async def _run() -> int:
        await store.connect()
        try:
            result = await run_eval_batch(
                instance_ids=instance_ids,
                batch_run_id=batch_id,
                store=store,
                predictions_path=Path(args.predictions) if args.predictions else None,
                dataset_name=args.dataset,
                split=args.split,
                run_harness=not args.skip_harness,
                resume=not args.no_resume,
            )
        finally:
            await store.close()

        ledger = UsageLedger()
        cumulative = ledger.total_tokens_used()
        for inst in result.instances:
            print(
                f"instance={inst.instance_id} agent_run_id={inst.agent_run_id} "
                f"resolved={inst.resolved} harness_run_id={inst.harness_run_id} "
                f"patch_files={inst.patch_files} error={inst.agent_error}"
            )
        if result.harness_skipped_reason:
            print(f"harness_skipped={result.harness_skipped_reason}")
        print(f"predictions={result.predictions_path}")
        print(f"ledger_cumulative_tokens={cumulative}")
        return 0

    return asyncio.run(_run())


def _cmd_net_check(_: argparse.Namespace) -> int:
    rows, reachable, cumulative = asyncio.run(run_net_check())
    print_net_check_table(rows, reachable, cumulative)
    print(f"ledger_cumulative_tokens={cumulative}")
    return 0 if reachable else 2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a task")
    p_run.add_argument("--provider", default="fake", choices=["fake", "deepseek"])
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--db", default=None, help="SQLite path (default .harness/harness.db)")
    p_run.add_argument("--recover-from", default=None, help="Recover from interrupted run id")
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Print run metrics")
    p_report.add_argument("run_id")
    p_report.add_argument("--db", default=None)
    p_report.add_argument("--eval", action="store_true", help="SWE-bench eval report")
    p_report.set_defaults(func=_cmd_report)

    p_eval = sub.add_parser("eval", help="SWE-bench evaluation")
    p_eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)
    p_eval_run = p_eval_sub.add_parser("run", help="Run agent + harness for instances")
    p_eval_run.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="SWE-bench instance ids",
    )
    p_eval_run.add_argument("--run-id", default=None, help="Agent run id (default: generated)")
    p_eval_run.add_argument("--predictions", default=None, help="Predictions JSONL path")
    p_eval_run.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    p_eval_run.add_argument("--split", default="test")
    p_eval_run.add_argument("--db", default=None)
    p_eval_run.add_argument(
        "--skip-harness",
        action="store_true",
        help="Only run agent and write predictions",
    )
    p_eval_run.add_argument(
        "--no-resume",
        action="store_true",
        help="Rewrite predictions even if instance already present",
    )
    p_eval_run.set_defaults(func=_cmd_eval_run)

    p_nc = sub.add_parser("net-check", help="Check connectivity to api.deepseek.com")
    p_nc.set_defaults(func=_cmd_net_check)

    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
