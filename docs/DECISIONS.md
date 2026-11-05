# DECISIONS.md — decided 2026-11-05 (UTC+8)

Status: **DECIDED** = Barry decided; **PROVISIONAL** = default chosen by the planner, may be changed by Barry later. Builder agents must follow both and must not change them; propose changes in the PR description instead.

| # | Topic | Decision | Status |
|---|---|---|---|
| D1 | Repository | New private repo `Fiuuuuu/Probe-harness`. All work lands here via PRs. | DECIDED |
| D2 | Language / runtime | Python 3.11+, asyncio. | DECIDED |
| D3 | Frontend | No Electron. A CLI plus a static, self-contained trace HTML page generated from the store. | DECIDED |
| D4 | Provider | MiniMax-M3 via its Anthropic-compatible endpoint (`https://api.minimaxi.com/anthropic`, Messages API). Other providers: adapters may exist, but only MiniMax-Anthropic is required and tested live. | DECIDED |
| D5 | Key | A new spend-limited key, stored as the Cursor cloud-agent secret `MINIMAX_API_KEY`. Never in the repo; code reads it from the environment only; never printed or logged. | DECIDED |
| D6 | API spending | Cloud agents MAY spend credits for live tests and small evals, with a **hard cap of 200,000 tokens per task** (input + output as reported by the API, summed over all calls in the task). The harness / test runner enforces the cap, logs cumulative usage, and stops when the cap is reached. Live tests stay behind `RUN_LIVE=1`, eval behind `RUN_EVAL=1`. | DECIDED |
| D7 | Reference source | No source code, prompts, tool schemas or database content of the reference product are given to builder agents. Only the design docs in `docs/`. | DECIDED |
| D8 | First batch | T0–T6 only. T7–T11 stay in `prompts/TASKS.md`, marked LATER. | DECIDED |
| D9 | Connectivity | Cloud VMs may not reach `api.minimaxi.com`. T0 includes a connectivity check and reports the result; do not work around it (no proxies, no other endpoints, no other providers). | DECIDED |
| D10 | Sandbox | SWE-bench / eval tasks: one Docker container per instance with `--cap-drop ALL --security-opt no-new-privileges --pids-limit 512`, network off where the harness allows. Local tool execution: bubblewrap when available; if no sandbox backend is available, side-effecting tools refuse to run unless the user passes an explicit `--no-sandbox` flag (recorded in config_hash). | PROVISIONAL |
| D11 | Eval size | In-cloud: smoke only, `pytest-dev__pytest-7432`, 1 run, within the 200k cap (it may stop on the cap: one earlier full run used 1.2M input tokens, mini-swe-agent used 287k total; a cap stop is a valid, reported outcome). First real comparison: 10 SWE-bench Lite instances chosen by fixed seed, 1 run per configuration, run on a machine with Docker and only with Barry's separate approval (exceeds the per-task cap). | PROVISIONAL |
| D12 | Licence | No LICENSE file for now; Barry decides later. | DECIDED |

## Token cap — implementation contract (applies to every task)
- `MAX_TASK_TOKENS` env var, default 200000; values above 200000 are clamped to 200000.
- A `UsageLedger` appends one JSON line per HTTP call to `.harness/usage_ledger.jsonl` (git-ignored): timestamp, task id, test/run id, input_total, output, cumulative.
- Before each live call: refuse if `cumulative + estimated_input + max_tokens > cap` and raise `TokenCapReached`; tests catch it and are reported as skipped-with-reason "token cap reached", never as passed.
- After each call: add the API-reported usage (normalised per CONSTRAINTS C7, input_total + output) to the ledger.
- Every live test session and eval run prints the cumulative total at the end. The PR description must state the task's total tokens used.
- The ledger is shared by all processes in the VM for the task (file lock).
