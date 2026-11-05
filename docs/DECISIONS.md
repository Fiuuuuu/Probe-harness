# DECISIONS.md — decided 2026-11-05 (UTC+8); provider changed to DeepSeek the same day

Status: **DECIDED** = Barry decided; **PROVISIONAL** = default chosen by the planner, may be changed by Barry later. Builder agents must follow both and must not change them; propose changes in the PR description instead.

| # | Topic | Decision | Status |
|---|---|---|---|
| D1 | Repository | New private repo `Fiuuuuu/Probe-harness`. All work lands here via PRs. | DECIDED |
| D2 | Language / runtime | Python 3.11+, asyncio. | DECIDED |
| D3 | Frontend | No Electron. A CLI plus a static, self-contained trace HTML page generated from the store. | DECIDED |
| D4 | Provider | **Primary: DeepSeek**, model `deepseek-flash`, via the **OpenAI-compatible** Chat Completions endpoint `https://api.deepseek.com` (`POST /chat/completions`). Thinking mode is set explicitly on every request (never rely on the default): `{"thinking": {"type": "enabled"}}` + `reasoning_effort: "high"` for agent runs; `{"type": "disabled"}` allowed for cheap protocol tests. `deepseek-v4-pro` is allowed only by explicit config. Secondary/optional: an Anthropic-format adapter (DeepSeek also documents `https://api.deepseek.com/anthropic`, unmeasured by us; MiniMax-M3 Anthropic endpoint measured earlier) — offline fixture tests only unless Barry enables it. See CONSTRAINTS C12 for sources. | DECIDED (model/endpoint choice: planner, per official docs 2026-11-05) |
| D5 | Key | A new spend-limited DeepSeek key, stored as the Cursor cloud-agent secret `DEEPSEEK_API_KEY`. Never in the repo; code reads it from the environment only; never printed or logged. `MINIMAX_API_KEY` is not required. | DECIDED |
| D6 | API spending | Cloud agents MAY spend credits for live tests and small evals, with a **hard cap of 200,000 tokens per task** (input + output as reported by the API, summed over all calls in the task). The harness / test runner enforces the cap, logs cumulative usage, and stops when the cap is reached. Live tests stay behind `RUN_LIVE=1`, eval behind `RUN_EVAL=1`. | DECIDED |
| D7 | Reference source | No source code, prompts, tool schemas or database content of the reference product are given to builder agents. Only the design docs in `docs/`. | DECIDED |
| D8 | First batch | T0–T6 only. T7–T11 stay in `prompts/TASKS.md`, marked LATER. | DECIDED |
| D9 | Connectivity | Cloud VMs may not reach `api.deepseek.com`. T0 includes a connectivity check and reports the result; do not work around it (no proxies, no other endpoints, no other providers). | DECIDED |
| D10 | Sandbox | SWE-bench / eval tasks: one Docker container per instance with `--cap-drop ALL --security-opt no-new-privileges --pids-limit 512`, network off where the harness allows. Local tool execution: bubblewrap when available; if no sandbox backend is available, side-effecting tools refuse to run unless the user passes an explicit `--no-sandbox` flag (recorded in config_hash). | PROVISIONAL |
| D11 | Eval size | In-cloud: smoke only, `pytest-dev__pytest-7432`, 1 run, within the 200k cap (it may stop on the cap: the earlier MiniMax-M3 runs used 1.2M input tokens for a full run and 287k for mini-swe-agent; no DeepSeek baseline exists; a cap stop is a valid, reported outcome). First real comparison: 10 SWE-bench Lite instances chosen by fixed seed, 1 run per configuration, run on a machine with Docker and only with Barry's separate approval (exceeds the per-task cap). | PROVISIONAL |
| D12 | Licence | No LICENSE file for now; Barry decides later. | DECIDED |

## Token cap — implementation contract (applies to every task)
- `MAX_TASK_TOKENS` env var, default 200000; values above 200000 are clamped to 200000.
- A `UsageLedger` appends one JSON line per HTTP call to `.harness/usage_ledger.jsonl` (git-ignored): timestamp, task id, test/run id, input_total, output, cumulative.
- Before each live call: refuse if `cumulative + estimated_input + max_tokens > cap` and raise `TokenCapReached`; tests catch it and are reported as skipped-with-reason "token cap reached", never as passed.
- After each call: add the API-reported usage to the ledger: DeepSeek OpenAI format `prompt_tokens + completion_tokens` (prompt includes cache hits; completion includes reasoning tokens); Anthropic format per CONSTRAINTS C7.
- Reservation: always send an explicit `max_tokens` (DeepSeek's default is 64K in thinking mode, which would block every call under the reservation rule); recommended 8192 for agent steps, small values for protocol tests.
- Every live test session and eval run prints the cumulative total at the end. The PR description must state the task's total tokens used.
- The ledger is shared by all processes in the VM for the task (file lock).
