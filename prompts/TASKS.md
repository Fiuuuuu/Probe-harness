# Task prompts for cloud coding agents (one run = one PR)

Repo: `Fiuuuuu/Probe-harness` (private). First batch: **T0–T6** (run in order; each starts after the previous PR is merged). T7–T11 are **LATER** — do not start them.

## Common preamble (prepend to every task prompt verbatim)

> You are working in the private repo Fiuuuuu/Probe-harness. Read docs/DECISIONS.md, docs/PLAN.md, docs/CONSTRAINTS.md, docs/ACCEPTANCE.md and docs/BASELINE.md before writing code; DECISIONS.md is binding. Python 3.11+, asyncio, package `probe_harness`, CLI `harness`.
> Secrets: the only key is the env var `MINIMAX_API_KEY` (Cursor secret). Read it from the environment only. Never print, log, store or commit it; scrub `x-api-key` / `Authorization` from anything you store.
> API spending: you MAY make live calls to MiniMax-M3 (Anthropic-compatible endpoint `https://api.minimaxi.com/anthropic`) only for live tests and small evals, gated by `RUN_LIVE=1` / `RUN_EVAL=1`. HARD CAP: 200,000 tokens for this whole task (input + output as reported by the API, summed over every call). Every live call must go through the UsageLedger (DECISIONS.md "Token cap"); stop all live work when the cap is reached. Report the task's total tokens in the PR description.
> If `api.minimaxi.com` is unreachable from this VM, report it and mark live items as not verified. Do not work around it: no proxies, no other hosts, no other providers.
> Do not invent metric numbers; any target is "goal, to be measured". The only real baseline numbers are in docs/BASELINE.md.
> Deliver exactly one PR against `main`: code + offline tests (`pytest -q tests/unit` must pass, `ruff check` clean) + a PR description listing what you verified and how, what you could not verify, and total tokens used. Do not merge it. Do not edit docs/DECISIONS.md. If a requirement is ambiguous, stop and say so in the PR instead of guessing.

## First batch

### T0 — skeleton, store, ledger, connectivity check (P0)
> Task T0. Build the project skeleton and the measurement base.
> 1. Packaging: `pyproject.toml` (Python >=3.11; deps: httpx, aiosqlite, pytest, pytest-asyncio, ruff), package `probe_harness` with the layout in docs/PLAN.md §3 (empty modules are fine for later parts), console script `harness`.
> 2. Store: SQLite (WAL) with the schema in docs/PLAN.md §4; events have UNIQUE(run_id, seq) and a single writer.
> 3. UsageLedger + token cap exactly as in docs/DECISIONS.md "Token cap" (`MAX_TASK_TOKENS`, default and hard max 200000, `.harness/usage_ledger.jsonl`, file lock, `TokenCapReached`). Add `.harness/` to .gitignore.
> 4. `FakeProvider` that replays scripted responses, and a minimal async loop (learn-claude-code s01/s02 mechanics; tools bash/read_file/write_file/edit_file/glob) that writes runs/events/model_calls/tool_calls rows.
> 5. CLI: `harness run --provider fake --task <file>`, `harness report <run_id>` (metric table: calls, input/cached/uncached/output, cache ratio, stop_reason, agent time), and `harness net-check`.
> 6. `harness net-check`: check DNS, TCP 443, TLS and one unauthenticated HTTPS request to `api.minimaxi.com` (expect an HTTP auth error, which counts as reachable); if `RUN_LIVE=1` and `MINIMAX_API_KEY` is set, also one authenticated Messages call with max_tokens 1 through the ledger. Print a table; exit 0 if reachable, 2 if not. No retries via other hosts or proxies.
> 7. Run `harness net-check` once in this VM (with `RUN_LIVE=1` if the secret is present) and paste its output (no secrets) into the PR description. If reachable, also run the P0 live item from docs/ACCEPTANCE.md under the cap; if unreachable, do not attempt any workaround.
> Done when: the P0 items 1 and 3, the UsageLedger item and the generic/token-cap DoD in docs/ACCEPTANCE.md pass, and the PR reports the net-check result and total tokens used.

### T1 — provider adapter + usage normalisation (P0)
> Task T1. Implement the Anthropic Messages adapter for MiniMax-M3 (streaming and non-streaming; base URL configurable, default `https://api.minimaxi.com/anthropic`), usage normalisation per docs/CONSTRAINTS.md C7, retry per C9, one model_calls row per HTTP call, every live call through the UsageLedger. Optional: an OpenAI-Chat adapter tested offline only. Unit tests use the shapes in fixtures/probe_results.json. Done when: ACCEPTANCE P0 item 2 passes offline; if reachable, P0 item 4 live under the cap.

### T2 — protocol guards (P1)
> Task T2. Implement CONSTRAINTS C1 pairing repair, C2 one user message per tool-result turn, C3 system merge (never overwrite), C4 name normalisation map (never split MCP names on `__`), C5 reasoning/thinking storage and replay, C6 truncation detection (never execute, never substitute `{}`), C8 stream assembly, with `protocol_repair` events. Done when: ACCEPTANCE P1 items 1–5 pass offline; live pairing check under the cap if reachable.

### T3 — hooks + live probe suite (P1)
> Task T3. Add an in-process hook/event bus (learn-claude-code s04: PreToolUse/PostToolUse/Stop), move metric collection onto hooks, and port the MiniMax-Anthropic parts of fixtures/probe_scripts into `tests/live` (pytest marker `live`, gated by `RUN_LIVE=1`, every call through the ledger). The scripts in fixtures/ are reference only; do not run them as-is. Done when: ACCEPTANCE P1 item 6 passes; the live suite runs end to end within the cap or stops cleanly at it.

### T4 — run state machine, checkpoints, recovery (P2)
> Task T4. Implement the run state machine, checkpoints at pre_model and post_tools, recovery as a NEW run linked by `recovered_from`, the startup sweep that marks every `running` run `interrupted`, and a per-call journal so side-effecting tools are never replayed. Done when: ACCEPTANCE P2 items 1–2 pass (SIGKILL tests offline with FakeProvider).

### T5 — budget, cache-stable finalization, submit + goal check (P2)
> Task T5. Implement three-level budgets for tokens and model calls (checked before and after each call), the repeated-identical-call guard, max_steps, cache-stable finalization (tools and system prompt unchanged; instruction appended as user text; tool calls rejected with synthetic results), the `submit` tool and a goal evaluator (learn-claude-code s17 mechanics, minimal task goal from s10). Run budgets are separate from the per-task API token cap. Done when: ACCEPTANCE P2 items 3–5 pass offline.

### T6 — SWE-bench runner + report (P2)
> Task T6. Implement the SWE-bench runner: one Docker container per instance with `--cap-drop ALL --security-opt no-new-privileges --pids-limit 512` (DECISIONS D10, provisional), patch collection with file list and hygiene filter, predictions JSONL with empty-patch lines and resume, a wrapper around the official harness that always uses a new run_id, and `harness report --eval`. If Docker is not available in this VM, implement and unit-test everything with Docker mocked, skip the eval, and report that; do not try to install or emulate Docker. If Docker is available and `RUN_EVAL=1`, run the smoke on `pytest-dev__pytest-7432` under the 200k cap (a cap stop is a valid result) and report resolved / stop_reason / tokens. Done when: runner unit tests pass; eval result or skip reason reported honestly.

## LATER (not in the first batch — do not start)
| Task | Phase | Scope |
|---|---|---|
| T7 | P3 | Context manager: window-based budget, tool-result budget with head/tail truncation, pairwise compaction, summary validation, context_prepared breakdown; todo reminder (s05). |
| T8 | P4 | Permissions: async approval, argument-hash-bound rules with scopes, denial as tool result; bubblewrap sandbox with fail-closed escape tests (D10). |
| T9 | P5 | Real MCP stdio client per CONSTRAINTS C10, sandboxed launch, name map from T2. |
| T10 | P5 | Skills on demand (s07); memory/skill candidates with review; disabled in eval config. |
| T11 | P6 | `harness trace <run_id> --html` static self-contained trace page; optional subagent/background with separate accounting. No Electron. |
