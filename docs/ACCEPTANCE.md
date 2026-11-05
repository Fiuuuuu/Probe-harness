# ACCEPTANCE.md — definition of done per phase

First batch: P0–P2 (tasks T0–T6). P3–P6 criteria are kept for later tasks (T7–T11).

Test tiers (all commands run from repo root):
- `pytest -q tests/unit` — offline, no network, no keys. Required to pass in every PR. Uses a scripted `FakeProvider` that replays recorded response shapes (from `fixtures/probe_results.json`).
- `RUN_LIVE=1 pytest -q tests/live -m live` — MiniMax-M3 Anthropic-compatible endpoint only; needs env `MINIMAX_API_KEY`; every call goes through the UsageLedger; stops at `MAX_TASK_TOKENS` (default and hard max 200000 per task, input+output as reported by the API).
- `RUN_EVAL=1 harness eval run --instances pytest-dev__pytest-7432 --run-id <unique>` then the official SWE-bench harness — needs Docker (x86_64); same ledger and cap.
- `harness net-check` — connectivity check to `api.minimaxi.com` (DNS, TCP 443, TLS, one unauthenticated request expecting an HTTP auth error, optionally one 1-token authenticated call if `MINIMAX_API_KEY` is set and `RUN_LIVE=1`). Prints a result table; exit code 0 reachable / 2 unreachable. It never retries via another host or proxy.

Generic DoD for every task: unit tests pass; `ruff check` clean; no secrets in repo, logs, or stored requests (`tests/unit/test_no_secrets.py` scans the store, logs and ledger); docs updated; PR description lists (a) what was verified and how, (b) what was NOT verified, (c) total tokens used by the task from the ledger (0 if no live calls), (d) the `harness net-check` result if any live step was attempted.
Token-cap DoD (every task): unit tests prove the ledger refuses a call that would exceed the cap, raises `TokenCapReached`, logs cumulative usage, and clamps `MAX_TASK_TOKENS` > 200000 to 200000.

## P0 Measurement base
- [ ] `harness run --provider fake --task tests/fixtures/echo_task.json` completes; DB has runs/events/model_calls rows; `harness report <run_id>` prints the metric table (calls, input/cached/uncached/output, cache ratio, stop_reason, agent time).
- [ ] usage normalisation unit tests reproduce C7 exactly: Anthropic(input 1, cache_read 2136) → input_total 2137 / cached 2136 / uncached 1; OpenAI(prompt 2137, cached 2136) → same; streaming Anthropic start-all-zero + delta merge; OpenAI stream with/without include_usage.
- [ ] events have UNIQUE(run_id, seq); concurrency test with 2 writers × 500 events → no duplicate/missing seq.
- [ ] Live (`RUN_LIVE=1`): `harness net-check` result reported in the PR; if reachable, one 3-step tool task on MiniMax-M3 records non-zero usage matching the provider's response; ledger total printed. If unreachable: report it, mark live items not verified, do not work around it.
- [ ] UsageLedger + token cap implemented per DECISIONS.md, with unit tests.
## P1 Protocol correctness
- [ ] Pairing: unit tests for orphan after (a) policy stop, (b) loop guard, (c) cancel during tool exec, (d) cancel during approval → history passed to provider always pairs; `protocol_repair` event recorded. Live (MiniMax-Anthropic, capped): probe a/b/c-style histories built by the harness never 400.
- [ ] Anthropic conversion: two system messages → single top-level system containing both; all tool_results of a turn in ONE user message.
- [ ] Names: `mcp.server/tool`, 70-char and colliding names map to `^[a-zA-Z0-9_-]{1,64}$` and back (offline; fixture f shows a strict provider rejects `.` and `/`).
- [ ] Reasoning: thinking blocks (with signature) stored and replayed byte-identical (unit); OpenAI-format `reasoning_content` passback covered offline from fixtures g7/g9.
- [ ] Truncation: recorded shapes of probe i for all endpoints in the fixtures are classified truncated and NOT executed.
- [ ] Hooks: PreToolUse/PostToolUse/Stop hooks fire in order; metric collection implemented as hooks.
## P2 State, budget, stopping
- [ ] Kill the process (SIGKILL) at: before model call, during tool exec, after tools → recover creates a new run, no side-effecting tool re-executed (test tool increments a counter file), usage not double counted.
- [ ] Startup marks every `running` run `interrupted`.
- [ ] Budget: warning/finalizing/hard for tokens and calls fire at configured thresholds (fake provider with scripted usage); a tool call during finalizing yields synthetic results + stop_reason run_budget, history still pairs.
- [ ] Finalization keeps the tools array and system prompt byte-identical (request_sha256 of prefix unchanged).
- [ ] `submit` tool → goal check; failing check returns feedback and continues; passing check ends with status submitted.
- [ ] Eval (`RUN_EVAL=1`, T6, capped): smoke on pytest-7432; report resolved / cap stop / stop_reason, tokens, steps after first passing repro. Target is a GOAL TO MEASURE.
## P3 Context (LATER)
- [ ] Compaction never separates a tool_call from its result; recent segment starts on a user message; summary validation; context_prepared breakdown recorded; tool-result budget enforced with head/tail truncation marker.
- [ ] Todo reminder after 3 rounds without update.
## P4 Permissions & sandbox (LATER)
- [ ] Denied call returns a tool result and the loop continues; rules bound to argument hash cannot be reused with changed args; scope priority run > conversation > global.
- [ ] Linux sandbox (bwrap locally, Docker per instance for eval — PROVISIONAL D10): escape tests fail closed: write outside workspace, network access, read ~/.ssh, fork bomb (pids limit). If the sandbox backend is unavailable the tool refuses to run.
## P5 Extensions (LATER)
- [ ] MCP stdio against a real SDK server (pinned `mcp` 2.x, `MCPServer`): handshake order, single-line framing, stderr drained (server writing 1 MB to stderr does not hang), server→client request not mistaken for a response, failed call → restart without replay.
- [ ] Skills loaded on demand; memory/skill candidates require review with all checks; injection sample rejected; knowledge disabled in eval config.
## P6 Shell (LATER)
- [ ] `harness trace <run_id> --html out.html` writes one self-contained static HTML file (no network, no server) showing steps, context breakdown, tool calls, budget state; optional subagent/background with separate accounting. No Electron.
