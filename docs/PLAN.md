# PLAN.md — what to build (self-contained; no access to the reference product required)

Repo: `Fiuuuuu/Probe-harness` (private). Python package: `probe_harness`, CLI command: `harness`. Python ≥3.11, asyncio. Decisions binding on this plan: docs/DECISIONS.md. Base course: learn-claude-code (public, https://github.com/shareAI-lab/learn-claude-code, commit 7226f555d48900076eba126acf37c9ed5f002b5d) used as a **mechanism reference, not as a library** (its s01–s15 are synchronous single files with globals; 15 chapters re-define `run_bash`).

## 1 Design principles to reproduce (behaviour, not code)
1. **Durable run state machine.** Run statuses: queued → running → completed | submitted | failed | cancelled | interrupted. A checkpoint is saved before every model call (input boundary) and after tools (next-step start). Recovery creates a NEW run from the latest checkpoint and links `recovered_from`; the old run is never mutated. Only failed/cancelled/interrupted runs are recoverable. On host restart every `running` run becomes `interrupted` (never silently resumed).
2. **Defence-in-depth budget.** Three levels per dimension (tokens, model calls): warning → finalizing → hard stop. Checked before AND after each model call. Finalizing = model must answer without tools; a tool call then is a policy stop — but first write synthetic results for pairing (C1). Plus repeated-identical-call guard (3 in a row) and max_steps. Interactive and benchmark defaults are configured separately and recorded in `config_hash`.
3. **Knowledge only after review.** Memory/skill/wiki items are created as candidates; approval requires a reviewer and explicit checks (relevance, source, applicability, sensitive info, prompt injection), an automated secret/injection scan, and a verifiable source run. Disabled in evaluation.
4. **Evidence and trace are first-class data.** Append-only events with per-run unique sequence; every tool output hashed (sha256) and stored as evidence; every model request hashed; a per-step context breakdown (system / messages / tool results / tool schemas). All metrics are derived from the store, never computed ad hoc.
5. **Provider abstraction.** One internal message format; required adapter: OpenAI-compatible Chat Completions for DeepSeek `deepseek-flash` (`https://api.deepseek.com`, key from env `DEEPSEEK_API_KEY`; thinking mode set explicitly per request; `reasoning_content` stored and replayed; see CONSTRAINTS C12). Optional secondary: an Anthropic Messages adapter (MiniMax-M3 or DeepSeek's Anthropic endpoint), tested offline from fixtures only. Normalised usage (C7); stream-aware retry (C9); protocol guards (C1–C8); every live call goes through the UsageLedger (token cap, DECISIONS.md).
6. **Shell separate from core.** Core runs headless (CLI + eval runner). The only UI is a static, self-contained trace HTML page generated from the store (no Electron, no server required).
7. **Approval before side effects, least privilege.** Shell/write/MCP calls go through policy; denial returns a tool result so the model can continue; persistent rules bind to a hash of the exact arguments; sandbox fails closed.
8. **Honest idempotency.** Never replay a side-effecting call after a crash or MCP failure; mark outcome uncertain.
9. **Context budget by model window.** Tool-result budget, head/tail truncation of long outputs, pairwise compaction (never split a tool_call from its result), at most N activated tools.

## 2 Improvements over the reference (each has a test in ACCEPTANCE.md)
submit tool + goal evaluator (s17) · pairing repair (C1) · system never overwritten (C3) · one user message per tool-result turn (C2) · tool-name normalisation (C4) · reasoning passback (C5) · truncation detection (C6) · cache-stable finalization (keep tools AND system) · unique event sequence · MCP stderr drain + request/response split (C10) · Linux sandbox · patch hygiene (C11) · normal runs marked interrupted on restart.

## 3 Architecture (packages)
```
probe_harness/
  providers/   openai_chat.py (DeepSeek, primary) anthropic.py (optional) usage.py guards.py ledger.py (C1–C8, C12, token cap)
  core/        loop.py budget.py goal.py context.py hooks.py
  runs/        service.py state.py checkpoints.py recovery.py
  tools/       registry.py builtin.py names.py mcp_stdio.py
  policy/      permissions.py approvals.py sandbox.py
  store/       db.py schema.sql events.py metrics.py
  knowledge/   candidates.py review.py skills.py          (later)
  eval/        swebench_runner.py report.py
  trace_html.py  cli.py
tests/ unit/ (offline, default)  live/ (RUN_LIVE=1, capped)  eval/ (RUN_EVAL=1, Docker, capped)
```

## 4 Data model (SQLite, WAL)
- runs(id, task_ref, status, stop_reason, provider, model, config_hash, recovered_from, started_at, ended_at)
- events(run_id, seq, step, type, ts, data_json) — UNIQUE(run_id, seq), single writer
- model_calls(run_id, step, attempt, input_total, cached_read, cache_write, uncached, output, reasoning, stop_reason, latency_ms, request_sha256, http_status, request_id)
- tool_calls(run_id, step, call_id, raw_name, norm_name, args_sha256, status, exit_code, duration_ms, output_tokens, truncated)
- checkpoints(run_id, step, boundary pre_model|post_tools, messages_json, usage_json, budget_json)
- evidence(id, run_id, call_id, sha256, kind, data_json)
- approvals(id, run_id, call_id, decision, scope, rule_hash)
- eval_results(run_id, instance_id, patch_sha256, patch_files, resolved, harness_run_id, eval_log_path)
Event types: run_started, run_finished, context_prepared, model_request, model_response, protocol_repair, tool_call, tool_result, budget_state, checkpoint_saved, goal_check, submitted, approval_requested, approval_resolved.

## 5 Main loop
start/recover → budget check → prepare context (compaction, pairing repair, context_prepared) → checkpoint(pre_model) → model call (normalise usage; model_calls row) → truncated? (retry larger / error to model; never execute) → tool_calls? no → goal check (pass → done; fail → feedback, continue) · yes → policy/approval/hooks → sandboxed exec → evidence + tool_result → checkpoint(post_tools) → budget check → loop. `submit` tool triggers the goal check explicitly. Finalizing: keep tools + system identical, append instruction as user text, reject tool execution with synthetic results.

## 6 lcc chapter mapping
s01 loop → extend (async, events, checkpoints, stop_reason) P0 · s02 tools → keep+extend (bash/read_file/write_file/edit_file/glob; name normalisation) P0 · s03 permission → extend (async approval, arg-hash rules) P4 · s04 hooks → NEW vs reference (event bus; metrics on hooks) P1 · s05 todo → keep (3-round reminder) P3 · s06 subagent → optional P6 · s07 skills → keep P5 · s08 compact → extend (pairwise, budgeted) P3 · s09 memory → extend (candidate+review) P5 · s10 tasks → trim (goal + success criteria for s17) P2 · s11 background → optional (long test runs) P6 · s12 cron → last P6 · s13 teams → skip · s14 MCP → new (real stdio; lcc uses in-process mocks) P5 · s15 integrated → reference only · s16 workflow → extend (journal, budgets) P2 · s17 goal loop → NEW vs reference (submit + evaluator) P2.

## 7 Phases (dependency order; no time estimates)
P0 measurement base → P1 protocol correctness → P2 state, budget, stopping → (P3 context ∥ P4 permissions & sandbox) → P5 MCP/skills/memory → P6 shell (CLI + static trace HTML, optional subagent/background/cron).
First batch = tasks T0–T6 (P0–P2 + SWE-bench runner). T7–T11 are LATER.
Every phase ends with the same metrics report on the same eval set and a comparison to the previous phase.

## 8 Evaluation protocol (SWE-bench Lite)
1 smoke: pytest-dev__pytest-7432 + DeepSeek `deepseek-flash` (thinking enabled) within the 200k per-task token cap; produce the metric table from the new store (a cap stop is a valid reported outcome). This is the first DeepSeek baseline; it is not comparable 1:1 with the MiniMax-M3 numbers in BASELINE.md (different model). 2 fixed subset from SWE-bench Lite by fixed seed, instance list committed; provisional size 10 instances (DECISIONS D11), run only with separate approval because it exceeds the per-task cap. 3 fixed config (model, temperature, max_steps, budgets, tools, skills; knowledge off; config_hash). 4 repeats per config. 5 official harness, new run_id every time, patch file list stored. 6 auto report: resolved, empty-patch rate, calls, input/uncached/cached/output, cache hit ratio, steps after first passing repro, stop_reason distribution, tool error rate, protocol repairs, wall clock (agent-internal AND process, labelled).
A/B designs (expected directions are hypotheses): submit vs none; guards vs none; cache-stable finalization vs tool-clearing; tool-result budget vs head/tail only; vs mini-swe-agent. Paired per-instance comparison, alternate order (shared provider cache).
