# CONSTRAINTS.md — verified API / protocol rules

Status legend: **MEASURED** = our probe on a live endpoint (2026-11-04/05: `deepseek-flash` on the OpenAI-compatible endpoint `https://api.deepseek.com`; MiniMax-M3 on api.minimaxi.com Anthropic + OpenAI endpoints; max_tokens ≤ 64; 53 requests, 30,552 tokens total). **DOC** = vendor documentation only (DeepSeek docs read 2026-11-05). **UNTESTED** = neither.
Raw results: `fixtures/probe_results.json` (sanitized). Probe IDs below match that file.

Primary provider is DeepSeek (DECISIONS D4). Every rule here needs an offline regression test using the fixture shapes. Rules marked **LIVE TEST** also need a live test against DeepSeek in `tests/live` (gated by `RUN_LIVE=1`, within the 200k per-task cap). MiniMax rules are kept for the optional Anthropic-format adapter (offline only).

## C12 DeepSeek (primary) — summary of what the harness must do
| Rule | Status | Evidence | LIVE TEST |
|---|---|---|---|
| Use model `deepseek-flash` on `https://api.deepseek.com` (OpenAI format). It supports tool calls, thinking and non-thinking modes, 1M context. | DOC | https://api-docs.deepseek.com/quick_start/pricing | net-check only |
| Set thinking explicitly per request: `{"thinking":{"type":"enabled"}}` + `reasoning_effort` (low/high/max) or `{"type":"disabled"}`. Thinking is ON by default (effort high). In thinking mode temperature/penalties have no effect. | DOC; default-on consistent with probe g1 (13 reasoning tokens with the param omitted) | https://api-docs.deepseek.com/guides/thinking_mode | yes (both modes) |
| Thinking enabled + tools: on continuation after a tool result, an assistant tool-call turn in the request must include that turn's `reasoning_content` when replaying a prior model tool call — missing it returns HTTP **400** (MEASURED **g7**); with passback returns **200** (MEASURED **g9**). | MEASURED g7/g9; DOC | https://api-docs.deepseek.com/guides/thinking_mode#tool-calls | yes (g7/g9) |
| Stripping `reasoning_content` from prior assistant tool turns in a longer multi-turn chat (then continuing with thinking enabled) has **measured HTTP 200** on DeepSeek (Round 4 live); do not rely on the API to enforce passback in that shape — the harness must always store and replay `reasoning_content` verbatim. | MEASURED Round 4 live + g2/g6 (thinking param omitted → drop still 200) | fixtures | yes (regression) |
| Tool names: only `a-z A-Z 0-9 _ -`, max length 128 (DOC). A name with `.` or `/` → 400 (MEASURED f). A 70-char name → 200 (MEASURED f; 128 itself not tested). Normalise all names (see C4). | MEASURED + DOC | https://api-docs.deepseek.com/api/create-chat-completion | yes |
| Usage: `prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens`; `prompt_tokens_details.cached_tokens` = hit; `completion_tokens_details.reasoning_tokens` is inside `completion_tokens`. Normalise: input_total = prompt_tokens, cached_read = hit, uncached = miss, output = completion_tokens. Repeat of an identical 1,972-token prompt hit 1,792 cached (h2/h3). | MEASURED d/h/g; DOC | create-chat-completion reference | yes |
| Streaming: a usage chunk arrived even WITHOUT `stream_options.include_usage` (MEASURED j_stream_no_include_usage). The API reference says usage is absent unless include_usage=true → always send `include_usage: true` and also accept usage when it appears unrequested. | MEASURED j; DOC conflicts | create-chat-completion reference | yes |
| Truncation: max_tokens hit inside a tool call → `finish_reason: "length"` with incomplete/invalid JSON arguments (MEASURED i). Treat as truncated, never execute. | MEASURED i | fixtures | yes |
| Pairing: missing / mismatched / non-adjacent tool result → 400 (MEASURED a/b/c). Chat Completions does not support inserting client-made tool calls mid-conversation (DOC) → never fabricate assistant tool calls; synthetic results only for calls the model actually made. | MEASURED + DOC | https://api-docs.deepseek.com/guides/tool_calls | yes (a/b/c) |
| `max_tokens`: 1–393,216; default 8K non-thinking, 64K thinking. Always send it explicitly (token-cap reservation). | DOC | create-chat-completion reference | — |

Pricing reference (DOC, read 2026-11-05, per 1M tokens, `deepseek-flash`, off-peak / peak): input cache hit $0.003 / $0.006; input cache miss $0.15 / $0.30; output $0.60 / $1.20. Peak = 01:00–04:00 and 06:00–10:00 UTC Mon–Fri (09:00–12:00 and 14:00–18:00 UTC+8). Source: https://api-docs.deepseek.com/quick_start/pricing. Prices may change; the token cap, not price, is the binding limit.

## C1 Tool-call pairing — hard rule, all providers
- Every assistant tool_call / tool_use id must be answered by exactly one tool result before the next model turn; ids must match; the result must come right after the call. MEASURED: probes a/b/c → HTTP 400 on MiniMax-Anthropic, MiniMax-OpenAI, DeepSeek.
- Paths that leave an orphan: an exception after the assistant message is persisted but before results are written (budget finalization, loop guard, cancel during tool exec or approval). MEASURED once in the reference system's run (finalization error left 1 orphan id).
- Required: before every model call, scan history; for each unanswered id append a synthetic `is_error` result ("tool call interrupted: <reason>"); record a `protocol_repair` event. Prefer committing assistant message + results atomically.

## C2 Consecutive same-role messages
- Two consecutive user messages → 200 on all three (MEASURED d). Still: merge all tool_results of one turn into ONE user message (Anthropic format) for portability to strict proxies (DOC).

## C3 System prompt
- Anthropic: system is a top-level field. Never let a later "system" message overwrite the original — concatenate, or put late instructions in the last user text. (Observed failure mode in the reference design: a later system message replaced the original one.)
- MiniMax-Anthropic accepted a system message inside `messages` (MEASURED e). Do not rely on it.

## C4 Tool names
- Normalise every tool name to `^[a-zA-Z0-9_-]{1,64}$` (OpenAI DOC limit 64). Keep a bidirectional map; on collision append a short hash.
- MEASURED f: DeepSeek → 400 for a name containing `.` or `/`; MiniMax accepted it. A 70-char name was accepted by all three (OpenAI documents max 64, DeepSeek max 128; normalising to ≤64 satisfies both).
- MCP names `mcp__{server}__{tool}`: parse from the map, never by splitting on `__`.

## C5 Reasoning / thinking passback
- DeepSeek live contract (MEASURED): **g7** — `thinking` enabled, messages end with user → assistant tool call (no `reasoning_content`) → tool result; next completion → HTTP **400** (`reasoning_content` must be passed back). **g9** — same with `reasoning_content` on the assistant tool turn → HTTP **200**.
- Longer multi-turn histories with `reasoning_content` removed from earlier assistant tool turns may still get HTTP **200** from DeepSeek when continuing (measured Round 4); that is not permission to omit passback in the harness.
- MiniMax (optional adapter): DOC requires passing back the full content incl. thinking blocks (with signature). MiniMax-M3 thinking is OFF by default on the Anthropic endpoint and ON by default on the OpenAI endpoint (`<think>` in content). Passback-failure behaviour UNTESTED.
- Required: store reasoning/thinking blocks verbatim per assistant message and replay them on every model request when thinking is enabled.

## C6 Truncation (stop_reason)
- With max_tokens=40 mid tool call (MEASURED i): MiniMax-Anthropic stop_reason `tool_use` with valid-but-truncated input; MiniMax-OpenAI finish_reason `tool_calls`; DeepSeek finish_reason `length` with invalid JSON.
- So stop_reason alone is NOT enough. Treat `output_tokens >= max_tokens` OR stop/finish reason in {max_tokens, length} OR JSON parse failure as truncated → do not execute; retry with a larger limit or return a clear error to the model. Never silently substitute `{}`.
- Thinking tokens count against max_tokens on MiniMax (DOC). Do not hard-code 4096.

## C7 Usage / cache accounting
- Normalised record per HTTP call: input_total, cached_read, cache_write, uncached, output, reasoning.
- Anthropic format (MiniMax-Anthropic MEASURED; DeepSeek-Anthropic DOC says the same): `input_tokens` EXCLUDES cache reads → input_total = input_tokens + cache_read + cache_creation (MEASURED h: input 1 + cache_read 2136).
- OpenAI-format (DeepSeek primary, MiniMax-OpenAI): `prompt_tokens` INCLUDES cached (`prompt_tokens_details.cached_tokens`; DeepSeek also `prompt_cache_hit_tokens`/`miss`) (MEASURED h).
- Streaming: MiniMax-Anthropic `message_start` usage is all zeros; real numbers arrive in `message_delta` → merge, do not sum (MEASURED j). MiniMax-OpenAI sends no usage unless `stream_options.include_usage=true` (MEASURED j); DeepSeek sends it either way.
- MiniMax shares cache across its two endpoints (first OpenAI request after Anthropic hit 2136 cached) and auto-caches without cache_control (MEASURED h). DeepSeek repeat hit 1792/1972 (MEASURED h). → A/B runs must alternate order and report uncached.
- Cache prefix order tools → system → messages (Anthropic DOC): changing tools or system mid-run invalidates the cache (observed drop 9,906→128 when both changed).

## C8 Streaming tool-call assembly
- Argument fragments arrive split (DeepSeek 8 fragments, MiniMax 4 for the same call, MEASURED j). Concatenate arguments; assign the id on first appearance only (do not concatenate ids — UNTESTED risk on gateways that resend ids).

## C9 Retry
- Retry only 429 / ≥500 / network errors, only if nothing was streamed to the user yet, max 3 attempts with backoff. Error messages carry HTTP status, request id header and the first 800 chars of the body.

## C10 MCP stdio (MEASURED locally with SDK `mcp` 2.x)
- One JSON-RPC message per line; a multi-line JSON message got NO response (hang) → always serialise single-line, with timeouts.
- `tools/list` before `initialize` → error -32602. A missing `notifications/initialized` was tolerated by the SDK server — still send it.
- SDK 2.x renamed `FastMCP` → `MCPServer` (`from mcp.server.mcpserver import MCPServer`). Pin versions.
- Read the server's stderr continuously (a never-read PIPE can block the child — UNTESTED). Distinguish server→client requests (have `method` + `id`) from responses.
- On a failed call: restart the server, do NOT replay the call (side effects); report "outcome uncertain".

## C11 SWE-bench
- Prediction line: `{"instance_id", "model_name_or_path", "model_patch"}`; emit an empty-patch line on failure so predictions stay 1:1 with instances; support resume.
- Harness applies with fallbacks (git apply → --3way → --reject → patch fuzz) (DOC/source).
- Same `run_id` + same instance reuses the cached result even with a different patch (evaluation.md:123) → new run_id every evaluation.
- Patch hygiene: list changed files; exclude scratch files; record the file list.

## C11b Safety constraints for the builder agent itself
- Never print or log API keys; keys only via env/secrets; scrub `Authorization`/`x-api-key` from stored requests.
- No live API calls by default; live tests require `RUN_LIVE=1`, eval requires `RUN_EVAL=1`; both go through the UsageLedger and stop at `MAX_TASK_TOKENS` (hard max 200,000 per task, see DECISIONS.md).
- Required provider for live tests: DeepSeek `deepseek-flash`, OpenAI-compatible endpoint `https://api.deepseek.com`, key from env `DEEPSEEK_API_KEY`. MiniMax results above are reference data for the optional Anthropic-format adapter (offline tests only).
