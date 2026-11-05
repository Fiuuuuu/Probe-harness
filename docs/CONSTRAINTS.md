# CONSTRAINTS.md — verified API / protocol rules

Status legend: **MEASURED** = probe on a live endpoint (2026-11-04/05, model MiniMax-M3 on api.minimaxi.com Anthropic + OpenAI endpoints, `deepseek-flash` on api.deepseek.com, max_tokens ≤ 64; 53 requests, 30,552 tokens total). **DOC** = vendor documentation only. **UNTESTED** = neither.
Raw results: `fixtures/probe_results.json` (sanitized). Probe IDs below match that file.

Every rule here must have a regression test (offline, using the fixture shapes) and, where marked, a live probe test gated by `RUN_LIVE=1` (MiniMax-Anthropic only; within the token cap).

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
- MEASURED f: DeepSeek → 400 for a name containing `.` or `/`; MiniMax accepted it. A 70-char name was accepted by all three (beyond the documented 64 — do not rely on it).
- MCP names `mcp__{server}__{tool}`: parse from the map, never by splitting on `__`.

## C5 Reasoning / thinking passback
- DeepSeek with thinking explicitly enabled and a prior tool turn lacking `reasoning_content` → 400 (MEASURED g7); with it passed back → 200 (g9). With the thinking param omitted, dropping reasoning still returned 200 (g2/g6) — unexplained, do not rely on it.
- MiniMax: DOC requires passing back the full content incl. thinking blocks (with signature). MiniMax-M3 thinking is OFF by default on the Anthropic endpoint and ON by default on the OpenAI endpoint (`<think>` in content). Passback-failure behaviour UNTESTED.
- Required: store reasoning/thinking blocks verbatim per assistant message and replay them.

## C6 Truncation (stop_reason)
- With max_tokens=40 mid tool call (MEASURED i): MiniMax-Anthropic stop_reason `tool_use` with valid-but-truncated input; MiniMax-OpenAI finish_reason `tool_calls`; DeepSeek finish_reason `length` with invalid JSON.
- So stop_reason alone is NOT enough. Treat `output_tokens >= max_tokens` OR stop/finish reason in {max_tokens, length} OR JSON parse failure as truncated → do not execute; retry with a larger limit or return a clear error to the model. Never silently substitute `{}`.
- Thinking tokens count against max_tokens on MiniMax (DOC). Do not hard-code 4096.

## C7 Usage / cache accounting
- Normalised record per HTTP call: input_total, cached_read, cache_write, uncached, output, reasoning.
- MiniMax-Anthropic: `input_tokens` EXCLUDES cache reads → input_total = input_tokens + cache_read + cache_creation (MEASURED h: input 1 + cache_read 2136).
- OpenAI-format (MiniMax-OpenAI, DeepSeek): `prompt_tokens` INCLUDES cached (`prompt_tokens_details.cached_tokens`; DeepSeek also `prompt_cache_hit_tokens`/`miss`) (MEASURED h).
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
- Required provider for live tests: MiniMax-M3, Anthropic-compatible endpoint, key from env `MINIMAX_API_KEY`. DeepSeek/OpenAI-format results above are reference data for adapters only.
