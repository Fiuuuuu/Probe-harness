# BASELINE.md — real measured numbers (the only numbers the project may quote)

Task: SWE-bench Lite `pytest-dev__pytest-7432`, model MiniMax-M3 (Anthropic-compatible endpoint), 2026-11-02..04. One run per configuration — anecdotal, not statistically meaningful.

| Metric | R1: reference system, benchmark defaults | R2: reference system, token budget unlimited | M: mini-swe-agent 2.4.6 |
|---|---|---|---|
| Budget config | tokens 80k/100k/120k; calls 48/56/64; max_steps 50 | tokens 1e9/2e9/3e9; calls 48/56/64; max_steps 50 | mini defaults |
| Outcome | failed, stop_reason run_budget (tool call during finalization); empty patch | failed, stop_reason max_steps; **resolved** by official harness | Submitted; **resolved** |
| Model calls | 18 | 50 | 27 |
| Input tokens total | 112,479 | 1,199,151 | 278,238 |
| Cached input | 93,228 (82.9%) | 1,160,964 (96.8%) | 262,032 (94.2%) |
| Uncached input | 19,251 | 38,187 | 16,206 |
| Output tokens | 1,587 | 8,866 | 9,208 |
| Tool calls | 19 | 64 (shell 52, read_file 5, task_update 4, list_files 1, task_create 1, write_file 1) | — |
| Agent time (run row / trajectory) | 97.0 s | 606.7 s | 399.5 s |
| Process wall clock | 106.9 s | 617.6 s | 409 s |

R1 budget events: warning at step 16 (84,280 chargeable tokens, 15 calls); finalizing at step 18 (104,613, 17 calls); then the model called a tool → run-policy error; 1 orphan tool_call left in history. Cached input at step 17 = 9,906, step 18 = 128 (finalization changed both tools and system).

R2 timeline: fix written step 15; first passing repro step 17; bug re-introduced as a check step 33 (2 failed); test file green step 34 (80 passed); steps 35–50 full test-suite runs, changelog, lint. Input by phase: steps 1–15 160,305; 16–34 488,605; 35–50 550,241. Input after step 15 = 86.6% of total. Per-call input grew from 1,419 (step 1) to 37,842 (step 50). Step-50 context estimate: system 86, messages 9,799, tool results 28,013 (72.5%), tool schemas 740. Compaction ran only at steps 5–6.

R2 per-call input tokens (steps 1–50):
1419,1741,4287,4544,8327,10853,11311,12494,12887,13248,13543,13817,16437,17174,18223,19021,21568,21819,22441,23550,23775,24464,25075,25536,26041,26608,27099,27456,27905,28544,28953,29278,29585,29887,30053,31410,32295,32771,33070,33259,33593,33820,34839,35088,35280,35950,36598,37042,37331,37842

Rules: any target is a GOAL TO BE MEASURED. Never present a target or estimate as a result.
