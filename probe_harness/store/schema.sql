PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    stop_reason TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    config_hash TEXT,
    recovered_from TEXT,
    started_at REAL NOT NULL,
    ended_at REAL
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    step INTEGER NOT NULL,
    type TEXT NOT NULL,
    ts REAL NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    input_total INTEGER NOT NULL,
    cached_read INTEGER NOT NULL DEFAULT 0,
    cache_write INTEGER NOT NULL DEFAULT 0,
    uncached INTEGER NOT NULL DEFAULT 0,
    output INTEGER NOT NULL DEFAULT 0,
    reasoning INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    latency_ms INTEGER,
    request_sha256 TEXT,
    http_status INTEGER,
    request_id TEXT,
    response_model TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    call_id TEXT NOT NULL,
    raw_name TEXT NOT NULL,
    norm_name TEXT NOT NULL,
    args_sha256 TEXT,
    status TEXT NOT NULL,
    exit_code INTEGER,
    duration_ms INTEGER,
    output_tokens INTEGER,
    truncated INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    boundary TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    usage_json TEXT,
    budget_json TEXT,
    PRIMARY KEY (run_id, step, boundary),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS tool_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lineage_run_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    norm_name TEXT NOT NULL,
    args_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    ts REAL NOT NULL,
    UNIQUE (lineage_run_id, call_id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    call_id TEXT,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    data_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    scope TEXT,
    rule_hash TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    patch_sha256 TEXT,
    patch_files TEXT,
    resolved INTEGER,
    harness_run_id TEXT,
    eval_log_path TEXT,
    PRIMARY KEY (run_id, instance_id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
