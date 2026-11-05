# Probe-harness

一个可观测、可量化的编码 Agent harness：以 learn-claude-code 为机制参考，用 Python 3.11 asyncio 重新实现，从第一阶段起内置事件、用量和 SWE-bench 评测。

An observable, measurable coding-agent harness, rebuilt in Python 3.11 asyncio using learn-claude-code as the mechanism reference, with events, usage accounting and SWE-bench evaluation built in from the first phase.

## CLI（T0）

安装：`pip install -e ".[dev]"`

```bash
# 离线 fake 任务（写入 .harness/harness.db）
harness run --provider fake --task tests/fixtures/echo_task.json

# 指标表（来自 store，非临时计算）
harness report <run_id>

# DeepSeek 连通性（401 亦视为 reachable；RUN_LIVE=1 且存在 DEEPSEEK_API_KEY 时走 ledger 做一次认证探测）
harness net-check
```

本地状态与用量：

- `.harness/harness.db` — SQLite WAL 运行库（已在 `.gitignore`）
- `.harness/usage_ledger.jsonl` — 跨进程 token 账本（200k 硬顶，git 忽略）

测试：

```bash
pytest -q tests/unit
RUN_LIVE=1 pytest -q tests/live -m live   # 需要 DEEPSEEK_API_KEY
ruff check probe_harness tests
```

## 文档 / Docs
- `docs/DECISIONS.md` — binding decisions (provider, key, 200k token cap per task, scope).
- `docs/PLAN.md` — design: principles, architecture, data model, loop, phases, eval protocol.
- `docs/CONSTRAINTS.md` — API/protocol rules, each marked MEASURED / DOC / UNTESTED.
- `docs/BASELINE.md` — the only real measured numbers the project may quote.
- `docs/ACCEPTANCE.md` — definition of done and commands per phase.
- `prompts/TASKS.md` — task prompts: T0–T6 first batch, T7–T11 later.
- `fixtures/` — sanitized API probe results and the probe scripts that produced them (reference only; keys come from environment variables).

## 规则 / Rules
- 主模型：DeepSeek `deepseek-flash`（OpenAI 兼容接口 https://api.deepseek.com）。Primary model: DeepSeek `deepseek-flash` via the OpenAI-compatible endpoint.
- 密钥只从环境变量 `DEEPSEEK_API_KEY` 读取，绝不入库。Keys only from the environment; never committed.
- 实时调用需 `RUN_LIVE=1` / `RUN_EVAL=1`，每个任务上限 200,000 token。Live calls are gated and capped at 200,000 tokens per task.
- 不编造评测数字。No invented metrics.

No LICENSE yet.
