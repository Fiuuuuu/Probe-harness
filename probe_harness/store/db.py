"""SQLite store with WAL and a single-writer event append path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_event_writer_lock = asyncio.Lock()


def default_db_path() -> Path:
    return Path(".harness") / "harness.db"


class Store:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_db_path()
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        await self._conn.executescript(schema)
        await self._migrate(self._conn)
        await self._conn.commit()
        from probe_harness.runs.service import ensure_startup_sweep

        await ensure_startup_sweep(self)

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        cur = await conn.execute("PRAGMA table_info(model_calls)")
        cols = {row[1] for row in await cur.fetchall()}
        if "response_model" not in cols:
            await conn.execute("ALTER TABLE model_calls ADD COLUMN response_model TEXT")
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_journal'"
        )
        if await cur.fetchone() is None:
            await conn.execute(
                """
                CREATE TABLE tool_journal (
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
                )
                """
            )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store not connected")
        return self._conn

    async def execute(self, sql: str, params: tuple | list = ()) -> aiosqlite.Cursor:
        cur = await self.conn.execute(sql, params)
        await self.conn.commit()
        return cur

    async def fetchone(self, sql: str, params: tuple | list = ()) -> aiosqlite.Row | None:
        cur = await self.conn.execute(sql, params)
        return await cur.fetchone()

    async def fetchall(self, sql: str, params: tuple | list = ()) -> list[aiosqlite.Row]:
        cur = await self.conn.execute(sql, params)
        return await cur.fetchall()


async def next_event_seq(conn: aiosqlite.Connection, run_id: str) -> int:
    cur = await conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE run_id = ?",
        (run_id,),
    )
    row = await cur.fetchone()
    return int(row[0])


class EventWriter:
    """Single-writer append for events (UNIQUE run_id, seq)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    async def append(
        self,
        run_id: str,
        step: int,
        event_type: str,
        ts: float,
        data_json: str,
    ) -> int:
        async with _event_writer_lock:
            seq = await next_event_seq(self.store.conn, run_id)
            await self.store.conn.execute(
                """
                INSERT INTO events (run_id, seq, step, type, ts, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, seq, step, event_type, ts, data_json),
            )
            await self.store.conn.commit()
            return seq
