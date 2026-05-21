"""SQLite-backed usage logging and query helpers.

Cost is intentionally **not** stored in the database — it is computed at
query time from the current config so that pricing changes apply retroactively.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    ts_epoch           REAL    NOT NULL,
    model              TEXT    NOT NULL DEFAULT '',
    provider           TEXT    NOT NULL DEFAULT '',
    backend_model      TEXT    NOT NULL DEFAULT '',
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    streaming          INTEGER NOT NULL DEFAULT 0,
    status             INTEGER NOT NULL DEFAULT 200,
    duration_ms        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_ts    ON usage (ts_epoch);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage (model);
"""


class UsageDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate(self) -> None:
        """Drop legacy precomputed cost columns if present."""
        cursor = self._conn.execute("PRAGMA table_info(usage)")
        cols = {row[1] for row in cursor.fetchall()}
        if "cost_cny" in cols or "cost_usd" in cols:
            self._conn.execute("ALTER TABLE usage RENAME TO _usage_migration")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                """
                INSERT INTO usage (
                    id, ts, ts_epoch, model, provider, backend_model,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    streaming, status, duration_ms
                )
                SELECT
                    id, ts, ts_epoch, model, provider, backend_model,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    streaming, status, duration_ms
                FROM _usage_migration
                """
            )
            self._conn.execute("DROP TABLE _usage_migration")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def log(self, record: dict) -> None:
        row = self._normalize_record(record)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO usage (
                    ts, ts_epoch, model, provider, backend_model,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    streaming, status, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ts"],
                    row["ts_epoch"],
                    row["model"],
                    row["provider"],
                    row["backend_model"],
                    row["input_tokens"],
                    row["output_tokens"],
                    row["cache_read_tokens"],
                    row["cache_write_tokens"],
                    row["streaming"],
                    row["status"],
                    row["duration_ms"],
                ),
            )
            self._conn.commit()

    def query_raw(self, since_epoch: float, until_epoch: float) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ts, model, provider, backend_model,
                       input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens,
                       streaming, status, duration_ms
                FROM usage
                WHERE ts_epoch >= ? AND ts_epoch < ?
                ORDER BY ts_epoch ASC, id ASC
                """,
                (since_epoch, until_epoch),
            ).fetchall()
        return [self._raw_row_to_dict(row) for row in rows]

    def query_by_model(self, since_epoch: float, until_epoch: float) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT model,
                       provider,
                       COUNT(*) AS requests,
                       SUM(input_tokens) AS input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(cache_read_tokens) AS cache_read_tokens,
                       SUM(cache_write_tokens) AS cache_write_tokens
                FROM usage
                WHERE ts_epoch >= ? AND ts_epoch < ?
                GROUP BY model, provider
                ORDER BY requests DESC, model ASC
                """,
                (since_epoch, until_epoch),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_by_day(self, since_epoch: float, until_epoch: float) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT substr(ts, 1, 10) AS day,
                       COUNT(*) AS requests,
                       SUM(input_tokens) AS input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(cache_read_tokens) AS cache_read_tokens,
                       SUM(cache_write_tokens) AS cache_write_tokens
                FROM usage
                WHERE ts_epoch >= ? AND ts_epoch < ?
                GROUP BY day
                ORDER BY day DESC
                """,
                (since_epoch, until_epoch),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_recent(self, limit: int = 100, offset: int = 0) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage ORDER BY ts_epoch DESC, id DESC LIMIT ? OFFSET ?",
                (max(0, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._full_row_to_dict(row) for row in rows]

    def count_all(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]

    def _normalize_record(self, record: dict) -> Dict[str, Any]:
        ts = str(record.get("ts") or datetime.now().astimezone().isoformat(timespec="seconds"))
        model = str(record.get("model") or "")
        provider = str(record.get("provider") or (model.split("/", 1)[0] if "/" in model else ""))
        return {
            "ts": ts,
            "ts_epoch": self._to_epoch(ts),
            "model": model,
            "provider": provider,
            "backend_model": str(record.get("backend_model") or ""),
            "input_tokens": int(record.get("input_tokens", 0) or 0),
            "output_tokens": int(record.get("output_tokens", 0) or 0),
            "cache_read_tokens": int(record.get("cache_read_tokens", 0) or 0),
            "cache_write_tokens": int(record.get("cache_write_tokens", 0) or 0),
            "streaming": int(bool(record.get("streaming", False))),
            "status": int(record.get("status", 200) or 200),
            "duration_ms": int(record.get("duration_ms", 0) or 0),
        }

    @staticmethod
    def _to_epoch(ts: str) -> float:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return time.time()

    @staticmethod
    def _raw_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "ts": row["ts"],
            "model": row["model"],
            "provider": row["provider"],
            "backend_model": row["backend_model"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cache_read_tokens": row["cache_read_tokens"],
            "cache_write_tokens": row["cache_write_tokens"],
            "streaming": bool(row["streaming"]),
            "status": row["status"],
            "duration_ms": row["duration_ms"],
        }

    @staticmethod
    def _full_row_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["streaming"] = bool(data["streaming"])
        return data


_db_instance: Optional[UsageDB] = None


def setup_usage_db(db_path: Path) -> None:
    global _db_instance
    db_path = Path(db_path).expanduser()
    if _db_instance is not None and _db_instance.db_path == db_path:
        return
    if _db_instance is not None:
        try:
            _db_instance.close()
        except Exception:
            pass
    _db_instance = UsageDB(db_path)


def get_usage_db() -> Optional[UsageDB]:
    return _db_instance


def log_usage(record: dict) -> None:
    db = get_usage_db()
    if db is None:
        return
    try:
        db.log(record)
    except Exception:
        pass
