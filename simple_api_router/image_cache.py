"""Persistent image description cache backed by SQLite.

Cache keys and TTLs
-------------------
- **URL images**: key = URL string, TTL = 1 hour (content at a URL can change,
  but web assets are usually stable for short periods)
- **Content images** (base64 / local file): key = MD5 hex of raw bytes, TTL = 30 days
  (same bytes always produce the same image)

Expired entries are evicted on every read and write, so the DB stays small.
The database file lives alongside the router logs.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

URL_TTL: float = 3600.0          # 1 hour
CONTENT_TTL: float = 30 * 24 * 3600.0  # 30 days

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_cache (
    key         TEXT    PRIMARY KEY,
    description TEXT    NOT NULL,
    expires_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_expires ON image_cache (expires_at);
"""


class ImageDescriptionCache:
    """Thread-safe SQLite-backed cache for image descriptions."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._init()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _evict(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM image_cache WHERE expires_at < ?", (time.time(),))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Return the cached description for *key*, or ``None`` if absent / expired."""
        with self._conn() as conn:
            self._evict(conn)
            row = conn.execute(
                "SELECT description FROM image_cache WHERE key = ? AND expires_at >= ?",
                (key, time.time()),
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, description: str, ttl: float) -> None:
        """Store *description* under *key* with the given *ttl* in seconds."""
        expires_at = time.time() + ttl
        with self._conn() as conn:
            self._evict(conn)
            conn.execute(
                "INSERT OR REPLACE INTO image_cache (key, description, expires_at)"
                " VALUES (?, ?, ?)",
                (key, description, expires_at),
            )
