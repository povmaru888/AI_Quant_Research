"""Cross-process request budget backed by a small SQLite sidecar."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock


class SqliteRequestLimiter:
    """Reserve request slots inside a rolling window across local processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int = 540,
        window_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("request limit and window must be positive")
        self.path = Path(path)
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self._setup_lock = Lock()
        self._ready = False

    def _setup(self) -> None:
        if self._ready:
            return
        with self._setup_lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vendor_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        requested_at REAL NOT NULL,
                        dataset TEXT NOT NULL,
                        status_code INTEGER
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vendor_requests_time "
                    "ON vendor_requests (requested_at)"
                )
            self._ready = True

    def acquire(self, dataset: str) -> int:
        self._setup()
        while True:
            now = self.clock()
            cutoff = now - self.window_seconds
            with sqlite3.connect(self.path, timeout=30, isolation_level=None) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM vendor_requests WHERE requested_at < ?", (cutoff,))
                rows = conn.execute(
                    "SELECT id, requested_at FROM vendor_requests ORDER BY requested_at"
                ).fetchall()
                if len(rows) < self.limit:
                    cursor = conn.execute(
                        "INSERT INTO vendor_requests (requested_at, dataset) VALUES (?, ?)",
                        (now, dataset),
                    )
                    conn.commit()
                    return int(cursor.lastrowid)
                wait = max(0.05, self.window_seconds - (now - float(rows[0][1])) + 0.05)
                conn.commit()
            self.sleeper(min(wait, 60.0))

    def finish(self, request_id: int, status_code: int | None) -> None:
        self._setup()
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute(
                "UPDATE vendor_requests SET status_code = ? WHERE id = ?",
                (status_code, request_id),
            )

    def usage(self) -> int:
        self._setup()
        cutoff = self.clock() - self.window_seconds
        with sqlite3.connect(self.path, timeout=30) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM vendor_requests WHERE requested_at >= ?", (cutoff,)
                ).fetchone()[0]
            )


_DEFAULT_LIMITER: SqliteRequestLimiter | None = None


def default_finmind_limiter() -> SqliteRequestLimiter:
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        path = os.environ.get("FINMIND_RATE_LEDGER_PATH", "database/finmind_rate_limit.sqlite3")
        limit = int(os.environ.get("FINMIND_REQUESTS_PER_HOUR", "540"))
        _DEFAULT_LIMITER = SqliteRequestLimiter(path, limit=limit)
    return _DEFAULT_LIMITER
