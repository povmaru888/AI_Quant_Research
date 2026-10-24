"""Small persistence store for live snapshots, separate from research data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MonitorStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS fund_snapshots (
                    fund_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    nav REAL NOT NULL,
                    unit_nav REAL NOT NULL,
                    equity_exposure REAL NOT NULL,
                    valuation_quality TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (fund_id, observed_at)
                );
                CREATE INDEX IF NOT EXISTS idx_fund_snapshots_time
                    ON fund_snapshots (fund_id, observed_at);
                CREATE TABLE IF NOT EXISTS monitor_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def save_snapshot(self, fund_id: str, snapshot: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO fund_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fund_id,
                    snapshot["as_of"],
                    snapshot["nav"],
                    snapshot["unit_nav"],
                    snapshot["equity_exposure"],
                    snapshot["valuation_quality"],
                    json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def recent_snapshots(self, fund_id: str, limit: int = 500) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM fund_snapshots WHERE fund_id = ? "
                "ORDER BY observed_at DESC LIMIT ?",
                (fund_id, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def log_event(self, severity: str, event_type: str, message: str) -> None:
        from datetime import datetime, timezone

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO monitor_events (observed_at, severity, event_type, message) "
                "VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), severity, event_type, message),
            )
