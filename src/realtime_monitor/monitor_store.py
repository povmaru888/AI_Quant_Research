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
                CREATE TABLE IF NOT EXISTS trading_calendar_years (
                    calendar_year INTEGER PRIMARY KEY,
                    holiday_dates_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS intraday_sessions (
                    fund_id TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (fund_id, session_date)
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

    def save_calendar_year(self, year: int, holiday_dates: list[str], source: str) -> None:
        from datetime import datetime, timezone

        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO trading_calendar_years VALUES (?, ?, ?, ?)",
                (
                    year,
                    json.dumps(holiday_dates, separators=(",", ":")),
                    source,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_calendar_year(self, year: int) -> list[str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT holiday_dates_json FROM trading_calendar_years WHERE calendar_year = ?",
                (year,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_intraday_session(self, fund_id: str, payload: dict) -> None:
        from datetime import datetime, timezone

        session_date = payload.get("session_date")
        if not session_date:
            return
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO intraday_sessions VALUES (?, ?, ?, ?)",
                (
                    fund_id,
                    session_date,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_intraday_session(self, fund_id: str, session_date: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM intraday_sessions WHERE fund_id = ? AND session_date = ?",
                (fund_id, session_date),
            ).fetchone()
        return json.loads(row[0]) if row else None
