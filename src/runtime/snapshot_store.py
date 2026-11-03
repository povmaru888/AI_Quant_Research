"""Read-only dashboard store backed by a compact deployment snapshot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


class SnapshotStore:
    """Serve the dashboard contracts without the research database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"dashboard snapshot not found: {self.path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _payload(self, run_id: str, kind: str, as_of: str = ""):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM payloads WHERE run_id=? AND kind=? AND as_of=?",
                (run_id, kind, as_of),
            ).fetchone()
        if row is None:
            raise ValueError(f"snapshot payload missing: {run_id!r}/{kind!r}/{as_of!r}")
        return json.loads(row["payload_json"])

    def list_runs(self, status: str | None = None) -> list[dict]:
        query = (
            "SELECT run_id, status, parameter_version, feature_version "
            "FROM runs"
        )
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY sort_order, run_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def get_run_status(self, run_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown run: {run_id!r}")
        return str(row["status"])

    def load_run_summary(self, run_id: str) -> dict:
        return dict(self._payload(run_id, "overview"))

    def list_holding_dates(self, run_id: str) -> list[str]:
        return list(self._payload(run_id, "holding_dates"))

    def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
        payload = self._payload(run_id, "portfolio", as_of)
        return pd.DataFrame(payload["holdings"])

    def load_portfolio_risk(self, run_id: str, as_of: str) -> dict:
        payload = self._payload(run_id, "portfolio", as_of)
        return dict(payload["risk_state"])

    def load_model_data(self, run_id: str) -> dict:
        return dict(self._payload(run_id, "model"))

    def load_risk(self, run_id: str) -> dict:
        return dict(self._payload(run_id, "risk"))

    def load_comparison(self, run_id: str) -> pd.DataFrame:
        return pd.DataFrame(self._payload(run_id, "comparison"))
