"""PIT monthly-revenue ingestion and persistence tests."""

from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from database import create_engine_from_settings, session_scope
from integrations.finmind_monthly_revenue import fetch_monthly_revenues
from repositories.monthly_revenues import load_monthly_revenues, upsert_monthly_revenues
from repositories.stocks import upsert_stocks


class _Response:
    status_code = 200

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def json(self) -> dict:
        return {"status": 200, "msg": "success", "data": self.rows}


def _requester(rows: list[dict]):
    def request(*args, **kwargs):
        return _Response(rows)

    return request


def _migration(name: str):
    path = Path(__file__).resolve().parents[1] / "database" / "migrations" / "versions" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finmind_monthly_revenue_uses_observed_time_and_conservative_fallback() -> None:
    rows = [
        {
            "date": "2026-09-01",
            "stock_id": "2330",
            "country": "Taiwan",
            "revenue": 100,
            "revenue_month": 8,
            "revenue_year": 2026,
            "create_time": "2026-09-09 18:01:02",
        },
        {
            "date": "2020-02-01",
            "stock_id": "2330",
            "country": "Taiwan",
            "revenue": 80,
            "revenue_month": 1,
            "revenue_year": 2020,
            "create_time": None,
        },
    ]
    frame = fetch_monthly_revenues(
        "2020-01-01", "2026-10-16", "token", requester=_requester(rows)
    )

    assert frame[["revenue_year", "revenue_month"]].values.tolist() == [[2020, 1], [2026, 8]]
    assert frame["available_date"].tolist() == ["2020-02-11", "2026-09-09"]
    assert frame["source_date"].tolist() == ["2020-02-01", "2026-09-01"]


def test_monthly_revenue_repository_is_idempotent_and_pit_filtered(
    settings, temp_db_path: Path
) -> None:
    conn = sqlite3.connect(temp_db_path)
    try:
        _migration("001_initial_schema.py").upgrade(conn)
        _migration("010_monthly_revenues.py").upgrade(conn)
    finally:
        conn.close()
    configured = replace(
        settings, data=replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(configured)
    rows = pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "revenue_year": 2026,
                "revenue_month": 7,
                "revenue": 90.0,
                "source_date": "2026-08-01",
                "create_time": "2026-08-08",
                "available_date": "2026-08-08",
                "country": "Taiwan",
                "source": "finmind",
            },
            {
                "stock_id": "2330",
                "revenue_year": 2026,
                "revenue_month": 8,
                "revenue": 100.0,
                "source_date": "2026-09-01",
                "create_time": "2026-09-09",
                "available_date": "2026-09-09",
                "country": "Taiwan",
                "source": "finmind",
            },
        ]
    )
    try:
        with session_scope(engine) as session:
            upsert_stocks(session, pd.DataFrame({"stock_id": ["2330"], "market": ["TWSE"]}))
            assert upsert_monthly_revenues(session, rows) == 2
            assert upsert_monthly_revenues(session, rows) == 2
        with session_scope(engine) as session:
            early = load_monthly_revenues(session, date(2026, 9, 1))
            late = load_monthly_revenues(session, date(2026, 9, 30))
        assert early["revenue_month"].tolist() == [7]
        assert late["revenue_month"].tolist() == [7, 8]
    finally:
        engine.dispose()
