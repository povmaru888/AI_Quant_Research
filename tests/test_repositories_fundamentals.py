"""P1-08 acceptance: fundamentals repository upsert and PIT load."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from repositories.fundamentals import (
    FINANCIAL_COLUMNS,
    INSTITUTIONAL_COLUMNS,
    load_pit_financials,
    upsert_financials,
    upsert_institutional,
)
from repositories.stocks import upsert_stocks
from settings import Settings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _engine_for(settings: Settings, path: Path):
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{path}")
    )
    return create_engine_from_settings(db_settings)


def _init_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()


def _seeded_engine(settings: Settings, path: Path):
    _init_schema(path)
    engine = _engine_for(settings, path)
    parents = pd.DataFrame(
        [
            {"stock_id": "2330", "market": "TWSE"},
            {"stock_id": "0050", "market": "TWSE"},
        ]
    )
    with session_scope(engine) as session:
        upsert_stocks(session, parents)
    return engine


def _financial_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "report_period": "2019Q3",
                "announcement_date": "2019-11-01",
                "available_date": "2019-11-05",
                "revenue": 90.0,
                "source": "test",
            },
            {
                "stock_id": "2330",
                "report_period": "2019Q4",
                "announcement_date": "2020-02-01",
                "available_date": "2020-02-05",
                "revenue": 100.0,
                "source": "test",
            },
            {
                "stock_id": "2330",
                "report_period": "2020Q1",
                "announcement_date": "2020-05-01",
                "available_date": "2020-05-05",
                "revenue": 110.0,
                "source": "test",
            },
        ]
    )


def _institutional_rows() -> pd.DataFrame:
    rows = []
    for stock_id in ("2330", "0050"):
        for i, trade_date in enumerate(("2020-01-02", "2020-01-03")):
            rows.append(
                {
                    "trade_date": trade_date,
                    "stock_id": stock_id,
                    "foreign_net_buy": float(10 + i),
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _table_count(engine, table: str) -> int:
    with session_scope(engine) as session:
        return session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def test_upsert_financials_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = _financial_rows()
        with session_scope(engine) as session:
            assert upsert_financials(session, rows) == 3
        with session_scope(engine) as session:
            assert upsert_financials(session, rows) == 3
        assert _table_count(engine, "financials") == 3

        changed = rows.copy()
        changed.loc[1, "revenue"] = 123.0
        with session_scope(engine) as session:
            upsert_financials(session, changed)
        assert _table_count(engine, "financials") == 3
        with session_scope(engine) as session:
            pit = load_pit_financials(session, date(2020, 3, 1))
        assert list(pit["report_period"]) == ["2019Q4"]
        assert pit.iloc[0]["revenue"] == 123.0
    finally:
        engine.dispose()


def test_upsert_institutional_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = _institutional_rows()
        with session_scope(engine) as session:
            assert upsert_institutional(session, rows) == 4
        with session_scope(engine) as session:
            assert upsert_institutional(session, rows) == 4
        assert _table_count(engine, "institutional") == 4
    finally:
        engine.dispose()


def test_upsert_validation(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                upsert_financials(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown Financial columns"):
                upsert_financials(
                    session,
                    pd.DataFrame(
                        [
                            {
                                "stock_id": "2330",
                                "report_period": "2019Q4",
                                "available_date": "2020-02-05",
                                "nope": 1,
                            }
                        ]
                    ),
                )
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                upsert_institutional(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            empty_fin = pd.DataFrame({col: pd.Series(dtype=str) for col in FINANCIAL_COLUMNS})
            assert upsert_financials(session, empty_fin) == 0
            empty_inst = pd.DataFrame({col: pd.Series(dtype=str) for col in INSTITUTIONAL_COLUMNS})
            assert upsert_institutional(session, empty_inst) == 0
        assert _table_count(engine, "financials") == 0
        assert _table_count(engine, "institutional") == 0
    finally:
        engine.dispose()


def test_pit_only_available(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            upsert_financials(session, _financial_rows())
        with session_scope(engine) as session:
            early = load_pit_financials(session, date(2020, 1, 15))
        assert list(early["report_period"]) == ["2019Q3"]
        with session_scope(engine) as session:
            late = load_pit_financials(session, date(2020, 12, 31))
        assert list(late["report_period"]) == ["2020Q1"]
        assert "2020Q1" not in set(early["report_period"])
    finally:
        engine.dispose()


def test_pit_one_row_per_stock(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "report_period": "2019Q3",
                    "announcement_date": "2020-02-01",
                    "available_date": "2020-02-05",
                    "revenue": 100.0,
                    "source": "test",
                },
                {
                    "stock_id": "2330",
                    "report_period": "2019Q4",
                    "announcement_date": "2020-02-03",
                    "available_date": "2020-02-05",
                    "revenue": 101.0,
                    "source": "test",
                },
                {
                    "stock_id": "0050",
                    "report_period": "2019Q4",
                    "announcement_date": "2020-01-15",
                    "available_date": "2020-01-20",
                    "revenue": 50.0,
                    "source": "test",
                },
            ]
        )
        with session_scope(engine) as session:
            # Same available_date: the later announcement_date wins.
            assert upsert_financials(session, rows) == 3
        with session_scope(engine) as session:
            pit = load_pit_financials(session, date(2020, 12, 31))
        assert list(pit.columns) == list(FINANCIAL_COLUMNS)
        assert list(pit["stock_id"]) == ["0050", "2330"]
        assert pit.loc[pit["stock_id"] == "2330", "revenue"].iloc[0] == 101.0
    finally:
        engine.dispose()


def test_pit_check_through_repository(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        bad = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "report_period": "2019Q4",
                    "announcement_date": "2020-02-01",
                    "available_date": "2020-01-15",
                    "source": "test",
                }
            ]
        )
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                upsert_financials(session, bad)
    finally:
        engine.dispose()
