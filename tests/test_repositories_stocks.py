"""P1-06 acceptance: stock repository upsert and active query."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from database import create_engine_from_settings, session_scope
from models.security import Stock
from repositories.stocks import get_active_stocks, upsert_stocks
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


def _stock_count(engine) -> int:
    with session_scope(engine) as session:
        return session.query(Stock).count()


def test_upsert_insert_and_update(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        first = pd.DataFrame(
            [
                {"stock_id": "2330", "stock_name": "台積電", "market": "TWSE"},
                {"stock_id": "0050", "stock_name": "元大台灣50", "market": "TWSE"},
            ]
        )
        with session_scope(engine) as session:
            assert upsert_stocks(session, first) == 2
        assert _stock_count(engine) == 2
        with session_scope(engine) as session:
            created = session.execute(
                select(Stock.created_at).where(Stock.stock_id == "2330")
            ).scalar_one()

        second = pd.DataFrame(
            [
                {"stock_id": "2330", "stock_name": "TSMC", "market": "TWSE"},
                {"stock_id": "0050", "stock_name": "元大台灣50", "market": "TWSE"},
            ]
        )
        with session_scope(engine) as session:
            assert upsert_stocks(session, second) == 2
        assert _stock_count(engine) == 2
        with session_scope(engine) as session:
            row = session.execute(select(Stock).where(Stock.stock_id == "2330")).scalar_one()
            assert row.stock_name == "TSMC"
            assert row.created_at == created
    finally:
        engine.dispose()


def test_upsert_empty(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        empty = pd.DataFrame({"stock_id": pd.Series(dtype=str)})
        with session_scope(engine) as session:
            assert upsert_stocks(session, empty) == 0
        assert _stock_count(engine) == 0
    finally:
        engine.dispose()


def test_upsert_validation(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="stock_id"):
                upsert_stocks(session, pd.DataFrame([{"market": "TWSE"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown Stock columns"):
                upsert_stocks(
                    session,
                    pd.DataFrame([{"stock_id": "2330", "nope": 1}]),
                )
        assert _stock_count(engine) == 0
    finally:
        engine.dispose()


def test_nan_to_null(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        rows = pd.DataFrame([{"stock_id": "2330", "market": "TWSE", "industry": float("nan")}])
        with session_scope(engine) as session:
            assert upsert_stocks(session, rows) == 1
        with session_scope(engine) as session:
            row = session.execute(select(Stock).where(Stock.stock_id == "2330")).scalar_one()
            assert row.industry is None
    finally:
        engine.dispose()


def test_get_active_stocks(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        rows = pd.DataFrame(
            [
                {"stock_id": "2330", "market": "TWSE"},
                {
                    "stock_id": "2317",
                    "market": "TWSE",
                    "listed_date": "1990-01-01",
                },
                {
                    "stock_id": "1101",
                    "market": "TWSE",
                    "listed_date": "1990-01-01",
                    "delisted_date": "2019-12-31",
                },
                {
                    "stock_id": "1301",
                    "market": "TWSE",
                    "delisted_date": "2020-06-30",
                },
                {
                    "stock_id": "1402",
                    "market": "TWSE",
                    "listed_date": "2020-06-01",
                },
            ]
        )
        with session_scope(engine) as session:
            assert upsert_stocks(session, rows) == 5
        with session_scope(engine) as session:
            active = get_active_stocks(session, date(2020, 1, 15))
        assert list(active["stock_id"]) == ["1301", "2317", "2330"]
        assert list(active.columns) == [
            "stock_id",
            "stock_name",
            "market",
            "listed_date",
            "delisted_date",
            "industry",
        ]
    finally:
        engine.dispose()
