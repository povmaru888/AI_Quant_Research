"""P1-03 acceptance: Stock ORM create/query/conflict/serialization."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.security import Stock
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


def test_create_and_query(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            session.add(
                Stock(
                    stock_id="2330",
                    stock_name="台積電",
                    market="TWSE",
                    listed_date="1994-09-05",
                    industry="半導體",
                )
            )
        with session_scope(engine) as session:
            row = session.execute(select(Stock).where(Stock.stock_id == "2330")).scalar_one()
            assert row.stock_name == "台積電"
            assert row.market == "TWSE"
            assert row.listed_date == "1994-09-05"
            assert row.delisted_date is None
            assert row.created_at is not None
    finally:
        engine.dispose()


def test_pk_conflict(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            session.add(Stock(stock_id="2330", market="TWSE"))
        with session_scope(engine) as session:
            session.add(Stock(stock_id="2330", market="TWSE"))
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            count = session.query(Stock).count()
        assert count == 1
    finally:
        engine.dispose()


def test_delisted_nullable_and_serializable(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            session.add(
                Stock(
                    stock_id="0050",
                    market="TWSE",
                    listed_date="2003-06-30",
                    delisted_date=None,
                )
            )
        with session_scope(engine) as session:
            row = session.execute(select(Stock).where(Stock.stock_id == "0050")).scalar_one()
            payload = {
                "stock_id": row.stock_id,
                "market": row.market,
                "listed_date": row.listed_date,
                "delisted_date": row.delisted_date,
            }
            assert json.loads(json.dumps(payload))["stock_id"] == "0050"
    finally:
        engine.dispose()


def test_market_not_null(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            session.add(Stock(stock_id="2330", market=None))  # type: ignore[arg-type]
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
    finally:
        engine.dispose()


def test_mapping_matches_migration(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    info = temp_db_conn.execute("PRAGMA table_info(stocks)").fetchall()
    ddl_columns = {col[1] for col in info}
    orm_columns = {col.name for col in Stock.__table__.columns}
    assert Stock.__tablename__ == "stocks"
    assert orm_columns == ddl_columns
