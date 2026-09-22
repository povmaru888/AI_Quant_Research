"""P1-04 acceptance: Price/Financial/Institutional ORM constraints."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.market import Financial, Institutional, Price
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


def _seed_stock(engine) -> None:
    with session_scope(engine) as session:
        session.add(Stock(stock_id="2330", market="TWSE"))


def _valid_price() -> Price:
    return Price(
        trade_date="2020-01-02",
        stock_id="2330",
        open=500.0,
        high=505.0,
        low=495.0,
        close=502.0,
        volume=1000.0,
        traded_value=502000.0,
        source="test",
    )


def test_price_create_and_query(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_stock(engine)
        with session_scope(engine) as session:
            session.add(_valid_price())
        with session_scope(engine) as session:
            row = session.execute(
                select(Price).where(Price.trade_date == "2020-01-02", Price.stock_id == "2330")
            ).scalar_one()
            assert row.close == 502.0
            assert row.source == "test"
    finally:
        engine.dispose()


def test_price_composite_pk(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_stock(engine)
        with session_scope(engine) as session:
            session.add(_valid_price())
        with session_scope(engine) as session:
            session.add(_valid_price())
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            count = session.query(Price).count()
        assert count == 1
    finally:
        engine.dispose()


def test_price_fk_and_check(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_stock(engine)
        with session_scope(engine) as session:
            orphan = _valid_price()
            orphan.stock_id = "9999"
            session.add(orphan)
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            bad_open = _valid_price()
            bad_open.open = 0.0
            session.add(bad_open)
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
    finally:
        engine.dispose()


def test_financial_pit_check(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_stock(engine)
        with session_scope(engine) as session:
            session.add(
                Financial(
                    stock_id="2330",
                    report_period="2019Q4",
                    announcement_date="2020-02-01",
                    available_date="2020-01-15",
                    source="test",
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            session.add(
                Financial(
                    stock_id="2330",
                    report_period="2019Q4",
                    announcement_date="2020-02-01",
                    available_date="2020-02-05",
                    revenue=100.0,
                    net_income=None,
                    source="test",
                )
            )
        with session_scope(engine) as session:
            row = session.execute(
                select(Financial).where(Financial.stock_id == "2330")
            ).scalar_one()
            assert row.revenue == 100.0
            assert row.net_income is None
    finally:
        engine.dispose()


def test_institutional_nullable_and_query(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_stock(engine)
        with session_scope(engine) as session:
            session.add(
                Institutional(
                    trade_date="2020-01-02",
                    stock_id="2330",
                    foreign_net_buy=10.0,
                    source="test",
                )
            )
        with session_scope(engine) as session:
            session.add(Institutional(trade_date="2020-01-02", stock_id="2330", source="test"))
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            row = session.execute(
                select(Institutional).where(Institutional.stock_id == "2330")
            ).scalar_one()
            assert row.trust_net_buy is None
    finally:
        engine.dispose()


def test_mappings_match_migration(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    for model in (Price, Financial, Institutional):
        info = temp_db_conn.execute(f"PRAGMA table_info({model.__tablename__})").fetchall()
        ddl_columns = {col[1] for col in info}
        orm_columns = {col.name for col in model.__table__.columns}
        assert orm_columns == ddl_columns, model.__tablename__
