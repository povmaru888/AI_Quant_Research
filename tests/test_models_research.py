"""P1-05 acceptance: research/trading ORM constraints and traceability."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.research import (
    Feature,
    Order,
    PipelineRun,
    PortfolioDaily,
    Position,
    Prediction,
    Signal,
)
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


def _seed_parents(engine) -> None:
    with session_scope(engine) as session:
        session.add(Stock(stock_id="2330", market="TWSE"))
        session.add(
            PipelineRun(
                run_id="run-001",
                run_time="2020-01-01",
                data_end_date="2019-12-31",
                feature_version="factor_v1",
                parameter_version="p1",
                status="started",
            )
        )


def _valid_order() -> Order:
    return Order(
        order_id="o-1",
        run_id="run-001",
        signal_date="2020-01-31",
        execution_date="2020-02-03",
        stock_id="2330",
        side="BUY",
        quantity=10.0,
        open_price=500.0,
        executed_price=500.5,
        notional=5000.0,
        broker_fee=7.125,
        transaction_tax=0.0,
        slippage_cost=5.0,
        total_cost=12.125,
    )


def test_run_status(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_parents(engine)
        with session_scope(engine) as session:
            session.add(
                PipelineRun(
                    run_id="run-002",
                    run_time="2020-01-01",
                    data_end_date="2019-12-31",
                    feature_version="factor_v1",
                    parameter_version="p1",
                    status="failed",
                    error_message="covariance missing",
                )
            )
        with session_scope(engine) as session:
            session.add(
                PipelineRun(
                    run_id="run-003",
                    run_time="2020-01-01",
                    data_end_date="2019-12-31",
                    feature_version="factor_v1",
                    parameter_version="p1",
                    status="done",
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            row = session.execute(
                select(PipelineRun).where(PipelineRun.run_id == "run-002")
            ).scalar_one()
            assert row.error_message == "covariance missing"
    finally:
        engine.dispose()


def test_prediction_requires_run(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_parents(engine)
        with session_scope(engine) as session:
            session.add(
                Prediction(
                    prediction_date="2020-01-31",
                    stock_id="2330",
                    run_id="run-999",
                    model_version="xgb_v1",
                    prediction_probability=0.8,
                    rank=1,
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            session.add(
                Prediction(
                    prediction_date="2020-01-31",
                    stock_id="2330",
                    run_id="run-001",
                    model_version="xgb_v1",
                    prediction_probability=0.8,
                    rank=1,
                )
            )
        with session_scope(engine) as session:
            row = session.execute(
                select(Prediction, PipelineRun)
                .join(PipelineRun, Prediction.run_id == PipelineRun.run_id)
                .where(Prediction.stock_id == "2330")
            ).one()
            assert row[0].rank == 1
            assert row[1].status == "started"
    finally:
        engine.dispose()


def test_order_timing(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_parents(engine)
        with session_scope(engine) as session:
            bad = _valid_order()
            bad.execution_date = "2020-01-31"
            session.add(bad)
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            session.add(_valid_order())
        with session_scope(engine) as session:
            row = session.execute(select(Order).where(Order.order_id == "o-1")).scalar_one()
            assert row.total_cost == 12.125
            assert row.run_id == "run-001"
    finally:
        engine.dispose()


def test_feature_pk_and_missing_flag(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_parents(engine)
        with session_scope(engine) as session:
            session.add(
                Feature(
                    rebalance_date="2020-01-31",
                    stock_id="2330",
                    feature_version="factor_v1",
                    momentum_20d=0.05,
                    volatility_60d=0.2,
                )
            )
        with session_scope(engine) as session:
            session.add(
                Feature(
                    rebalance_date="2020-01-31",
                    stock_id="2330",
                    feature_version="factor_v1",
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            row = session.execute(select(Feature).where(Feature.stock_id == "2330")).scalar_one()
            session.refresh(row)
            assert row.missing_flag == 0
            assert row.momentum_20d == 0.05
    finally:
        engine.dispose()


def test_signal_position_daily(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        _seed_parents(engine)
        with session_scope(engine) as session:
            session.add(
                Signal(
                    signal_date="2020-01-31",
                    stock_id="2330",
                    run_id="run-001",
                    signal="SIDEWAYS",
                    rank=1,
                    target_weight=0.1,
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            session.add(
                Position(
                    position_date="2020-02-03",
                    stock_id="2330",
                    shares=-1.0,
                    weight=0.1,
                    market_value=5000.0,
                )
            )
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
        with session_scope(engine) as session:
            session.add(
                Signal(
                    signal_date="2020-01-31",
                    stock_id="2330",
                    run_id="run-001",
                    signal="BUY",
                    rank=1,
                    target_weight=0.1,
                )
            )
            session.add(
                Position(
                    position_date="2020-02-03",
                    stock_id="2330",
                    shares=10.0,
                    weight=0.1,
                    market_value=5000.0,
                )
            )
            session.add(
                PortfolioDaily(
                    trade_date="2020-02-03",
                    run_id="run-001",
                    nav=1000000.0,
                    equity_exposure=0.9,
                    market_regime="bull",
                )
            )
        with session_scope(engine) as session:
            daily = session.execute(
                select(PortfolioDaily).where(PortfolioDaily.trade_date == "2020-02-03")
            ).scalar_one()
            assert daily.nav == 1000000.0
    finally:
        engine.dispose()


def test_mappings_match_migration(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    models = (PipelineRun, Feature, Prediction, Signal, Position, Order, PortfolioDaily)
    for model in models:
        info = temp_db_conn.execute(f"PRAGMA table_info({model.__tablename__})").fetchall()
        ddl_columns = {col[1] for col in info}
        orm_columns = {col.name for col in model.__table__.columns}
        assert orm_columns == ddl_columns, model.__tablename__
