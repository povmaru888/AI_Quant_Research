"""P1-10 acceptance: trading repository save semantics."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.research import Order, PipelineRun
from repositories.stocks import upsert_stocks
from repositories.trading import save_orders, save_positions, save_signals
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
    with session_scope(engine) as session:
        upsert_stocks(
            session,
            pd.DataFrame(
                [
                    {"stock_id": "2330", "market": "TWSE"},
                    {"stock_id": "0050", "market": "TWSE"},
                ]
            ),
        )
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
    return engine


def _signal_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2020-01-31",
                "stock_id": "2330",
                "run_id": "run-001",
                "signal": "BUY",
                "rank": 1,
                "target_weight": 0.1,
            },
            {
                "signal_date": "2020-01-31",
                "stock_id": "0050",
                "run_id": "run-001",
                "signal": "NONE",
                "rank": 40,
                "target_weight": 0.0,
            },
        ]
    )


def _order_row(order_id: str, stock_id: str) -> dict:
    return {
        "order_id": order_id,
        "run_id": "run-001",
        "signal_date": "2020-01-31",
        "execution_date": "2020-02-03",
        "stock_id": stock_id,
        "side": "BUY",
        "quantity": 10.0,
        "open_price": 500.0,
        "executed_price": 500.5,
        "notional": 5000.0,
        "broker_fee": 7.125,
        "transaction_tax": 0.0,
        "slippage_cost": 5.0,
        "total_cost": 12.125,
    }


def _table_count(engine, table: str) -> int:
    with session_scope(engine) as session:
        return session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def test_save_signals_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = _signal_rows()
        with session_scope(engine) as session:
            assert save_signals(session, rows) == 2
        with session_scope(engine) as session:
            assert save_signals(session, rows) == 2
        assert _table_count(engine, "signals") == 2

        changed = rows.copy()
        changed.loc[0, "target_weight"] = 0.2
        with session_scope(engine) as session:
            save_signals(session, changed)
        assert _table_count(engine, "signals") == 2

        bad = rows.copy()
        bad.loc[0, "signal"] = "SIDEWAYS"
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                save_signals(session, bad)
    finally:
        engine.dispose()


def test_save_positions_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = pd.DataFrame(
            [
                {
                    "position_date": "2020-02-03",
                    "stock_id": "2330",
                    "shares": 10.0,
                    "weight": 0.1,
                    "market_value": 5000.0,
                }
            ]
        )
        with session_scope(engine) as session:
            assert save_positions(session, rows) == 1
        with session_scope(engine) as session:
            assert save_positions(session, rows) == 1
        assert _table_count(engine, "positions") == 1

        for bad_value, column in ((-1.0, "shares"), (-0.5, "weight")):
            bad = rows.copy()
            bad.loc[0, column] = bad_value
            with session_scope(engine) as session:
                with pytest.raises(IntegrityError):
                    save_positions(session, bad)
    finally:
        engine.dispose()


def test_save_orders_idempotent_and_shrinking(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        batch = pd.DataFrame([_order_row("o-1", "2330"), _order_row("o-2", "0050")])
        with session_scope(engine) as session:
            assert save_orders(session, batch) == 2
        with session_scope(engine) as session:
            assert save_orders(session, batch) == 2
        assert _table_count(engine, "orders") == 2
        with session_scope(engine) as session:
            row = session.execute(select(Order).where(Order.order_id == "o-1")).scalar_one()
            assert row.total_cost == 12.125
            assert row.execution_date == "2020-02-03"

        shrunk = pd.DataFrame([_order_row("o-1", "2330")])
        with session_scope(engine) as session:
            assert save_orders(session, shrunk) == 1
        assert _table_count(engine, "orders") == 1

        bad_timing = pd.DataFrame([_order_row("o-3", "2330")])
        bad_timing.loc[0, "execution_date"] = "2020-01-31"
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                save_orders(session, bad_timing)

        blank_id = pd.DataFrame([_order_row("  ", "2330")])
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="non-empty 'order_id'"):
                save_orders(session, blank_id)

        mixed = pd.DataFrame([_order_row("o-1", "2330"), _order_row("o-2", "0050")])
        mixed.loc[1, "signal_date"] = "2020-02-29"
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="single run_id and signal_date"):
                save_orders(session, mixed)
    finally:
        engine.dispose()


def test_validation(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                save_signals(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                save_positions(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                save_orders(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown Order columns"):
                save_orders(
                    session,
                    pd.DataFrame([_order_row("o-9", "2330") | {"nope": 1}]),
                )
        with session_scope(engine) as session:
            assert (
                save_signals(
                    session,
                    pd.DataFrame(
                        {
                            "signal_date": pd.Series(dtype=str),
                            "stock_id": pd.Series(dtype=str),
                        }
                    ),
                )
                == 0
            )
            assert (
                save_positions(
                    session,
                    pd.DataFrame(
                        {
                            "position_date": pd.Series(dtype=str),
                            "stock_id": pd.Series(dtype=str),
                        }
                    ),
                )
                == 0
            )
            assert save_orders(session, pd.DataFrame({"order_id": pd.Series(dtype=str)})) == 0
    finally:
        engine.dispose()
