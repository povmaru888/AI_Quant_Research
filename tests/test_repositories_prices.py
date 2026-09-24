"""P1-07 acceptance: price repository upsert and sorted load."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from repositories.prices import (
    PRICE_HISTORY_COLUMNS,
    load_prices,
    upsert_price_adj,
    upsert_prices,
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
ADJ_MIGRATION_PATH = MIGRATION_PATH.with_name("003_price_adj.py")


def _load_adj_migration():
    spec = importlib.util.spec_from_file_location("price_adj_repository", ADJ_MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adj_migration = _load_adj_migration()


def _engine_for(settings: Settings, path: Path):
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{path}")
    )
    return create_engine_from_settings(db_settings)


def _init_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        migration.upgrade(conn)
        adj_migration.upgrade(conn)
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


def _price_count(engine) -> int:
    with session_scope(engine) as session:
        frame = load_prices(session, ["0050", "2330"], date(2000, 1, 1), date(2030, 1, 1))
    return len(frame)


def test_upsert_and_reload_sorted(
    settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame
) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            assert upsert_prices(session, sample_prices) == len(sample_prices)
        with session_scope(engine) as session:
            loaded = load_prices(session, ["2330", "0050"], date(2000, 1, 1), date(2030, 1, 1))
        assert list(loaded.columns) == list(PRICE_HISTORY_COLUMNS)
        assert len(loaded) == len(sample_prices)
        pairs = list(loaded[["stock_id", "trade_date"]].itertuples(index=False, name=None))
        assert pairs == sorted(pairs)
        first = sample_prices.iloc[0]
        match = loaded[
            (loaded["stock_id"] == first["stock_id"])
            & (loaded["trade_date"] == first["trade_date"])
        ]
        assert match.iloc[0]["close"] == first["close"]
    finally:
        engine.dispose()


def test_upsert_idempotent(
    settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame
) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            upsert_prices(session, sample_prices)
        with session_scope(engine) as session:
            assert upsert_prices(session, sample_prices) == len(sample_prices)
        assert _price_count(engine) == len(sample_prices)

        changed = sample_prices.copy()
        changed.loc[0, "close"] = 9999.0
        with session_scope(engine) as session:
            upsert_prices(session, changed)
        assert _price_count(engine) == len(sample_prices)
        with session_scope(engine) as session:
            loaded = load_prices(session, ["0050"], date(2000, 1, 1), date(2030, 1, 1))
        assert changed.iloc[0]["close"] in set(loaded["close"])
    finally:
        engine.dispose()


def test_load_prices_exposes_adjusted_values_without_raw_fallback(
    settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame
) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            upsert_prices(session, sample_prices)
            first = sample_prices.iloc[0]
            upsert_price_adj(
                session,
                pd.DataFrame(
                    [
                        {
                            "trade_date": first["trade_date"],
                            "stock_id": first["stock_id"],
                            "open_adj": 10.0,
                            "high_adj": 11.0,
                            "low_adj": 9.0,
                            "close_adj": 10.5,
                        }
                    ]
                ),
            )
        with session_scope(engine) as session:
            loaded = load_prices(
                session, [first["stock_id"]], date(2000, 1, 1), date(2030, 1, 1)
            )
        assert loaded.iloc[0]["close_adj"] == 10.5
        assert loaded.iloc[0]["close"] == first["close"]
        assert loaded["close_adj"].isna().sum() == len(loaded) - 1
    finally:
        engine.dispose()


def test_upsert_validation(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                upsert_prices(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown Price columns"):
                upsert_prices(
                    session,
                    pd.DataFrame([{"trade_date": "2020-01-02", "stock_id": "2330", "nope": 1}]),
                )
        with session_scope(engine) as session:
            empty = sample_prices_empty_frame()
            assert upsert_prices(session, empty) == 0
        assert _price_count(engine) == 0
    finally:
        engine.dispose()


def sample_prices_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Series(dtype=str),
            "stock_id": pd.Series(dtype=str),
            "open": pd.Series(dtype=float),
            "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float),
            "close": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
            "traded_value": pd.Series(dtype=float),
            "source": pd.Series(dtype=str),
        }
    )


def test_load_filters(settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            upsert_prices(session, sample_prices)
        with session_scope(engine) as session:
            sub = load_prices(session, ["2330"], date(2020, 1, 2), date(2020, 1, 10))
        assert set(sub["stock_id"]) == {"2330"}
        assert sub["trade_date"].min() >= "2020-01-02"
        assert sub["trade_date"].max() <= "2020-01-10"
        pairs = list(sub[["stock_id", "trade_date"]].itertuples(index=False, name=None))
        assert pairs == sorted(pairs)
        with session_scope(engine) as session:
            empty = load_prices(session, [], date(2020, 1, 2), date(2020, 1, 10))
        assert len(empty) == 0
        assert list(empty.columns) == list(PRICE_HISTORY_COLUMNS)
    finally:
        engine.dispose()


def test_fk_through_repository(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        orphan = pd.DataFrame(
            [
                {
                    "trade_date": "2020-01-02",
                    "stock_id": "9999",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 100.0,
                    "traded_value": 1000.0,
                    "source": "test",
                }
            ]
        )
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                upsert_prices(session, orphan)
    finally:
        engine.dispose()
