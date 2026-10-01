"""Bounded OOS market reads preserve the legacy adjusted-price semantics."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from database import create_engine_from_settings
from repositories import stocks as stocks_repo
from runtime.db_store import DbStore
from runtime.oos_data import PreparedOOSData, month_end_signal_dates


def _migration(name: str):
    path = Path(__file__).resolve().parents[1] / f"database/migrations/versions/{name}.py"
    spec = importlib.util.spec_from_file_location(f"oos_data_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_oos_data_matches_adjusted_returns_and_bounds_symbols(
    settings, temp_db_path: Path
) -> None:
    connection = sqlite3.connect(temp_db_path)
    try:
        _migration("001_initial_schema").upgrade(connection)
        _migration("003_price_adj").upgrade(connection)
    finally:
        connection.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    store = DbStore(engine, db_settings)
    with store._scope() as session:  # noqa: SLF001
        stocks_repo.upsert_stocks(
            session,
            pd.DataFrame(
                {
                    "stock_id": ["A", "B", "TAIEX"],
                    "stock_name": ["A", "B", "TAIEX"],
                    "market": ["TWSE", "TWSE", "INDEX"],
                    "listed_date": ["2019-01-01"] * 3,
                }
            ),
        )
    rows = []
    for stock_id, closes in {"A": [10.0, np.nan, 12.0], "B": [20.0, 21.0, 22.0]}.items():
        for day, close in zip(("2020-01-30", "2020-01-31", "2020-02-03"), closes, strict=True):
            raw = 10.0 if not np.isfinite(close) else close
            rows.append(
                {"trade_date": day, "stock_id": stock_id, "open": raw, "high": raw,
                 "low": raw, "close": raw, "open_adj": raw, "high_adj": raw,
                 "low_adj": raw, "close_adj": close, "volume": 1.0,
                 "traded_value": raw, "source": "test"}
            )
    for day, close in zip(("2020-01-30", "2020-01-31", "2020-02-03"), (100.0, 101.0, 102.0), strict=True):
        rows.append(
            {"trade_date": day, "stock_id": "TAIEX", "open": close, "high": close,
             "low": close, "close": close, "open_adj": None, "high_adj": None,
             "low_adj": None, "close_adj": None, "volume": 1.0,
             "traded_value": close, "source": "test"}
        )
    store.upsert_prices(pd.DataFrame(rows))

    assert month_end_signal_dates(engine, ["2020-01"]) == ["2020-01-31"]
    prepared = PreparedOOSData.load(engine, ["A"], ["2020-01-31"])
    assert set(prepared.returns["stock_id"]) <= {"A"}
    assert prepared.returns.empty  # missing adjusted bar must not bridge the gap
    assert prepared.next_open("2020-01-31")["trade_date"].unique().tolist() == ["2020-02-03"]
    assert set(prepared.quotes["stock_id"]) == {"A"}
    engine.dispose()
