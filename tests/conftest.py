"""P0-07: shared test fixtures (temp DB, settings, trading days, prices)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from settings import Settings, load_settings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FIXTURE_STOCK_IDS = ("2330", "0050")
FIXTURE_BASE_PRICES = {"2330": 500.0, "0050": 140.0}
TRADING_DAY_COUNT = 25


@pytest.fixture()
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
def temp_db_conn(temp_db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(temp_db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def settings() -> Settings:
    return load_settings(CONFIG_PATH, env={})


@pytest.fixture()
def trading_days() -> list[str]:
    days: list[str] = []
    current = date(2020, 1, 2)
    while len(days) < TRADING_DAY_COUNT:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


@pytest.fixture()
def sample_prices(trading_days: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for stock_id in FIXTURE_STOCK_IDS:
        base = FIXTURE_BASE_PRICES[stock_id]
        for i, trade_date in enumerate(trading_days):
            close = round(base * (1 + 0.002 * i), 2)
            volume = 1_000_000 + i * 10_000
            rows.append(
                {
                    "trade_date": trade_date,
                    "stock_id": stock_id,
                    "open": round(close * 0.999, 2),
                    "high": round(close * 1.005, 2),
                    "low": round(close * 0.995, 2),
                    "close": close,
                    "volume": volume,
                    "traded_value": round(close * volume, 2),
                    "source": "test",
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["stock_id", "trade_date"], ignore_index=True)
