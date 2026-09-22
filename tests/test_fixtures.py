"""P0-07 acceptance: fixtures are independent and well-formed."""

from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from settings import Settings


def test_temp_databases_are_independent(temp_db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    temp_db_conn.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY, v TEXT)")
    temp_db_conn.execute("INSERT INTO marker (v) VALUES ('a')")
    temp_db_conn.commit()
    rows = temp_db_conn.execute("SELECT v FROM marker").fetchall()
    assert rows == [("a",)]

    other_path = tmp_path / "other.db"
    assert not other_path.exists()
    other = sqlite3.connect(str(other_path))
    try:
        tables = other.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        other.close()
    assert tables == []


def test_trading_days_cover_months_and_count(trading_days: list[str]) -> None:
    assert len(trading_days) >= 20
    assert trading_days == sorted(trading_days)
    assert len(set(trading_days)) == len(trading_days)
    for day in trading_days:
        parsed = date.fromisoformat(day)
        assert parsed.weekday() < 5
    months = {day[:7] for day in trading_days}
    assert len(months) >= 2


def test_sample_prices_valid(sample_prices: pd.DataFrame, trading_days: list[str]) -> None:
    assert set(sample_prices["trade_date"]) <= set(trading_days)
    assert sample_prices["stock_id"].map(lambda v: isinstance(v, str)).all()
    assert not sample_prices.duplicated(["trade_date", "stock_id"]).any()
    ordered = sample_prices.sort_values(["stock_id", "trade_date"], ignore_index=True)
    pd.testing.assert_frame_equal(sample_prices, ordered)
    for column in ("open", "high", "low", "close"):
        assert (sample_prices[column] > 0).all(), column
    assert (sample_prices["high"] >= sample_prices["low"]).all()
    assert (sample_prices["volume"] >= 0).all()
    assert (sample_prices["traded_value"] >= 0).all()


def test_settings_fixture(settings: Settings) -> None:
    assert settings.portfolio.top_n == 15
    assert settings.validation.purge_trading_days == 20
    with pytest.raises(FrozenInstanceError):
        settings.portfolio.top_n = 99  # type: ignore[misc]
