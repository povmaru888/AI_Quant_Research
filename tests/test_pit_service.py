"""P2-03 acceptance: point-in-time snapshot service."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from contracts import UniverseEntry, UniverseSnapshot
from services.pit_service import build_pit_snapshot


def _universe(as_of: str = "2019-12-31") -> UniverseSnapshot:
    return UniverseSnapshot(
        run_id="run-001",
        as_of=as_of,
        entries=[
            UniverseEntry("2330", True, "pass"),
            UniverseEntry("2317", True, "pass"),
            UniverseEntry("9999", False, "price:5.00"),
        ],
    )


def _financials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "report_period": "2019Q3",
                "announcement_date": "2019-11-10",
                "available_date": "2019-11-11",
                "roe": 0.20,
            },
            # Newer period but announced AFTER as_of: must NOT leak in.
            {
                "stock_id": "2330",
                "report_period": "2019Q4",
                "announcement_date": "2020-01-15",
                "available_date": "2020-01-16",
                "roe": 0.99,
            },
            {
                "stock_id": "2317",
                "report_period": "2019Q3",
                "announcement_date": "2019-11-12",
                "available_date": "2019-11-13",
                "roe": 0.10,
            },
        ]
    )


def _institutional() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stock_id": "2330", "trade_date": "2019-12-30", "foreign_net_buy": 100.0},
            {"stock_id": "2330", "trade_date": "2019-12-31", "foreign_net_buy": 200.0},
            # Dated after as_of: must NOT leak in.
            {"stock_id": "2330", "trade_date": "2020-01-02", "foreign_net_buy": 999.0},
            {"stock_id": "2317", "trade_date": "2019-12-31", "foreign_net_buy": -50.0},
        ]
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stock_id": "2330", "trade_date": "2019-12-31", "close": 900.0, "close_adj": 300.0},
            {"stock_id": "2317", "trade_date": "2019-12-31", "close": 800.0, "close_adj": 80.0},
        ]
    )


def test_build_pit_snapshot_shape() -> None:
    snapshot = build_pit_snapshot(
        _universe(), date(2019, 12, 31), _financials(), _institutional(), _prices()
    )
    assert list(snapshot["stock_id"]) == ["2330", "2317"]
    assert set(snapshot.columns) >= {
        "stock_id",
        "report_period",
        "roe",
        "foreign_net_buy",
        "as_of_close",
        "as_of_close_adj",
    }


def test_build_pit_snapshot_delayed_announcement_stays_out() -> None:
    snapshot = build_pit_snapshot(
        _universe(), date(2019, 12, 31), _financials(), _institutional(), _prices()
    )
    row = snapshot.loc[snapshot["stock_id"] == "2330"].iloc[0]
    assert row["report_period"] == "2019Q3"
    assert row["roe"] == pytest.approx(0.20)
    assert row["foreign_net_buy"] == pytest.approx(200.0)
    assert row["as_of_close"] == pytest.approx(900.0)
    assert row["as_of_close_adj"] == pytest.approx(300.0)


def test_build_pit_snapshot_missing_financials_kept_as_nan() -> None:
    financials = _financials().loc[_financials()["stock_id"] == "2317"]
    snapshot = build_pit_snapshot(
        _universe(), date(2019, 12, 31), financials, _institutional(), _prices()
    )
    row = snapshot.loc[snapshot["stock_id"] == "2330"].iloc[0]
    assert pd.isna(row["roe"])
    assert row["foreign_net_buy"] == pytest.approx(200.0)


def test_build_pit_snapshot_missing_adjusted_close_stays_missing() -> None:
    prices = _prices()
    prices.loc[prices["stock_id"] == "2330", "close_adj"] = pd.NA
    snapshot = build_pit_snapshot(
        _universe(), date(2019, 12, 31), _financials(), _institutional(), prices
    )
    row = snapshot.loc[snapshot["stock_id"] == "2330"].iloc[0]
    assert row["as_of_close"] == pytest.approx(900.0)
    assert pd.isna(row["as_of_close_adj"])


def test_build_pit_snapshot_ignores_other_stocks() -> None:
    # Eligible data exists, but none belongs to the selected universe.
    financials = _financials().assign(stock_id="9999")
    institutional = _institutional().assign(stock_id="9999")
    snapshot = build_pit_snapshot(
        _universe(), date(2019, 12, 31), financials, institutional, _prices()
    )
    assert list(snapshot["stock_id"]) == ["2330", "2317"]
    assert snapshot["roe"].isna().all()
    assert snapshot["foreign_net_buy"].isna().all()
    assert "available_date" not in snapshot.columns
    assert "trade_date" not in snapshot.columns


def test_build_pit_snapshot_empty_universe() -> None:
    empty = UniverseSnapshot(run_id="run-001", as_of="2019-12-31", entries=[])
    snapshot = build_pit_snapshot(
        empty, date(2019, 12, 31), _financials(), _institutional(), _prices()
    )
    assert snapshot.empty
    assert "stock_id" in snapshot.columns


def test_build_pit_snapshot_rejects_mismatched_universe() -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_pit_snapshot(
            _universe("2019-11-29"), date(2019, 12, 31), _financials(), _institutional(), _prices()
        )
    with pytest.raises(ValueError, match="missing columns"):
        build_pit_snapshot(
            _universe(),
            date(2019, 12, 31),
            _financials().drop(columns=["available_date"]),
            _institutional(),
            _prices(),
        )
