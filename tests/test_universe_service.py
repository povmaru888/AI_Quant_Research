"""P2-02 acceptance: tradable universe service."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from services.universe_service import build_universe


def _prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stocks(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _history(
    stock_id: str, close: float, value: float, days: int = 20, end: str = "2019-12-31"
) -> list[dict]:
    end_date = date.fromisoformat(end)
    added = 0
    day = date.fromordinal(end_date.toordinal() - 1)
    prefix = []
    while added < days - 1:
        if day.weekday() < 5:
            prefix.append(
                {
                    "stock_id": stock_id,
                    "trade_date": day.isoformat(),
                    "close": close,
                    "close_adj": close,
                    "traded_value": value,
                }
            )
            added += 1
        day = date.fromordinal(day.toordinal() - 1)
    prefix.reverse()
    prefix.append(
        {
            "stock_id": stock_id,
            "trade_date": end,
            "close": close,
            "close_adj": close,
            "traded_value": value,
        }
    )
    return prefix


def _good_stock(stock_id: str = "2330", **overrides) -> dict:
    row = {
        "stock_id": stock_id,
        "listed_date": "2010-01-01",
        "delisted_date": "",
        "flags": "",
        "market_cap": 600_000_000_000.0,
    }
    row.update(overrides)
    return row


def test_build_universe_pass(settings) -> None:
    prices = _prices(_history("2330", 300.0, 500_000_000.0))
    stocks = _stocks([_good_stock()])
    snapshot = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert snapshot.included_ids == ("2330",)
    assert snapshot.as_of == "2019-12-31"


def test_build_universe_exclusion_reasons(settings) -> None:
    prices = _prices(
        _history("good", 300.0, 500_000_000.0)
        + _history("cheap", 5.0, 500_000_000.0)
        + _history("small", 300.0, 500_000_000.0)
        + _history("thin", 300.0, 1_000_000.0)
        + _history("ky", 300.0, 500_000_000.0)
    )
    stocks = _stocks(
        [
            _good_stock("good"),
            _good_stock("cheap"),
            _good_stock("small", market_cap=1_000_000_000.0),
            _good_stock("thin"),
            _good_stock("ky", flags="KY"),
            _good_stock("gone", delisted_date="2019-06-01"),
            _good_stock("future", listed_date="2020-01-01"),
            _good_stock("noday"),
        ]
    )
    snapshot = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    by_id = {e.stock_id: e for e in snapshot.entries}
    assert by_id["good"].included and by_id["good"].reason == "pass"
    assert by_id["cheap"].reason.startswith("price:")
    assert by_id["small"].reason.startswith("market_cap:")
    assert by_id["thin"].reason.startswith("avg_traded_value:")
    assert by_id["ky"].reason == "flag:KY"
    assert by_id["gone"].reason == "delisted:2019-06-01"
    assert by_id["future"].reason.startswith("not_listed:")
    assert by_id["noday"].reason == "no_price"


def test_build_universe_insufficient_history(settings) -> None:
    prices = _prices(
        [
            {
                "stock_id": "new",
                "trade_date": "2019-12-31",
                "close": 300.0,
                "close_adj": 300.0,
                "traded_value": 500_000_000.0,
            }
        ]
    )
    stocks = _stocks([_good_stock("new", listed_date="2019-12-01")])
    snapshot = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert snapshot.entries[0].reason.startswith("avg_traded_value:insufficient_history")


def test_build_universe_uses_each_stocks_last_20_bars(settings) -> None:
    rows = _history("steady", 300.0, 500_000_000.0, days=21) + _history(
        "rising", 300.0, 500_000_000.0, days=20
    )
    rows[0]["traded_value"] = 0.0  # Outside steady's 20-bar window.
    for row in rows:
        if row["stock_id"] == "rising":
            row["traded_value"] = 1_000_000.0
    prices = _prices(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    stocks = _stocks([_good_stock("steady"), _good_stock("rising")])
    result = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    by_id = {entry.stock_id: entry for entry in result.entries}
    assert by_id["steady"].included
    assert by_id["rising"].reason.startswith("avg_traded_value:")


def test_build_universe_missing_market_cap_fails_loud(settings) -> None:
    prices = _prices(_history("2330", 300.0, 500_000_000.0))
    stocks = _stocks(
        [
            {
                "stock_id": "2330",
                "listed_date": "2010-01-01",
                "delisted_date": "",
                "flags": "",
            }
        ]
    )
    snapshot = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert snapshot.included_ids == ()
    assert snapshot.entries[0].reason == "market_cap:missing"


def test_build_universe_historical_view_no_survivorship_bias(settings) -> None:
    prices = _prices(_history("gone", 300.0, 500_000_000.0, end="2019-03-29"))
    stocks = _stocks([_good_stock("gone", delisted_date="2019-06-01")])
    before = build_universe(prices, stocks, date(2019, 3, 29), settings, "run-001")
    assert before.included_ids == ("gone",)
    after = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert after.entries[0].reason == "delisted:2019-06-01"


def test_build_universe_rejects_bad_inputs(settings) -> None:
    prices = _prices(_history("2330", 300.0, 500_000_000.0))
    stocks = _stocks([_good_stock()])
    with pytest.raises(ValueError, match="run_id"):
        build_universe(prices, stocks, date(2019, 12, 31), settings, " ")
    with pytest.raises(ValueError, match="as_of"):
        build_universe(prices, stocks, "2019-12-31", settings, "run-001")
    with pytest.raises(ValueError, match="missing columns"):
        build_universe(prices.drop(columns=["close_adj"]), stocks, date(2019, 12, 31), settings, "r")


def test_build_universe_uses_nominal_tradable_price_and_requires_adjusted_close(settings) -> None:
    prices = _prices(_history("2330", 300.0, 500_000_000.0))
    stocks = _stocks([_good_stock()])
    prices.loc[prices["trade_date"] == "2019-12-31", "close_adj"] = 5.0
    result = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert result.entries[0].included

    prices.loc[prices["trade_date"] == "2019-12-31", "close_adj"] = 300.0
    prices.loc[prices["trade_date"] == "2019-12-31", "close"] = 5.0
    result = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert result.entries[0].reason == "price:5.00"

    prices.loc[prices["trade_date"] == "2019-12-31", "close_adj"] = pd.NA
    result = build_universe(prices, stocks, date(2019, 12, 31), settings, "run-001")
    assert result.entries[0].reason == "no_price"
