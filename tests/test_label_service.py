"""P2-06 acceptance: label service."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from services.label_service import build_labels

AS_OF = date(2019, 12, 31)


def _prices(spec: dict[str, float], start: str = "2019-10-01", periods: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods).strftime("%Y-%m-%d")
    frames = []
    for stock_id, drift in spec.items():
        closes = 100.0 + drift * pd.Series(range(periods))
        frames.append(
            pd.DataFrame(
                {
                    "stock_id": stock_id,
                    "trade_date": dates,
                    "close": closes,
                    "close_adj": closes,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_labels_quantile_cut(settings) -> None:
    prices = _prices({f"S{i:02d}": float(i) for i in range(10)})
    labels = build_labels(prices, [f"S{i:02d}" for i in range(10)], AS_OF, settings)
    assert labels.sum() == 2
    assert labels["S09"] == 1 and labels["S08"] == 1
    assert labels["S07"] == 0
    assert labels.name == "2019-12-31"


def test_build_labels_uses_trading_days_not_calendar(settings) -> None:
    # 2019-12-31 is a Tuesday; +20 calendar days lands on a holiday-adjacent
    # Monday, but the label must use the 20th traded row instead.
    prices = _prices({"A": 1.0, "B": 0.0})
    labels = build_labels(prices, ["A", "B"], AS_OF, settings)
    assert labels["A"] == 1 and labels["B"] == 0
    as_of_pos = prices.loc[
        (prices["stock_id"] == "A") & (prices["trade_date"] == "2019-12-31")
    ].index[0]
    t20 = prices.loc[as_of_pos + 20, "trade_date"]
    assert t20 != "2020-01-20"


def test_build_labels_missing_horizon_yields_empty(settings) -> None:
    prices = _prices({"A": 1.0, "B": 0.0}, periods=70)
    labels = build_labels(prices, ["A", "B"], AS_OF, settings)
    assert labels.empty


def test_build_labels_ignores_non_universe_stocks(settings) -> None:
    prices = _prices({"A": 1.0, "B": 0.0, "Z": 999.0})
    labels = build_labels(prices, ["A", "B"], AS_OF, settings)
    assert set(labels.index) == {"A", "B"}


def test_build_labels_sorts_interleaved_stock_histories(settings) -> None:
    prices = _prices({"A": 1.0, "B": 0.0, "Z": 999.0})
    expected = build_labels(prices, ["B", "A"], AS_OF, settings)
    interleaved = prices.sample(frac=1, random_state=7).reset_index(drop=True)
    actual = build_labels(interleaved, ["B", "A"], AS_OF, settings)
    pd.testing.assert_series_equal(actual, expected)


def test_build_labels_rejects_bad_inputs(settings) -> None:
    prices = _prices({"A": 1.0})
    with pytest.raises(ValueError, match="stock_ids"):
        build_labels(prices, [], AS_OF, settings)
    with pytest.raises(ValueError, match="as_of"):
        build_labels(prices, ["A"], "2019-12-31", settings)
    with pytest.raises(ValueError, match="missing columns"):
        build_labels(prices.drop(columns=["close_adj"]), ["A"], AS_OF, settings)


def test_build_labels_uses_adjusted_prices_and_rejects_missing_adj(settings) -> None:
    prices = _prices({"A": 0.0, "B": 0.0})
    a = prices["stock_id"] == "A"
    b = prices["stock_id"] == "B"
    prices.loc[a, "close"] = 9999.0
    prices.loc[b, "close"] = 1.0
    dates = prices.loc[a, "trade_date"].to_numpy()
    as_of_idx = list(dates).index(AS_OF.isoformat())
    a_rows = prices.index[a]
    prices.loc[a_rows[as_of_idx + settings.label.horizon_trading_days], "close_adj"] = 120.0
    labels = build_labels(prices, ["A", "B"], AS_OF, settings)
    assert labels["A"] == 1 and labels["B"] == 0

    prices.loc[a_rows[as_of_idx], "close_adj"] = pd.NA
    assert build_labels(prices, ["A", "B"], AS_OF, settings).empty
