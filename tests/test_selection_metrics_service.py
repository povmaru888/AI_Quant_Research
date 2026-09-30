"""Model selection diagnostics use the label's adjusted-price horizon."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from services.selection_metrics_service import (
    build_selection_metrics,
    validate_selection_metrics,
)


def _prices_and_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = []
    day = date(2020, 1, 2)
    while len(dates) < 23:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)

    price_rows = []
    prediction_rows = []
    for index in range(31):
        stock_id = f"S{index:02d}"
        stock_dates = dates.copy()
        if index == 0:
            stock_dates.pop(5)  # Each stock advances on its own price rows.
        expected_return = (30 - index) / 1000
        for position, trade_date in enumerate(stock_dates):
            adjusted = 100.0
            if position == 20:
                adjusted *= 1 + expected_return
            elif position == 21:
                adjusted = 9_000.0  # Must not replace the t+20 endpoint.
            if index == 1 and position == 10:
                adjusted = None  # Keep this missing bar in the 20-row offset.
            price_rows.append(
                {
                    "stock_id": stock_id,
                    "trade_date": trade_date,
                    "close_adj": adjusted,
                    "close": 50_000.0 + position,  # Raw close must be ignored.
                }
            )
        prediction_rows.append(
            {
                "prediction_date": dates[0],
                "stock_id": stock_id,
                "prediction_probability": 1.0 - index / 100,
                "rank": index + 1,
            }
        )

    # Has a next price row, but no exact signal-date price, so it is ineligible.
    for position, trade_date in enumerate(dates[1:]):
        price_rows.append(
            {
                "stock_id": "NO_EXACT_T",
                "trade_date": trade_date,
                "close_adj": 100.0 + position,
                "close": 500.0,
            }
        )
    prediction_rows.append(
        {
            "prediction_date": dates[0],
            "stock_id": "NO_EXACT_T",
            "prediction_probability": 0.01,
            "rank": 32,
        }
    )
    return pd.DataFrame(prediction_rows), pd.DataFrame(price_rows)


def test_selection_metrics_match_simple_adjusted_t_plus_20_definition() -> None:
    predictions, prices = _prices_and_predictions()

    payload = build_selection_metrics(predictions, prices)

    row = payload["monthly"][0]
    assert row["eligible_count"] == 31
    assert row["continuous_rank_ic"] == pytest.approx(1.0)
    assert row["top15_actual_return"] == pytest.approx(0.023)
    assert row["universe_actual_return"] == pytest.approx(0.015)
    assert row["top15_excess_return"] == pytest.approx(0.008)
    assert row["bottom15_actual_return"] == pytest.approx(0.007)
    assert row["top_bottom_spread"] == pytest.approx(0.016)
    assert payload["summary"]["ic_positive_month_ratio"] == 1.0
    assert validate_selection_metrics(payload) == []


def test_t_plus_21_and_raw_close_do_not_change_diagnostics() -> None:
    predictions, prices = _prices_and_predictions()
    original = build_selection_metrics(predictions, prices)
    changed = prices.copy()
    changed["close"] *= 7
    # The last adjusted close is beyond t+20 for every stock with a full window.
    for stock_id, group in changed.groupby("stock_id"):
        ordered = group.sort_values("trade_date", kind="mergesort")
        if len(ordered) > 21:
            changed.loc[ordered.index[21], "close_adj"] = 123_456.0

    revised = build_selection_metrics(predictions, changed)

    assert revised["monthly"] == original["monthly"]
    assert revised["summary"] == original["summary"]


def test_missing_t_plus_20_excludes_stock_and_thirty_eligible_still_scores() -> None:
    predictions, prices = _prices_and_predictions()
    date_t = predictions.iloc[0]["prediction_date"]
    ordered = prices.loc[prices["stock_id"] == "S30"].sort_values("trade_date")
    t_plus_20 = ordered.iloc[20]["trade_date"]
    prices.loc[
        (prices["stock_id"] == "S30") & (prices["trade_date"] == t_plus_20),
        "close_adj",
    ] = None

    payload = build_selection_metrics(predictions, prices)
    row = payload["monthly"][0]

    assert row["signal_date"] == date_t
    assert row["eligible_count"] == 30
    assert row["top15_actual_return"] is not None
    assert validate_selection_metrics(payload) == []


def test_fewer_than_thirty_eligible_does_not_shrink_top_and_bottom_groups() -> None:
    predictions, prices = _prices_and_predictions()
    predictions = predictions.loc[predictions["rank"] <= 29]

    payload = build_selection_metrics(predictions, prices)

    row = payload["monthly"][0]
    assert row["eligible_count"] == 29
    assert row["top15_actual_return"] is None
    assert row["bottom15_actual_return"] is None
    assert validate_selection_metrics(payload)
