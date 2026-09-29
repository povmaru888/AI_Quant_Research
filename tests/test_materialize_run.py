"""Adjusted forward-return diagnostics keep trading-day positions intact."""

from __future__ import annotations

import pandas as pd
import pytest

from services.backtest_service import run_backtest
from tools.materialize_run import (
    _close_panel,
    _forward_returns,
    _prices_through_oos_horizon,
)


def test_forward_returns_do_not_skip_missing_adjusted_day() -> None:
    prices = pd.DataFrame(
        [
            {"stock_id": "A", "trade_date": "2020-01-01", "close_adj": 10.0},
            {"stock_id": "A", "trade_date": "2020-01-02", "close_adj": None},
            {"stock_id": "A", "trade_date": "2020-01-03", "close_adj": 12.0},
            {"stock_id": "B", "trade_date": "2020-01-01", "close_adj": 20.0},
            {"stock_id": "B", "trade_date": "2020-01-02", "close_adj": 21.0},
            {"stock_id": "B", "trade_date": "2020-01-03", "close_adj": 22.0},
        ]
    )
    panel = _close_panel(prices, "2020-01-01")
    assert len(panel["A"]) == 3
    assert _forward_returns(panel, 1).index.tolist() == ["B"]
    assert set(_forward_returns(panel, 2).index) == {"A", "B"}


def test_backtest_prices_cover_full_final_execution_month() -> None:
    prices = pd.DataFrame(
        [
            {"stock_id": "A", "trade_date": "2024-01-02"},
            {"stock_id": "A", "trade_date": "2024-01-31"},
            {"stock_id": "A", "trade_date": "2024-02-01"},
        ]
    )

    window, valuation_end = _prices_through_oos_horizon(prices, "2023-12-29")

    assert valuation_end == "2024-01-31"
    assert window["trade_date"].tolist() == ["2024-01-02", "2024-01-31"]


def test_oos_horizon_requires_a_complete_final_month() -> None:
    prices = pd.DataFrame(
        [{"stock_id": "A", "trade_date": "2024-01-15"}]
    )

    with pytest.raises(ValueError, match="does not prove the full valuation month"):
        _prices_through_oos_horizon(prices, "2023-12-29")


def test_backtest_values_last_entry_through_month_end(settings) -> None:
    prices = pd.DataFrame(
        [
            {
                "stock_id": "A",
                "trade_date": "2024-01-02",
                "open": 100.0,
                "open_adj": 100.0,
                "close_adj": 100.0,
            },
            {
                "stock_id": "A",
                "trade_date": "2024-01-31",
                "open": 120.0,
                "open_adj": 120.0,
                "close_adj": 120.0,
            },
            {
                "stock_id": "A",
                "trade_date": "2024-02-01",
                "open": 999.0,
                "open_adj": 999.0,
                "close_adj": 999.0,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "oos|2024-01-02|A|BUY",
                "signal_date": "2023-12-29",
                "execution_date": "2024-01-02",
                "stock_id": "A",
                "side": "BUY",
                "target_shares": 1,
                "executed_price": 100.0,
                "total_cost": 0.0,
            }
        ]
    )

    window, _ = _prices_through_oos_horizon(prices, "2023-12-29")
    result = run_backtest(orders, window, 1_000.0, settings, "oos")

    assert result.end_date == "2024-01-31"
    assert result.nav.iloc[-1] == pytest.approx(1_020.0)
