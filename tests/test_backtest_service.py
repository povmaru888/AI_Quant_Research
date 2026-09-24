"""P2-14 acceptance: backtest service."""

from __future__ import annotations

import pandas as pd
import pytest

from contracts import BacktestResult
from services.backtest_service import run_backtest


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range(start="2020-01-02", periods=5).strftime("%Y-%m-%d")
    rows = []
    for stock_id, base in (("A", 100.0), ("B", 50.0)):
        for i, day in enumerate(dates):
            rows.append(
                {
                    "stock_id": stock_id,
                    "trade_date": day,
                    "open": base + i,
                    "open_adj": base + i,
                    "close_adj": base + i,
                }
            )
    return pd.DataFrame(rows)


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "r|2020-01-02|A|BUY",
                "signal_date": "2019-12-31",
                "execution_date": "2020-01-02",
                "stock_id": "A",
                "side": "BUY",
                "target_shares": 100,
                "executed_price": 100.1,
                "total_cost": 24.25,
            },
            {
                "order_id": "r|2020-01-03|B|BUY",
                "signal_date": "2019-12-31",
                "execution_date": "2020-01-03",
                "stock_id": "B",
                "side": "BUY",
                "target_shares": 50,
                "executed_price": 51.051,
                "total_cost": 12.0,
            },
        ]
    )


def test_run_backtest_nav_and_cost(settings) -> None:
    result = run_backtest(_orders(), _prices(), 1_000_000.0, settings, "run-001")
    assert isinstance(result, BacktestResult)
    # Day 1: 100 x 100.1 + 24.25 outlay; 100 x 100.0 equity (0.1 slippage drag).
    assert result.nav.iloc[0] == pytest.approx(1_000_000.0 - 100 * 0.1 - 24.25)
    # Day 2: second fill lands; equity 100x101 + 50x51.
    day2 = 1_000_000.0 - (100 * 100.1 + 24.25) - (50 * 51.051 + 12.0) + 100 * 101.0 + 50 * 51.0
    assert result.nav.iloc[1] == pytest.approx(day2)
    assert result.total_cost == pytest.approx(36.25)
    assert result.start_date == "2020-01-02"
    assert result.end_date == "2020-01-08"
    assert len(result.nav) == 5


def test_run_backtest_insufficient_cash_skips(settings) -> None:
    result = run_backtest(_orders(), _prices(), 1_000.0, settings, "r")
    assert (result.nav == 1_000.0).all()
    assert result.total_cost == pytest.approx(0.0)


def test_run_backtest_empty_orders_stays_cash(settings) -> None:
    empty = pd.DataFrame(columns=["order_id", "executed_price", "total_cost"])
    result = run_backtest(empty, _prices(), 500_000.0, settings, "r")
    assert (result.nav == 500_000.0).all()
    assert result.total_cost == pytest.approx(0.0)


def test_run_backtest_sell_liquidates(settings) -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "r|2020-01-02|A|BUY",
                "signal_date": "2019-12-31",
                "execution_date": "2020-01-02",
                "stock_id": "A",
                "side": "BUY",
                "target_shares": 100,
                "executed_price": 100.0,
                "total_cost": 0.0,
            },
            {
                "order_id": "r|2020-01-06|A|SELL",
                "signal_date": "2019-12-31",
                "execution_date": "2020-01-06",
                "stock_id": "A",
                "side": "SELL",
                "target_shares": 0,
                "executed_price": 103.0,
                "total_cost": 0.0,
            },
        ]
    )
    result = run_backtest(orders, _prices(), 1_000_000.0, settings, "r")
    assert result.nav.iloc[-1] == pytest.approx(1_000_000.0 - 10_000.0 + 10_300.0)
    assert result.total_cost == pytest.approx(0.0)


def test_run_backtest_rejects_bad_inputs(settings) -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        run_backtest(_orders(), _prices(), 0.0, settings, "r")
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(_orders().drop(columns=["side"]), _prices(), 1_000_000.0, settings, "r")
    bad_time = _orders().copy()
    bad_time.loc[0, "execution_date"] = "2019-12-31"
    with pytest.raises(ValueError, match="must exceed signal_date"):
        run_backtest(bad_time, _prices(), 1_000_000.0, settings, "r")


def test_run_backtest_carries_last_close_over_gaps(settings) -> None:
    prices = _prices()
    # Drop B's day-4 bar and append an index-only day: holdings must carry
    # forward, never read as worthless.
    prices = prices.loc[~((prices["stock_id"] == "B") & (prices["trade_date"] == "2020-01-07"))]
    extra = pd.DataFrame([{"stock_id": "TAIEX", "trade_date": "2020-01-09"}])
    prices = pd.concat([prices, extra], ignore_index=True)
    result = run_backtest(_orders(), prices, 1_000_000.0, settings, "r")
    assert result.end_date == "2020-01-09"
    # Day 4 (2020-01-07): B carried at 52.0 (day-3 close), A at 103.0.
    day4 = 1_000_000.0 - (100 * 100.1 + 24.25) - (50 * 51.051 + 12.0) + 100 * 103.0 + 50 * 52.0
    assert result.nav.loc["2020-01-07"] == pytest.approx(day4)
    # Index-only day carries both positions: no fake crash to cash.
    assert result.nav.loc["2020-01-09"] == pytest.approx(result.nav.loc["2020-01-08"])


def test_run_backtest_uses_adjusted_total_return_across_dividend(settings) -> None:
    prices = pd.DataFrame(
        [
            # A 20% cash distribution makes raw prices fall from 100 to 80.
            # Back-adjusted prices correctly stay at 80 across the event.
            {"stock_id": "A", "trade_date": "2020-01-02", "open": 100.0,
             "open_adj": 80.0, "close_adj": 80.0},
            {"stock_id": "A", "trade_date": "2020-01-03", "open": 80.0,
             "open_adj": 80.0, "close_adj": 80.0},
            {"stock_id": "A", "trade_date": "2020-01-06", "open": 80.0,
             "open_adj": 80.0, "close_adj": 80.0},
        ]
    )
    orders = pd.DataFrame(
        [
            {"order_id": "buy", "signal_date": "2020-01-01", "execution_date": "2020-01-02",
             "stock_id": "A", "side": "BUY", "target_shares": 100,
             "executed_price": 100.0, "total_cost": 0.0},
            {"order_id": "sell", "signal_date": "2020-01-03", "execution_date": "2020-01-06",
             "stock_id": "A", "side": "SELL", "target_shares": 0,
             "executed_price": 80.0, "total_cost": 0.0},
        ]
    )
    result = run_backtest(orders, prices, 20_000.0, settings, "r")
    assert result.nav.tolist() == pytest.approx([20_000.0] * 3)
    assert result.orders.loc[0, "executed_price"] == 100.0  # real exchange quote retained
    assert result.orders.loc[1, "executed_price"] == 80.0


def test_run_backtest_partial_sell_preserves_adjusted_wealth(settings) -> None:
    prices = pd.DataFrame(
        [
            {"stock_id": "A", "trade_date": "2020-01-02", "open": 100.0,
             "open_adj": 80.0, "close_adj": 80.0},
            {"stock_id": "A", "trade_date": "2020-01-03", "open": 80.0,
             "open_adj": 80.0, "close_adj": 80.0},
            {"stock_id": "A", "trade_date": "2020-01-06", "open": 80.0,
             "open_adj": 80.0, "close_adj": 80.0},
        ]
    )
    orders = pd.DataFrame(
        [
            {"order_id": "buy", "signal_date": "2020-01-01", "execution_date": "2020-01-02",
             "stock_id": "A", "side": "BUY", "target_shares": 100,
             "executed_price": 100.0, "total_cost": 0.0},
            {"order_id": "half", "signal_date": "2020-01-02", "execution_date": "2020-01-03",
             "stock_id": "A", "side": "SELL", "target_shares": 50,
             "executed_price": 80.0, "total_cost": 0.0},
            {"order_id": "rest", "signal_date": "2020-01-03", "execution_date": "2020-01-06",
             "stock_id": "A", "side": "SELL", "target_shares": 0,
             "executed_price": 80.0, "total_cost": 0.0},
        ]
    )
    result = run_backtest(orders, prices, 20_000.0, settings, "r")
    assert result.nav.tolist() == pytest.approx([20_000.0] * 3)


def test_run_backtest_rejects_missing_adjusted_quotes(settings) -> None:
    with pytest.raises(ValueError, match="open_adj"):
        run_backtest(_orders(), _prices().drop(columns=["open_adj"]), 1_000_000.0, settings, "r")
    prices = _prices()
    on_day = (prices["stock_id"] == "A") & (prices["trade_date"] == "2020-01-03")
    prices.loc[on_day, "close_adj"] = None
    with pytest.raises(ValueError, match="missing close_adj"):
        run_backtest(_orders(), prices, 1_000_000.0, settings, "r")
    prices = _prices()
    on_day = (prices["stock_id"] == "A") & (prices["trade_date"] == "2020-01-02")
    prices.loc[on_day, "open_adj"] = None
    with pytest.raises(ValueError, match="raw/adjusted open"):
        run_backtest(_orders(), prices, 1_000_000.0, settings, "r")
