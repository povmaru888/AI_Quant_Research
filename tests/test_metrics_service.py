"""P2-15 acceptance: performance and sensitivity service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import BacktestResult
from services.metrics_service import (
    METRIC_COLUMNS,
    SENSITIVITY_SLIPPAGES,
    calculate_metrics,
    run_cost_sensitivity,
)


def _result() -> BacktestResult:
    dates = pd.bdate_range(start="2020-01-02", periods=252).strftime("%Y-%m-%d")
    nav = 1_000_000.0 * (1.0005 ** np.arange(252))
    orders = pd.DataFrame(
        {
            "order_id": ["r|2020-01-02|A|BUY"],
            "signal_date": ["2019-12-31"],
            "execution_date": ["2020-01-02"],
            "stock_id": ["A"],
            "side": ["BUY"],
            "target_shares": [100],
            "executed_price": [100.1],
            "total_cost": [24.25],
        }
    )
    return BacktestResult(
        run_id="r",
        start_date=dates[0],
        end_date=dates[-1],
        initial_cash=1_000_000.0,
        nav=pd.Series(nav, index=pd.Index(dates, name="trade_date"), name="nav"),
        orders=orders,
        total_cost=24.25,
    )


def test_calculate_metrics_steady_growth() -> None:
    metrics = calculate_metrics(_result(), rank_ic=0.05, icir=0.8)
    assert list(metrics.columns) == list(METRIC_COLUMNS)
    row = metrics.iloc[0]
    assert row["cagr"] == pytest.approx(1.0005**251 - 1, rel=1e-6)
    assert row["max_drawdown"] == pytest.approx(0.0)
    assert row["win_rate"] == pytest.approx(1.0)
    assert np.isnan(row["sortino"])  # no losing days: undefined Sortino.
    assert row["sharpe"] > 100  # near-constant growth: exploding Sharpe.
    assert row["rank_ic"] == pytest.approx(0.05)
    assert row["icir"] == pytest.approx(0.8)
    assert np.isnan(calculate_metrics(_result()).iloc[0]["rank_ic"])


def test_calculate_metrics_drawdown_and_sortino() -> None:
    dates = pd.bdate_range(start="2020-01-02", periods=10).strftime("%Y-%m-%d")
    nav = pd.Series(
        [100.0, 110.0, 105.0, 90.0, 95.0, 100.0, 108.0, 104.0, 112.0, 115.0],
        index=pd.Index(dates, name="trade_date"),
        name="nav",
    )
    result = BacktestResult(
        run_id="r",
        start_date=dates[0],
        end_date=dates[-1],
        initial_cash=100.0,
        nav=nav,
        orders=pd.DataFrame(columns=["order_id", "executed_price", "total_cost"]),
        total_cost=0.0,
    )
    row = calculate_metrics(result).iloc[0]
    assert row["max_drawdown"] == pytest.approx(90.0 / 110.0 - 1)
    assert row["calmar"] == pytest.approx(row["cagr"] / abs(row["max_drawdown"]))
    assert 0 < row["win_rate"] < 1
    assert np.isfinite(row["sortino"])


def test_run_cost_sensitivity_monotonic_cost(settings) -> None:
    dates = pd.bdate_range(start="2020-01-02", periods=5).strftime("%Y-%m-%d")
    prices = pd.DataFrame(
        {
            "stock_id": ["A"] * 5,
            "trade_date": dates,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["r|2020-01-02|A|BUY"],
            "signal_date": ["2019-12-31"],
            "execution_date": ["2020-01-02"],
            "stock_id": ["A"],
            "side": ["BUY"],
            "target_shares": [100],
            "executed_price": [100.1],
            "broker_fee": [14.25],
            "transaction_tax": [0.0],
            "slippage_cost": [10.0],
            "total_cost": [24.25],
        }
    )
    table = run_cost_sensitivity(orders, prices, 1_000_000.0, settings, "r")
    assert table["slippage_rate"].tolist() == list(SENSITIVITY_SLIPPAGES)
    assert (table["total_cost"].diff().dropna() > 0).all()
    assert set(METRIC_COLUMNS) <= set(table.columns)


def test_run_cost_sensitivity_rejects_bad_inputs(settings) -> None:
    prices = pd.DataFrame(
        {"stock_id": ["A"], "trade_date": ["2020-01-02"], "open": [1.0], "close": [1.0]}
    )
    with pytest.raises(ValueError, match="non-empty"):
        run_cost_sensitivity(pd.DataFrame(columns=["order_id"]), prices, 1_000_000.0, settings, "r")
    with pytest.raises(ValueError, match="open"):
        orders = pd.DataFrame(
            {
                "order_id": ["x"],
                "stock_id": ["A"],
                "signal_date": ["2019-12-31"],
                "execution_date": ["2020-01-02"],
                "side": ["BUY"],
                "target_shares": [1],
            }
        )
        run_cost_sensitivity(orders, prices.drop(columns=["open"]), 1_000_000.0, settings, "r")
