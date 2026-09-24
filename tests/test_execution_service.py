"""P2-13 acceptance: execution and cost service."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from contracts import PortfolioTarget
from services.execution_service import (
    ORDER_COLUMNS,
    create_orders,
    executed_price,
    transaction_cost,
)

SIGNAL = date(2019, 12, 31)


def _target() -> PortfolioTarget:
    return PortfolioTarget(
        run_id="run-001",
        signal_date="2019-12-31",
        top_n=15,
        actions={"A": "BUY", "B": "SELL", "C": "NONE"},
        weights={"A": 0.6},
        cash_weight=0.4,
        equity_exposure=0.6,
    )


def _opens(execution_date: str = "2020-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["A", "B"],
            "trade_date": [execution_date, execution_date],
            "open": [100.0, 50.0],
        }
    )


def _holdings() -> pd.DataFrame:
    return pd.DataFrame({"stock_id": ["B"], "shares": [100]})


def test_executed_price_and_cost_math(settings) -> None:
    assert executed_price(100.0, "BUY", 0.001) == pytest.approx(100.1)
    assert executed_price(100.0, "SELL", 0.001) == pytest.approx(99.9)
    with pytest.raises(ValueError, match="HOLD"):
        executed_price(100.0, "HOLD", 0.001)
    cfg = settings.execution
    buy = transaction_cost(10_000.0, "BUY", cfg)
    assert buy["broker_fee"] == pytest.approx(14.25)
    assert buy["transaction_tax"] == pytest.approx(0.0)
    assert buy["slippage_cost"] == pytest.approx(10.0)
    assert buy["total_cost"] == pytest.approx(24.25)
    sell = transaction_cost(10_000.0, "SELL", cfg)
    assert sell["transaction_tax"] == pytest.approx(30.0)
    assert sell["total_cost"] == pytest.approx(14.25 + 30.0 + 10.0)


def test_create_orders_buy_and_sell(settings) -> None:
    orders = create_orders(
        _target(), SIGNAL, _opens(), _holdings(), 1_000_000.0, settings, "run-001"
    )
    assert list(orders.columns) == list(ORDER_COLUMNS)
    buy = orders.loc[orders["stock_id"] == "A"].iloc[0]
    assert buy["side"] == "BUY"
    assert buy["execution_date"] == "2020-01-02"
    assert buy["target_shares"] == round(0.6 * 1_000_000.0 / 100.0)
    assert buy["executed_price"] == pytest.approx(100.0 * 1.001)
    assert buy["order_id"] == "run-001|2020-01-02|A|BUY"
    sell = orders.loc[orders["stock_id"] == "B"].iloc[0]
    assert sell["side"] == "SELL"
    assert sell["target_shares"] == 0
    assert sell["executed_price"] == pytest.approx(50.0 * 0.999)
    assert sell["transaction_tax"] > 0


def test_create_orders_executes_at_raw_open_when_adjusted_differs(settings) -> None:
    opens = _opens()
    opens["open_adj"] = [80.0, 40.0]
    orders = create_orders(
        _target(), SIGNAL, opens, _holdings(), 1_000_000.0, settings, "run-001"
    )
    buy = orders.set_index("stock_id").loc["A"]
    assert buy["target_shares"] == round(0.6 * 1_000_000.0 / 100.0)
    assert buy["executed_price"] == pytest.approx(100.0 * 1.001)


def test_create_orders_hold_rebalance(settings) -> None:
    target = PortfolioTarget(
        run_id="r",
        signal_date="2019-12-31",
        top_n=15,
        actions={"A": "HOLD"},
        weights={"A": 0.6},
        cash_weight=0.4,
        equity_exposure=0.6,
    )
    steady = pd.DataFrame({"stock_id": ["A"], "shares": [round(0.6 * 1_000_000.0 / 100.0)]})
    assert create_orders(target, SIGNAL, _opens(), steady, 1_000_000.0, settings, "r").empty
    light = pd.DataFrame({"stock_id": ["A"], "shares": [100]})
    top_up = create_orders(target, SIGNAL, _opens(), light, 1_000_000.0, settings, "r")
    assert top_up.iloc[0]["side"] == "BUY"
    heavy = pd.DataFrame({"stock_id": ["A"], "shares": [999_999]})
    trim = create_orders(target, SIGNAL, _opens(), heavy, 1_000_000.0, settings, "r")
    assert trim.iloc[0]["side"] == "SELL"


def test_create_orders_execution_must_follow_signal(settings) -> None:
    with pytest.raises(ValueError, match="must exceed signal_date"):
        create_orders(
            _target(), SIGNAL, _opens("2019-12-31"), _holdings(), 1_000_000.0, settings, "r"
        )
    mixed = pd.DataFrame(
        {
            "stock_id": ["A", "B"],
            "trade_date": ["2020-01-02", "2020-01-03"],
            "open": [100.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="single execution date"):
        create_orders(_target(), SIGNAL, mixed, _holdings(), 1_000_000.0, settings, "r")


def test_create_orders_skips_missing_price(settings) -> None:
    opens = pd.DataFrame({"stock_id": ["B"], "trade_date": ["2020-01-02"], "open": [50.0]})
    orders = create_orders(_target(), SIGNAL, opens, _holdings(), 1_000_000.0, settings, "r")
    assert set(orders["stock_id"]) == {"B"}


def test_create_orders_rejects_bad_inputs(settings) -> None:
    with pytest.raises(ValueError, match="portfolio_value"):
        create_orders(_target(), SIGNAL, _opens(), _holdings(), 0.0, settings, "r")
    with pytest.raises(ValueError, match="run_id"):
        create_orders(_target(), SIGNAL, _opens(), _holdings(), 1_000_000.0, settings, " ")
    with pytest.raises(ValueError, match="missing columns"):
        create_orders(
            _target(),
            SIGNAL,
            _opens().drop(columns=["open"]),
            _holdings(),
            1_000_000.0,
            settings,
            "r",
        )
