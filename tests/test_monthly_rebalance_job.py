"""P3-06 acceptance: monthly rebalance CLI job (mocked store)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from contracts import BacktestResult
from jobs.monthly_rebalance import main, run_monthly_rebalance


class FakeStore:
    def __init__(self, month_end: bool = True, fail_replace: bool = False) -> None:
        self.month_end = month_end
        self.fail_replace = fail_replace
        self.replaced: pd.DataFrame | None = None

    def verify_month_end(self, signal_date: date) -> bool:
        return self.month_end

    def replace_orders(self, run_id: str, signal_date: str, orders: pd.DataFrame) -> int:
        if self.fail_replace:
            raise RuntimeError("disk full")
        self.replaced = orders.copy()
        return len(orders)


def _result(execution_date: str = "2020-01-02") -> BacktestResult:
    dates = pd.bdate_range(start="2020-01-02", periods=3).strftime("%Y-%m-%d")
    return BacktestResult(
        run_id="r",
        start_date=dates[0],
        end_date=dates[-1],
        initial_cash=100.0,
        nav=pd.Series([100.0, 101.0, 102.0], index=pd.Index(dates, name="t"), name="nav"),
        orders=pd.DataFrame(
            {
                "order_id": ["r|2020-01-02|A|BUY"],
                "signal_date": ["2019-12-31"],
                "execution_date": [execution_date],
                "stock_id": ["A"],
                "side": ["BUY"],
                "target_shares": [1],
                "executed_price": [10.0],
                "total_cost": [1.0],
            }
        ),
        total_cost=1.0,
    )


def test_run_monthly_rebalance_success(settings) -> None:
    store = FakeStore()
    result = run_monthly_rebalance(
        date(2019, 12, 31),
        settings,
        store,
        research_fn=lambda *a, **k: _result(),
    )
    assert result["run_id"] == "rebalance-2019-12-31"
    assert result["execution_date"] == "2020-01-02"
    assert result["orders"] == 1
    assert store.replaced is not None and len(store.replaced) == 1


def test_run_monthly_rebalance_rejects_non_month_end(settings) -> None:
    store = FakeStore(month_end=False)
    with pytest.raises(ValueError, match="month-end"):
        run_monthly_rebalance(date(2019, 12, 15), settings, store)
    assert store.replaced is None


def test_run_monthly_rebalance_bad_execution_dates(settings) -> None:
    store = FakeStore()
    with pytest.raises(RuntimeError, match="execution dates"):
        run_monthly_rebalance(
            date(2019, 12, 31), settings, store, research_fn=lambda *a, **k: _result("2019-12-31")
        )
    assert store.replaced is None


def test_run_monthly_rebalance_replace_failure_leaves_nothing(settings) -> None:
    store = FakeStore(fail_replace=True)
    with pytest.raises(RuntimeError, match="disk full"):
        run_monthly_rebalance(
            date(2019, 12, 31), settings, store, research_fn=lambda *a, **k: _result()
        )
    assert store.replaced is None


def test_main_exit_codes(tmp_path, capsys) -> None:
    assert main(["--signal-date", "nope"], store_factory=lambda settings: FakeStore()) == 2
    assert main([], store_factory=lambda settings: FakeStore()) == 2
    code = main(
        ["--signal-date", "2019-12-15", "--config", str(tmp_path / "missing.yaml")],
        store_factory=lambda settings: FakeStore(),
    )
    assert code == 2


def test_main_success_and_replace_failure(capsys) -> None:
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "config.yaml"
    store = FakeStore()
    code = main(
        ["--signal-date", "2019-12-31", "--config", str(config)],
        store_factory=lambda settings: store,
        research_fn=lambda *a, **k: _result(),
    )
    assert code == 0
    assert "orders=1" in capsys.readouterr().out

    failing = FakeStore(fail_replace=True)
    code = main(
        ["--signal-date", "2019-12-31", "--config", str(config)],
        store_factory=lambda settings: failing,
        research_fn=lambda *a, **k: _result(),
    )
    assert code == 1
    assert failing.replaced is None
