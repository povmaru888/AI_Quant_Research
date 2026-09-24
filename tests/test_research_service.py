"""P2-16 acceptance: research pipeline end-to-end smoke test."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date

import numpy as np
import pandas as pd
import pytest

from contracts import BacktestResult
from services.metrics_service import METRIC_COLUMNS, calculate_metrics
from services.research_service import get_dashboard_snapshot, run_research

AS_OF = date(2020, 2, 28)
STOCKS = [f"S{i:02d}" for i in range(12)]


def _build_fixture(delist_all: bool = False) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range(end="2020-03-31", periods=175).strftime("%Y-%m-%d")
    as_of_str = AS_OF.isoformat()
    price_rows, return_rows = [], []
    for i, stock_id in enumerate(STOCKS):
        closes = 100.0 + (i - 5.5) * 0.05 * np.arange(175) + rng.normal(0, 0.2, 175)
        closes = np.maximum(closes, 20.0)
        for day, close in zip(dates, closes, strict=True):
            price_rows.append(
                {
                    "stock_id": stock_id,
                    "trade_date": day,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 2_000_000.0,
                    "traded_value": close * 2_000_000.0,
                }
            )
        idx = int(dates.get_loc(as_of_str))
        ret_dates = dates[idx - 59 : idx + 1]
        window = closes[idx - 60 : idx + 1]
        for day, prev, cur in zip(ret_dates, window[:-1], window[1:], strict=True):
            return_rows.append(
                {"stock_id": stock_id, "trade_date": day, "log_return": np.log(cur / prev)}
            )
    prices = pd.DataFrame(price_rows)
    month_end_close = prices.loc[prices["trade_date"] == as_of_str].set_index("stock_id")["close"]
    taiex_dates = pd.bdate_range(end=as_of_str, periods=60).strftime("%Y-%m-%d")
    fixture = {
        "prices": prices,
        "stocks": pd.DataFrame(
            {
                "stock_id": STOCKS,
                "listed_date": ["2010-01-01"] * 12,
                "delisted_date": ["2019-01-01" if delist_all else ""] * 12,
                "flags": [""] * 12,
                "market_cap": [6_000_000_000.0] * 12,
            }
        ),
        "financials_snapshot": pd.DataFrame(
            {
                "stock_id": STOCKS,
                "report_period": ["2019Q4"] * 12,
                "announcement_date": ["2020-01-15"] * 12,
                "available_date": ["2020-01-16"] * 12,
                "revenue": [50e9] * 12,
                "net_income": [10e9] * 12,
                "equity": [100e9] * 12,
                "assets": [200e9] * 12,
                "operating_income": [8e9] * 12,
                "operating_cash_flow": [6e9] * 12,
            }
        ),
        "institutional_snapshot": pd.DataFrame(
            {
                "stock_id": STOCKS,
                "trade_date": [as_of_str] * 12,
                "foreign_net_buy": [1000.0] * 12,
                "trust_net_buy": [500.0] * 12,
                "margin_balance": [1_000_000.0] * 12,
                "short_balance": [100_000.0] * 12,
                "float_shares": [100_000_000.0] * 12,
            }
        ),
        "financials_history": pd.DataFrame(
            {
                "stock_id": [s for s in STOCKS for _ in range(6)],
                "report_period": ["2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4"] * 12,
                "available_date": [
                    "2018-11-14",
                    "2019-03-20",
                    "2019-05-15",
                    "2019-08-14",
                    "2019-11-14",
                    "2020-01-16",
                ]
                * 12,
                "revenue": [40e9, 42e9, 44e9, 46e9, 48e9, 50e9] * 12,
                "operating_income": [6e9, 6.4e9, 6.8e9, 7.2e9, 7.6e9, 8e9] * 12,
            }
        ),
        "institutional_history": pd.DataFrame(
            {
                "stock_id": [s for s in STOCKS for _ in range(25)],
                "trade_date": list(pd.bdate_range(end=as_of_str, periods=25).strftime("%Y-%m-%d"))
                * 12,
                "foreign_net_buy": [1000.0] * (25 * 12),
                "trust_net_buy": [500.0] * (25 * 12),
                "margin_balance": list(np.linspace(800_000.0, 1_000_000.0, 25)) * 12,
            }
        ),
        "returns": pd.DataFrame(return_rows),
        "taiex": pd.DataFrame(
            {
                "trade_date": taiex_dates,
                "close": np.linspace(10000.0, 11000.0, 60),
            }
        ),
        "next_open": pd.DataFrame(
            {
                "stock_id": STOCKS,
                "trade_date": ["2020-03-02"] * 12,
                "open": [month_end_close[s] for s in STOCKS],
            }
        ),
    }
    return fixture


class FakeStore:
    """In-memory ResearchStore for the smoke test."""

    def __init__(self, fixture: dict[str, pd.DataFrame]) -> None:
        self.fixture = fixture
        self.started: dict | None = None
        self.finished: tuple[str, str | None] | None = None
        self.saved: dict[str, object] = {}
        self.portfolio_value_loads = 0

    def load_prices(self) -> pd.DataFrame:
        return self.fixture["prices"]

    def load_stocks(self) -> pd.DataFrame:
        return self.fixture["stocks"]

    def load_financials_snapshot(self) -> pd.DataFrame:
        return self.fixture["financials_snapshot"]

    def load_institutional_snapshot(self) -> pd.DataFrame:
        return self.fixture["institutional_snapshot"]

    def load_financials_history(self) -> pd.DataFrame:
        return self.fixture["financials_history"]

    def load_institutional_history(self) -> pd.DataFrame:
        return self.fixture["institutional_history"]

    def load_returns(self, stock_ids: Collection[str] | None = None) -> pd.DataFrame:
        frame = self.fixture["returns"]
        return frame if stock_ids is None else frame.loc[frame["stock_id"].isin(stock_ids)]

    def load_taiex(self) -> pd.DataFrame:
        return self.fixture["taiex"]

    def load_next_open(self, as_of: date) -> pd.DataFrame:
        assert as_of == AS_OF
        return self.fixture["next_open"]

    def load_current_holdings(self) -> pd.DataFrame:
        return pd.DataFrame({"stock_id": [], "shares": []})

    def load_previous_positions(self) -> pd.DataFrame:
        return pd.DataFrame({"stock_id": []})

    def load_portfolio_value(self) -> float:
        self.portfolio_value_loads += 1
        return 30_000_000.0

    def start_run(self, metadata: dict) -> None:
        self.started = dict(metadata)

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        self.finished = (status, error)

    def save_predictions(self, predictions: pd.DataFrame, model_version: str) -> int:
        self.saved["predictions"] = predictions
        self.saved["model_version"] = model_version
        return len(predictions)

    def save_target_holdings(self, target) -> None:
        self.saved["target"] = target

    def save_orders(self, orders: pd.DataFrame) -> int:
        self.saved["orders"] = orders
        return len(orders)

    def load_run_summary(self, run_id: str) -> dict:
        assert self.started is not None and self.started["run_id"] == run_id
        return {
            **self.started,
            "model_version": self.saved.get("model_version", ""),
            "metrics": {},
            "oos_months": [AS_OF.strftime("%Y-%m")],
        }


def _tiny_optimize(monkeypatch) -> None:
    import services.optimization_service as opt_module

    real_suggest = opt_module._suggest

    def tiny_suggest(trial) -> dict:
        params = real_suggest(trial)
        params["n_estimators"] = 5
        return params

    monkeypatch.setattr(opt_module, "_suggest", tiny_suggest)


def test_run_research_smoke(settings, monkeypatch) -> None:
    _tiny_optimize(monkeypatch)
    store = FakeStore(_build_fixture())
    result = run_research(AS_OF, settings, store, "run-001", n_trials=1)
    assert isinstance(result, BacktestResult)
    assert len(result.orders) == 12
    assert store.started is not None and store.started["run_id"] == "run-001"
    assert store.finished == ("succeeded", None)
    assert store.portfolio_value_loads == 1
    metrics = calculate_metrics(result)
    assert list(metrics.columns) == list(METRIC_COLUMNS)


def test_run_research_deterministic(settings, monkeypatch) -> None:
    _tiny_optimize(monkeypatch)
    first = run_research(AS_OF, settings, FakeStore(_build_fixture()), "run-a", n_trials=1)
    second = run_research(AS_OF, settings, FakeStore(_build_fixture()), "run-b", n_trials=1)
    assert first.nav.equals(second.nav)
    assert first.total_cost == second.total_cost


def test_run_research_failure_closes_run(settings, monkeypatch) -> None:
    _tiny_optimize(monkeypatch)
    store = FakeStore(_build_fixture(delist_all=True))
    with pytest.raises(RuntimeError, match="empty universe"):
        run_research(AS_OF, settings, store, "run-002", n_trials=1)
    assert store.started is not None
    assert store.finished is not None and store.finished[0] == "failed"
    assert "empty universe" in (store.finished[1] or "")


def test_get_dashboard_snapshot_keys(settings, monkeypatch) -> None:
    _tiny_optimize(monkeypatch)
    store = FakeStore(_build_fixture())
    run_research(AS_OF, settings, store, "run-001", n_trials=1)
    snapshot = get_dashboard_snapshot("run-001", store)
    assert snapshot["run_id"] == "run-001"
    assert snapshot["data_end_date"] == "2020-02-28"
    assert snapshot["feature_version"] == settings.features.feature_version
    assert snapshot["model_version"].startswith("xgb_202002")
    assert snapshot["parameter_version"].startswith("params_")
    assert snapshot["oos_months"] == ["2020-02"]
