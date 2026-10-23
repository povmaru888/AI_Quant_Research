"""Adjusted forward-return diagnostics keep trading-day positions intact."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from services.backtest_service import run_backtest
from tools import materialize_run
from tools.materialize_run import (
    _close_panel,
    _forward_returns,
    _frozen_model_explain,
    _prices_through_oos_horizon,
)


def test_frozen_lightgbm_explain_uses_persisted_model(tmp_path) -> None:
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"])
    target = matrix["a"] - matrix["b"]
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=10,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(matrix, target)
    model.booster_.save_model(str(tmp_path / "booster.txt"))
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "model_family": "lightgbm",
                "training_objective": "lightgbm:huber",
                "model_version": "lgbm-test",
                "feature_version": "factor_v4",
                "common_features": ["a", "b"],
                "model_file": "booster.txt",
            }
        ),
        encoding="utf-8",
    )
    features = SimpleNamespace(
        feature_version="factor_v4",
        frame=pd.concat([pd.Series([f"S{i}" for i in range(30)], name="stock_id"), matrix], axis=1),
    )

    payload = _frozen_model_explain(features, tmp_path, {"model_version": "lgbm-test"})

    assert payload["method"] == "frozen_lightgbm_lightgbm:huber"
    assert {row["feature"] for row in payload["shap_top"]} == {"a", "b"}
    assert {row["feature"] for row in payload["feature_importance"]} == {"a", "b"}


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
    prices = pd.DataFrame([{"stock_id": "A", "trade_date": "2024-01-15"}])

    with pytest.raises(ValueError, match="does not prove the full valuation month"):
        _prices_through_oos_horizon(prices, "2023-12-29")


def test_oos_horizon_can_value_an_in_progress_final_month() -> None:
    prices = pd.DataFrame(
        [
            {"stock_id": "A", "trade_date": "2024-01-02"},
            {"stock_id": "A", "trade_date": "2024-01-15"},
        ]
    )

    window, valuation_end = _prices_through_oos_horizon(
        prices, "2023-12-29", allow_partial=True
    )

    assert valuation_end == "2024-01-15"
    assert window["trade_date"].tolist() == ["2024-01-02", "2024-01-15"]


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


def test_save_features_omits_stable_missing_indicators_from_dashboard_table(
    monkeypatch,
) -> None:
    captured = {}

    @contextmanager
    def fake_session_scope(_engine):
        yield object()

    def fake_save_features(_session, frame):
        captured["frame"] = frame.copy()
        return len(frame)

    monkeypatch.setattr(materialize_run, "session_scope", fake_session_scope)
    monkeypatch.setattr(materialize_run.research_repo, "save_features", fake_save_features)
    features = SimpleNamespace(
        frame=pd.DataFrame(
            {
                "stock_id": ["2330"],
                "momentum_20d": [0.1],
                "momentum_20d__missing": [0.0],
                "missing_flag": [0],
            }
        )
    )

    written = materialize_run._save_features(
        object(), features, "2020-12-31", "factor_adj_pit_v3_stable"
    )

    assert written == 1
    assert "momentum_20d__missing" not in captured["frame"].columns
    assert captured["frame"].loc[0, "momentum_20d"] == pytest.approx(0.1)
    assert captured["frame"].loc[0, "missing_flag"] == 0
