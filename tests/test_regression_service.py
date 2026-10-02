from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from services.regression_service import (
    build_excess_return_labels_by_month,
    train_xgb_regressor,
)
from services.xgb_service import predict_xgb


def test_excess_labels_use_each_stocks_exact_twentieth_subsequent_row() -> None:
    days = pd.bdate_range("2020-01-01", periods=22).strftime("%Y-%m-%d").tolist()
    rows = []
    for stock_id, end in (("A", 1.2), ("B", 1.0)):
        closes = np.ones(22)
        closes[20] = end
        for day, close in zip(days, closes, strict=True):
            rows.append({"stock_id": stock_id, "trade_date": day, "close_adj": close})
    panels = {
        "2020-01": {
            "signal_date": days[0],
            "frame": pd.DataFrame({"stock_id": ["A", "B"]}),
        }
    }
    labels = build_excess_return_labels_by_month(panels, pd.DataFrame(rows))["2020-01"]
    assert labels["A"] == pytest.approx(0.1)
    assert labels["B"] == pytest.approx(-0.1)
    assert labels.mean() == pytest.approx(0.0)


def test_pseudo_huber_regressor_scores_for_existing_prediction_contract() -> None:
    rng = np.random.default_rng(12)
    train_x = pd.DataFrame({"f": rng.normal(size=24)})
    train_y = pd.Series(rng.normal(scale=0.05, size=24))
    valid_x = pd.DataFrame({"f": rng.normal(size=12)})
    valid_y = pd.Series(rng.normal(scale=0.05, size=12))
    artifact, booster = train_xgb_regressor(
        train_x, train_y, [6] * 4, valid_x, valid_y, [6] * 2,
        {"n_estimators": 5}, "regressor", "factor_v4", "p", "r",
    )
    assert isinstance(booster, XGBRegressor)
    features = valid_x.copy()
    features.insert(0, "stock_id", [f"S{i}" for i in range(len(features))])
    scored = predict_xgb(artifact, features, booster)
    assert scored["rank"].is_unique
    assert scored["probability"].between(0.0, 1.0).all()
