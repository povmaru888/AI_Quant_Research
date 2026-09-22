"""P2-08 acceptance: XGBoost model service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.xgb_service import DEFAULT_PARAMS, predict_xgb, rank_ic, train_xgb

rng = np.random.default_rng(42)


def _dataset(n: int, shift: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    frame = pd.DataFrame({"f1": f1, "f2": f2})
    labels = pd.Series(((f1 + shift * f2) > 0).astype(int))
    return frame, labels


def test_train_xgb_artifact_complete() -> None:
    train_x, train_y = _dataset(60)
    valid_x, valid_y = _dataset(20)
    artifact, booster = train_xgb(
        train_x,
        train_y,
        valid_x,
        valid_y,
        {"n_estimators": 10},
        "xgb_v1",
        "factor_v1",
        "p1",
        "run-001",
    )
    assert artifact.feature_columns == ("f1", "f2")
    assert artifact.best_params["n_estimators"] == 10
    assert artifact.best_params["max_depth"] == DEFAULT_PARAMS["max_depth"]
    assert artifact.best_params["random_state"] == 42
    assert np.isfinite(artifact.validation_rank_ic)
    assert booster is not None


def test_predict_xgb_rank_unique_and_ordered() -> None:
    train_x, train_y = _dataset(60)
    valid_x, valid_y = _dataset(20)
    artifact, booster = train_xgb(
        train_x, train_y, valid_x, valid_y, {"n_estimators": 10}, "v", "f", "p", "r"
    )
    features = pd.DataFrame(
        {
            "stock_id": [f"S{i:02d}" for i in range(20)],
            "f1": valid_x["f1"].tolist(),
            "f2": valid_x["f2"].tolist(),
        }
    )
    out = predict_xgb(artifact, features, booster)
    assert list(out.columns) == ["stock_id", "probability", "rank"]
    assert out["rank"].is_unique
    assert set(out["rank"]) == set(range(1, 21))
    best = out.loc[out["rank"] == 1, "probability"].iloc[0]
    assert best == out["probability"].max()


def test_predict_xgb_rejects_column_skew() -> None:
    train_x, train_y = _dataset(60)
    valid_x, valid_y = _dataset(20)
    artifact, booster = train_xgb(
        train_x, train_y, valid_x, valid_y, {"n_estimators": 10}, "v", "f", "p", "r"
    )
    with pytest.raises(ValueError, match="missing columns"):
        predict_xgb(artifact, pd.DataFrame({"stock_id": ["S"], "f1": [0.0]}), booster)
    with pytest.raises(ValueError, match="stock_id"):
        predict_xgb(artifact, valid_x, booster)


def test_train_xgb_rejects_bad_params() -> None:
    train_x, train_y = _dataset(60)
    valid_x, valid_y = _dataset(20)
    with pytest.raises(ValueError, match="unknown keys"):
        train_xgb(train_x, train_y, valid_x, valid_y, {"nope": 1}, "v", "f", "p", "r")
    with pytest.raises(ValueError, match="n_estimators"):
        train_xgb(train_x, train_y, valid_x, valid_y, {"n_estimators": 0}, "v", "f", "p", "r")
    with pytest.raises(ValueError, match="0/1"):
        bad_y = pd.Series([0, 2] * 30)
        train_xgb(train_x, bad_y, valid_x, valid_y, None, "v", "f", "p", "r")


def test_train_xgb_degenerate_validation_floors_to_minus_one() -> None:
    train_x, train_y = _dataset(60)
    valid_x, _ = _dataset(20)
    single_class_y = pd.Series([0] * 20)
    artifact, _ = train_xgb(
        train_x, train_y, valid_x, single_class_y, {"n_estimators": 10}, "v", "f", "p", "r"
    )
    assert artifact.validation_rank_ic == pytest.approx(-1.0)


def test_rank_ic_known_values() -> None:
    scores = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert rank_ic(scores, pd.Series([0, 0, 1, 1])) == pytest.approx(0.894427, abs=1e-5)
    assert np.isnan(rank_ic(scores, pd.Series([1, 1, 1, 1])))
