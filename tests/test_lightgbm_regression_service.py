from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from services.lightgbm_regression_service import (
    predict_lightgbm,
    train_lightgbm_regressor,
)


def _data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    train_x = pd.DataFrame(rng.normal(size=(32, 3)), columns=["a", "b", "c"])
    train_y = pd.Series(0.03 * train_x["a"] - 0.01 * train_x["b"])
    valid_x = pd.DataFrame(rng.normal(size=(16, 3)), columns=train_x.columns)
    valid_y = pd.Series(0.03 * valid_x["a"] - 0.01 * valid_x["b"])
    return train_x, train_y, valid_x, valid_y


def test_lightgbm_huber_native_reload_preserves_predictions(tmp_path) -> None:
    train_x, train_y, valid_x, valid_y = _data()
    artifact, model = train_lightgbm_regressor(
        train_x,
        train_y,
        [8] * 4,
        valid_x,
        valid_y,
        [8] * 2,
        {"n_estimators": 20, "random_state": 42},
        "lgbm",
        "factor_v4",
        "p",
        "r",
    )
    features = valid_x.copy()
    features.insert(0, "stock_id", [f"S{i}" for i in range(len(features))])
    before = predict_lightgbm(artifact, features, model)
    path = tmp_path / "booster.txt"
    model.booster_.save_model(str(path))
    after = predict_lightgbm(artifact, features, lgb.Booster(model_file=str(path)))

    np.testing.assert_allclose(before["probability"], after["probability"], rtol=1e-12, atol=1e-12)
    assert before["rank"].tolist() == after["rank"].tolist()
    assert before["probability"].between(0.0, 1.0).all()


def test_lightgbm_huber_same_seed_is_deterministic() -> None:
    train_x, train_y, valid_x, valid_y = _data()
    arguments = (
        train_x,
        train_y,
        [8] * 4,
        valid_x,
        valid_y,
        [8] * 2,
        {"n_estimators": 20, "random_state": 42},
        "lgbm",
        "factor_v4",
        "p",
        "r",
    )
    artifact_a, model_a = train_lightgbm_regressor(*arguments)
    artifact_b, model_b = train_lightgbm_regressor(*arguments)
    features = valid_x.copy()
    features.insert(0, "stock_id", [f"S{i}" for i in range(len(features))])
    score_a = predict_lightgbm(artifact_a, features, model_a)
    score_b = predict_lightgbm(artifact_b, features, model_b)
    np.testing.assert_array_equal(score_a["probability"], score_b["probability"])
    assert score_a["rank"].tolist() == score_b["rank"].tolist()


def test_lightgbm_huber_rejects_invalid_inputs() -> None:
    train_x, train_y, valid_x, valid_y = _data()
    with pytest.raises(ValueError, match="Huber alpha"):
        train_lightgbm_regressor(
            train_x,
            train_y,
            [8] * 4,
            valid_x,
            valid_y,
            [8] * 2,
            {"alpha": 1.0},
            "lgbm",
            "factor_v4",
            "p",
            "r",
        )
    broken = train_x.copy()
    broken.loc[0, "a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        train_lightgbm_regressor(
            broken,
            train_y,
            [8] * 4,
            valid_x,
            valid_y,
            [8] * 2,
            None,
            "lgbm",
            "factor_v4",
            "p",
            "r",
        )
    with pytest.raises(ValueError, match="group"):
        train_lightgbm_regressor(
            train_x,
            train_y,
            [31],
            valid_x,
            valid_y,
            [8] * 2,
            None,
            "lgbm",
            "factor_v4",
            "p",
            "r",
        )
