from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from services.logistic_service import predict_logistic, train_logistic


def _data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    train_x = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    train_y = pd.Series((train_x["a"] + 0.2 * train_x["b"] > 0).astype(int))
    valid_x = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    valid_y = pd.Series((valid_x["a"] + 0.2 * valid_x["b"] > 0).astype(int))
    return train_x, train_y, valid_x, valid_y


def test_logistic_round_trip_preserves_probabilities_and_ranks(tmp_path) -> None:
    train_x, train_y, valid_x, valid_y = _data()
    artifact, model = train_logistic(
        train_x, train_y, valid_x, valid_y, "logit", "factor_v4", "fixed", "run"
    )
    features = valid_x.copy()
    features.insert(0, "stock_id", [f"S{i}" for i in range(len(features))])
    expected = predict_logistic(artifact, features, model)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    actual = predict_logistic(artifact, features, joblib.load(path))
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["probability"].between(0.0, 1.0).all()
    assert actual["rank"].is_unique


@pytest.mark.parametrize("bad", ["nan", "missing", "order"])
def test_logistic_rejects_invalid_features(bad: str) -> None:
    train_x, train_y, valid_x, valid_y = _data()
    if bad == "nan":
        train_x.loc[0, "a"] = np.nan
    elif bad == "missing":
        valid_x = valid_x.drop(columns="b")
    else:
        valid_x = valid_x[["b", "a"]]
    with pytest.raises(ValueError):
        train_logistic(
            train_x, train_y, valid_x, valid_y, "logit", "factor_v4", "fixed", "run"
        )


def test_logistic_rejects_single_class_labels() -> None:
    train_x, train_y, valid_x, valid_y = _data()
    with pytest.raises(ValueError, match="both classes"):
        train_logistic(
            train_x,
            pd.Series(np.zeros(len(train_y), dtype=int)),
            valid_x,
            valid_y,
            "logit",
            "factor_v4",
            "fixed",
            "run",
        )


def test_validation_labels_do_not_change_fitted_coefficients() -> None:
    train_x, train_y, valid_x, valid_y = _data()
    _, first = train_logistic(
        train_x, train_y, valid_x, valid_y, "logit", "factor_v4", "fixed", "run"
    )
    _, second = train_logistic(
        train_x, train_y, valid_x, 1 - valid_y, "logit", "factor_v4", "fixed", "run"
    )
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.intercept_, second.intercept_)


def test_logistic_fails_on_convergence_warning(monkeypatch) -> None:
    train_x, train_y, valid_x, valid_y = _data()
    original_fit = LogisticRegression.fit

    def fit_with_warning(self, *args, **kwargs):
        fitted = original_fit(self, *args, **kwargs)
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        return fitted

    monkeypatch.setattr(LogisticRegression, "fit", fit_with_warning)
    with pytest.raises(RuntimeError, match="converge"):
        train_logistic(
            train_x, train_y, valid_x, valid_y, "logit", "factor_v4", "fixed", "run"
        )
