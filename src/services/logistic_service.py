"""Formal fixed-parameter Logistic Regression baseline for walk-forward research."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from contracts import ModelArtifact
from services.xgb_service import rank_ic

LOGISTIC_PARAMS: dict[str, object] = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": None,
    "random_state": 42,
}


def _matrix(name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"invalid {name}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"invalid {name}: must be non-empty")
    selected = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if selected.isna().any().any():
        raise ValueError(f"invalid {name}: must not contain NaN")
    return selected.to_numpy(dtype=float)


def _labels(name: str, labels: pd.Series, size: int) -> np.ndarray:
    values = labels.to_numpy()
    if len(values) != size or set(np.unique(values)) - {0, 1}:
        raise ValueError(f"invalid {name}: must be 0/1 with one label per row")
    if len(np.unique(values)) != 2:
        raise ValueError(f"invalid {name}: must contain both classes")
    return values.astype(int)


def train_logistic(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
) -> tuple[ModelArtifact, LogisticRegression]:
    """Fit on training rows only and use validation rows only for Rank IC."""
    for name, value in (
        ("model_version", model_version),
        ("feature_version", feature_version),
        ("parameter_version", parameter_version),
        ("run_id", run_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid {name}: {value!r}")
    columns = tuple(train_x.columns)
    if not columns or "stock_id" in columns or len(set(columns)) != len(columns):
        raise ValueError("invalid train_x: needs unique feature columns only, no stock_id")
    if tuple(valid_x.columns) != columns:
        raise ValueError("invalid valid_x: feature columns and order must match train_x")
    x_train = _matrix("train_x", train_x, columns)
    y_train = _labels("train_y", train_y, len(x_train))
    x_valid = _matrix("valid_x", valid_x, columns)
    y_valid = _labels("valid_y", valid_y, len(x_valid))

    model = LogisticRegression(**LOGISTIC_PARAMS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(pd.DataFrame(x_train, columns=columns), y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("logistic regression failed to converge")
    probability = model.predict_proba(pd.DataFrame(x_valid, columns=columns))[:, 1]
    validation_ic = rank_ic(pd.Series(probability), pd.Series(y_valid))
    if not np.isfinite(validation_ic):
        raise ValueError("invalid validation scores: Rank IC is not finite")
    artifact = ModelArtifact(
        run_id=run_id,
        model_version=model_version,
        feature_version=feature_version,
        parameter_version=parameter_version,
        feature_columns=columns,
        best_params=dict(LOGISTIC_PARAMS),
        validation_rank_ic=validation_ic,
    )
    return artifact, model


def predict_logistic(
    artifact: ModelArtifact, features: pd.DataFrame, model: LogisticRegression
) -> pd.DataFrame:
    """Return probabilities and unique cross-sectional ranks."""
    if "stock_id" not in features.columns:
        raise ValueError("invalid features: missing 'stock_id'")
    matrix = _matrix("features", features, artifact.feature_columns)
    if tuple(model.feature_names_in_) != artifact.feature_columns:
        raise ValueError("invalid model: feature columns do not match artifact")
    probability = model.predict_proba(
        pd.DataFrame(matrix, columns=artifact.feature_columns)
    )[:, 1]
    out = pd.DataFrame(
        {"stock_id": features["stock_id"].astype(str).tolist(), "probability": probability}
    )
    out["rank"] = out["probability"].rank(ascending=False, method="first").astype(int)
    return out
