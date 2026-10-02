"""P2-08: XGBoost model service (SDD 10.2, 10.4).

Trains an ``XGBClassifier`` for top-quantile probability and scores new
months. ``ModelArtifact`` carries only metadata; the fitted booster
travels explicitly so train/serve skew is caught by column checks in
``predict_xgb`` instead of hiding inside the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

from contracts import ModelArtifact

DEFAULT_RANDOM_STATE = 42

DEFAULT_PARAMS: dict[str, object] = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def rank_ic(scores: pd.Series, labels: pd.Series) -> float:
    """Spearman correlation between scores and labels."""
    aligned = pd.DataFrame({"s": scores, "y": labels}).dropna()
    if len(aligned) < 3 or aligned["s"].nunique() < 2 or aligned["y"].nunique() < 2:
        return float("nan")
    return float(aligned["s"].corr(aligned["y"], method="spearman"))


def _resolve_params(params: dict | None) -> dict:
    merged = dict(DEFAULT_PARAMS)
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("invalid params: must be a dict")
        unknown = [k for k in params if k not in DEFAULT_PARAMS and k != "random_state"]
        if unknown:
            raise ValueError(f"invalid params: unknown keys {unknown}")
        merged.update(params)
    merged.setdefault("random_state", DEFAULT_RANDOM_STATE)
    n_estimators = merged["n_estimators"]
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError(f"invalid n_estimators: {n_estimators!r}")
    return merged


def _frame_inputs(name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"invalid {name}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"invalid {name}: must be non-empty")
    selected = frame.loc[:, columns]
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        # Processed features are numeric. Avoid per-column pandas conversion on
        # every Optuna trial and prediction; nullable numeric columns still map
        # missing values to NaN for the same validation below.
        matrix = selected.to_numpy(dtype=float, copy=False, na_value=np.nan)
        if np.isnan(matrix).any():
            raise ValueError(f"invalid {name}: must not contain NaN")
        return matrix
    converted = selected.apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        raise ValueError(f"invalid {name}: must not contain NaN")
    return converted.to_numpy(dtype=float)


def _label_inputs(name: str, labels: pd.Series, size: int) -> np.ndarray:
    values = labels.to_numpy(dtype=int) if labels.dtype == bool else labels.to_numpy()
    if len(values) != size or set(np.unique(values)) - {0, 1}:
        raise ValueError(f"invalid {name}: must be 0/1 with one label per row")
    return values.astype(int)


@dataclass(frozen=True)
class _TrainingData:
    feature_columns: tuple[str, ...]
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray


def _prepare_training(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
) -> _TrainingData:
    """Validate and convert a fold once before its hyperparameter trials."""
    feature_columns = tuple(train_x.columns)
    if not feature_columns or "stock_id" in feature_columns:
        raise ValueError("invalid train_x: needs feature columns only, no stock_id")
    x_train = _frame_inputs("train_x", train_x, feature_columns)
    y_train = _label_inputs("train_y", train_y, len(x_train))
    x_valid = _frame_inputs("valid_x", valid_x, feature_columns)
    y_valid = _label_inputs("valid_y", valid_y, len(x_valid))
    return _TrainingData(feature_columns, x_train, y_train, x_valid, y_valid)


def _validate_metadata(
    model_version: str, feature_version: str, parameter_version: str, run_id: str
) -> None:
    for name, value in (
        ("model_version", model_version),
        ("feature_version", feature_version),
        ("parameter_version", parameter_version),
        ("run_id", run_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid {name}: {value!r}")


def _fit_prepared(
    data: _TrainingData,
    resolved: dict,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
) -> tuple[ModelArtifact, XGBClassifier]:
    booster = XGBClassifier(**resolved)  # type: ignore[arg-type]
    booster.fit(data.x_train, data.y_train)
    valid_prob = booster.predict_proba(data.x_valid)[:, 1]
    valid_ic = rank_ic(pd.Series(valid_prob), pd.Series(data.y_valid, dtype=int))
    if not np.isfinite(valid_ic):
        # Degenerate validation (constant scores or single class) carries no
        # rank information: record the Spearman minimum instead of NaN so the
        # artifact stays finite and optimizers can rank the trial last.
        valid_ic = -1.0
    artifact = ModelArtifact(
        run_id=run_id,
        model_version=model_version,
        feature_version=feature_version,
        parameter_version=parameter_version,
        feature_columns=data.feature_columns,
        best_params={k: resolved[k] for k in (*DEFAULT_PARAMS, "random_state")},
        validation_rank_ic=valid_ic,
    )
    return artifact, booster


def train_xgb(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    params: dict | None,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
) -> tuple[ModelArtifact, XGBClassifier]:
    """Train and score on validation; return artifact plus fitted booster."""
    _validate_metadata(model_version, feature_version, parameter_version, run_id)
    if not train_x.columns.size or "stock_id" in train_x.columns:
        raise ValueError("invalid train_x: needs feature columns only, no stock_id")
    resolved = _resolve_params(params)
    data = _prepare_training(train_x, train_y, valid_x, valid_y)
    return _fit_prepared(data, resolved, model_version, feature_version, parameter_version, run_id)


def predict_xgb(
    artifact: ModelArtifact,
    features: pd.DataFrame,
    booster: XGBClassifier | XGBRanker | XGBRegressor,
) -> pd.DataFrame:
    """Score one month; probability is a classifier probability or ranker score."""
    if "stock_id" not in features.columns:
        raise ValueError("invalid features: missing 'stock_id'")
    matrix = _frame_inputs("features", features, artifact.feature_columns)
    if isinstance(booster, (XGBRanker, XGBRegressor)):
        margin = booster.predict(matrix)
        # LambdaMART emits unbounded ranking margins.  Persist a monotonic
        # sigmoid transform so the existing 0..1 prediction contract remains
        # valid without changing any cross-sectional ordering.
        probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -709.0, 709.0)))
    else:
        probability = booster.predict_proba(matrix)[:, 1]
    out = pd.DataFrame({"stock_id": features["stock_id"].tolist(), "probability": probability})
    out["rank"] = out["probability"].rank(ascending=False, method="first").astype(int)
    return out
