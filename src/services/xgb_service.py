"""P2-08: XGBoost model service (SDD 10.2, 10.4).

Trains an ``XGBClassifier`` for top-quantile probability and scores new
months. ``ModelArtifact`` carries only metadata; the fitted booster
travels explicitly so train/serve skew is caught by column checks in
``predict_xgb`` instead of hiding inside the contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

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
    if params is None:
        return merged
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
    matrix = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        raise ValueError(f"invalid {name}: must not contain NaN")
    if matrix.empty:
        raise ValueError(f"invalid {name}: must be non-empty")
    return matrix.to_numpy(dtype=float)


def _label_inputs(name: str, labels: pd.Series, size: int) -> np.ndarray:
    values = labels.to_numpy(dtype=int) if labels.dtype == bool else labels.to_numpy()
    if len(values) != size or set(np.unique(values)) - {0, 1}:
        raise ValueError(f"invalid {name}: must be 0/1 with one label per row")
    return values.astype(int)


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
    for name, value in (
        ("model_version", model_version),
        ("feature_version", feature_version),
        ("parameter_version", parameter_version),
        ("run_id", run_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid {name}: {value!r}")
    feature_columns = tuple(train_x.columns)
    if not feature_columns or "stock_id" in feature_columns:
        raise ValueError("invalid train_x: needs feature columns only, no stock_id")
    resolved = _resolve_params(params)
    x_train = _frame_inputs("train_x", train_x, feature_columns)
    y_train = _label_inputs("train_y", train_y, len(x_train))
    x_valid = _frame_inputs("valid_x", valid_x, feature_columns)
    y_valid = _label_inputs("valid_y", valid_y, len(x_valid))

    booster = XGBClassifier(**resolved)  # type: ignore[arg-type]
    booster.fit(x_train, y_train)
    valid_prob = booster.predict_proba(x_valid)[:, 1]
    valid_ic = rank_ic(pd.Series(valid_prob), pd.Series(y_valid, dtype=int))
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
        feature_columns=feature_columns,
        best_params={k: resolved[k] for k in (*DEFAULT_PARAMS, "random_state")},
        validation_rank_ic=valid_ic,
    )
    return artifact, booster


def predict_xgb(
    artifact: ModelArtifact, features: pd.DataFrame, booster: XGBClassifier
) -> pd.DataFrame:
    """Score one month; returns stock_id, probability, rank (1 = best)."""
    if "stock_id" not in features.columns:
        raise ValueError("invalid features: missing 'stock_id'")
    matrix = _frame_inputs("features", features, artifact.feature_columns)
    probability = booster.predict_proba(matrix)[:, 1]
    out = pd.DataFrame({"stock_id": features["stock_id"].tolist(), "probability": probability})
    out["rank"] = out["probability"].rank(ascending=False, method="first").astype(int)
    return out
