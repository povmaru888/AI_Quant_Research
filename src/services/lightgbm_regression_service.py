"""LightGBM Huber regression trained against monthly validation Rank IC."""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from contracts import ModelArtifact
from services.ranking_service import _mean_group_rank_ic, _validate_groups
from services.regression_service import _numeric_labels
from services.xgb_service import _frame_inputs, _validate_metadata
from settings import Settings

LIGHTGBM_SEARCH_SPACE: tuple[str, ...] = (
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "alpha",
)

LIGHTGBM_DEFAULT_PARAMS: dict[str, object] = {
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "alpha": 0.9,
}


def _resolve_lightgbm_params(params: dict | None, random_state: int = 42) -> dict:
    resolved = dict(LIGHTGBM_DEFAULT_PARAMS)
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("invalid params: must be a dict")
        allowed = set(LIGHTGBM_DEFAULT_PARAMS) | {"random_state"}
        if unknown := sorted(set(params) - allowed):
            raise ValueError(f"invalid params: unknown keys {unknown}")
        resolved.update(params)
    resolved.setdefault("random_state", random_state)
    if isinstance(resolved["n_estimators"], bool) or int(resolved["n_estimators"]) <= 0:
        raise ValueError(f"invalid n_estimators: {resolved['n_estimators']!r}")
    alpha = float(resolved["alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"invalid Huber alpha: {resolved['alpha']!r}")
    return resolved


def _suggest_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.70, 0.99),
    }


def train_lightgbm_regressor(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_groups: Sequence[int],
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    valid_groups: Sequence[int],
    params: dict | None,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
) -> tuple[ModelArtifact, LGBMRegressor]:
    """Fit deterministic LightGBM Huber and score mean monthly Rank IC."""
    _validate_metadata(model_version, feature_version, parameter_version, run_id)
    columns = tuple(train_x.columns)
    if not columns or "stock_id" in columns:
        raise ValueError("invalid train_x: needs feature columns only, no stock_id")
    x_train = _frame_inputs("train_x", train_x, columns)
    y_train = _numeric_labels("train_y", train_y, len(x_train))
    x_valid = _frame_inputs("valid_x", valid_x, columns)
    y_valid = _numeric_labels("valid_y", valid_y, len(x_valid))
    _validate_groups("train_groups", train_groups, len(x_train))
    valid_group = _validate_groups("valid_groups", valid_groups, len(x_valid))
    resolved = _resolve_lightgbm_params(params)
    booster = LGBMRegressor(
        objective="huber",
        deterministic=True,
        force_col_wise=True,
        subsample_freq=1,
        verbosity=-1,
        n_jobs=-1,
        **resolved,
    )
    booster.fit(x_train, y_train)
    score = _mean_group_rank_ic(booster.predict(x_valid), y_valid, valid_group)
    artifact = ModelArtifact(
        run_id=run_id,
        model_version=model_version,
        feature_version=feature_version,
        parameter_version=parameter_version,
        feature_columns=columns,
        best_params={key: resolved[key] for key in (*LIGHTGBM_DEFAULT_PARAMS, "random_state")},
        validation_rank_ic=score,
    )
    return artifact, booster


def optimize_lightgbm_regressor(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_groups: Sequence[int],
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    valid_groups: Sequence[int],
    settings: Settings,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
    n_trials: int | None = None,
) -> tuple[ModelArtifact, LGBMRegressor, pd.DataFrame]:
    """Tune LightGBM Huber against mean monthly validation Rank IC."""
    trials = settings.validation.optuna_trials if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError(f"invalid n_trials: {n_trials!r}")
    records: list[dict] = []
    best_score = -np.inf
    best_artifact: ModelArtifact | None = None
    best_booster: LGBMRegressor | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_artifact, best_booster
        params = _suggest_lightgbm(trial)
        params["random_state"] = settings.project.random_state
        artifact, booster = train_lightgbm_regressor(
            train_x,
            train_y,
            train_groups,
            valid_x,
            valid_y,
            valid_groups,
            params,
            model_version,
            feature_version,
            parameter_version,
            run_id,
        )
        score = float(artifact.validation_rank_ic)
        records.append({"trial": trial.number, **params, "rank_ic": score})
        if score > best_score:
            best_score, best_artifact, best_booster = score, artifact, booster
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.project.random_state),
    )
    study.optimize(objective, n_trials=trials)
    assert best_artifact is not None and best_booster is not None
    columns = ["trial", *LIGHTGBM_SEARCH_SPACE, "random_state", "rank_ic"]
    return best_artifact, best_booster, pd.DataFrame(records, columns=columns)


def predict_lightgbm(
    artifact: ModelArtifact,
    features: pd.DataFrame,
    booster: LGBMRegressor | lgb.Booster,
) -> pd.DataFrame:
    """Return the existing probability/rank contract without changing ordering."""
    if "stock_id" not in features.columns:
        raise ValueError("invalid features: missing 'stock_id'")
    matrix = _frame_inputs("features", features, artifact.feature_columns)
    raw = np.asarray(booster.predict(matrix), dtype=float)
    if raw.shape != (len(features),) or not np.isfinite(raw).all():
        raise ValueError("invalid LightGBM prediction: expected one finite score per row")
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -709.0, 709.0)))
    out = pd.DataFrame({"stock_id": features["stock_id"].tolist(), "probability": probability})
    out["rank"] = out["probability"].rank(ascending=False, method="first").astype(int)
    return out
