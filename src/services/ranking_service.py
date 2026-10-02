"""Monthly grouped XGBoost learning-to-rank training and optimization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRanker

from contracts import ModelArtifact
from services.optimization_service import SEARCH_SPACE, _suggest
from services.xgb_service import (
    DEFAULT_PARAMS,
    _frame_inputs,
    _label_inputs,
    _resolve_params,
    _validate_metadata,
    rank_ic,
)
from settings import Settings


def _validate_groups(name: str, groups: Sequence[int], row_count: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in groups)
    if not values or any(value <= 0 for value in values) or sum(values) != row_count:
        raise ValueError(f"invalid {name}: positive group sizes must sum to {row_count}")
    return values


def _mean_group_rank_ic(scores: np.ndarray, labels: np.ndarray, groups: Sequence[int]) -> float:
    values: list[float] = []
    start = 0
    for size in groups:
        stop = start + size
        value = rank_ic(pd.Series(scores[start:stop]), pd.Series(labels[start:stop]))
        if np.isfinite(value):
            values.append(float(value))
        start = stop
    return float(np.mean(values)) if values else -1.0


def train_xgb_ranker(
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
) -> tuple[ModelArtifact, XGBRanker]:
    """Fit rank:pairwise with one query group per signal month."""
    _validate_metadata(model_version, feature_version, parameter_version, run_id)
    columns = tuple(train_x.columns)
    if not columns or "stock_id" in columns:
        raise ValueError("invalid train_x: needs feature columns only, no stock_id")
    x_train = _frame_inputs("train_x", train_x, columns)
    y_train = _label_inputs("train_y", train_y, len(x_train))
    x_valid = _frame_inputs("valid_x", valid_x, columns)
    y_valid = _label_inputs("valid_y", valid_y, len(x_valid))
    train_group = _validate_groups("train_groups", train_groups, len(x_train))
    valid_group = _validate_groups("valid_groups", valid_groups, len(x_valid))
    resolved = _resolve_params(params)
    booster = XGBRanker(objective="rank:pairwise", **resolved)  # type: ignore[arg-type]
    booster.fit(x_train, y_train, group=train_group, verbose=False)
    score = _mean_group_rank_ic(booster.predict(x_valid), y_valid, valid_group)
    artifact = ModelArtifact(
        run_id=run_id,
        model_version=model_version,
        feature_version=feature_version,
        parameter_version=parameter_version,
        feature_columns=columns,
        best_params={k: resolved[k] for k in (*DEFAULT_PARAMS, "random_state")},
        validation_rank_ic=score,
    )
    return artifact, booster


def optimize_xgb_ranker(
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
) -> tuple[ModelArtifact, XGBRanker, pd.DataFrame]:
    """Tune rank:pairwise against mean monthly validation Rank IC."""
    trials = settings.validation.optuna_trials if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError(f"invalid n_trials: {n_trials!r}")
    records: list[dict] = []
    best_score = -np.inf
    best_artifact: ModelArtifact | None = None
    best_booster: XGBRanker | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_artifact, best_booster
        params = _suggest(trial)
        params["random_state"] = settings.project.random_state
        artifact, booster = train_xgb_ranker(
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
    log = pd.DataFrame(records, columns=["trial", *SEARCH_SPACE, "random_state", "rank_ic"])
    return best_artifact, best_booster, log
