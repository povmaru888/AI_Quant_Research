"""P2-09: Optuna optimization service (SDD 10.4).

Searches the eight SDD hyperparameters with Validation Mean Rank IC as
the sole objective. Test data can never enter: the signature has no
test parameter at all. Trial count defaults to
``settings.validation.optuna_trials`` (MVP 50).
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from contracts import ModelArtifact
from services.xgb_service import (
    _fit_prepared,
    _prepare_training,
    _resolve_params,
    _validate_metadata,
)
from settings import Settings

SEARCH_SPACE: tuple[str, ...] = (
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
)


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def optimize_xgb(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    settings: Settings,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
    n_trials: int | None = None,
) -> tuple[ModelArtifact, XGBClassifier, pd.DataFrame]:
    """Run the search; return best (artifact, booster) plus the trial log."""
    trials = settings.validation.optuna_trials if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError(f"invalid n_trials: {n_trials!r}")

    _validate_metadata(model_version, feature_version, parameter_version, run_id)
    data = _prepare_training(train_x, train_y, valid_x, valid_y)
    records: list[dict] = []
    best_score = -np.inf
    best_artifact: ModelArtifact | None = None
    best_booster: XGBClassifier | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_artifact, best_booster
        params = _suggest(trial)
        params["random_state"] = settings.project.random_state
        artifact, booster = _fit_prepared(
            data,
            _resolve_params(params),
            model_version,
            feature_version,
            parameter_version,
            run_id,
        )
        score = artifact.validation_rank_ic
        # Optuna rejects NaN objectives; floor to the Spearman minimum.
        score = -1.0 if not np.isfinite(score) else float(score)
        records.append({"trial": trial.number, **params, "rank_ic": score})
        # Keep the first winner on ties, matching numpy.argmax without
        # retaining every fitted model in a long search.
        if score > best_score:
            best_score, best_artifact, best_booster = score, artifact, booster
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.project.random_state),
    )
    study.optimize(objective, n_trials=trials)
    trial_log = pd.DataFrame(records, columns=["trial", *SEARCH_SPACE, "random_state", "rank_ic"])
    assert best_artifact is not None and best_booster is not None
    return best_artifact, best_booster, trial_log
