"""Continuous monthly excess-return labels and pseudo-Huber XGBoost regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from contracts import ModelArtifact
from services.optimization_service import SEARCH_SPACE, _suggest
from services.ranking_service import _mean_group_rank_ic, _validate_groups
from services.xgb_service import (
    DEFAULT_PARAMS,
    _frame_inputs,
    _resolve_params,
    _validate_metadata,
)
from settings import Settings


def build_excess_return_labels_by_month(
    panels: Mapping[str, dict], prices: pd.DataFrame, horizon: int = 20
) -> dict[str, pd.Series]:
    """Return stock 20-row simple return minus its panel-universe mean."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"invalid horizon: {horizon!r}")
    required = {"stock_id", "trade_date", "close_adj"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"invalid prices: missing columns {sorted(missing)}")
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    clean = prices.loc[:, ["stock_id", "trade_date", "close_adj"]].copy()
    clean["stock_id"] = clean["stock_id"].astype(str)
    clean["trade_date"] = clean["trade_date"].astype(str)
    clean["close_adj"] = pd.to_numeric(clean["close_adj"], errors="coerce")
    clean = clean.sort_values(["stock_id", "trade_date"], kind="mergesort")
    for stock_id, group in clean.groupby("stock_id", sort=False):
        histories[str(stock_id)] = (
            group["trade_date"].to_numpy(dtype=str),
            group["close_adj"].to_numpy(dtype=float),
        )

    output: dict[str, pd.Series] = {}
    for month, panel in panels.items():
        signal_date = str(panel["signal_date"])
        stock_ids = panel["frame"]["stock_id"].astype(str).tolist()
        returns: dict[str, float] = {}
        for stock_id in stock_ids:
            history = histories.get(stock_id)
            if history is None:
                continue
            dates, closes = history
            position = int(np.searchsorted(dates, signal_date))
            end = position + horizon
            if position >= len(dates) or dates[position] != signal_date or end >= len(dates):
                continue
            start_close, end_close = closes[position], closes[end]
            if not np.isfinite(start_close) or not np.isfinite(end_close) or start_close <= 0:
                continue
            returns[stock_id] = float(end_close / start_close - 1.0)
        raw = pd.Series(returns, dtype=float)
        output[month] = raw - float(raw.mean()) if not raw.empty else raw
    return output


def _numeric_labels(name: str, labels: pd.Series, size: int) -> np.ndarray:
    values = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
    if len(values) != size or not np.isfinite(values).all():
        raise ValueError(f"invalid {name}: needs one finite numeric label per row")
    return values


def train_xgb_regressor(
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
) -> tuple[ModelArtifact, XGBRegressor]:
    """Fit reg:pseudohubererror and score mean monthly validation Rank IC."""
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
    resolved = _resolve_params(params)
    booster = XGBRegressor(objective="reg:pseudohubererror", **resolved)  # type: ignore[arg-type]
    booster.fit(x_train, y_train, verbose=False)
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


def optimize_xgb_regressor(
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
) -> tuple[ModelArtifact, XGBRegressor, pd.DataFrame]:
    """Tune pseudo-Huber regression against mean monthly validation Rank IC."""
    trials = settings.validation.optuna_trials if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError(f"invalid n_trials: {n_trials!r}")
    records: list[dict] = []
    best_score = -np.inf
    best_artifact: ModelArtifact | None = None
    best_booster: XGBRegressor | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_artifact, best_booster
        params = _suggest(trial)
        params["random_state"] = settings.project.random_state
        artifact, booster = train_xgb_regressor(
            train_x, train_y, train_groups, valid_x, valid_y, valid_groups, params,
            model_version, feature_version, parameter_version, run_id,
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
