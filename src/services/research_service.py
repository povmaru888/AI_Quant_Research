"""P2-16: research pipeline service (SDD 14.3 monthly rotation).

Single-month research flow wiring P2-02..P2-15 through a
``ResearchStore`` seam: services stay DB-free, and the Phase 5
scheduler provides the database-backed store. Every run opens a
pipeline run first and always closes it (succeeded/failed) so results
trace to one run id (SDD 15.2). Multi-month walk-forward concatenation
stays in ``run_walk_forward`` (P2-10) for the Phase 5 scheduler.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Collection
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from contracts import BacktestResult, PortfolioTarget
from services.backtest_service import run_backtest
from services.execution_service import create_orders
from services.feature_preprocess_service import preprocess_features
from services.feature_service import calculate_raw_features
from services.label_service import build_labels
from services.optimization_service import optimize_xgb
from services.pit_service import build_pit_snapshot
from services.portfolio_service import build_target_holdings
from services.risk_service import apply_risk_controls
from services.universe_service import build_universe
from services.xgb_service import predict_xgb
from settings import Settings


class ResearchStore(Protocol):
    """Data seam between the pipeline and its persistence."""

    def load_prices(self) -> pd.DataFrame: ...
    def load_stocks(self) -> pd.DataFrame: ...
    def load_financials_snapshot(self) -> pd.DataFrame: ...
    def load_institutional_snapshot(self) -> pd.DataFrame: ...
    def load_financials_history(self) -> pd.DataFrame: ...
    def load_institutional_history(self) -> pd.DataFrame: ...
    def load_returns(self, stock_ids: Collection[str] | None = None) -> pd.DataFrame: ...
    def load_taiex(self) -> pd.DataFrame: ...
    def load_next_open(self, as_of: date) -> pd.DataFrame: ...
    def load_current_holdings(self) -> pd.DataFrame: ...
    def load_previous_positions(self) -> pd.DataFrame: ...
    def load_portfolio_value(self) -> float: ...
    def start_run(self, metadata: dict) -> None: ...
    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None: ...
    def save_predictions(self, predictions: pd.DataFrame, model_version: str) -> int: ...
    def save_target_holdings(self, target: PortfolioTarget) -> None: ...
    def save_orders(self, orders: pd.DataFrame) -> int: ...
    def load_run_summary(self, run_id: str) -> dict: ...


def _parameter_version(settings: Settings) -> str:
    """Short hash of every strategy parameter section (SDD 5.3 traceability)."""
    payload = {
        section: dataclasses.asdict(getattr(settings, section))
        for section in ("universe", "features", "label", "validation", "portfolio", "execution")
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"params_{digest[:12]}"


def run_research(
    as_of: date,
    settings: Settings,
    store: ResearchStore,
    run_id: str,
    n_trials: int | None = None,
    optimize=optimize_xgb,
) -> BacktestResult:
    """Execute one research month; always close the run, then return/raise."""
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    parameter_version = _parameter_version(settings)
    store.start_run(
        {
            "run_id": run_id,
            "data_end_date": as_of.isoformat(),
            "feature_version": settings.features.feature_version,
            "parameter_version": parameter_version,
        }
    )
    try:
        result = _execute(as_of, settings, store, run_id, parameter_version, n_trials, optimize)
    except Exception as exc:
        store.finish_run(run_id, "failed", str(exc))
        raise
    store.finish_run(run_id, "succeeded")
    return result


def _execute(
    as_of: date,
    settings: Settings,
    store: ResearchStore,
    run_id: str,
    parameter_version: str,
    n_trials: int | None,
    optimize,
) -> BacktestResult:
    prices = store.load_prices()
    universe = build_universe(prices, store.load_stocks(), as_of, settings, run_id)
    if not universe.included_ids:
        raise RuntimeError(f"empty universe for {as_of.isoformat()}; halting month")
    snapshot = build_pit_snapshot(
        universe,
        as_of,
        store.load_financials_snapshot(),
        store.load_institutional_snapshot(),
        prices,
    )
    raw = calculate_raw_features(
        snapshot,
        prices,
        as_of,
        store.load_financials_history(),
        store.load_institutional_history(),
    )
    features = preprocess_features(raw, settings, run_id, as_of)
    labels = build_labels(prices, universe.included_ids, as_of, settings)
    if labels.empty:
        raise RuntimeError(f"no labels for {as_of.isoformat()}; halting month")

    model_version = f"xgb_{as_of.strftime('%Y%m')}"
    feature_names = list(features.feature_columns)
    matrix = features.frame[feature_names]
    aligned_labels = labels.reindex(features.frame["stock_id"]).astype(int)
    train_x, train_y, valid_x, valid_y = _stratified_split(
        matrix, aligned_labels, settings.project.random_state
    )
    artifact, booster, _ = optimize(
        train_x,
        train_y,
        valid_x,
        valid_y,
        settings,
        model_version,
        features.feature_version,
        parameter_version,
        run_id,
        n_trials=n_trials,
    )
    scored = predict_xgb(artifact, features.frame, booster)
    scored["prediction_date"] = as_of.isoformat()
    store.save_predictions(scored, model_version)

    target = build_target_holdings(scored, store.load_previous_positions(), settings, run_id, as_of)
    active_ids = [
        stock_id for stock_id, action in target.actions.items() if action in ("BUY", "HOLD")
    ]
    controlled = apply_risk_controls(
        target, store.load_returns(active_ids), store.load_taiex(), as_of, settings
    )
    store.save_target_holdings(controlled)
    portfolio_value = store.load_portfolio_value()
    orders = create_orders(
        controlled,
        as_of,
        store.load_next_open(as_of),
        store.load_current_holdings(),
        portfolio_value,
        settings,
        run_id,
    )
    store.save_orders(orders)
    return run_backtest(orders, prices, portfolio_value, settings, run_id)


def _stratified_split(
    matrix: pd.DataFrame, labels: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    train_parts: list[int] = []
    valid_parts: list[int] = []
    values = labels.to_numpy(dtype=int)
    for cls in (0, 1):
        positions = np.where(values == cls)[0]
        rng.shuffle(positions)
        count = len(positions)
        n_train = min(max(1, int(0.7 * count)), max(count - 1, 1)) if count else 0
        train_parts.extend(positions[:n_train].tolist())
        valid_parts.extend(positions[n_train:].tolist())
    if not train_parts or not valid_parts:
        raise RuntimeError("single-class split; cannot train")
    train_x = matrix.iloc[train_parts].reset_index(drop=True)
    train_y = labels.iloc[train_parts].reset_index(drop=True)
    valid_x = matrix.iloc[valid_parts].reset_index(drop=True)
    valid_y = labels.iloc[valid_parts].reset_index(drop=True)
    if train_y.nunique() < 2:
        raise RuntimeError("single-class training split; cannot train")
    return train_x, train_y, valid_x, valid_y


def get_dashboard_snapshot(run_id: str, store: ResearchStore) -> dict:
    """Fetch the dashboard-facing summary for one run."""
    summary = store.load_run_summary(run_id)
    required = (
        "run_id",
        "data_end_date",
        "feature_version",
        "model_version",
        "parameter_version",
    )
    missing = [k for k in required if k not in summary]
    if missing:
        raise ValueError(f"invalid run summary: missing {missing}")
    return {
        **{key: summary[key] for key in required},
        "metrics": summary.get("metrics", {}),
        "oos_months": summary.get("oos_months", []),
    }
