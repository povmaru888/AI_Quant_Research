"""P4-02: dashboard read service (SDD 14.1).

DB-free read layer between the Streamlit shell/pages and persistence.
Every reader first gates on ``get_run_status`` so only completed
(``succeeded``) runs are ever shown. Return values are plain dicts and
DataFrames with explicit fields: ORM entities, unknown keys, and NaN
never leak to the UI (NaN -> None, fixed column order, reset index).
"""

from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

SUCCEEDED = "succeeded"

OVERVIEW_KEYS: tuple[str, ...] = (
    "run_id",
    "data_end_date",
    "feature_version",
    "model_version",
    "parameter_version",
)

OVERVIEW_OPTIONAL_KEYS: tuple[str, ...] = ("equity_curve", "monthly_returns")

HOLDING_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "rank",
    "prediction_probability",
    "weight",
    "volatility_60d",
    "beta_60d",
)

MODEL_KEYS: tuple[str, ...] = (
    "shap_top",
    "feature_importance",
    "monthly_ic",
    "prediction_dist",
)

RISK_KEYS: tuple[str, ...] = (
    "equity_exposure",
    "predicted_volatility",
    "realized_volatility",
    "max_drawdown",
    "turnover",
    "market_regime",
    "exposure_cap",
)


class DashboardStore(Protocol):
    """Persistence seam for dashboard reads (Phase 5 implements with DB)."""

    def get_run_status(self, run_id: str) -> str: ...
    def load_run_summary(self, run_id: str) -> dict: ...
    def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame: ...
    def load_model_data(self, run_id: str) -> dict: ...
    def load_risk(self, run_id: str) -> dict: ...
    def load_comparison(self, run_id: str) -> pd.DataFrame: ...


def _require_completed(store: DashboardStore, run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: must be a non-empty string, got {run_id!r}")
    try:
        status = store.get_run_status(run_id)
    except Exception as exc:
        raise ValueError(f"invalid run_id: unknown run {run_id!r}") from exc
    if status != SUCCEEDED:
        raise ValueError(f"invalid run_id: run {run_id!r} is not completed ({status!r})")


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def get_overview(run_id: str, store: DashboardStore) -> dict:
    """Return the overview payload for one completed run."""
    _require_completed(store, run_id)
    summary = store.load_run_summary(run_id)
    if not isinstance(summary, dict):
        raise ValueError("invalid run summary: must be a dict")
    missing = [k for k in OVERVIEW_KEYS if k not in summary]
    if missing:
        raise ValueError(f"invalid run summary: missing {missing}")
    payload = {key: summary[key] for key in OVERVIEW_KEYS}
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("invalid run summary: 'metrics' must be a dict")
    payload["metrics"] = {key: _clean(value) for key, value in metrics.items()}
    oos_months = summary.get("oos_months", [])
    if not isinstance(oos_months, list):
        raise ValueError("invalid run summary: 'oos_months' must be a list")
    payload["oos_months"] = oos_months
    for key in OVERVIEW_OPTIONAL_KEYS:
        payload[key] = summary.get(key)
    return payload


def get_holdings(run_id: str, as_of: str, store: DashboardStore) -> pd.DataFrame:
    """Return holdings with explicit columns; never ORM entities."""
    _require_completed(store, run_id)
    if not isinstance(as_of, str):
        raise ValueError(f"invalid as_of: must be a string, got {as_of!r}")
    frame = store.load_holdings(run_id, as_of)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("invalid holdings: must be a DataFrame")
    if "stock_id" not in frame.columns and "stock_id" in frame.index.names:
        frame = frame.reset_index()
    missing = [c for c in HOLDING_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"invalid holdings: missing columns {missing}")
    return frame.loc[:, list(HOLDING_COLUMNS)].reset_index(drop=True)


def get_model_data(run_id: str, store: DashboardStore) -> dict:
    """Return model explainability payload for one completed run."""
    _require_completed(store, run_id)
    data = store.load_model_data(run_id)
    if not isinstance(data, dict):
        raise ValueError("invalid model data: must be a dict")
    return {key: data.get(key) for key in MODEL_KEYS}


def get_risk(run_id: str, store: DashboardStore) -> dict:
    """Return risk payload for one completed run."""
    _require_completed(store, run_id)
    data = store.load_risk(run_id)
    if not isinstance(data, dict):
        raise ValueError("invalid risk data: must be a dict")
    missing = [k for k in RISK_KEYS if k not in data]
    if missing:
        raise ValueError(f"invalid risk data: missing {missing}")
    return {key: _clean(data[key]) for key in RISK_KEYS}


def get_comparison(run_id: str, store: DashboardStore) -> pd.DataFrame:
    """Return scenario comparison with cost on/off side by side."""
    _require_completed(store, run_id)
    frame = store.load_comparison(run_id)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("invalid comparison: must be a DataFrame")
    if "scenario" not in frame.columns:
        raise ValueError("invalid comparison: missing 'scenario' column")
    return frame.reset_index(drop=True)
