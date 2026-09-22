"""P2-01: cross-module domain contracts.

Pure value objects shared by Phase 2 services. Every contract carries
``run_id`` and/or date fields so results trace back to one pipeline run
(SDD section 2, principle 6). All invariants raise ``ValueError``.

This module is dependency-free on purpose: it must not import
``settings`` or the ORM, so services, tests, and reports can all rely on
it without import cycles.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from numbers import Real

import pandas as pd

__all__ = [
    "ACTIONS",
    "BacktestResult",
    "FeatureSet",
    "ModelArtifact",
    "PortfolioTarget",
    "UniverseEntry",
    "UniverseSnapshot",
]

ACTIONS = ("BUY", "HOLD", "SELL", "NONE")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WEIGHT_EPS = 1e-9


def _require_run_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid run_id: must be a non-empty string, got {value!r}")
    return value


def _require_date(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ValueError(f"invalid {field_name}: must be YYYY-MM-DD, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: not a calendar date: {value!r}") from exc
    return value


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invalid {field_name}: must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"invalid {field_name}: must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class UniverseEntry:
    """One stock's keep/drop verdict for a rebalance date."""

    stock_id: str
    included: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.stock_id, str) or not self.stock_id.strip():
            raise ValueError(f"invalid stock_id: {self.stock_id!r}")
        if not isinstance(self.included, bool):
            raise ValueError(f"invalid included flag: {self.included!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("invalid reason: must be a non-empty string")


@dataclass(frozen=True)
class UniverseSnapshot:
    """Tradable universe for one rebalance date."""

    run_id: str
    as_of: str
    entries: tuple[UniverseEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_date(self.as_of, "as_of")
        entries = tuple(self.entries) if isinstance(self.entries, Sequence) else self.entries
        if not isinstance(entries, tuple) or not all(isinstance(e, UniverseEntry) for e in entries):
            raise ValueError("invalid entries: must be a sequence of UniverseEntry")
        seen = [e.stock_id for e in entries]
        if len(set(seen)) != len(seen):
            raise ValueError("invalid entries: duplicate stock_id")
        object.__setattr__(self, "entries", entries)

    @property
    def included_ids(self) -> tuple[str, ...]:
        """Stock ids that passed every universe filter."""
        return tuple(e.stock_id for e in self.entries if e.included)

    @property
    def excluded_ids(self) -> tuple[str, ...]:
        """Stock ids rejected by at least one filter."""
        return tuple(e.stock_id for e in self.entries if not e.included)


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """One month of processed features plus coverage diagnostics."""

    run_id: str
    as_of: str
    feature_version: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    coverage: dict[str, float]
    missing_flag_column: str = "missing_flag"

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_date(self.as_of, "as_of")
        if not isinstance(self.feature_version, str) or not self.feature_version.strip():
            raise ValueError(f"invalid feature_version: {self.feature_version!r}")
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            raise ValueError("invalid frame: must be a non-empty DataFrame")
        if "stock_id" not in self.frame.columns:
            raise ValueError("invalid frame: missing 'stock_id' column")
        if self.missing_flag_column not in self.frame.columns:
            raise ValueError(f"invalid frame: missing flag column {self.missing_flag_column!r}")
        columns = tuple(self.feature_columns)
        if not columns:
            raise ValueError("invalid feature_columns: must be non-empty")
        if len(set(columns)) != len(columns):
            raise ValueError("invalid feature_columns: duplicate column")
        unknown = [c for c in columns if c not in self.frame.columns]
        if unknown:
            raise ValueError(f"invalid feature_columns: not in frame: {unknown}")
        object.__setattr__(self, "feature_columns", columns)
        if not isinstance(self.coverage, Mapping) or set(self.coverage) != set(columns):
            raise ValueError("invalid coverage: keys must equal feature_columns")
        for name, rate in self.coverage.items():
            rate_value = _require_finite_number(rate, f"coverage[{name!r}]")
            if not 0.0 <= rate_value <= 1.0:
                raise ValueError(f"invalid coverage[{name!r}]: must be in [0, 1]")


@dataclass(frozen=True)
class ModelArtifact:
    """Trained classifier plus the metadata needed to reproduce it (SDD 10.2)."""

    run_id: str
    model_version: str
    feature_version: str
    parameter_version: str
    feature_columns: tuple[str, ...]
    best_params: dict
    validation_rank_ic: float

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        for field_name in ("model_version", "feature_version", "parameter_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {field_name}: {value!r}")
        columns = tuple(self.feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("invalid feature_columns: must be non-empty and unique")
        object.__setattr__(self, "feature_columns", columns)
        if not isinstance(self.best_params, dict):
            raise ValueError("invalid best_params: must be a dict")
        object.__setattr__(
            self,
            "validation_rank_ic",
            _require_finite_number(self.validation_rank_ic, "validation_rank_ic"),
        )


@dataclass(frozen=True)
class PortfolioTarget:
    """Target holdings for one signal date (SDD 12.1 rank buffer)."""

    run_id: str
    signal_date: str
    top_n: int
    actions: dict[str, str]
    weights: dict[str, float]
    cash_weight: float
    equity_exposure: float

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_date(self.signal_date, "signal_date")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int):
            raise ValueError(f"invalid top_n: must be an int, got {self.top_n!r}")
        if self.top_n <= 0:
            raise ValueError(f"invalid top_n: must be positive, got {self.top_n!r}")
        if not isinstance(self.actions, Mapping) or not self.actions:
            raise ValueError("invalid actions: must be a non-empty mapping")
        for stock_id, action in self.actions.items():
            if not isinstance(stock_id, str) or not stock_id.strip():
                raise ValueError(f"invalid actions key: {stock_id!r}")
            if action not in ACTIONS:
                raise ValueError(f"invalid action for {stock_id!r}: {action!r}")
        if not isinstance(self.weights, Mapping):
            raise ValueError("invalid weights: must be a mapping")
        checked: dict[str, float] = {}
        for stock_id, weight in self.weights.items():
            if not isinstance(stock_id, str) or not stock_id.strip():
                raise ValueError(f"invalid weights key: {stock_id!r}")
            value = _require_finite_number(weight, f"weights[{stock_id!r}]")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"invalid weights[{stock_id!r}]: must be in (0, 1]")
            checked[stock_id] = value
        active = {s for s, a in self.actions.items() if a in ("BUY", "HOLD")}
        if set(checked) != active:
            raise ValueError("invalid weights: keys must equal BUY/HOLD stocks")
        exposure = _require_finite_number(self.equity_exposure, "equity_exposure")
        cash = _require_finite_number(self.cash_weight, "cash_weight")
        if not 0.0 <= exposure <= 1.0:
            raise ValueError("invalid equity_exposure: must be in [0, 1]")
        if not 0.0 <= cash <= 1.0:
            raise ValueError("invalid cash_weight: must be in [0, 1]")
        if abs(sum(checked.values()) - exposure) > _WEIGHT_EPS:
            raise ValueError("invalid weights: sum must equal equity_exposure")
        if abs(cash - (1.0 - exposure)) > _WEIGHT_EPS:
            raise ValueError("invalid cash_weight: must equal 1 - equity_exposure")


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Auditable outcome of one simulated run (SDD 13.3 traceability)."""

    run_id: str
    start_date: str
    end_date: str
    initial_cash: float
    nav: pd.Series
    orders: pd.DataFrame
    total_cost: float

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_date(self.start_date, "start_date")
        _require_date(self.end_date, "end_date")
        if self.start_date > self.end_date:
            raise ValueError("invalid dates: start_date must not exceed end_date")
        cash = _require_finite_number(self.initial_cash, "initial_cash")
        if cash <= 0:
            raise ValueError("invalid initial_cash: must be positive")
        if not isinstance(self.nav, pd.Series) or self.nav.empty:
            raise ValueError("invalid nav: must be a non-empty Series")
        if self.nav.isna().any():
            raise ValueError("invalid nav: must not contain NaN")
        if (self.nav < 0).any():
            raise ValueError("invalid nav: long-only NAV must not be negative")
        if not isinstance(self.orders, pd.DataFrame):
            raise ValueError("invalid orders: must be a DataFrame")
        if not self.orders.empty:
            missing = {"order_id", "executed_price", "total_cost"} - set(self.orders.columns)
            if missing:
                raise ValueError(f"invalid orders: missing columns {sorted(missing)}")
        cost = _require_finite_number(self.total_cost, "total_cost")
        if cost < 0:
            raise ValueError("invalid total_cost: must be >= 0")
