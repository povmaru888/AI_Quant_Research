"""P2-12: weight and market risk service (SDD 12.2).

Inverse-volatility relative weights first; total exposure from the
portfolio covariance and the volatility target; per-name cap with
iterative redistribution; TAIEX MA60 regime filter last. Unmeasurable
risk means no position: affected names flip to SELL. Unusable
covariance raises RiskFailure (a ValueError) so the pipeline can fail
the run loudly instead of allocating blind (SDD section 16).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from contracts import PortfolioTarget
from settings import Settings

_TRADING_DAYS_PER_YEAR = 252
_RISK_WINDOW = 60
_MIN_OVERLAP = 30


class RiskFailure(ValueError):
    """Raised when portfolio risk cannot be estimated."""


def apply_risk_controls(
    target: PortfolioTarget,
    returns: pd.DataFrame,
    taiex: pd.DataFrame,
    signal_date: date,
    settings: Settings,
) -> PortfolioTarget:
    """Reshape target weights under volatility, cap, and regime controls."""
    if not isinstance(signal_date, date):
        raise ValueError(f"invalid signal_date: must be a date, got {signal_date!r}")
    if signal_date.isoformat() != target.signal_date:
        raise ValueError("signal_date does not match target.signal_date")
    for name, frame, columns in (
        ("returns", returns, ("stock_id", "trade_date", "log_return")),
        ("taiex", taiex, ("trade_date", "close")),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"invalid {name}: must be a DataFrame")
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"invalid {name}: missing columns {missing}")

    active = [s for s, action in target.actions.items() if action in ("BUY", "HOLD")]
    if not active:
        return target

    portfolio = settings.portfolio
    signal_day = signal_date.isoformat()
    active_returns = returns.loc[
        (returns["trade_date"] <= signal_day) & returns["stock_id"].isin(active)
    ]
    volatilities: dict[str, float] = {}
    for stock_id in active:
        series = _window(active_returns, stock_id, signal_day)
        if series is None:
            continue
        sigma = float(series.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if np.isfinite(sigma) and sigma > 0:
            volatilities[stock_id] = sigma

    actions = dict(target.actions)
    for stock_id in active:
        if stock_id not in volatilities:
            actions[stock_id] = "SELL"
    active = [s for s in active if s in volatilities]
    if not active:
        return PortfolioTarget(
            run_id=target.run_id,
            signal_date=target.signal_date,
            top_n=target.top_n,
            actions=actions,
            weights={},
            cash_weight=1.0,
            equity_exposure=0.0,
        )

    raw = {s: 1.0 / volatilities[s] for s in active}
    total = sum(raw.values())
    relative = {s: w / total for s, w in raw.items()}
    covariance = _covariance(active_returns, active, signal_day)
    weights_vector = np.array([relative[s] for s in active])
    portfolio_variance = float(weights_vector @ covariance @ weights_vector)
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
        raise RiskFailure("unusable covariance: non-positive portfolio variance")
    sigma_p = float(np.sqrt(portfolio_variance * _TRADING_DAYS_PER_YEAR))
    exposure = min(portfolio.max_equity_exposure, portfolio.target_annual_volatility / sigma_p)

    capped = _apply_cap(relative, portfolio.max_individual_weight)
    allocated = sum(capped.values())
    # When N * cap < 1 no redistribution can place the residual: it stays cash.
    exposure = exposure * allocated
    exposure = _apply_regime(taiex, signal_date.isoformat(), exposure, settings)
    weights = {s: capped[s] / allocated * exposure for s in active}
    return PortfolioTarget(
        run_id=target.run_id,
        signal_date=target.signal_date,
        top_n=target.top_n,
        actions=actions,
        weights=weights,
        cash_weight=1.0 - exposure,
        equity_exposure=exposure,
    )


def _window(returns: pd.DataFrame, stock_id: str, signal_date: str) -> pd.Series | None:
    rows = returns.loc[
        (returns["stock_id"] == stock_id) & (returns["trade_date"] <= signal_date)
    ].sort_values("trade_date")
    if len(rows) < _RISK_WINDOW:
        return None
    series = pd.to_numeric(rows["log_return"].tail(_RISK_WINDOW), errors="coerce")
    if series.isna().any():
        return None
    return series


def _covariance(returns: pd.DataFrame, active: list[str], signal_date: str) -> np.ndarray:
    eligible = returns.loc[returns["trade_date"] <= signal_date].sort_values("trade_date")
    pivot = (
        eligible.loc[eligible["stock_id"].isin(active)]
        .pivot_table(index="trade_date", columns="stock_id", values="log_return", aggfunc="last")
        .tail(_RISK_WINDOW)
    )
    pivot = pivot.reindex(columns=active)
    if len(pivot) < _MIN_OVERLAP or pivot.isna().any().any():
        raise RiskFailure(f"insufficient overlapping returns: {len(pivot)} rows")
    matrix = pivot.to_numpy(dtype=float)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    if not np.all(np.isfinite(covariance)):
        raise RiskFailure("non-finite covariance entries")
    return np.atleast_2d(covariance)


def _apply_cap(relative: dict[str, float], cap: float) -> dict[str, float]:
    weights = dict(relative)
    if len(weights) * cap < 1:
        # Infeasible: even full capacity cannot place 1.0. Cap everyone and
        # leave the residual to cash via the allocated scaler upstream.
        return {stock_id: cap for stock_id in weights}
    for _ in range(len(weights) + 1):
        over = [s for s, w in weights.items() if w > cap]
        if not over:
            return weights
        surplus = sum(weights[s] - cap for s in over)
        for stock_id in over:
            weights[stock_id] = cap
        room = [s for s in weights if s not in over]
        room_total = sum(weights[s] for s in room)
        if room_total <= 0:
            return weights
        for stock_id in room:
            weights[stock_id] += surplus * weights[stock_id] / room_total
    return weights


def _apply_regime(
    taiex: pd.DataFrame, signal_date: str, exposure: float, settings: Settings
) -> float:
    window = settings.portfolio.taiex_ma_window
    eligible = taiex.loc[taiex["trade_date"] <= signal_date].sort_values("trade_date")
    if len(eligible) < window:
        raise RiskFailure(f"insufficient TAIEX history: {len(eligible)} < {window}")
    closes = pd.to_numeric(eligible["close"].tail(window), errors="coerce")
    if closes.isna().any() or (closes <= 0).any():
        raise RiskFailure("invalid TAIEX closes in MA window")
    if float(closes.iloc[-1]) <= float(closes.mean()):
        return min(exposure, settings.portfolio.reduced_exposure_below_ma)
    return exposure
