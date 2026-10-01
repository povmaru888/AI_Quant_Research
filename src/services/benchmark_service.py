"""Build like-for-like TAIEX performance for dashboard comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def build_benchmark_payload(equity_curve: object, taiex: pd.DataFrame) -> dict:
    """Align TAIEX raw closes to strategy NAV dates and calculate matching metrics."""
    if not isinstance(equity_curve, list) or not isinstance(taiex, pd.DataFrame):
        return {}
    strategy_rows: list[tuple[str, float]] = []
    for row in equity_curve:
        if not isinstance(row, dict):
            continue
        day, nav = row.get("date"), row.get("nav")
        if not isinstance(day, str) or not day.strip():
            continue
        if isinstance(nav, bool) or not isinstance(nav, (int, float)):
            continue
        value = float(nav)
        if math.isfinite(value) and value > 0:
            strategy_rows.append((day, value))
    required = {"trade_date", "close"}
    if len(strategy_rows) < 2 or not required.issubset(taiex.columns):
        return {}

    strategy = pd.Series(
        {day: nav for day, nav in strategy_rows}, dtype="float64"
    ).sort_index()
    market = taiex.loc[:, ["trade_date", "close"]].copy()
    market["trade_date"] = market["trade_date"].astype(str)
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    market = market.reindex(strategy.index).dropna()
    market = market.loc[np.isfinite(market) & (market > 0)]
    if len(market) < 2:
        return {}

    strategy = strategy.reindex(market.index)
    benchmark_nav = market / float(market.iloc[0]) * float(strategy.iloc[0])
    returns = benchmark_nav.pct_change().dropna()
    years = len(benchmark_nav) / _TRADING_DAYS_PER_YEAR
    total_return = float(benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else float("nan")
    std = float(returns.std(ddof=1))
    downside = returns.loc[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
    drawdown = float((benchmark_nav / benchmark_nav.cummax() - 1).min())
    metrics = {
        "cagr": _finite(float(cagr)),
        "sharpe": _finite(float(returns.mean()) / std * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if std > 0
        else None,
        "sortino": _finite(
            float(returns.mean()) / downside_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )
        if downside_std > 0
        else None,
        "max_drawdown": _finite(drawdown),
    }
    curve = [
        {"date": str(day), "nav": _finite(float(nav))}
        for day, nav in benchmark_nav.items()
    ]
    monthly = []
    months = sorted({str(day)[:7] for day in benchmark_nav.index})
    index_months = benchmark_nav.index.to_series().astype(str).str[:7].to_numpy()
    for month in months:
        leg = benchmark_nav.loc[index_months == month]
        monthly.append(
            {
                "month": month,
                "return": _finite(float(leg.iloc[-1] / leg.iloc[0] - 1)),
            }
        )
    return {
        "benchmark_metrics": metrics,
        "benchmark_equity_curve": curve,
        "benchmark_monthly_returns": monthly,
    }
