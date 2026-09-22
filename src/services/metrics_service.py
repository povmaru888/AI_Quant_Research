"""P2-15: performance and sensitivity service (SDD 11.2, 13.3).

``calculate_metrics`` reduces one NAV curve (plus optional OOS rank IC
inputs) to the nine SDD metrics. ``run_cost_sensitivity`` replays the
same orders under fixed one-side slippage scenarios so cost/no-cost
results sit side by side. Annualization uses 252 trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import BacktestResult
from services.backtest_service import run_backtest
from settings import Settings

METRIC_COLUMNS: tuple[str, ...] = (
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "turnover",
    "rank_ic",
    "icir",
)

SENSITIVITY_SLIPPAGES: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.003)

_TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    result: BacktestResult,
    rank_ic: float | None = None,
    icir: float | None = None,
) -> pd.DataFrame:
    """Reduce one backtest to a single-row nine-metric frame."""
    nav = pd.to_numeric(result.nav, errors="coerce")
    if nav.isna().any() or len(nav) < 2 or (nav <= 0).any():
        raise ValueError("invalid result.nav: needs 2+ positive dated values")
    returns = nav.pct_change().dropna()
    years = len(nav) / _TRADING_DAYS_PER_YEAR
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1)
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else float("nan")
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    downside = returns.loc[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
    drawdown = float((nav / nav.cummax() - 1).min())
    turnover = _turnover(result)
    row = {
        "cagr": cagr,
        "sharpe": mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR) if std > 0 else float("nan"),
        "sortino": mean / downside_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
        if downside_std > 0
        else float("nan"),
        "calmar": cagr / abs(drawdown) if drawdown < 0 else float("nan"),
        "max_drawdown": drawdown,
        "win_rate": float((returns > 0).mean()),
        "turnover": turnover,
        "rank_ic": float(rank_ic) if rank_ic is not None else float("nan"),
        "icir": float(icir) if icir is not None else float("nan"),
    }
    return pd.DataFrame([row], columns=[*METRIC_COLUMNS])


def _turnover(result: BacktestResult) -> float:
    if result.orders.empty or "executed_price" not in result.orders.columns:
        return 0.0
    orders = result.orders
    if "target_shares" in orders.columns:
        notional = (orders["target_shares"].fillna(0).abs() * orders["executed_price"]).sum()
    else:
        notional = pd.Series(0.0, index=orders.index).sum()
    avg_nav = float(pd.to_numeric(result.nav, errors="coerce").mean())
    if not avg_nav > 0:
        return float("nan")
    return float(notional / avg_nav)


def run_cost_sensitivity(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    initial_cash: float,
    settings: Settings,
    run_id: str,
) -> pd.DataFrame:
    """Replay orders under each fixed slippage; one row per scenario."""
    if not isinstance(orders, pd.DataFrame) or orders.empty:
        raise ValueError("invalid orders: must be a non-empty DataFrame")
    required = ("stock_id", "side", "target_shares", "signal_date")
    missing = [c for c in required if c not in orders.columns]
    if missing:
        raise ValueError(f"invalid orders: missing columns {missing}")
    if "open" not in prices.columns:
        raise ValueError("invalid prices: missing 'open' column; slippage replays need opens")
    opens = prices.set_index(["trade_date", "stock_id"])["open"].apply(
        pd.to_numeric, errors="coerce"
    )

    frames: list[pd.DataFrame] = []
    for slippage in SENSITIVITY_SLIPPAGES:
        replayed = _replay_with_slippage(orders, opens, slippage, settings)
        result = run_backtest(replayed, prices, initial_cash, settings, run_id)
        metrics = calculate_metrics(result)
        metrics.insert(0, "slippage_rate", slippage)
        metrics.insert(1, "total_cost", result.total_cost)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def _replay_with_slippage(
    orders: pd.DataFrame, opens: pd.Series, slippage: float, settings: Settings
) -> pd.DataFrame:
    from services.execution_service import executed_price, transaction_cost

    replayed = orders.sort_values(["execution_date", "order_id"]).copy()
    holdings: dict[str, int] = {}
    prices, costs = [], []
    for _, row in replayed.iterrows():
        stock_id, side = str(row["stock_id"]), str(row["side"])
        open_price = float(opens.get((str(row["execution_date"]), stock_id), float("nan")))
        if not np.isfinite(open_price) or open_price <= 0:
            raise ValueError(f"missing open for order {row.get('order_id')!r}")
        price = executed_price(open_price, side, slippage)
        goal = abs(int(row["target_shares"]))
        if side == "BUY":
            shares = goal - holdings.get(stock_id, 0)
            holdings[stock_id] = holdings.get(stock_id, 0) + max(shares, 0)
        else:
            held = holdings.get(stock_id, 0)
            shares = held - goal if goal > 0 else held
            holdings[stock_id] = held - max(shares, 0)
        notional = max(shares, 0) * price
        leg = transaction_cost(notional, side, settings.execution)
        # Slippage leg follows the scenario rate, not the config rate.
        leg["slippage_cost"] = notional * slippage
        leg["total_cost"] = leg["broker_fee"] + leg["transaction_tax"] + leg["slippage_cost"]
        prices.append(price)
        costs.append(leg)
    replayed["executed_price"] = prices
    for key in ("broker_fee", "transaction_tax", "slippage_cost", "total_cost"):
        replayed[key] = [leg[key] for leg in costs]
    return replayed
