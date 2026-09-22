"""P2-06: label service (SDD 10.1).

``future return 20d = close[t+20] / close[t] - 1``; the top quantile of
the cross-section is 1, the rest 0. The 20-day offset counts each
stock's own traded rows (positional), never calendar days. If any
universe stock lacks t or t+20, the whole date yields no labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from settings import Settings


def build_labels(
    prices: pd.DataFrame,
    stock_ids: Sequence[str],
    as_of: date,
    settings: Settings,
) -> pd.Series:
    """Build 0/1 labels for one signal date; empty Series when unbuildable."""
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("invalid prices: must be a DataFrame")
    missing = [c for c in ("stock_id", "trade_date", "close") if c not in prices.columns]
    if missing:
        raise ValueError(f"invalid prices: missing columns {missing}")
    members = list(dict.fromkeys(stock_ids))
    if not members:
        raise ValueError("invalid stock_ids: must be non-empty")

    horizon = settings.label.horizon_trading_days
    quantile = settings.label.top_quantile
    as_of_str = as_of.isoformat()

    future_returns: dict[str, float] = {}
    for stock_id in members:
        rows = (
            prices.loc[prices["stock_id"] == stock_id]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        hit = rows.index[rows["trade_date"] == as_of_str]
        if len(hit) == 0:
            return pd.Series(dtype=int, name=as_of_str)
        pos = int(hit[0])
        if pos + horizon >= len(rows):
            return pd.Series(dtype=int, name=as_of_str)
        base = float(rows.iloc[pos]["close"])
        ahead = float(rows.iloc[pos + horizon]["close"])
        if not base > 0 or not ahead > 0:
            return pd.Series(dtype=int, name=as_of_str)
        future_returns[str(stock_id)] = ahead / base - 1

    ranks = pd.Series(future_returns).rank(pct=True)
    labels = (ranks > 1 - quantile).astype(int)
    labels.name = as_of_str
    labels.index.name = "stock_id"
    return labels
