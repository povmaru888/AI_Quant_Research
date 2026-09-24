"""P2-06: label service (SDD 10.1).

``future return 20d = close_adj[t+20] / close_adj[t] - 1``; the top quantile of
the cross-section is 1, the rest 0. The 20-day offset counts each
stock's own traded rows (positional), never calendar days. If any
universe stock lacks t or t+20, the whole date yields no labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
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
    missing = [c for c in ("stock_id", "trade_date", "close_adj") if c not in prices.columns]
    if missing:
        raise ValueError(f"invalid prices: missing columns {missing}")
    members = list(dict.fromkeys(stock_ids))
    if not members:
        raise ValueError("invalid stock_ids: must be non-empty")

    horizon = settings.label.horizon_trading_days
    quantile = settings.label.top_quantile
    as_of_str = as_of.isoformat()

    # Group once so each stock does not rescan the entire price table.
    target_prices = prices.loc[
        prices["stock_id"].isin(members), ["stock_id", "trade_date", "close_adj"]
    ]
    prices_by_stock = {
        stock_id: rows.sort_values("trade_date")
        for stock_id, rows in target_prices.groupby("stock_id", sort=False)
    }

    future_returns: dict[str, float] = {}
    for stock_id in members:
        rows = prices_by_stock.get(stock_id)
        if rows is None:
            return pd.Series(dtype=int, name=as_of_str)
        dates = rows["trade_date"].to_numpy()
        hit = np.flatnonzero(dates == as_of_str)
        if not len(hit):
            return pd.Series(dtype=int, name=as_of_str)
        pos = int(hit[0])
        if pos + horizon >= len(rows):
            return pd.Series(dtype=int, name=as_of_str)
        closes = pd.to_numeric(rows["close_adj"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        base = closes[pos]
        ahead = closes[pos + horizon]
        if not np.isfinite(base) or not np.isfinite(ahead) or base <= 0 or ahead <= 0:
            return pd.Series(dtype=int, name=as_of_str)
        future_returns[str(stock_id)] = ahead / base - 1

    ranks = pd.Series(future_returns).rank(pct=True)
    labels = (ranks > 1 - quantile).astype(int)
    labels.name = as_of_str
    labels.index.name = "stock_id"
    return labels
