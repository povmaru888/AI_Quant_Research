"""P2-03: point-in-time snapshot service (SDD section 6.3).

Pandas implementation of the PIT join semantics: for each stock in the
universe, keep at most one financial row (latest with
``available_date <= as_of``) and one institutional row (latest with
``trade_date <= as_of``), plus the as-of close price. Mirrors the
ordering of ``load_pit_financials`` (P1-08) without touching the DB.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from contracts import UniverseSnapshot

_FINANCIAL_KEYS = ("stock_id", "report_period", "announcement_date", "available_date")
_INSTITUTIONAL_KEYS = ("stock_id", "trade_date")
_PRICE_KEYS = ("stock_id", "trade_date", "close")


def build_pit_snapshot(
    universe: UniverseSnapshot,
    as_of: date,
    financials: pd.DataFrame,
    institutional: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Join one PIT row per universe stock; unknown future data never leaks."""
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    if universe.as_of != as_of.isoformat():
        raise ValueError(f"universe as_of {universe.as_of!r} does not match {as_of.isoformat()!r}")
    for name, frame, keys in (
        ("financials", financials, _FINANCIAL_KEYS),
        ("institutional", institutional, _INSTITUTIONAL_KEYS),
        ("prices", prices, _PRICE_KEYS),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"invalid {name}: must be a DataFrame")
        missing = [c for c in keys if c not in frame.columns]
        if missing:
            raise ValueError(f"invalid {name}: missing columns {missing}")

    as_of_str = as_of.isoformat()
    stock_ids = list(universe.included_ids)
    snapshot = pd.DataFrame({"stock_id": stock_ids})

    eligible_fin = financials.loc[financials["available_date"] <= as_of_str]
    if not eligible_fin.empty:
        latest_fin = (
            eligible_fin.sort_values(["available_date", "announcement_date"])
            .groupby("stock_id", as_index=False)
            .tail(1)
            .drop(columns=["announcement_date", "available_date"])
        )
        snapshot = snapshot.merge(latest_fin, on="stock_id", how="left")
    else:
        for column in [c for c in financials.columns if c not in ("stock_id",)]:
            snapshot[column] = pd.NA

    eligible_inst = institutional.loc[institutional["trade_date"] <= as_of_str]
    if not eligible_inst.empty:
        latest_inst = (
            eligible_inst.sort_values("trade_date")
            .groupby("stock_id", as_index=False)
            .tail(1)
            .drop(columns=["trade_date"])
        )
        snapshot = snapshot.merge(latest_inst, on="stock_id", how="left", suffixes=("", "_inst"))
    else:
        for column in [c for c in institutional.columns if c != "stock_id"]:
            snapshot[column] = pd.NA

    day_close = prices.loc[prices["trade_date"] == as_of_str, ["stock_id", "close"]]
    snapshot = snapshot.merge(day_close, on="stock_id", how="left")
    return snapshot.rename(columns={"close": "as_of_close"})
