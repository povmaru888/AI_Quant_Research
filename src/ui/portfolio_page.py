"""P4-04: portfolio page (SDD 14.1 投組).

Renders stock id, Chinese stock name, rank, predicted probability, weight,
volatility, and beta from a ``get_holdings`` frame, plus the weight
total and holding count for the selected run.
Missing columns or non-finite weights fail fast; an empty frame is a
legal "no positions" state and degrades to an info line.
"""

from __future__ import annotations

import math

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "stock_name",
    "rank",
    "prediction_probability",
    "weight",
    "volatility_60d",
    "beta_60d",
)


def summarize_holdings(holdings: pd.DataFrame) -> dict:
    """Return ``{count, total_weight}`` for a validated holdings frame."""
    _validate(holdings)
    weights = [float(w) for w in holdings["weight"].tolist()]
    return {"count": int(len(holdings)), "total_weight": float(sum(weights))}


def render_portfolio(holdings: pd.DataFrame, st=None) -> None:
    """Render the holdings table plus weight/count consistency lines."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(holdings, pd.DataFrame):
        raise ValueError("invalid holdings: must be a DataFrame")
    if holdings.empty:
        st.info("此截止日無持股。")
        return
    _validate(holdings)
    active = holdings.loc[holdings["weight"].astype(float) > 0]
    if active.empty:
        st.info("此截止日無持股。")
        return
    ordered = active.sort_values("rank", ignore_index=True)
    st.header("投組")
    table = ordered.loc[:, list(REQUIRED_COLUMNS)].rename(columns={"stock_name": "股票名稱"})
    st.dataframe(table)
    summary = summarize_holdings(ordered)
    st.write(f"權重總和：{summary['total_weight']:.4f}")
    st.write(f"股票數量：{summary['count']}")


def _validate(holdings: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in holdings.columns]
    if missing:
        raise ValueError(f"invalid holdings: missing columns {missing}")
    for value in holdings["weight"].tolist():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid holdings: bad weight {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"invalid holdings: non-finite weight {value!r}")
