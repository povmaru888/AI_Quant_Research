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


def render_portfolio(holdings: pd.DataFrame | dict, st=None) -> None:
    """Render the holdings table plus weight/count consistency lines."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if isinstance(holdings, dict) and holdings.get("comparison_mode") is True:
        st.header("投組比較")
        st.subheader(f"目前 OOS：{holdings.get('primary_run_id', '')}")
        render_portfolio(holdings.get("primary"), st)
        divider = getattr(st, "divider", None)
        if callable(divider):
            divider()
        st.subheader(f"比較 OOS：{holdings.get('comparison_run_id', '')}")
        render_portfolio(holdings.get("comparison"), st)
        return
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
    summary = summarize_holdings(ordered)
    columns_fn = getattr(st, "columns", None)
    if callable(columns_fn):
        metrics = (
            ("持股數", str(summary["count"])),
            ("權重總和", f"{summary['total_weight']:.2%}"),
            ("平均 60D 波動", f"{ordered['volatility_60d'].astype(float).mean():.2%}"),
            ("平均 Beta", f"{ordered['beta_60d'].astype(float).mean():.2f}"),
        )
        for col, (label, value) in zip(columns_fn(4), metrics, strict=False):
            col.metric(label, value)
    table = ordered.loc[:, list(REQUIRED_COLUMNS)].rename(
        columns={
            "stock_id": "股票代碼",
            "stock_name": "股票名稱",
            "rank": "排名",
            "prediction_probability": "模型分數",
            "weight": "權重",
            "volatility_60d": "60D 波動",
            "beta_60d": "60D Beta",
        }
    )
    table["權重"] = table["權重"].astype(float) * 100
    table["60D 波動"] = table["60D 波動"].astype(float) * 100
    try:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "排名": st.column_config.NumberColumn(format="%d"),
                "模型分數": st.column_config.NumberColumn(format="%.4f"),
                "權重": st.column_config.ProgressColumn(
                    format="%.2f%%", min_value=0, max_value=100
                ),
                "60D 波動": st.column_config.NumberColumn(format="%.2f%%"),
                "60D Beta": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    except (TypeError, AttributeError):
        st.dataframe(table)
    if not callable(columns_fn):
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
