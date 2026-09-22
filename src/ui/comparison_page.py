"""P4-07: research comparison page (SDD 14.1 研究比較).

Compares baseline models, factor ablations, Top N, and cost sensitivity
from a ``get_comparison`` frame. The ``scenario`` column is required;
other columns pair as ``<metric>_before`` / ``<metric>_after`` so cost
on/off results sit side by side with a self-computed delta. A frame
with no pairs degrades to an info line instead of an exception.
"""

from __future__ import annotations

import math

import pandas as pd

_SUFFIX_BEFORE = "_before"
_SUFFIX_AFTER = "_after"


def paired_metrics(columns: list[str]) -> list[str]:
    """Return metric stems present on both cost sides, preserving order."""
    names = list(dict.fromkeys(columns))
    stems: list[str] = []
    for name in names:
        if name.endswith(_SUFFIX_BEFORE):
            stem = name[: -len(_SUFFIX_BEFORE)]
            if stem and f"{stem}{_SUFFIX_AFTER}" in names and stem not in stems:
                stems.append(stem)
    return stems


def _format(value: object) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def render_comparison(metrics: pd.DataFrame, st=None) -> None:
    """Render the scenario comparison table plus before/after deltas."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("invalid metrics: must be a DataFrame")
    if "scenario" not in metrics.columns:
        raise ValueError("invalid metrics: missing 'scenario' column")
    st.header("研究比較")
    if metrics.empty:
        st.info("尚無可比較的研究情境。")
        return
    st.dataframe(metrics.reset_index(drop=True))
    stems = paired_metrics(list(metrics.columns))
    if not stems:
        st.info("尚無成本前後對照資料。")
        return
    for _, row in metrics.iterrows():
        scenario = row["scenario"]
        st.write(f"情境：{scenario}")
        for stem in stems:
            before, after = (
                _format(row[f"{stem}{_SUFFIX_BEFORE}"]),
                _format(row[f"{stem}{_SUFFIX_AFTER}"]),
            )
            before_v, after_v = row[f"{stem}{_SUFFIX_BEFORE}"], row[f"{stem}{_SUFFIX_AFTER}"]
            if before == "N/A" or after == "N/A":
                st.write(f"{stem}: {before} → {after}")
            else:
                st.write(f"{stem}: {before} → {after}（Δ{after_v - before_v:.4f}）")
