"""P4-05: model page (SDD 14.1 模型).

Renders SHAP Top 10, feature importance, monthly IC, and the prediction
distribution from a ``get_model_data`` dict. A disclaimer is always shown
first: SHAP reflects feature importance, not factor return attribution.
Each of the four sections degrades independently to an info line when
its data is missing or corrupt. Figure builders are pure (no Streamlit).
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

SHAP_DISCLAIMER = "SHAP 反映特徵重要性，不是因子收益歸因。"

_TOP_N = 10


def _clean_pairs(rows: object, name_key: str, value_key: str) -> list[tuple[str, float]]:
    cleaned: list[tuple[str, float]] = []
    if not isinstance(rows, list):
        return cleaned
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, value = row.get(name_key), row.get(value_key)
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        cleaned.append((name, float(value)))
    return cleaned


def _clean_floats(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    cleaned: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        cleaned.append(float(value))
    return cleaned


def shap_figure(shap_top: object) -> go.Figure:
    """Top-10 SHAP bar chart; corrupt rows skipped, capped at ten."""
    figure = go.Figure()
    cleaned = _clean_pairs(shap_top, "feature", "value")[:_TOP_N]
    if not cleaned:
        figure.update_layout(title="SHAP Top 10 (no data)")
        return figure
    figure.add_trace(
        go.Bar(
            x=[value for _, value in cleaned],
            y=[name for name, _ in cleaned],
            orientation="h",
            name="SHAP",
        )
    )
    figure.update_layout(title="SHAP Top 10")
    return figure


def monthly_ic_figure(monthly_ic: object) -> go.Figure:
    """Monthly IC curve; corrupt rows skipped."""
    figure = go.Figure()
    cleaned = _clean_pairs(monthly_ic, "month", "ic")
    if not cleaned:
        figure.update_layout(title="Monthly IC (no data)")
        return figure
    figure.add_trace(
        go.Scatter(
            x=[month for month, _ in cleaned],
            y=[value for _, value in cleaned],
            mode="lines+markers",
            name="IC",
        )
    )
    figure.update_layout(title="Monthly IC")
    return figure


def prediction_dist_figure(prediction_dist: object) -> go.Figure:
    """Prediction probability histogram; non-finite values skipped."""
    figure = go.Figure()
    cleaned = _clean_floats(prediction_dist)
    if not cleaned:
        figure.update_layout(title="Prediction distribution (no data)")
        return figure
    figure.add_trace(go.Histogram(x=cleaned, name="probability"))
    figure.update_layout(title="Prediction distribution")
    return figure


def render_model(model_data: dict, st=None) -> None:
    """Render the model explainability section."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(model_data, dict):
        raise ValueError("invalid model_data: must be a dict")
    st.header("模型")
    st.write(SHAP_DISCLAIMER)
    shap = _clean_pairs(model_data.get("shap_top"), "feature", "value")[:_TOP_N]
    if shap:
        st.plotly_chart(shap_figure(model_data.get("shap_top")))
    else:
        st.info("尚無 SHAP 資料。")
    importance = _clean_pairs(model_data.get("feature_importance"), "feature", "gain")
    if importance:
        st.dataframe(pd.DataFrame(importance, columns=["feature", "gain"]))
    else:
        st.info("尚無特徵重要性資料。")
    if _clean_pairs(model_data.get("monthly_ic"), "month", "ic"):
        st.plotly_chart(monthly_ic_figure(model_data.get("monthly_ic")))
    else:
        st.info("尚無月 IC 資料。")
    if _clean_floats(model_data.get("prediction_dist")):
        st.plotly_chart(prediction_dist_figure(model_data.get("prediction_dist")))
    else:
        st.info("尚無預測分布資料。")
