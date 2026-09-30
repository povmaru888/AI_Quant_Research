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
from plotly.subplots import make_subplots

SHAP_DISCLAIMER = "SHAP 反映特徵重要性，不是因子收益歸因。"
SELECTION_DISCLAIMER = "以下指標只衡量模型選股排序，不代表實際持股或投資組合損益；報酬以訊號日收盤至第 20 個個股交易日的還原收盤價計算。"

_TOP_N = 10
_SELECTION_SUMMARIES = (
    ("mean_continuous_rank_ic", "平均連續 Rank IC", False),
    ("ic_positive_month_ratio", "IC > 0 月份比例", True),
    ("mean_top15_actual_return", "Top15 平均實際報酬", True),
    ("mean_top15_excess_return", "Top15 平均 Excess Return", True),
    ("top15_excess_positive_month_ratio", "Top15 Excess > 0 月份比例", True),
    ("mean_top_bottom_spread", "平均 Top-Bottom Spread", True),
)


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


def _clean_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _monthly_records(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        if payload.get("schema_version") != 2:
            return []
        source = payload.get("monthly")
    else:
        source = payload
    if not isinstance(source, list):
        return []
    records = []
    for row in source:
        if not isinstance(row, dict):
            continue
        month = row.get("month")
        if not isinstance(month, str) or not month.strip():
            signal_date = row.get("signal_date")
            month = signal_date[:7] if isinstance(signal_date, str) else None
        if not isinstance(month, str) or not month.strip():
            continue
        record = {
            "month": month,
            "continuous_rank_ic": _clean_number(
                row.get("continuous_rank_ic", row.get("ic"))
            ),
            "top15_actual_return": _clean_number(row.get("top15_actual_return")),
            "top15_excess_return": _clean_number(row.get("top15_excess_return")),
            "top_bottom_spread": _clean_number(row.get("top_bottom_spread")),
        }
        if any(value is not None for key, value in record.items() if key != "month"):
            records.append(record)
    return records


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
    cleaned = [
        (row["month"], row["continuous_rank_ic"])
        for row in _monthly_records(monthly_ic)
        if row["continuous_rank_ic"] is not None
    ]
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


def selection_metrics_figure(monthly_metrics: object) -> go.Figure:
    """Monthly model IC and realized-return diagnostics with separate axes."""
    rows = _monthly_records(monthly_metrics)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    series = (
        ("continuous_rank_ic", "Continuous Rank IC", False),
        ("top15_excess_return", "Top15 Excess Return", True),
        ("top_bottom_spread", "Top-Bottom Spread", True),
    )
    for field, name, secondary_y in series:
        valid = [row for row in rows if row[field] is not None]
        if valid:
            figure.add_trace(
                go.Scatter(
                    x=[row["month"] for row in valid],
                    y=[row[field] for row in valid],
                    mode="lines+markers",
                    name=name,
                ),
                secondary_y=secondary_y,
            )
    if not figure.data:
        figure.update_layout(title="Monthly model selection diagnostics (no data)")
        return figure
    figure.update_layout(title="Monthly model selection diagnostics", legend_title_text="指標")
    figure.update_yaxes(title_text="Rank IC", tickformat=".2f", secondary_y=False)
    figure.update_yaxes(title_text="Simple return", tickformat=".1%", secondary_y=True)
    return figure


def _format_value(value: object, *, percentage: bool) -> str:
    number = _clean_number(value)
    if number is None:
        return "N/A"
    return f"{number:.2%}" if percentage else f"{number:.4f}"


def _render_selection_metrics(st, payload: object) -> None:
    st.subheader("模型選股能力")
    st.caption(SELECTION_DISCLAIMER)
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        summary = {}
    metrics = iter(_SELECTION_SUMMARIES)
    while batch := [next(metrics, None) for _ in range(3)]:
        batch = [item for item in batch if item is not None]
        if not batch:
            break
        columns = st.columns(3)
        for column, (key, label, percentage) in zip(columns, batch, strict=False):
            column.metric(label, _format_value(summary.get(key), percentage=percentage))

    rows = _monthly_records(payload)
    if not rows:
        st.info("尚無完整的 20 交易日選股績效資料。")
        return
    figure = selection_metrics_figure(payload)
    if figure.data:
        st.plotly_chart(figure)
    table = pd.DataFrame(
        [
            {
                "月份": row["month"],
                "Continuous IC": _format_value(row["continuous_rank_ic"], percentage=False),
                "Excess Return": _format_value(row["top15_excess_return"], percentage=True),
                "Top-Bottom Spread": _format_value(row["top_bottom_spread"], percentage=True),
            }
            for row in rows
        ],
        columns=["月份", "Continuous IC", "Excess Return", "Top-Bottom Spread"],
    )
    st.dataframe(table)


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
    _render_selection_metrics(st, model_data.get("monthly_ic"))
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
    if _clean_floats(model_data.get("prediction_dist")):
        st.plotly_chart(prediction_dist_figure(model_data.get("prediction_dist")))
    else:
        st.info("尚無預測分布資料。")
