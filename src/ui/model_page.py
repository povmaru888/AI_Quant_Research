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

from ui.theme import DARK, DashboardTheme, active_theme, apply_plotly_theme

SHAP_DISCLAIMER = "SHAP 反映特徵重要性，不是因子收益歸因。"
SELECTION_DISCLAIMER = (
    "以下指標只衡量模型選股排序，不代表實際持股或投資組合損益；"
    "報酬以訊號日收盤至第 20 個個股交易日的還原收盤價計算。"
)

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
            "continuous_rank_ic": _clean_number(row.get("continuous_rank_ic", row.get("ic"))),
            "top15_actual_return": _clean_number(row.get("top15_actual_return")),
            "top15_excess_return": _clean_number(row.get("top15_excess_return")),
            "top_bottom_spread": _clean_number(row.get("top_bottom_spread")),
        }
        if any(value is not None for key, value in record.items() if key != "month"):
            records.append(record)
    return records


def shap_figure(shap_top: object, theme: DashboardTheme = DARK) -> go.Figure:
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
            marker_color=theme.primary,
            hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(title="SHAP Top 10", yaxis={"autorange": "reversed"})
    return apply_plotly_theme(figure, theme)


def monthly_ic_figure(monthly_ic: object, theme: DashboardTheme = DARK) -> go.Figure:
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
            line={"color": theme.primary, "width": 2.5},
        )
    )
    figure.update_layout(title="Monthly IC")
    figure.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor=theme.border)
    return apply_plotly_theme(figure, theme)


def selection_metrics_figure(monthly_metrics: object, theme: DashboardTheme = DARK) -> go.Figure:
    """Monthly model IC and realized-return diagnostics with separate axes."""
    rows = _monthly_records(monthly_metrics)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    series = (
        ("continuous_rank_ic", "Continuous Rank IC", False, theme.benchmark),
        ("top15_excess_return", "Top15 Excess Return", True, theme.primary),
        ("top_bottom_spread", "Top-Bottom Spread", True, theme.warning),
    )
    for field, name, secondary_y, color in series:
        valid = [row for row in rows if row[field] is not None]
        if valid:
            figure.add_trace(
                go.Scatter(
                    x=[row["month"] for row in valid],
                    y=[row[field] for row in valid],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2.4},
                    marker={"size": 6},
                ),
                secondary_y=secondary_y,
            )
    if not figure.data:
        figure.update_layout(title="Monthly model selection diagnostics (no data)")
        return figure
    figure.update_layout(title="月度選股能力", legend_title_text="指標", hovermode="x unified")
    figure.update_yaxes(
        title_text="Rank IC",
        tickformat=".2f",
        zeroline=True,
        zerolinecolor=theme.border,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Simple return",
        tickformat=".1%",
        zeroline=True,
        zerolinecolor=theme.border,
        secondary_y=True,
    )
    return apply_plotly_theme(figure, theme)


def _format_value(value: object, *, percentage: bool) -> str:
    number = _clean_number(value)
    if number is None:
        return "N/A"
    return f"{number:.2%}" if percentage else f"{number:.4f}"


def _render_selection_metrics(st, payload: object, theme: DashboardTheme) -> None:
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
    figure = selection_metrics_figure(payload, theme)
    if figure.data:
        _show_chart(st, figure)
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
    _show_dataframe(st, table)


def prediction_dist_figure(prediction_dist: object, theme: DashboardTheme = DARK) -> go.Figure:
    """Prediction probability histogram; non-finite values skipped."""
    figure = go.Figure()
    cleaned = _clean_floats(prediction_dist)
    if not cleaned:
        figure.update_layout(title="Prediction distribution (no data)")
        return figure
    median = float(pd.Series(cleaned).median())
    figure.add_trace(
        go.Histogram(
            x=cleaned,
            name="probability",
            marker_color=theme.primary,
            opacity=0.82,
            hovertemplate="分數 %{x:.3f}<br>數量 %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=median, line_color=theme.warning, line_dash="dash", annotation_text=f"中位數 {median:.3f}"
    )
    figure.update_layout(title="預測分數分布", bargap=0.05)
    return apply_plotly_theme(figure, theme)


def _show_chart(st, figure: go.Figure) -> None:
    try:
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(figure)


def _show_dataframe(st, frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(frame)


def _render_model_comparison(model_data: dict, st, theme: DashboardTheme) -> None:
    primary = model_data.get("primary")
    comparison = model_data.get("comparison")
    if not isinstance(primary, dict) or not isinstance(comparison, dict):
        raise ValueError("invalid model comparison payload")
    st.header("模型比較")
    st.caption(
        f"目前：{model_data.get('primary_run_id', '')}｜"
        f"比較：{model_data.get('comparison_run_id', '')}"
    )
    st.write(SHAP_DISCLAIMER)
    first_monthly, second_monthly = primary.get("monthly_ic"), comparison.get("monthly_ic")
    first_summary = first_monthly.get("summary", {}) if isinstance(first_monthly, dict) else {}
    second_summary = (
        second_monthly.get("summary", {}) if isinstance(second_monthly, dict) else {}
    )
    for start in range(0, len(_SELECTION_SUMMARIES), 3):
        cols = st.columns(3)
        for col, (key, label, percentage) in zip(
            cols, _SELECTION_SUMMARIES[start : start + 3], strict=False
        ):
            first = _clean_number(first_summary.get(key))
            second = _clean_number(second_summary.get(key))
            delta = first - second if first is not None and second is not None else None
            col.metric(
                label,
                _format_value(first, percentage=percentage),
                delta=(
                    f"相差 {_format_value(delta, percentage=percentage)}"
                    if delta is not None
                    else None
                ),
                help=f"比較 OOS：{_format_value(second, percentage=percentage)}",
            )

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    series = (
        ("continuous_rank_ic", "Continuous IC", False),
        ("top15_excess_return", "Excess Return", True),
        ("top_bottom_spread", "Top-Bottom Spread", True),
    )
    colors = (theme.primary, theme.benchmark)
    for payload, prefix, color, dash in (
        (first_monthly, "目前", colors[0], "solid"),
        (second_monthly, "比較", colors[1], "dash"),
    ):
        rows = _monthly_records(payload)
        for field, name, secondary in series:
            valid = [row for row in rows if row[field] is not None]
            if valid:
                figure.add_trace(
                    go.Scatter(
                        x=[row["month"] for row in valid],
                        y=[row[field] for row in valid],
                        mode="lines+markers",
                        name=f"{prefix} {name}",
                        line={"color": color, "dash": dash},
                    ),
                    secondary_y=secondary,
                )
    figure.update_layout(title="月度選股能力比較", hovermode="x unified")
    figure.update_yaxes(zeroline=True, secondary_y=False)
    figure.update_yaxes(tickformat=".1%", zeroline=True, secondary_y=True)
    _show_chart(st, apply_plotly_theme(figure, theme))

    shap_first = dict(_clean_pairs(primary.get("shap_top"), "feature", "value"))
    shap_second = dict(_clean_pairs(comparison.get("shap_top"), "feature", "value"))
    shap_features = list(dict.fromkeys([*shap_first, *shap_second]))
    if shap_features:
        st.subheader("SHAP 比較")
        _show_dataframe(
            st,
            pd.DataFrame(
                {
                    "feature": shap_features,
                    "目前 OOS": [shap_first.get(name) for name in shap_features],
                    "比較 OOS": [shap_second.get(name) for name in shap_features],
                }
            ),
        )
    gain_first = dict(_clean_pairs(primary.get("feature_importance"), "feature", "gain"))
    gain_second = dict(
        _clean_pairs(comparison.get("feature_importance"), "feature", "gain")
    )
    gain_features = list(dict.fromkeys([*gain_first, *gain_second]))
    if gain_features:
        st.subheader("Feature Gain 比較")
        _show_dataframe(
            st,
            pd.DataFrame(
                {
                    "feature": gain_features,
                    "目前 OOS": [gain_first.get(name) for name in gain_features],
                    "比較 OOS": [gain_second.get(name) for name in gain_features],
                }
            ),
        )
    dist_figure = go.Figure()
    for values, name, color in (
        (primary.get("prediction_dist"), "目前 OOS", theme.primary),
        (comparison.get("prediction_dist"), "比較 OOS", theme.benchmark),
    ):
        clean = _clean_floats(values)
        if clean:
            dist_figure.add_trace(
                go.Histogram(x=clean, name=name, opacity=0.55, marker_color=color)
            )
    if dist_figure.data:
        dist_figure.update_layout(title="預測分數分布比較", barmode="overlay")
        _show_chart(st, apply_plotly_theme(dist_figure, theme))


def render_model(model_data: dict, st=None) -> None:
    """Render the model explainability section."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(model_data, dict):
        raise ValueError("invalid model_data: must be a dict")
    theme = active_theme(st)
    if model_data.get("comparison_mode") is True:
        _render_model_comparison(model_data, st, theme)
        return
    st.header("模型")
    st.write(SHAP_DISCLAIMER)
    _render_selection_metrics(st, model_data.get("monthly_ic"), theme)
    shap = _clean_pairs(model_data.get("shap_top"), "feature", "value")[:_TOP_N]
    if shap:
        st.subheader("特徵解釋")
        st.caption("SHAP 衡量個別預測的平均影響；Gain 衡量樹模型分裂時帶來的改善，兩者定義不同。")
        _show_chart(st, shap_figure(model_data.get("shap_top"), theme))
    else:
        st.info("尚無 SHAP 資料。")
    importance = _clean_pairs(model_data.get("feature_importance"), "feature", "gain")
    if importance:
        _show_dataframe(st, pd.DataFrame(importance, columns=["feature", "gain"]))
    else:
        st.info("尚無特徵重要性資料。")
    if _clean_floats(model_data.get("prediction_dist")):
        _show_chart(st, prediction_dist_figure(model_data.get("prediction_dist"), theme))
    else:
        st.info("尚無預測分布資料。")
