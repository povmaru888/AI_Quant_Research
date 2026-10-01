"""P4-03: overview page (SDD 14.1 總覽).

Renders CAGR/Sharpe/Sortino/MDD/Turnover/IC plus the equity curve,
drawdown, and monthly-return heatmap from a ``get_overview`` snapshot.
Missing metrics show ``N/A``; missing or corrupt curve data degrades to
an info line instead of raising a chart exception. Figure builders are
pure (no Streamlit) so tests assert on traces, not screenshots.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("cagr", "CAGR"),
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("max_drawdown", "MDD"),
    ("turnover", "Turnover"),
    ("rank_ic", "IC"),
)
BENCHMARK_METRICS = frozenset({"cagr", "sharpe", "sortino", "max_drawdown"})


def _clean_points(points: object) -> list[tuple[str, float]]:
    cleaned: list[tuple[str, float]] = []
    if not isinstance(points, list):
        return cleaned
    for point in points:
        if not isinstance(point, dict):
            continue
        day, nav = point.get("date"), point.get("nav")
        if not isinstance(day, str) or not day.strip():
            continue
        if isinstance(nav, bool) or not isinstance(nav, (int, float)):
            continue
        if not math.isfinite(nav) or nav <= 0:
            continue
        cleaned.append((day, float(nav)))
    return cleaned


def equity_curve_figure(points: object, benchmark: object = None) -> go.Figure:
    """Strategy and TAIEX NAV curves on the same starting capital."""
    cleaned = _clean_points(points)
    figure = go.Figure()
    if not cleaned:
        figure.update_layout(title="Equity curve (no data)")
        return figure
    days = [day for day, _ in cleaned]
    navs = [nav for _, nav in cleaned]
    figure.add_trace(go.Scatter(x=days, y=navs, mode="lines", name="策略 NAV"))
    market = _clean_points(benchmark)
    if market:
        figure.add_trace(
            go.Scatter(
                x=[day for day, _ in market],
                y=[nav for _, nav in market],
                mode="lines",
                name="TAIEX NAV",
            )
        )
    figure.update_layout(title="NAV：策略 vs TAIEX")
    return figure


def drawdown_figure(points: object, benchmark: object = None) -> go.Figure:
    """Strategy and TAIEX drawdowns recomputed from their NAV curves."""
    cleaned = _clean_points(points)
    figure = go.Figure()
    if not cleaned:
        figure.update_layout(title="Drawdown (no data)")
        return figure
    days = [day for day, _ in cleaned]
    peak = 0.0
    drawdowns: list[float] = []
    for _, nav in cleaned:
        peak = max(peak, nav)
        drawdowns.append(nav / peak - 1.0)
    figure.add_trace(go.Scatter(x=days, y=drawdowns, mode="lines", name="策略 Drawdown"))
    market = _clean_points(benchmark)
    if market:
        market_peak = 0.0
        market_drawdowns: list[float] = []
        for _, nav in market:
            market_peak = max(market_peak, nav)
            market_drawdowns.append(nav / market_peak - 1.0)
        figure.add_trace(
            go.Scatter(
                x=[day for day, _ in market],
                y=market_drawdowns,
                mode="lines",
                name="TAIEX Drawdown",
            )
        )
    figure.update_layout(title="Drawdown：策略 vs TAIEX")
    return figure


def _clean_monthly(monthly: object) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(monthly, list):
        for entry in monthly:
            if not isinstance(entry, dict):
                continue
            month, value = entry.get("month"), entry.get("return")
            if not isinstance(month, str) or not month.strip():
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            rows.append((month, float(value)))
    return rows


def monthly_heatmap_figure(monthly: object, benchmark: object = None) -> go.Figure:
    """Strategy and TAIEX monthly-return heatmap."""
    rows = _clean_monthly(monthly)
    figure = go.Figure()
    if not rows:
        figure.update_layout(title="Monthly returns (no data)")
        return figure
    market_by_month = dict(_clean_monthly(benchmark))
    months = [month for month, _ in rows]
    z = [[value for _, value in rows]]
    labels = ["策略"]
    if market_by_month:
        z.append([market_by_month.get(month) for month in months])
        labels.append("TAIEX")
    figure.add_trace(
        go.Heatmap(
            x=months,
            y=labels,
            z=z,
            name="Monthly returns",
        )
    )
    figure.update_layout(title="每月報酬：策略 vs TAIEX")
    return figure


def _format_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def render_overview(snapshot: dict, st=None) -> None:
    """Render the overview section for one snapshot."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(snapshot, dict):
        raise ValueError("invalid snapshot: must be a dict")
    run_id = snapshot.get("run_id", "n/a")
    st.header(f"總覽（{run_id}）")
    metrics = snapshot.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    benchmark_metrics = snapshot.get("benchmark_metrics", {})
    if not isinstance(benchmark_metrics, dict):
        benchmark_metrics = {}
    for key, label in METRIC_LABELS:
        strategy_value = _format_metric(metrics.get(key))
        if key in BENCHMARK_METRICS:
            market_value = _format_metric(benchmark_metrics.get(key))
            st.write(f"{label}: 策略 {strategy_value}｜TAIEX {market_value}")
        else:
            st.write(f"{label}: {strategy_value}")
    equity = snapshot.get("equity_curve")
    benchmark_equity = snapshot.get("benchmark_equity_curve")
    if _clean_points(equity):
        st.plotly_chart(equity_curve_figure(equity, benchmark_equity))
        st.plotly_chart(drawdown_figure(equity, benchmark_equity))
    else:
        st.info("尚無淨值曲線資料。")
    monthly = snapshot.get("monthly_returns")
    rows = monthly if isinstance(monthly, list) and monthly else None
    if rows:
        figure = monthly_heatmap_figure(rows, snapshot.get("benchmark_monthly_returns"))
        if figure.data:
            st.plotly_chart(figure)
        else:
            st.info("月報酬資料格式不正確。")
    else:
        st.info("尚無月報酬資料。")
