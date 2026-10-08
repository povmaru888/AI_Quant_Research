"""Shared visual system for the Streamlit research dashboard."""

# ruff: noqa: E501 -- CSS declarations are intentionally kept readable as whole rules.

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import plotly.graph_objects as go


@dataclass(frozen=True)
class DashboardTheme:
    name: str
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    muted: str
    primary: str
    positive: str
    negative: str
    warning: str
    benchmark: str
    grid: str


DARK = DashboardTheme(
    "dark",
    "#06090f",
    "#0e141d",
    "#111b25",
    "#263341",
    "#f3f7fa",
    "#8b98a8",
    "#27d3bd",
    "#35d07f",
    "#ff6275",
    "#f5b84b",
    "#91a0b2",
    "#23303c",
)
LIGHT = DashboardTheme(
    "light",
    "#f4f7f9",
    "#ffffff",
    "#edf3f5",
    "#d6e0e5",
    "#14202b",
    "#607080",
    "#087f72",
    "#16874f",
    "#d83b50",
    "#a66700",
    "#66788a",
    "#dce5e9",
)


def get_theme(name: str | None) -> DashboardTheme:
    return LIGHT if name == "light" else DARK


def active_theme(st=None) -> DashboardTheme:
    """Read the session theme without requiring Streamlit in unit tests."""
    state = getattr(st, "session_state", {}) if st is not None else {}
    try:
        name = state.get("dashboard_theme", "dark")
    except AttributeError:
        name = "dark"
    return get_theme(name)


def format_value(value: object, kind: str = "ratio") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "N/A"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "multiple":
        return f"{value:.2f}×"
    return f"{value:.2f}"


def value_class(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "neutral"
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"


def apply_plotly_theme(figure: go.Figure, theme: DashboardTheme) -> go.Figure:
    figure.update_layout(
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font={"color": theme.text, "family": "Inter, Noto Sans TC, sans-serif", "size": 13},
        title={"font": {"size": 17, "color": theme.text}, "x": 0.02, "xanchor": "left"},
        margin={"l": 42, "r": 24, "t": 58, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        hoverlabel={
            "bgcolor": theme.surface_alt,
            "font_color": theme.text,
            "bordercolor": theme.border,
        },
    )
    figure.update_xaxes(gridcolor=theme.grid, zerolinecolor=theme.border)
    figure.update_yaxes(gridcolor=theme.grid, zerolinecolor=theme.border)
    return figure


def dashboard_css(theme: DashboardTheme) -> str:
    return f"""
    <style>
    :root {{ --bg:{theme.background}; --surface:{theme.surface}; --surface-alt:{theme.surface_alt};
      --border:{theme.border}; --text:{theme.text}; --muted:{theme.muted}; --primary:{theme.primary};
      --positive:{theme.positive}; --negative:{theme.negative}; --warning:{theme.warning}; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background:var(--bg); color:var(--text); }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
    [data-testid="stSidebar"] * {{ color:var(--text); }}
    .block-container {{ max-width:1480px; padding-top:1.5rem; padding-bottom:3rem; }}
    h1,h2,h3 {{ color:var(--text)!important; letter-spacing:-.02em; }}
    .dashboard-hero {{ padding:1.05rem 1.2rem; border:1px solid var(--border); border-radius:16px;
      background:linear-gradient(135deg,var(--surface),var(--surface-alt)); margin-bottom:1rem; }}
    .dashboard-title {{ font-size:1.5rem; font-weight:750; color:var(--text); }}
    .dashboard-subtitle {{ color:var(--muted); margin-top:.25rem; font-size:.9rem; }}
    .tag {{ display:inline-block; padding:.22rem .58rem; margin:.55rem .3rem 0 0; border-radius:999px;
      border:1px solid color-mix(in srgb,var(--primary) 45%,var(--border)); color:var(--primary);
      background:color-mix(in srgb,var(--primary) 10%,transparent); font-size:.78rem; }}
    [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
      padding:.9rem 1rem; min-height:112px; box-shadow:0 8px 24px rgba(0,0,0,.10); }}
    [data-testid="stMetricLabel"] {{ color:var(--muted); }}
    [data-testid="stMetricValue"] {{ color:var(--text); font-weight:750; }}
    .kpi-card {{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
      padding:1rem 1.05rem; min-height:118px; }}
    .kpi-label {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.07em; }}
    .kpi-value {{ color:var(--text); font-size:1.65rem; font-weight:760; margin:.25rem 0; }}
    .kpi-detail {{ color:var(--muted); font-size:.82rem; }}
    .positive {{ color:var(--positive)!important; }} .negative {{ color:var(--negative)!important; }}
    .neutral {{ color:var(--muted)!important; }}
    .status {{ display:inline-block; padding:.3rem .65rem; border-radius:999px; border:1px solid var(--border);
      background:var(--surface); font-size:.84rem; margin-right:.35rem; }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    [data-testid="stPlotlyChart"] {{ background:var(--surface); border:1px solid var(--border);
      border-radius:16px; padding:.25rem; margin:.4rem 0 1rem; overflow:hidden; }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ background:var(--surface)!important;
      border-color:var(--border)!important; }}
    @media(max-width:800px) {{ .block-container {{ padding:1rem .75rem; }} .dashboard-title {{font-size:1.2rem;}}
      .kpi-card {{min-height:100px;}} }}
    </style>
    """


def hero_html(run_id: str, feature_version: str, method: str, year: str | None) -> str:
    tags = [feature_version, method]
    if year:
        tags.append(f"OOS {year}")
    badges = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags if tag)
    return (
        '<div class="dashboard-hero"><div class="dashboard-title">台股多因子量化研究</div>'
        f'<div class="dashboard-subtitle">目前研究：{escape(run_id)}</div>{badges}</div>'
    )


def kpi_html(label: str, value: str, detail: str = "", tone: str = "neutral") -> str:
    return (
        f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value {tone}">{escape(value)}</div>'
        f'<div class="kpi-detail">{escape(detail)}</div></div>'
    )


def comparison_kpi_html(
    label: str,
    value: str,
    benchmark: str | None,
    delta: str | None,
    delta_tone: str = "neutral",
    benchmark_label: str = "TAIEX",
) -> str:
    detail = f"{escape(benchmark_label)} {escape(benchmark)}" if benchmark is not None else ""
    if delta is not None:
        detail += f' <span class="{delta_tone}">｜差異 {escape(delta)}</span>'
    return (
        f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-detail">{detail}</div></div>'
    )
