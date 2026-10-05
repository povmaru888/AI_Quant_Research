from __future__ import annotations

import plotly.graph_objects as go

from ui.theme import (
    DARK,
    LIGHT,
    apply_plotly_theme,
    dashboard_css,
    format_value,
    get_theme,
    value_class,
)


def test_theme_tokens_and_selection() -> None:
    assert get_theme(None) is DARK
    assert get_theme("dark") is DARK
    assert get_theme("light") is LIGHT
    for theme in (DARK, LIGHT):
        assert all(
            value.startswith("#")
            for value in (
                theme.background,
                theme.surface,
                theme.border,
                theme.text,
                theme.primary,
                theme.positive,
                theme.negative,
                theme.warning,
            )
        )
        css = dashboard_css(theme)
        assert theme.background in css
        assert "@media(max-width:800px)" in css


def test_value_formatting_and_tones() -> None:
    assert format_value(0.1234, "percent") == "12.34%"
    assert format_value(1.234, "multiple") == "1.23×"
    assert format_value(None) == "N/A"
    assert value_class(0.1) == "positive"
    assert value_class(-0.1) == "negative"
    assert value_class(0) == "neutral"


def test_plotly_theme_applies_surface_and_grid() -> None:
    figure = apply_plotly_theme(go.Figure(go.Scatter(x=[1], y=[2])), DARK)
    assert figure.layout.paper_bgcolor == DARK.surface
    assert figure.layout.plot_bgcolor == DARK.surface
    assert figure.layout.xaxis.gridcolor == DARK.grid
