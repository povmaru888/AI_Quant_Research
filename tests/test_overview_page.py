"""P4-03 acceptance: overview page (fake st, real figures)."""

from __future__ import annotations

import pytest

from ui.overview_page import (
    drawdown_figure,
    equity_curve_figure,
    monthly_heatmap_figure,
    render_overview,
)


class FakeSt:
    def __init__(self) -> None:
        self.calls: list = []

    def header(self, text) -> None:
        self.calls.append(("header", text))

    def write(self, text) -> None:
        self.calls.append(("write", text))

    def info(self, text) -> None:
        self.calls.append(("info", text))

    def plotly_chart(self, figure) -> None:
        self.calls.append(("plotly_chart", figure))


def _snapshot(**overrides):
    base = {
        "run_id": "run-001",
        "metrics": {
            "cagr": 0.1234,
            "sharpe": 1.5,
            "sortino": None,
            "max_drawdown": -0.1,
            "turnover": float("nan"),
            "rank_ic": 0.05,
        },
        "equity_curve": [
            {"date": "2020-02-03", "nav": 1.0},
            {"date": "2020-02-04", "nav": 1.02},
            {"date": "2020-02-05", "nav": 1.01},
        ],
        "monthly_returns": [{"month": "2020-02", "return": 0.01}],
        "benchmark_metrics": {
            "cagr": 0.08,
            "sharpe": 0.9,
            "sortino": 1.1,
            "max_drawdown": -0.2,
        },
        "benchmark_equity_curve": [
            {"date": "2020-02-03", "nav": 1.0},
            {"date": "2020-02-04", "nav": 0.99},
            {"date": "2020-02-05", "nav": 1.03},
        ],
        "benchmark_monthly_returns": [{"month": "2020-02", "return": 0.03}],
    }
    base.update(overrides)
    return base


def test_render_full_snapshot() -> None:
    st = FakeSt()
    render_overview(_snapshot(), st=st)
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert "CAGR: 策略 0.1234｜TAIEX 0.0800" in writes
    assert "Sharpe: 策略 1.5000｜TAIEX 0.9000" in writes
    assert "Sortino: 策略 N/A｜TAIEX 1.1000" in writes
    assert "Turnover: N/A" in writes
    charts = [c[1] for c in st.calls if c[0] == "plotly_chart"]
    assert len(charts) == 3
    assert [trace.name for trace in charts[0].data] == ["策略 NAV", "TAIEX NAV"]
    assert list(charts[0].data[0].y) == [1.0, 1.02, 1.01]
    assert [trace.name for trace in charts[1].data] == [
        "策略 Drawdown",
        "TAIEX Drawdown",
    ]
    assert list(charts[2].data[0].y) == ["策略", "TAIEX"]


def test_render_missing_values_degrade() -> None:
    st = FakeSt()
    render_overview({"run_id": "run-002"}, st=st)
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert writes and all(value.endswith("N/A") for value in writes)
    assert [c for c in st.calls if c[0] == "plotly_chart"] == []
    infos = [c[1] for c in st.calls if c[0] == "info"]
    assert len(infos) == 2

    st = FakeSt()
    render_overview(_snapshot(metrics=None, equity_curve="bad", monthly_returns=[]), st=st)
    assert [c for c in st.calls if c[0] == "plotly_chart"] == []

    with pytest.raises(ValueError, match="snapshot"):
        render_overview([], st=st)


def test_drawdown_recomputed_from_nav() -> None:
    figure = drawdown_figure(
        [
            {"date": "d1", "nav": 1.0},
            {"date": "d2", "nav": 1.1},
            {"date": "d3", "nav": 0.99},
        ]
    )
    assert figure.data[0].name == "策略 Drawdown"
    assert list(figure.data[0].y) == pytest.approx([0.0, 0.0, 0.99 / 1.1 - 1.0])


def test_builders_skip_corrupt_rows() -> None:
    figure = equity_curve_figure(
        [
            {"date": "d1", "nav": 1.0},
            {"date": "d2", "nav": float("nan")},
            {"date": "", "nav": 2.0},
            "junk",
            {"date": "d3", "nav": -1.0},
        ]
    )
    assert list(figure.data[0].y) == [1.0]
    assert equity_curve_figure(None).data == ()
    assert drawdown_figure([]).data == ()
    heat = monthly_heatmap_figure([{"month": "2020-02", "return": 0.02}, {"month": "x"}])
    assert heat.data[0].z == ((0.02,),) or list(heat.data[0].z[0]) == [0.02]
    assert monthly_heatmap_figure("bad").data == ()
