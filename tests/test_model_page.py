"""P4-05 acceptance: model page (fake st, real figures)."""

from __future__ import annotations

import pytest

from ui.model_page import (
    SHAP_DISCLAIMER,
    monthly_ic_figure,
    prediction_dist_figure,
    render_model,
    selection_metrics_figure,
    shap_figure,
)


class FakeSt:
    def __init__(self) -> None:
        self.calls: list = []

    def header(self, text) -> None:
        self.calls.append(("header", text))

    def write(self, text) -> None:
        self.calls.append(("write", text))

    def subheader(self, text) -> None:
        self.calls.append(("subheader", text))

    def caption(self, text) -> None:
        self.calls.append(("caption", text))

    def info(self, text) -> None:
        self.calls.append(("info", text))

    def dataframe(self, frame) -> None:
        self.calls.append(("dataframe", frame))

    def columns(self, count):
        parent = self

        class Column:
            def metric(self, label, value) -> None:
                parent.calls.append(("metric", label, value))

        return [Column() for _ in range(count)]

    def plotly_chart(self, figure) -> None:
        self.calls.append(("plotly_chart", figure))


def _model_data(**overrides):
    base = {
        "shap_top": [
            {"feature": "momentum_20d", "value": 0.05},
            {"feature": "roe", "value": 0.03},
        ],
        "feature_importance": [
            {"feature": "momentum_20d", "gain": 12.0},
            {"feature": "roe", "gain": 8.0},
        ],
        "monthly_ic": [
            {"month": "2020-01", "ic": 0.06},
            {"month": "2020-02", "ic": 0.08},
        ],
        "prediction_dist": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    base.update(overrides)
    return base


def test_render_full_model_data() -> None:
    st = FakeSt()
    render_model(_model_data(), st=st)
    assert ("write", SHAP_DISCLAIMER) in st.calls
    assert "因子收益歸因" in SHAP_DISCLAIMER
    charts = [c[1] for c in st.calls if c[0] == "plotly_chart"]
    assert len(charts) == 3
    assert [trace.name for trace in charts[0].data] == ["Continuous Rank IC"]
    assert charts[1].data[0].name == "SHAP"
    assert charts[2].data[0].name == "probability"
    shown = next(
        c[1]
        for c in st.calls
        if c[0] == "dataframe" and "feature" in c[1].columns
    )
    assert shown["feature"].tolist() == ["momentum_20d", "roe"]
    assert len([c for c in st.calls if c[0] == "metric"]) == 6
    selection_table = next(
        c[1]
        for c in st.calls
        if c[0] == "dataframe" and "Excess Return" in c[1].columns
    )
    assert list(selection_table.columns) == [
        "月份",
        "Continuous IC",
        "Excess Return",
        "Top-Bottom Spread",
    ]


def test_disclaimer_without_data() -> None:
    st = FakeSt()
    render_model({}, st=st)
    assert ("write", SHAP_DISCLAIMER) in st.calls
    assert [c for c in st.calls if c[0] == "plotly_chart"] == []
    assert len([c for c in st.calls if c[0] == "info"]) == 4
    with pytest.raises(ValueError, match="model_data"):
        render_model([], st=st)


def test_shap_capped_at_ten_and_skips_bad_rows() -> None:
    rows = [{"feature": f"f{i}", "value": 0.01 * i} for i in range(15)]
    rows.append({"feature": "", "value": 9.0})
    rows.append({"feature": "bad", "value": float("nan")})
    figure = shap_figure(rows)
    assert len(figure.data[0].y) == 10
    assert monthly_ic_figure(None).data == ()
    assert prediction_dist_figure("bad").data == ()
    dist = prediction_dist_figure([0.2, float("nan"), "x", 0.8])
    assert list(dist.data[0].x) == [0.2, 0.8]


def test_selection_figure_uses_three_monthly_series_and_secondary_return_axis() -> None:
    payload = {
        "schema_version": 2,
        "monthly": [
            {
                "month": "2020-01",
                "continuous_rank_ic": 0.1,
                "top15_excess_return": 0.02,
                "top_bottom_spread": 0.04,
            },
            {
                "month": "2020-02",
                "continuous_rank_ic": -0.03,
                "top15_excess_return": -0.01,
                "top_bottom_spread": 0.05,
            },
        ],
    }

    figure = selection_metrics_figure(payload)

    assert [trace.name for trace in figure.data] == [
        "Continuous Rank IC",
        "Top15 Excess Return",
        "Top-Bottom Spread",
    ]
    assert list(figure.data[1].y) == [0.02, -0.01]


def test_render_versioned_metrics_formats_returns_as_percentages() -> None:
    model_data = _model_data(
        monthly_ic={
            "schema_version": 2,
            "summary": {
                "mean_continuous_rank_ic": 0.1,
                "ic_positive_month_ratio": 0.5,
                "mean_top15_actual_return": 0.03,
                "mean_top15_excess_return": 0.02,
                "top15_excess_positive_month_ratio": 0.75,
                "mean_top_bottom_spread": 0.04,
            },
            "monthly": [
                {
                    "month": "2020-01",
                    "continuous_rank_ic": 0.1,
                    "top15_excess_return": 0.02,
                    "top_bottom_spread": 0.04,
                }
            ],
        }
    )
    st = FakeSt()

    render_model(model_data, st=st)

    displayed = {(call[1], call[2]) for call in st.calls if call[0] == "metric"}
    assert ("平均連續 Rank IC", "0.1000") in displayed
    assert ("IC > 0 月份比例", "50.00%") in displayed
    assert ("Top15 平均 Excess Return", "2.00%") in displayed
    selection_table = next(
        call[1]
        for call in st.calls
        if call[0] == "dataframe" and "Excess Return" in call[1].columns
    )
    assert selection_table.loc[0, "Excess Return"] == "2.00%"
    assert selection_table.loc[0, "Top-Bottom Spread"] == "4.00%"
