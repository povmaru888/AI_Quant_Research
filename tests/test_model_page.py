"""P4-05 acceptance: model page (fake st, real figures)."""

from __future__ import annotations

import pytest

from ui.model_page import (
    SHAP_DISCLAIMER,
    monthly_ic_figure,
    prediction_dist_figure,
    render_model,
    shap_figure,
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

    def dataframe(self, frame) -> None:
        self.calls.append(("dataframe", frame))

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
    assert charts[0].data[0].name == "SHAP"
    assert charts[1].data[0].name == "IC"
    assert charts[2].data[0].name == "probability"
    shown = next(c[1] for c in st.calls if c[0] == "dataframe")
    assert shown["feature"].tolist() == ["momentum_20d", "roe"]


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
