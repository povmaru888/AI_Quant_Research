"""P4-06 acceptance: risk page (fake st)."""

from __future__ import annotations

import pytest

from ui.risk_page import check_ma60_exposure, render_risk


class FakeSt:
    def __init__(self) -> None:
        self.calls: list = []

    def header(self, text) -> None:
        self.calls.append(("header", text))

    def write(self, text) -> None:
        self.calls.append(("write", text))


def _risk_data(**overrides):
    base = {
        "equity_exposure": 0.6,
        "predicted_volatility": 0.18,
        "realized_volatility": None,
        "max_drawdown": -0.1,
        "turnover": 0.5,
        "market_regime": "below_ma60",
        "exposure_cap": 0.6,
    }
    base.update(overrides)
    return base


def test_render_full_risk_data() -> None:
    st = FakeSt()
    render_risk(_risk_data(), st=st)
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert "股票曝險：0.6000" in writes
    assert "實際波動：N/A" in writes
    assert not any("TAIEX" in value or "MA60" in value for value in writes)


def test_ma60_verdict_branches() -> None:
    assert check_ma60_exposure("below_ma60", 0.6, 0.6) == "符合"
    assert check_ma60_exposure("below_ma60", 0.61, 0.6) == "超限"
    assert check_ma60_exposure("above_ma60", 1.0, 1.0) == "不適用"
    assert check_ma60_exposure(None, 0.6, 0.6) == "資料不足"
    assert check_ma60_exposure("below_ma60", None, 0.6) == "資料不足"
    assert check_ma60_exposure("below_ma60", 0.6, float("nan")) == "資料不足"

    st = FakeSt()
    render_risk({}, st=st)
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert all(v.endswith("N/A") or v.endswith("資料不足") for v in writes)

    with pytest.raises(ValueError, match="risk_data"):
        render_risk([], st=st)
