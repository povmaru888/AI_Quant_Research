"""P4-04 acceptance: portfolio page (fake st)."""

from __future__ import annotations

import pandas as pd
import pytest

from ui.portfolio_page import render_portfolio, summarize_holdings


class FakeSt:
    def __init__(self) -> None:
        self.calls: list = []

    def header(self, text) -> None:
        self.calls.append(("header", text))

    def dataframe(self, frame) -> None:
        self.calls.append(("dataframe", frame))

    def write(self, text) -> None:
        self.calls.append(("write", text))

    def info(self, text) -> None:
        self.calls.append(("info", text))


def _holdings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["2317", "2330", "2454"],
            "stock_name": ["鴻海", "台積電", "聯發科"],
            "rank": [2, 1, 3],
            "prediction_probability": [0.6, 0.7, 0.55],
            "weight": [0.15, 0.2, 0.1],
            "volatility_60d": [0.3, 0.25, 0.35],
            "beta_60d": [0.9, 1.1, 1.2],
        }
    )


def test_render_shows_sorted_table_and_consistency() -> None:
    st = FakeSt()
    frame = _holdings()
    render_portfolio(frame, st=st)
    shown = next(c[1] for c in st.calls if c[0] == "dataframe")
    assert shown["stock_id"].tolist() == ["2330", "2317", "2454"]
    assert shown["股票名稱"].tolist() == ["台積電", "鴻海", "聯發科"]
    assert list(shown.columns) == [
        "stock_id",
        "股票名稱",
        "rank",
        "prediction_probability",
        "weight",
        "volatility_60d",
        "beta_60d",
    ]
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert f"權重總和：{frame['weight'].sum():.4f}" in writes
    assert f"股票數量：{len(frame)}" in writes


def test_summarize_holdings() -> None:
    summary = summarize_holdings(_holdings())
    assert summary == {"count": 3, "total_weight": pytest.approx(0.45)}


def test_render_hides_zero_weight_signal_rows() -> None:
    st = FakeSt()
    frame = _holdings()
    frame.loc[2, "weight"] = 0.0
    render_portfolio(frame, st=st)
    shown = next(c[1] for c in st.calls if c[0] == "dataframe")
    assert shown["stock_id"].tolist() == ["2330", "2317"]
    assert ("write", "股票數量：2") in st.calls


def test_empty_frame_is_info_state() -> None:
    st = FakeSt()
    render_portfolio(_holdings().iloc[0:0], st=st)
    assert any(c[0] == "info" for c in st.calls)
    assert [c for c in st.calls if c[0] == "dataframe"] == []


def test_bad_inputs_rejected() -> None:
    st = FakeSt()
    with pytest.raises(ValueError, match="missing columns"):
        render_portfolio(_holdings().drop(columns=["beta_60d"]), st=st)
    bad = _holdings()
    bad.loc[0, "weight"] = float("nan")
    with pytest.raises(ValueError, match="weight"):
        render_portfolio(bad, st=st)
    with pytest.raises(ValueError, match="DataFrame"):
        render_portfolio([], st=st)  # type: ignore[arg-type]
