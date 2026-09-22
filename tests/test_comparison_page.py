"""P4-07 acceptance: research comparison page (fake st)."""

from __future__ import annotations

import pandas as pd
import pytest

from ui.comparison_page import paired_metrics, render_comparison


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


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["baseline", "ablation_momentum"],
            "cagr_before": [0.12, 0.10],
            "cagr_after": [0.09, 0.07],
            "sharpe_before": [1.2, 1.0],
            "sharpe_after": [0.9, 0.8],
            "note": ["a", "b"],
        }
    )


def test_render_shows_table_and_deltas() -> None:
    st = FakeSt()
    render_comparison(_metrics(), st=st)
    shown = next(c[1] for c in st.calls if c[0] == "dataframe")
    assert "cagr_before" in shown.columns and "cagr_after" in shown.columns
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert "情境：baseline" in writes
    assert "cagr: 0.1200 → 0.0900（Δ-0.0300）" in writes
    assert "sharpe: 1.2000 → 0.9000（Δ-0.3000）" in writes
    assert "情境：ablation_momentum" in writes


def test_paired_metrics_only_complete_pairs() -> None:
    assert paired_metrics(["scenario", "cagr_before", "cagr_after", "note"]) == ["cagr"]
    assert paired_metrics(["scenario", "cagr_before"]) == []
    assert paired_metrics(["scenario"]) == []


def test_degraded_states() -> None:
    st = FakeSt()
    render_comparison(pd.DataFrame({"scenario": ["baseline"]}), st=st)
    assert any(c[0] == "info" for c in st.calls)
    assert [c for c in st.calls if c[0] == "write" and "→" in c[1]] == []

    st = FakeSt()
    render_comparison(pd.DataFrame({"scenario": []}), st=st)
    assert any(c[0] == "info" for c in st.calls)

    st = FakeSt()
    partial = _metrics()
    partial.loc[0, "cagr_after"] = float("nan")
    render_comparison(partial, st=st)
    writes = [c[1] for c in st.calls if c[0] == "write"]
    assert "cagr: 0.1200 → N/A" in writes

    with pytest.raises(ValueError, match="scenario"):
        render_comparison(pd.DataFrame({"cagr": [0.1]}), st=st)
    with pytest.raises(ValueError, match="DataFrame"):
        render_comparison([], st=st)  # type: ignore[arg-type]
