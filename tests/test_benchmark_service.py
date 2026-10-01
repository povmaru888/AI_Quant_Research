"""TAIEX dashboard benchmark uses the strategy's exact active dates."""

from __future__ import annotations

import pandas as pd
import pytest

from services.benchmark_service import build_benchmark_payload


def test_build_benchmark_payload_aligns_dates_metrics_and_months() -> None:
    equity = [
        {"date": "2020-01-02", "nav": 1000.0},
        {"date": "2020-01-03", "nav": 1010.0},
        {"date": "2020-01-06", "nav": 1020.0},
    ]
    taiex = pd.DataFrame(
        {
            "trade_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
            "close": [90.0, 100.0, 110.0, 99.0],
        }
    )

    payload = build_benchmark_payload(equity, taiex)

    curve = payload["benchmark_equity_curve"]
    assert [row["date"] for row in curve] == ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert [row["nav"] for row in curve] == pytest.approx([1000.0, 1100.0, 990.0])
    assert payload["benchmark_metrics"]["max_drawdown"] == pytest.approx(-0.1)
    assert payload["benchmark_monthly_returns"] == [
        {"month": "2020-01", "return": pytest.approx(-0.01)}
    ]


def test_build_benchmark_payload_requires_two_matching_valid_days() -> None:
    equity = [{"date": "2020-01-02", "nav": 1000.0}]
    taiex = pd.DataFrame({"trade_date": ["2020-01-02"], "close": [100.0]})
    assert build_benchmark_payload(equity, taiex) == {}
    assert build_benchmark_payload("bad", taiex) == {}
