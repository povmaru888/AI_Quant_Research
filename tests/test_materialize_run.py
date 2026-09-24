"""Adjusted forward-return diagnostics keep trading-day positions intact."""

from __future__ import annotations

import pandas as pd

from tools.materialize_run import _close_panel, _forward_returns


def test_forward_returns_do_not_skip_missing_adjusted_day() -> None:
    prices = pd.DataFrame(
        [
            {"stock_id": "A", "trade_date": "2020-01-01", "close_adj": 10.0},
            {"stock_id": "A", "trade_date": "2020-01-02", "close_adj": None},
            {"stock_id": "A", "trade_date": "2020-01-03", "close_adj": 12.0},
            {"stock_id": "B", "trade_date": "2020-01-01", "close_adj": 20.0},
            {"stock_id": "B", "trade_date": "2020-01-02", "close_adj": 21.0},
            {"stock_id": "B", "trade_date": "2020-01-03", "close_adj": 22.0},
        ]
    )
    panel = _close_panel(prices, "2020-01-01")
    assert len(panel["A"]) == 3
    assert _forward_returns(panel, 1).index.tolist() == ["B"]
    assert set(_forward_returns(panel, 2).index) == {"A", "B"}
