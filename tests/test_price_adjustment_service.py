from __future__ import annotations

import pandas as pd

from services.price_adjustment_service import calibrate_adjusted_prices


def test_calibration_matches_finmind_scale_for_each_stock() -> None:
    yahoo = pd.DataFrame(
        {
            "stock_id": ["A"] * 4 + ["B"] * 4,
            "trade_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"] * 2,
            "open_adj": [8.0] * 4 + [5.0] * 4,
            "high_adj": [9.0] * 4 + [6.0] * 4,
            "low_adj": [7.0] * 4 + [4.0] * 4,
            "close_adj": [8.0, 8.1, 8.2, 8.3, 5.0, 5.1, 5.2, 5.3],
        }
    )
    reference = pd.DataFrame(
        {
            "stock_id": ["A"] * 3 + ["B"] * 3,
            "trade_date": ["2020-01-01", "2020-01-02", "2020-01-03"] * 2,
            "close_adj": [16.0, 16.2, 16.4, 15.0, 15.3, 15.6],
        }
    )

    result, diagnostics = calibrate_adjusted_prices(yahoo, reference)

    assert set(result["stock_id"]) == {"A", "B"}
    assert result.loc[result["stock_id"] == "A", "close_adj"].iloc[-1] == 16.6
    assert diagnostics["A"]["accepted"] is True
    assert diagnostics["A"]["scale"] == 2.0


def test_calibration_rejects_nonconstant_vendor_adjustment_ratio() -> None:
    yahoo = pd.DataFrame(
        {
            "stock_id": ["A"] * 3,
            "trade_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open_adj": [10.0] * 3,
            "high_adj": [11.0] * 3,
            "low_adj": [9.0] * 3,
            "close_adj": [10.0, 10.0, 10.0],
        }
    )
    reference = pd.DataFrame(
        {
            "stock_id": ["A"] * 3,
            "trade_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "close_adj": [10.0, 11.0, 13.0],
        }
    )

    result, diagnostics = calibrate_adjusted_prices(yahoo, reference)

    assert result.empty
    assert diagnostics["A"]["accepted"] is False
