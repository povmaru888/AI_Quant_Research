"""P2-04 acceptance: raw factor service."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.feature_service import FACTOR_COLUMNS, calculate_raw_features

AS_OF = date(2019, 12, 31)


def _price_history(
    stock_id: str, closes: np.ndarray, end: str = "2019-12-31", volume: float = 10_000.0
) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "stock_id": stock_id,
            "trade_date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": volume,
            "traded_value": closes * volume,
        }
    )


def _snapshot(stock_id: str = "2330", **overrides) -> pd.DataFrame:
    row = {
        "stock_id": stock_id,
        "as_of_close": 250.0,
        "float_shares": 1_000_000_000.0,
        "net_income": 10_000_000_000.0,
        "equity": 100_000_000_000.0,
        "revenue": 50_000_000_000.0,
        "assets": 200_000_000_000.0,
        "operating_income": 8_000_000_000.0,
        "operating_cash_flow": 6_000_000_000.0,
        "margin_balance": 1_000_000.0,
        "short_balance": 100_000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _financials(stock_id: str = "2330") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [stock_id] * 6,
            "report_period": [f"2018Q{i}" for i in (1, 2, 3, 4)] + ["2019Q1", "2019Q2"],
            "available_date": [
                "2018-05-15",
                "2018-08-14",
                "2018-11-14",
                "2019-03-20",
                "2019-05-15",
                "2019-08-14",
            ],
            "revenue": [40e9, 42e9, 44e9, 46e9, 48e9, 50e9],
            "operating_income": [6e9, 6.4e9, 6.8e9, 7.2e9, 7.6e9, 8e9],
        }
    )


def _institutional(stock_id: str = "2330") -> pd.DataFrame:
    dates = pd.bdate_range(end="2019-12-31", periods=25).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "stock_id": stock_id,
            "trade_date": dates,
            "foreign_net_buy": 1_000.0,
            "trust_net_buy": 500.0,
            "margin_balance": np.linspace(800_000.0, 1_000_000.0, 25),
        }
    )


def _full_prices() -> pd.DataFrame:
    closes = 100.0 + np.arange(150)
    market = 1000.0 + 2 * np.arange(150)
    return pd.concat(
        [
            _price_history("2330", closes),
            _price_history("TAIEX", market),
        ],
        ignore_index=True,
    )


def test_factor_columns_complete() -> None:
    assert len(FACTOR_COLUMNS) == 30
    assert len(set(FACTOR_COLUMNS)) == 30
    out = calculate_raw_features(
        _snapshot(), _full_prices(), AS_OF, _financials(), _institutional()
    )
    assert list(out.columns) == ["stock_id", *FACTOR_COLUMNS, "missing_flag"]
    assert len(out) == 1


def test_momentum_matches_hand_calculation() -> None:
    closes = 100.0 + np.arange(150)
    out = calculate_raw_features(
        _snapshot(), _full_prices(), AS_OF, _financials(), _institutional()
    )
    row = out.iloc[0]
    assert row["momentum_20d"] == pytest.approx(closes[-1] / closes[-21] - 1)
    assert row["momentum_60d"] == pytest.approx(closes[-1] / closes[-61] - 1)
    assert row["momentum_120d"] == pytest.approx(closes[-1] / closes[-121] - 1)
    assert row["momentum_20d_ex_5d"] == pytest.approx(closes[-6] / closes[-26] - 1)
    assert row["ma20_ma60_gap"] == pytest.approx(closes[-20:].mean() / closes[-60:].mean() - 1)
    assert row["price_ma20_gap"] == pytest.approx(closes[-1] / closes[-20:].mean() - 1)
    assert row["revenue_yoy"] == pytest.approx(50e9 / 42e9 - 1)
    assert row["revenue_mom"] == pytest.approx(50e9 / 48e9 - 1)
    assert row["operating_income_qoq"] == pytest.approx(8e9 / 7.6e9 - 1)
    assert row["foreign_net_buy_float"] == pytest.approx(20 * 1_000.0 / 1e9)
    balances = np.linspace(800_000.0, 1_000_000.0, 25)
    assert row["margin_balance_change"] == pytest.approx(balances[-1] / balances[-21] - 1)
    assert row["missing_flag"] == 1  # dividend_yield has no source yet


def test_zero_denominator_never_produces_inf() -> None:
    out = calculate_raw_features(
        _snapshot(margin_balance=0.0),
        _full_prices(),
        AS_OF,
        _financials(),
        _institutional(),
    )
    assert pd.isna(out.iloc[0]["short_margin_ratio"])
    assert not np.isinf(out[list(FACTOR_COLUMNS)].to_numpy(dtype=float)).any()


def test_negative_valuation_masked_with_flag() -> None:
    out = calculate_raw_features(_snapshot(net_income=-5_000_000_000.0), _full_prices(), AS_OF)
    row = out.iloc[0]
    assert pd.isna(row["earnings_yield"])
    assert row["missing_flag"] == 1


def test_future_prices_do_not_leak() -> None:
    base = calculate_raw_features(
        _snapshot(), _full_prices(), AS_OF, _financials(), _institutional()
    )
    spike = pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "trade_date": "2020-01-02",
                "open": 2500.0,
                "high": 2500.0,
                "low": 2500.0,
                "close": 2500.0,
                "volume": 10_000.0,
                "traded_value": 25_000_000.0,
            }
        ]
    )
    leaked = calculate_raw_features(
        _snapshot(),
        pd.concat([_full_prices(), spike], ignore_index=True),
        AS_OF,
        _financials(),
        _institutional(),
    )
    pd.testing.assert_frame_equal(base, leaked)


def test_missing_taiex_leaves_beta_nan() -> None:
    closes = 100.0 + np.arange(150)
    prices = _price_history("2330", closes)
    out = calculate_raw_features(_snapshot(), prices, AS_OF)
    row = out.iloc[0]
    assert pd.isna(row["beta_60d"])
    assert row["momentum_20d"] == pytest.approx(closes[-1] / closes[-21] - 1)


def test_beta_requires_all_60_matching_market_dates() -> None:
    closes = 100.0 + np.arange(150)
    stock = _price_history("2330", closes)
    market = _price_history("TAIEX", closes)
    complete = calculate_raw_features(_snapshot(), pd.concat([stock, market]), AS_OF)
    assert complete.iloc[0]["beta_60d"] == pytest.approx(1.0)

    missing_market_day = market.drop(market.index[-30])
    incomplete = calculate_raw_features(_snapshot(), pd.concat([stock, missing_market_day]), AS_OF)
    assert pd.isna(incomplete.iloc[0]["beta_60d"])
