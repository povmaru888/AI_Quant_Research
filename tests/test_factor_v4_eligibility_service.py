from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from contracts import UniverseEntry, UniverseSnapshot
from services.factor_v4_eligibility_service import filter_factor_v4_universe


def _prices(days: pd.DatetimeIndex, stock_ids: list[str]) -> pd.DataFrame:
    records = [
        {
            "stock_id": "TAIEX",
            "trade_date": day.date().isoformat(),
            "close_adj": np.nan,
            "high_adj": np.nan,
        }
        for day in days
    ]
    for stock_id in stock_ids:
        for day in days[-121:]:
            records.append(
                {
                    "stock_id": stock_id,
                    "trade_date": day.date().isoformat(),
                    "close_adj": 100.0,
                    "high_adj": 101.0,
                }
            )
    return pd.DataFrame(records)


def test_factor_v4_filters_before_labels_and_keeps_negative_financial_values(settings) -> None:
    days = pd.bdate_range("2023-01-02", periods=252)
    as_of = days[-1].date()
    ids = ["pass", "young", "missing_fin", "missing_label"]
    stocks = pd.DataFrame(
        {
            "stock_id": ids,
            "listed_date": [
                days[0].date().isoformat(),
                days[1].date().isoformat(),
                days[0].date().isoformat(),
                days[0].date().isoformat(),
            ],
        }
    )
    financials = pd.DataFrame(
        {
            "stock_id": ["pass", "young", "missing_fin", "missing_label"],
            "report_period": ["2023Q3"] * 4,
            "available_date": [as_of.isoformat()] * 4,
            "net_income": [-1.0, 1.0, np.nan, 1.0],
            "equity": [-2.0, 2.0, 2.0, 2.0],
        }
    )
    universe = UniverseSnapshot(
        run_id="v4",
        as_of=as_of.isoformat(),
        entries=tuple(UniverseEntry(stock_id=sid, included=True, reason="pass") for sid in ids),
    )
    v4_settings = replace(
        settings,
        universe=replace(settings.universe, min_listing_age_trading_days=252),
        features=replace(
            settings.features,
            feature_version="factor_v4",
            required_adjusted_price_rows=121,
            required_financial_fields=("net_income", "equity"),
        ),
    )

    filtered, report = filter_factor_v4_universe(
        universe,
        stocks,
        _prices(days, ids),
        financials,
        as_of,
        v4_settings,
        labelable_ids={"pass", "young", "missing_fin"},
    )

    assert filtered.included_ids == ("pass",)
    reasons = {entry.stock_id: entry.reason for entry in filtered.entries}
    assert reasons["young"] == "listing_age:insufficient:251"
    assert reasons["missing_fin"] == "financials:missing_net_income"
    assert reasons["missing_label"] == "label:missing_t_plus_horizon"
    assert report["initial_count"] == 4
    assert report["eligible_count"] == 1


def test_factor_v4_rejects_invalid_adjusted_window_and_missing_listing_date(settings) -> None:
    days = pd.bdate_range("2023-01-02", periods=252)
    as_of = days[-1].date()
    prices = _prices(days, ["bad_adj", "no_date"])
    prices.loc[
        prices["stock_id"].eq("bad_adj") & prices["trade_date"].eq(days[-5].date().isoformat()),
        "close_adj",
    ] = np.nan
    stocks = pd.DataFrame(
        {"stock_id": ["bad_adj", "no_date"], "listed_date": [days[0].date().isoformat(), None]}
    )
    financials = pd.DataFrame(
        {
            "stock_id": ["bad_adj", "no_date"],
            "available_date": [as_of.isoformat()] * 2,
            "net_income": [1.0, 1.0],
            "equity": [1.0, 1.0],
        }
    )
    universe = UniverseSnapshot(
        run_id="v4",
        as_of=as_of.isoformat(),
        entries=(
            UniverseEntry("bad_adj", True, "pass"),
            UniverseEntry("no_date", True, "pass"),
        ),
    )
    v4_settings = replace(
        settings,
        universe=replace(settings.universe, min_listing_age_trading_days=252),
        features=replace(
            settings.features,
            feature_version="factor_v4",
            required_adjusted_price_rows=121,
            required_financial_fields=("net_income", "equity"),
        ),
    )
    filtered, _ = filter_factor_v4_universe(
        universe, stocks, prices, financials, as_of, v4_settings
    )
    reasons = {entry.stock_id: entry.reason for entry in filtered.entries}
    assert reasons == {
        "bad_adj": "adjusted_price:invalid_close_adj",
        "no_date": "listing_age:missing_date",
    }
