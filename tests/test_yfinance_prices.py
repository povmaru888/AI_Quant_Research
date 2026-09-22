"""P3-03 acceptance: yfinance fallback client (mocked, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from integrations.yfinance_prices import (
    PRICE_COLUMNS,
    fetch_fallback_prices,
    to_yahoo_symbol,
)


def _bars(dates: list[str], close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close] * len(dates),
            "High": [close + 1] * len(dates),
            "Low": [close - 1] * len(dates),
            "Close": [close] * len(dates),
            "Volume": [1000] * len(dates),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_to_yahoo_symbol() -> None:
    assert to_yahoo_symbol("2330") == "2330.TW"
    assert to_yahoo_symbol("6488", "TPEX") == "6488.TWO"
    assert to_yahoo_symbol("2330.TW") == "2330.TW"
    with pytest.raises(ValueError, match="stock_id"):
        to_yahoo_symbol("  ")


def test_fetch_fallback_prices_mapping() -> None:
    seen: dict = {}

    def downloader(ticker, **kwargs):
        seen[ticker] = kwargs
        return _bars(["2020-01-02", "2020-01-03"], close=50.0)

    frame = fetch_fallback_prices(["2330"], "2020-01-01", "2020-01-04", downloader=downloader)
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert seen["2330.TW"]["auto_adjust"] is False
    row = frame.iloc[0].to_dict()
    assert row["stock_id"] == "2330"
    assert row["trade_date"] == "2020-01-02"
    assert row["traded_value"] == pytest.approx(50.0 * 1000)
    assert row["source"] == "yfinance"
    assert frame.attrs["failed"] == []


def test_fetch_fallback_prices_partial_failure() -> None:
    def downloader(ticker, **kwargs):
        if ticker == "9999.TW":
            raise ConnectionError("down")
        if ticker == "0000.TW":
            return pd.DataFrame()
        return _bars(["2020-01-02"])

    frame = fetch_fallback_prices(
        ["2330", "9999", "0000"], "2020-01-01", "2020-01-04", downloader=downloader
    )
    assert frame["stock_id"].unique().tolist() == ["2330"]
    assert sorted(frame.attrs["failed"]) == ["0000", "9999"]


def test_fetch_fallback_prices_all_fail_and_empty() -> None:
    frame = fetch_fallback_prices(
        ["9999"], "2020-01-01", "2020-01-04", downloader=lambda *a, **k: pd.DataFrame()
    )
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert frame.empty
    assert frame.attrs["failed"] == ["9999"]
    empty = fetch_fallback_prices([], "2020-01-01", "2020-01-04", downloader=lambda *a, **k: None)
    assert empty.empty and empty.attrs["failed"] == []


def test_fetch_fallback_prices_drops_bad_bars_and_bad_range() -> None:
    bars = _bars(["2020-01-02", "2020-01-03"], close=0.0)
    frame = fetch_fallback_prices(
        ["2330"], "2020-01-01", "2020-01-04", downloader=lambda *a, **k: bars
    )
    assert frame.empty
    assert frame.attrs["failed"] == ["2330"]
    with pytest.raises(ValueError, match="invalid range"):
        fetch_fallback_prices(["2330"], "2020-01-04", "2020-01-01")


def test_fetch_fallback_prices_flattens_multiindex_columns() -> None:
    bars = _bars(["2020-01-02", "2020-01-03"], close=50.0)
    bars.columns = pd.MultiIndex.from_product([bars.columns, ["2330.TW"]])
    frame = fetch_fallback_prices(
        ["2330"], "2020-01-01", "2020-01-04", downloader=lambda *a, **k: bars
    )
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert len(frame) == 2
    assert frame.attrs["failed"] == []
