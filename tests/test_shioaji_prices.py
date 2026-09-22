"""Shioaji price client acceptance (mocked api, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from integrations.shioaji_prices import (
    PRICE_COLUMNS,
    ShioajiError,
    _windows,
    fetch_daily_prices,
    fetch_taiex_daily,
    kbars_to_daily,
)


def _ns(day: str, hm: str) -> int:
    return pd.Timestamp(f"{day} {hm}", tz="Asia/Taipei").value


def _kbars(day: str, opens=(100.0, 101.0)) -> dict:
    n = len(opens)
    return {
        "ts": [_ns(day, f"09:0{i + 1}") for i in range(n)],
        "Open": list(opens),
        "High": [o + 1 for o in opens],
        "Low": [o - 1 for o in opens],
        "Close": [o + 0.5 for o in opens],
        "Volume": [1000] * n,
        "Amount": [o * 1000 for o in opens],
    }


class _Payload:
    def __init__(self, data: dict) -> None:
        self._data = data

    def dict(self) -> dict:
        return self._data


def _api(payloads: dict[tuple[str, str], dict], contracts=("2330",)):
    calls: list[tuple[str, str]] = []

    def kbars_fn(contract, start=None, end=None, **kwargs):
        calls.append((start, end))
        return _Payload(payloads[(start, end)])

    return (
        SimpleNamespace(
            Contracts=SimpleNamespace(
                Stocks={code: object() for code in contracts},
                Indexs=SimpleNamespace(TSE={"IX0001": object()}),
            ),
            kbars=kbars_fn,
        ),
        calls,
    )


def test_resample_math_and_taipei_dates() -> None:
    frame = kbars_to_daily("2330", _kbars("2020-01-02", opens=(100.0, 102.0)))
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["trade_date"] == "2020-01-02"
    assert row["open"] == pytest.approx(100.0)
    assert row["high"] == pytest.approx(103.0)
    assert row["low"] == pytest.approx(99.0)
    assert row["close"] == pytest.approx(102.5)
    assert row["volume"] == pytest.approx(2000.0)
    assert row["traded_value"] == pytest.approx(202000.0)
    assert row["source"] == "shioaji"


def test_windows_split_30_days() -> None:
    assert _windows("2020-01-01", "2020-01-30") == [("2020-01-01", "2020-01-30")]
    assert _windows("2020-01-01", "2020-03-01") == [
        ("2020-01-01", "2020-01-30"),
        ("2020-01-31", "2020-02-29"),
        ("2020-03-01", "2020-03-01"),
    ]


def test_fetch_chunks_and_concatenates() -> None:
    payloads = {
        ("2020-01-01", "2020-01-30"): _kbars("2020-01-02"),
        ("2020-01-31", "2020-02-29"): _kbars("2020-02-03"),
        ("2020-03-01", "2020-03-01"): _kbars("2020-03-02"),
    }
    api, calls = _api(payloads)
    frame = fetch_daily_prices("2330", "2020-01-01", "2020-03-01", api)
    assert [c for c in calls] == [
        ("2020-01-01", "2020-01-30"),
        ("2020-01-31", "2020-02-29"),
        ("2020-03-01", "2020-03-01"),
    ]
    assert frame["trade_date"].tolist() == ["2020-01-02", "2020-02-03", "2020-03-02"]
    assert (frame["stock_id"] == "2330").all()


def test_fetch_drops_invalid_minutes() -> None:
    payload = _kbars("2020-01-02", opens=(100.0, 0.0))
    api, _ = _api({("2020-01-01", "2020-01-01"): payload})
    frame = fetch_daily_prices("2330", "2020-01-01", "2020-01-01", api)
    assert len(frame) == 1  # day survives on the valid minute bar.
    assert frame.iloc[0]["close"] == pytest.approx(100.5)
    empty_payload = _kbars("2020-01-02", opens=(0.0,))
    api2, _ = _api({("2020-01-01", "2020-01-01"): empty_payload})
    frame2 = fetch_daily_prices("2330", "2020-01-01", "2020-01-01", api2)
    assert frame2.empty
    assert list(frame2.columns) == list(PRICE_COLUMNS)


def test_fetch_rejects_bad_inputs() -> None:
    api, _ = _api({("2020-01-01", "2020-01-01"): _kbars("2020-01-02")})
    with pytest.raises(ValueError, match="stock_id"):
        fetch_daily_prices("  ", "2020-01-01", "2020-01-01", api)
    with pytest.raises(ValueError, match="invalid range"):
        fetch_daily_prices("2330", "2020-01-02", "2020-01-01", api)
    with pytest.raises(ValueError, match="unknown shioaji stock"):
        fetch_daily_prices("9999", "2020-01-01", "2020-01-01", api)


def test_fetch_wraps_api_errors() -> None:
    api, _ = _api({})
    with pytest.raises(ShioajiError, match="kbars 2330"):
        fetch_daily_prices("2330", "2020-01-01", "2020-01-01", api)


def test_fetch_taiex_daily() -> None:
    payloads = {("2020-01-01", "2020-01-01"): _kbars("2020-01-02")}
    api, _ = _api(payloads, contracts=())
    frame = fetch_taiex_daily("2020-01-01", "2020-01-01", api)
    assert frame.iloc[0]["stock_id"] == "TAIEX"
    assert frame.iloc[0]["volume"] == 0.0
    assert frame.iloc[0]["traded_value"] == 0.0
    assert frame.iloc[0]["close"] == pytest.approx(101.5)
