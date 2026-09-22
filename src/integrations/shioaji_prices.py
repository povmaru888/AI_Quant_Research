"""Shioaji historical price client (bulk backfill source, SDD 7.1).

Shioaji serves MINUTE kbars (max 30 days per call, history from
2020-03-02 for stocks/indices). This module chunks long ranges and
resamples to daily OHLCV matching the Price Repository columns with
``source="shioaji"``. Timestamps are Unix-ns; trade dates are derived in
Asia/Taipei. Vendor field names never reach downstream modules.

Live notes (verified 2026-09-22, simulation key):

- Stock contract: ``api.Contracts.Stocks["2330"]`` (covers TSE and OTC).
- TAIEX contract: ``api.Contracts.Indexs.TSE["IX0001"]``; index kbars
  carry no volume, so ``volume``/``traded_value`` are stored as 0.0
  (allowed by the ``>= 0`` CHECKs).
- Simulation OHLC matches FinMind tick-for-tick, but simulation
  volume/amount look scaled down; values pass through untouched and the
  discrepancy is documented, not "corrected". Re-verify on a production
  key before trusting liquidity gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

SOURCE = "shioaji"

PRICE_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "traded_value",
    "source",
)

_TAIPEI = ZoneInfo("Asia/Taipei")
_MAX_WINDOW_DAYS = 30
_TAIEX_CODE = "IX0001"


class ShioajiError(Exception):
    """Shioaji fetch failure."""


def _require_range(start: str, end: str) -> None:
    for name, value in (("start", start), ("end", end)):
        try:
            date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name}: {value!r}") from exc
    if start > end:
        raise ValueError(f"invalid range: start {start!r} exceeds end {end!r}")


def _windows(start: str, end: str, days: int = _MAX_WINDOW_DAYS) -> list[tuple[str, str]]:
    """Split [start, end] into chunks of at most ``days`` calendar days."""
    cursor = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    step = timedelta(days=days - 1)
    one = timedelta(days=1)
    out: list[tuple[str, str]] = []
    while cursor <= stop:
        out.append((cursor.isoformat(), min(cursor + step, stop).isoformat()))
        cursor = min(cursor + step, stop) + one
    return out


def _resolve_stock_contract(api: Any, stock_id: str) -> Any:
    try:
        contracts = api.Contracts.Stocks
    except AttributeError as exc:
        raise ShioajiError("shioaji api has no Contracts.Stocks") from exc
    try:
        return contracts[stock_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown shioaji stock contract: {stock_id!r}") from exc


def _resolve_taiex_contract(api: Any) -> Any:
    try:
        return api.Contracts.Indexs.TSE[_TAIEX_CODE]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ShioajiError("shioaji api has no TAIEX index contract") from exc


def kbars_to_daily(stock_id: str, kbars: Mapping[str, Sequence]) -> pd.DataFrame:
    """Resample one minute-kbar payload to daily OHLCV (pure, testable)."""
    frame = pd.DataFrame(
        {
            "ts": list(kbars.get("ts", [])),
            "open": pd.Series(pd.to_numeric(kbars.get("Open", []), errors="coerce")),
            "high": pd.Series(pd.to_numeric(kbars.get("High", []), errors="coerce")),
            "low": pd.Series(pd.to_numeric(kbars.get("Low", []), errors="coerce")),
            "close": pd.Series(pd.to_numeric(kbars.get("Close", []), errors="coerce")),
            "volume": pd.Series(pd.to_numeric(kbars.get("Volume", []), errors="coerce")).fillna(
                0.0
            ),
            "amount": pd.Series(pd.to_numeric(kbars.get("Amount", []), errors="coerce")).fillna(
                0.0
            ),
        }
    )
    if frame.empty:
        return pd.DataFrame(columns=[*PRICE_COLUMNS])
    # Drop bad minutes before resampling so one bad tick never poisons a day.
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame[["volume", "amount"]] >= 0).all(axis=1)
        & (frame["high"] >= frame["low"])
    ]
    if frame.empty:
        return pd.DataFrame(columns=[*PRICE_COLUMNS])
    frame["trade_date"] = (
        pd.to_datetime(frame["ts"], unit="ns", utc=True)
        .dt.tz_convert(_TAIPEI)
        .dt.strftime("%Y-%m-%d")
    )
    grouped = frame.groupby("trade_date", sort=True)
    daily = pd.DataFrame(
        {
            "stock_id": stock_id,
            "trade_date": grouped["trade_date"].first(),
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "traded_value": grouped["amount"].sum(),
        }
    ).reset_index(drop=True)
    daily = daily.loc[
        (daily[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (daily[["volume", "traded_value"]] >= 0).all(axis=1)
        & (daily["high"] >= daily["low"])
    ]
    daily["source"] = SOURCE
    return daily[[*PRICE_COLUMNS]].reset_index(drop=True)


def _fetch_windows(
    api: Any,
    contract: Any,
    stock_id: str,
    start: str,
    end: str,
    timeout: float,
    kbars_fn: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    call = kbars_fn or api.kbars
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _windows(start, end):
        try:
            payload = call(
                contract,
                start=window_start,
                end=window_end,
                timeout=int(timeout * 1000),
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ShioajiError(
                f"shioaji kbars {stock_id} {window_start}..{window_end} failed: "
                f"{type(exc).__name__}"
            ) from exc
        data = payload.dict() if hasattr(payload, "dict") else payload
        frames.append(kbars_to_daily(stock_id, data))
    if not frames:
        return pd.DataFrame(columns=[*PRICE_COLUMNS])
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    return merged.sort_values("trade_date").reset_index(drop=True)


def fetch_daily_prices(
    stock_id: str,
    start: str,
    end: str,
    api: Any,
    timeout: float = 30.0,
    kbars_fn: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Fetch daily prices for one stock over [start, end] via Shioaji."""
    if not isinstance(stock_id, str) or not stock_id.strip():
        raise ValueError(f"invalid stock_id: {stock_id!r}")
    _require_range(start, end)
    contract = _resolve_stock_contract(api, stock_id.strip())
    return _fetch_windows(api, contract, stock_id.strip(), start, end, timeout, kbars_fn)


def fetch_taiex_daily(
    start: str,
    end: str,
    api: Any,
    timeout: float = 30.0,
    kbars_fn: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Fetch TAIEX daily closes (volume/traded_value stored as 0.0)."""
    _require_range(start, end)
    contract = _resolve_taiex_contract(api)
    frame = _fetch_windows(api, contract, "TAIEX", start, end, timeout, kbars_fn)
    if frame.empty:
        return frame
    frame["volume"] = 0.0
    frame["traded_value"] = 0.0
    return frame[[*PRICE_COLUMNS]].reset_index(drop=True)
