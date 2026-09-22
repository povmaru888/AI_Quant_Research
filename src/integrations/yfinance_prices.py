"""P3-03: yfinance fallback price client (SDD 7.1).

Price-only fallback when FinMind is unavailable. Vendor fields never
reach downstream modules; output columns match the Price Repository
(P1-07) with ``source="yfinance"``. Per-symbol isolation: one symbol's
failure never drops the others.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import pandas as pd
import yfinance as yf

SOURCE = "yfinance"

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

_SUFFIXES = {"TWSE": ".TW", "TPEX": ".TWO"}


def to_yahoo_symbol(stock_id: str, market: str = "TWSE") -> str:
    """Map a local stock id to its Yahoo ticker."""
    if not isinstance(stock_id, str) or not stock_id.strip():
        raise ValueError(f"invalid stock_id: {stock_id!r}")
    code = stock_id.strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    suffix = _SUFFIXES.get(market.upper(), ".TW")
    return f"{code}{suffix}"


def _require_range(start: str, end: str) -> None:
    for name, value in (("start", start), ("end", end)):
        try:
            date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name}: {value!r}") from exc
    if start > end:
        raise ValueError(f"invalid range: start {start!r} exceeds end {end!r}")


def fetch_fallback_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    downloader: Callable[..., pd.DataFrame] = yf.download,
    market_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Download per symbol; failures land in ``frame.attrs['failed']``."""
    _require_range(start, end)
    codes = list(dict.fromkeys(symbols))
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    mapping = dict(market_map) if market_map else {}
    for stock_id in codes:
        try:
            frame = _fetch_one(stock_id, start, end, downloader, mapping.get(stock_id, "TWSE"))
        except Exception:  # noqa: BLE001 - isolation is the contract.
            failed.append(stock_id)
            continue
        if frame.empty:
            failed.append(stock_id)
            continue
        frames.append(frame)
    if not frames:
        empty = pd.DataFrame(columns=[*PRICE_COLUMNS])
        empty.attrs["failed"] = failed
        return empty
    merged = pd.concat(frames, ignore_index=True).sort_values(["stock_id", "trade_date"])
    merged.attrs["failed"] = failed
    return merged.reset_index(drop=True)


def _fetch_one(
    stock_id: str,
    start: str,
    end: str,
    downloader: Callable[..., pd.DataFrame],
    market: str,
) -> pd.DataFrame:
    ticker = to_yahoo_symbol(stock_id, market)
    raw = downloader(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # Newer yfinance adds a ticker level even for single-ticker calls.
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    columns = {str(c).strip().lower(): c for c in raw.columns}
    needed = {"open": None, "high": None, "low": None, "close": None, "volume": None}
    for key in needed:
        if key not in columns:
            return pd.DataFrame()
        needed[key] = columns[key]
    frame = pd.DataFrame(
        {
            "stock_id": stock_id,
            "trade_date": pd.to_datetime(raw.index).strftime("%Y-%m-%d"),
            "open": pd.to_numeric(raw[needed["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[needed["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[needed["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[needed["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[needed["volume"]], errors="coerce"),
        }
    )
    frame = frame.dropna()
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame["low"])
    ]
    if frame.empty:
        return frame
    frame["traded_value"] = frame["close"] * frame["volume"]
    frame["source"] = SOURCE
    return frame[[*PRICE_COLUMNS]]
