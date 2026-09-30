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

ADJUSTED_PRICE_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "trade_date",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
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


def _standardize(raw: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """Map one Yahoo frame to PRICE_COLUMNS; empty when unusable."""
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


def _standardize_adjusted(raw: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """Map Yahoo's adjusted-close factor onto OHLC adjusted to the same basis."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    columns = {str(column).strip().lower(): column for column in raw.columns}
    required = {"open", "high", "low", "close", "adj close"}
    if not required.issubset(columns):
        return pd.DataFrame()
    close = pd.to_numeric(raw[columns["close"]], errors="coerce")
    adjusted_close = pd.to_numeric(raw[columns["adj close"]], errors="coerce")
    factor = adjusted_close / close.where(close > 0)
    frame = pd.DataFrame(
        {
            "stock_id": stock_id,
            "trade_date": pd.to_datetime(raw.index).strftime("%Y-%m-%d"),
            "open_adj": pd.to_numeric(raw[columns["open"]], errors="coerce") * factor,
            "high_adj": pd.to_numeric(raw[columns["high"]], errors="coerce") * factor,
            "low_adj": pd.to_numeric(raw[columns["low"]], errors="coerce") * factor,
            "close_adj": adjusted_close,
        }
    ).dropna(subset=["trade_date", *ADJUSTED_PRICE_COLUMNS[2:]])
    frame = frame.loc[
        (frame[list(ADJUSTED_PRICE_COLUMNS[2:])] > 0).all(axis=1)
        & (frame["high_adj"] >= frame["low_adj"])
    ]
    return frame[[*ADJUSTED_PRICE_COLUMNS]].reset_index(drop=True)


def _fetch_one(
    stock_id: str,
    start: str,
    end: str,
    downloader: Callable[..., pd.DataFrame],
    market: str,
) -> pd.DataFrame:
    ticker = to_yahoo_symbol(stock_id, market)
    raw = downloader(ticker, start=start, end=end, auto_adjust=False, progress=False)
    return _standardize(raw, stock_id)


def _slice_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker's sub-frame from a bulk download."""
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    try:
        sub = frame.xs(ticker, axis=1, level=1)
    except KeyError:
        return pd.DataFrame()
    if isinstance(sub.columns, pd.MultiIndex):  # defensive: deeper nesting.
        sub = sub.copy()
        sub.columns = sub.columns.get_level_values(0)
    return sub


def fetch_bulk_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    market_map: Mapping[str, str] | None = None,
    downloader: Callable[..., pd.DataFrame] = yf.download,
    batch: int = 150,
) -> pd.DataFrame:
    """Download many tickers in batches; failures land in ``attrs['failed']``.

    One bulk call per batch instead of one call per symbol (option C base:
    full-market daily prices, including pre-2020 history Shioaji lacks).
    """
    _require_range(start, end)
    if batch < 1:
        raise ValueError(f"invalid batch: {batch!r}")
    call = downloader
    codes = list(dict.fromkeys(s for s in symbols if isinstance(s, str) and s.strip()))
    mapping = dict(market_map) if market_map else {}
    tickers = {s: to_yahoo_symbol(s, mapping.get(s, "TWSE")) for s in codes}
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    unique_tickers = list(dict.fromkeys(tickers.values()))
    for offset in range(0, len(unique_tickers), batch):
        chunk = unique_tickers[offset : offset + batch]
        try:
            raw = call(chunk, start=start, end=end, auto_adjust=False, progress=False)
        except Exception:  # noqa: BLE001 - batch isolation like per-symbol.
            failed.extend(s for s, t in tickers.items() if t in chunk)
            continue
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            failed.extend(s for s, t in tickers.items() if t in chunk)
            continue
        for stock_id, ticker in tickers.items():
            if ticker not in chunk:
                continue
            if len(chunk) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                sub = raw
            else:
                sub = _slice_ticker(raw, ticker)
            standardized = _standardize(sub, stock_id)
            if standardized.empty:
                failed.append(stock_id)
                continue
            frames.append(standardized)
    if not frames:
        empty = pd.DataFrame(columns=[*PRICE_COLUMNS])
        empty.attrs["failed"] = sorted(set(failed))
        return empty
    merged = pd.concat(frames, ignore_index=True).sort_values(["stock_id", "trade_date"])
    merged.attrs["failed"] = sorted(set(failed))
    return merged.reset_index(drop=True)


def fetch_bulk_prices_with_adjustments(
    symbols: Sequence[str],
    start: str,
    end: str,
    market_map: Mapping[str, str] | None = None,
    downloader: Callable[..., pd.DataFrame] = yf.download,
    batch: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw bars and Yahoo adjusted OHLC in one batched download.

    The adjusted values are normalized to Yahoo's current basis. Callers
    combining them with another vendor's history must calibrate each stock
    against overlapping adjusted closes before storing them.
    """
    _require_range(start, end)
    if batch < 1:
        raise ValueError(f"invalid batch: {batch!r}")
    codes = list(dict.fromkeys(s for s in symbols if isinstance(s, str) and s.strip()))
    mapping = dict(market_map) if market_map else {}
    tickers = {stock_id: to_yahoo_symbol(stock_id, mapping.get(stock_id, "TWSE")) for stock_id in codes}
    raw_frames: list[pd.DataFrame] = []
    adjusted_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    unique_tickers = list(dict.fromkeys(tickers.values()))
    for offset in range(0, len(unique_tickers), batch):
        chunk = unique_tickers[offset : offset + batch]
        try:
            raw = downloader(chunk, start=start, end=end, auto_adjust=False, progress=False)
        except Exception:  # noqa: BLE001 - batch-level isolation.
            failed.extend(stock_id for stock_id, ticker in tickers.items() if ticker in chunk)
            continue
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            failed.extend(stock_id for stock_id, ticker in tickers.items() if ticker in chunk)
            continue
        for stock_id, ticker in tickers.items():
            if ticker not in chunk:
                continue
            sub = raw if len(chunk) == 1 and not isinstance(raw.columns, pd.MultiIndex) else _slice_ticker(raw, ticker)
            raw_frame = _standardize(sub, stock_id)
            adjusted_frame = _standardize_adjusted(sub, stock_id)
            if raw_frame.empty or adjusted_frame.empty:
                failed.append(stock_id)
                continue
            raw_frames.append(raw_frame)
            adjusted_frames.append(adjusted_frame)
    raw_result = (
        pd.concat(raw_frames, ignore_index=True).sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
        if raw_frames
        else pd.DataFrame(columns=list(PRICE_COLUMNS))
    )
    adjusted_result = (
        pd.concat(adjusted_frames, ignore_index=True)
        .sort_values(["stock_id", "trade_date"])
        .reset_index(drop=True)
        if adjusted_frames
        else pd.DataFrame(columns=list(ADJUSTED_PRICE_COLUMNS))
    )
    raw_result.attrs["failed"] = sorted(set(failed))
    adjusted_result.attrs["failed"] = sorted(set(failed))
    return raw_result, adjusted_result
