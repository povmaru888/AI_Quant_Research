"""P3-01: FinMind price client (SDD 7.1).

Fetches ``TaiwanStockPrice`` and maps vendor fields to the standard
price columns used by the Price Repository (P1-07). Vendor field names
never reach downstream modules.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import requests

from integrations.finmind import _get

DATASET = "TaiwanStockPrice"
ADJ_DATASET = "TaiwanStockPriceAdj"
SOURCE = "finmind"

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

ADJ_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "trade_date",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
)

_FIELD_MAP = {
    "stock_id": "stock_id",
    "date": "trade_date",
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
    "Trading_money": "traded_value",
}


def fetch_prices(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
    stock_id: str | None = None,
) -> pd.DataFrame:
    """Fetch and standardize prices; invalid bars are dropped (SDD 7.3)."""
    rows = _get(DATASET, start, end, token, requester, timeout, data_id=stock_id)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*PRICE_COLUMNS])
    missing = [c for c in _FIELD_MAP if c not in frame.columns]
    if missing:
        raise ValueError(f"finmind {DATASET} missing fields {missing}")
    frame = frame.rename(columns=_FIELD_MAP)
    numeric = ["open", "high", "low", "close", "volume", "traded_value"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "trade_date", *numeric])
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame[["volume", "traded_value"]] >= 0).all(axis=1)
        & (frame["high"] >= frame["low"])
    ]
    frame["source"] = SOURCE
    return frame[[*PRICE_COLUMNS]].reset_index(drop=True)


_ADJ_FIELD_MAP = {
    "stock_id": "stock_id",
    "date": "trade_date",
    "open": "open_adj",
    "max": "high_adj",
    "min": "low_adj",
    "close": "close_adj",
}


def fetch_price_adj(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
    stock_id: str | None = None,
) -> pd.DataFrame:
    """Fetch back-adjusted OHLC; rows without a raw bar are the caller's to skip.

    Adjusted closes are back-computed to the latest trading day, so the
    event day's adj price equals its raw price and history shifts. Bars
    with non-positive adjusted values are dropped (SDD 7.3).
    """
    rows = _get(ADJ_DATASET, start, end, token, requester, timeout, data_id=stock_id)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*ADJ_COLUMNS])
    missing = [c for c in _ADJ_FIELD_MAP if c not in frame.columns]
    if missing:
        raise ValueError(f"finmind {ADJ_DATASET} missing fields {missing}")
    frame = frame.rename(columns=_ADJ_FIELD_MAP)
    numeric = ["open_adj", "high_adj", "low_adj", "close_adj"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "trade_date", *numeric])
    frame = frame.loc[
        (frame[["open_adj", "high_adj", "low_adj", "close_adj"]] > 0).all(axis=1)
        & (frame["high_adj"] >= frame["low_adj"])
    ]
    return frame[["stock_id", "trade_date", *numeric]].reset_index(drop=True)
