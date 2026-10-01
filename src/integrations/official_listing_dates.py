"""Official TWSE/TPEx company-master listing dates."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

import pandas as pd
import requests

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"


def fetch_official_listing_dates(
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and normalize formal TWSE/TPEx listing dates."""
    frames = [
        _fetch_one(
            TWSE_URL,
            "TWSE",
            ("上市日期", "DateOfListing", "listing_date"),
            requester,
            timeout,
        ),
        _fetch_one(
            TPEX_URL,
            "TPEX",
            ("上櫃日期", "DateOfListing", "listing_date"),
            requester,
            timeout,
        ),
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise RuntimeError("official listing-date feeds returned no valid rows")
    frame = frame.sort_values(["stock_id", "listed_date"]).drop_duplicates("stock_id", keep="first")
    return frame.reset_index(drop=True)


def _fetch_one(
    url: str,
    market: str,
    date_keys: tuple[str, ...],
    requester: Callable[..., requests.Response],
    timeout: float,
) -> pd.DataFrame:
    failure: Exception | None = None
    payload = None
    for _ in range(3):
        try:
            response = requester(
                url,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "taiwan-quant-xgb/1.0",
                },
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            failure = exc
    if payload is None:
        raise RuntimeError(f"{market} listing-date feed failed after retries") from failure
    if not isinstance(payload, list):
        raise RuntimeError(f"{market} listing-date feed returned an invalid envelope")
    records: list[dict[str, str]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        stock_id = _first(row, "公司代號", "stock_id", "SecuritiesCompanyCode")
        raw_date = _first(row, *date_keys)
        parsed = _parse_date(raw_date)
        if stock_id and parsed:
            records.append(
                {"stock_id": str(stock_id).strip(), "listed_date": parsed, "market": market}
            )
    if not records:
        raise RuntimeError(f"{market} listing-date feed returned no valid rows")
    return pd.DataFrame(records)


def _first(row: dict, *keys: str) -> object | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _parse_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    digits = re.sub(r"\D", "", text)
    try:
        if len(digits) == 8:
            parsed = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        elif len(digits) == 7:
            parsed = date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7]))
        else:
            return None
    except ValueError:
        return None
    return parsed.isoformat()
