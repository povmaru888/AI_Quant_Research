"""Date-specific TWSE and TPEx official daily price adapters."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
import requests

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
_SESSION = requests.Session()
_REQUEST_LOCK = Lock()
_LAST_REQUEST: dict[str, float] = {}

PRICE_COLUMNS = (
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


@dataclass(frozen=True)
class OfficialPriceDay:
    frame: pd.DataFrame
    content_hash: str
    cache_hit: bool


_ALIASES = {
    "stock_id": ("證券代號", "代號", "股票代號"),
    "open": ("開盤價", "開盤"),
    "high": ("最高價", "最高"),
    "low": ("最低價", "最低"),
    "close": ("收盤價", "收盤"),
    "volume": ("成交股數", "成交量"),
    "traded_value": ("成交金額", "成交金額(元)", "成交金額（元）"),
}


def _require_day(day: str) -> str:
    try:
        return date.fromisoformat(day).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid trade date: {day!r}") from exc


def _field_index(fields: list[str], name: str) -> int:
    normalized = [str(field).strip().replace(" ", "") for field in fields]
    for alias in _ALIASES[name]:
        target = alias.replace(" ", "")
        if target in normalized:
            return normalized.index(target)
    raise ValueError(f"official price schema missing {name}; fields={fields}")


def _number(value: Any) -> float:
    text = str(value).strip().replace(",", "").replace("=", "")
    if text in {"", "--", "---", "除權", "除息", "除權息"}:
        return float("nan")
    return float(text)


def _table(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    candidates = payload.get("tables")
    if not isinstance(candidates, list):
        candidates = [payload]
    required = set(_ALIASES)
    for table in candidates:
        if not isinstance(table, dict):
            continue
        fields = table.get("fields")
        rows = table.get("data")
        if not isinstance(fields, list) or not isinstance(rows, list):
            continue
        found = set()
        for name in required:
            try:
                _field_index(fields, name)
                found.add(name)
            except ValueError:
                pass
        if found == required:
            return [str(field) for field in fields], rows
    raise ValueError("official response contains no all-stock OHLCV table")


def _parse(payload: dict[str, Any], day: str, market: str) -> pd.DataFrame:
    response_day = payload.get("date")
    if response_day is not None:
        digits = "".join(character for character in str(response_day) if character.isdigit())
        expected = day.replace("-", "")
        if len(digits) == 7:  # ROC year, for example 1150921.
            digits = f"{int(digits[:3]) + 1911:04d}{digits[3:]}"
        if len(digits) == 8 and digits != expected:
            raise ValueError(
                f"{market} official response date mismatch: expected {day}, got {response_day}"
            )
    fields, rows = _table(payload)
    indexes = {name: _field_index(fields, name) for name in _ALIASES}
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) <= max(indexes.values()):
            continue
        stock_id = str(row[indexes["stock_id"]]).strip().replace("=", "")
        if not stock_id:
            continue
        try:
            values = {
                name: _number(row[index]) for name, index in indexes.items() if name != "stock_id"
            }
        except (TypeError, ValueError):
            continue
        if (
            any(
                pd.isna(values[name]) or values[name] <= 0
                for name in ("open", "high", "low", "close")
            )
            or values["volume"] < 0
            or values["traded_value"] < 0
            or values["high"] < values["low"]
        ):
            continue
        records.append(
            {
                "stock_id": stock_id,
                "trade_date": day,
                **values,
                "source": f"{market}:official-daily-close",
            }
        )
    frame = pd.DataFrame(records, columns=list(PRICE_COLUMNS))
    if frame.empty:
        raise ValueError(f"{market} official response has no valid price rows for {day}")
    if frame["stock_id"].duplicated().any():
        raise ValueError(f"{market} official response contains duplicate stock IDs for {day}")
    return frame.sort_values("stock_id", ignore_index=True)


def _load_or_fetch(
    market: str,
    day: str,
    cache_dir: Path,
    requester: Callable[..., requests.Response],
    timeout: float,
) -> tuple[dict[str, Any], str, bool]:
    cache_path = cache_dir / market.lower() / f"{day}.json.gz"
    if cache_path.is_file():
        try:
            raw = gzip.decompress(cache_path.read_bytes())
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("cached response is not an object")
            return payload, hashlib.sha256(raw).hexdigest(), True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            cache_path.unlink(missing_ok=True)
    if market == "TWSE":
        url = TWSE_URL
        params = {"date": day.replace("-", ""), "type": "ALL", "response": "json"}
    elif market == "TPEX":
        url = TPEX_URL
        roc = f"{int(day[:4]) - 1911:03d}/{day[5:7]}/{day[8:10]}"
        params = {"date": roc, "id": "", "response": "json"}
    else:
        raise ValueError(f"unsupported market: {market}")
    is_default_session = getattr(requester, "__self__", None) is _SESSION
    response = None
    for attempt in range(3):
        if is_default_session:
            with _REQUEST_LOCK:
                wait = 3.0 - (time.monotonic() - _LAST_REQUEST.get(market, 0.0))
                if wait > 0:
                    time.sleep(wait)
                _LAST_REQUEST[market] = time.monotonic()
        response = requester(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0 AI-Quant-Research/1.0"},
            timeout=timeout,
        )
        if response.status_code == 200:
            break
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
        delay = float(retry_after) if retry_after else min(8.0, 1.5 * (2**attempt))
        time.sleep(delay)
    assert response is not None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{market} returned a non-object response")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(raw))
    temporary.replace(cache_path)
    return payload, hashlib.sha256(raw).hexdigest(), False


def fetch_official_price_day(
    market: str,
    day: str,
    *,
    cache_dir: str | Path = "data/cache/official_prices",
    requester: Callable[..., requests.Response] = _SESSION.get,
    timeout: float = 30.0,
) -> OfficialPriceDay:
    """Fetch and standardize one official full-market daily price response."""
    parsed_day = _require_day(day)
    normalized_market = market.upper()
    payload, content_hash, cache_hit = _load_or_fetch(
        normalized_market, parsed_day, Path(cache_dir), requester, timeout
    )
    frame = _parse(payload, parsed_day, normalized_market)
    return OfficialPriceDay(frame=frame, content_hash=content_hash, cache_hit=cache_hit)
