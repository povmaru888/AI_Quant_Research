"""Seed the stocks table from FinMind TaiwanStockInfo plus the TAIEX index.

Usage:
    python tools/seed_stocks.py [--config config.yaml] [--start 2015-01-01]

Needs FINMIND_TOKEN in the environment (see .env.example). The TAIEX
history (pseudo stock_id "TAIEX", needed for market beta and the MA60
regime filter) comes from FinMind TaiwanVariousIndices when available,
falling back to yfinance ^TWII. Idempotent: all writes are upserts.

Run this BEFORE the first sync: sync_market_data needs load_symbols(),
and prices have a foreign key to stocks.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import create_engine_from_settings, session_scope  # noqa: E402
from integrations.finmind import _get  # noqa: E402
from repositories import prices as prices_repo  # noqa: E402
from repositories import stocks as stocks_repo  # noqa: E402
from runtime.db_store import TAIEX_ID  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402

INFO_DATASET = "TaiwanStockInfo"
INDICES_DATASET = "TaiwanVariousIndices"

_MARKET_MAP = {"twse": "TWSE", "tpex": "TPEX", "otc": "TPEX", "emerging": "TPEX"}


def _map_market(raw: object) -> str:
    return _MARKET_MAP.get(str(raw).strip().lower(), "UNKNOWN")


def _seed_info(engine, token: str) -> int:
    rows = _get(INFO_DATASET, "2024-01-01", date.today().isoformat(), token)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"{INFO_DATASET} returned no rows")
    print(f"{INFO_DATASET} columns: {list(frame.columns)}")
    if "type" in frame.columns:
        unknown = sorted(set(frame["type"].astype(str)) - set(_MARKET_MAP))
        if unknown:
            print(f"unmapped market types (stored as UNKNOWN): {unknown}")
    records = pd.DataFrame(
        {
            "stock_id": frame["stock_id"].astype(str).str.strip(),
            "stock_name": frame.get("stock_name"),
            "market": frame.get("type", "UNKNOWN").map(_map_market)
            if "type" in frame.columns
            else "UNKNOWN",
            "industry": frame.get("industry_category"),
        }
    )
    records = records.loc[records["stock_id"] != ""].drop_duplicates("stock_id")
    with session_scope(engine) as session:
        return stocks_repo.upsert_stocks(session, records)


def _seed_taiex_finmind(engine, token: str, start: str) -> int:
    rows = _get(INDICES_DATASET, start, date.today().isoformat(), token)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return 0
    print(f"{INDICES_DATASET} columns: {list(frame.columns)}")
    taiex = frame.loc[frame["stock_id"] == "TAIEX"].copy()
    if taiex.empty:
        return 0
    rename = {"date": "trade_date"}
    taiex = taiex.rename(columns=rename)
    for column in ("open", "high", "low", "close"):
        if column not in taiex.columns:
            return 0
    prices = pd.DataFrame(
        {
            "stock_id": TAIEX_ID,
            "trade_date": taiex["trade_date"].astype(str),
            "open": pd.to_numeric(taiex["open"], errors="coerce"),
            "high": pd.to_numeric(taiex["high"], errors="coerce"),
            "low": pd.to_numeric(taiex["low"], errors="coerce"),
            "close": pd.to_numeric(taiex["close"], errors="coerce"),
            "volume": 0.0,
            "traded_value": 0.0,
            "source": "finmind",
        }
    ).dropna(subset=["trade_date", "open", "high", "low", "close"])
    prices = prices.loc[
        (prices[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (prices["high"] >= prices["low"])
    ]
    return _store_taiex(engine, prices, "finmind")


def _seed_taiex_yahoo(engine, start: str) -> int:
    import yfinance as yf

    raw = yf.download(
        "^TWII", start=start, end=date.today().isoformat(), auto_adjust=False, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance ^TWII returned no rows")
    columns = {str(c).strip().lower(): c for c in raw.columns}
    for key in ("open", "high", "low", "close"):
        if key not in columns:
            raise RuntimeError(f"yfinance ^TWII missing {key}")
    prices = pd.DataFrame(
        {
            "stock_id": TAIEX_ID,
            "trade_date": pd.to_datetime(raw.index).strftime("%Y-%m-%d"),
            "open": pd.to_numeric(raw[columns["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[columns["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[columns["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[columns["close"]], errors="coerce"),
            "volume": 0.0,
            "traded_value": 0.0,
            "source": "yfinance",
        }
    ).dropna(subset=["trade_date", "open", "high", "low", "close"])
    prices = prices.loc[
        (prices[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (prices["high"] >= prices["low"])
    ]
    # Index volume is unreliable; zeros pass the volume >= 0 CHECK.
    return _store_taiex(engine, prices, "yfinance")


def _store_taiex(engine, prices: pd.DataFrame, source: str) -> int:
    if prices.empty:
        raise RuntimeError(f"TAIEX {source} feed produced no valid rows")
    with session_scope(engine) as session:
        stocks_repo.upsert_stocks(
            session,
            pd.DataFrame({"stock_id": [TAIEX_ID], "stock_name": ["加權指數"], "market": ["INDEX"]}),
        )
        return prices_repo.upsert_prices(session, prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed stocks + TAIEX history.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN: copy .env.example to .env and fill it", file=sys.stderr)
        return 1
    engine = create_engine_from_settings(settings)
    try:
        count = _seed_info(engine, token)
        print(f"stocks seeded: {count}")
        try:
            taiex_rows = _seed_taiex_finmind(engine, token, args.start)
            print(f"TAIEX rows from FinMind: {taiex_rows}")
        except Exception as exc:
            print(f"FinMind indices unavailable ({exc}); trying yfinance ^TWII")
            taiex_rows = _seed_taiex_yahoo(engine, args.start)
            print(f"TAIEX rows from yfinance: {taiex_rows}")
        if taiex_rows == 0:
            print("FinMind indices feed had no TAIEX rows; trying yfinance ^TWII")
            taiex_rows = _seed_taiex_yahoo(engine, args.start)
            print(f"TAIEX rows from yfinance: {taiex_rows}")
    except Exception as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
