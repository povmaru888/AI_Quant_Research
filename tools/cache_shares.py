"""Cache shares outstanding per stock (yfinance fast_info, one-time + refresh).

Free FinMind tokens cannot fetch float shares, but the universe needs a
market-cap gate. This script caches ``{shares, market_cap, as_of}`` per
stock into ``database/shares.json`` (next to the DB file, gitignored).
ETFs publish no share count; their market cap is cached directly.

Usage:
    python tools/cache_shares.py [--config config.yaml] [--limit 5]
        [--refresh] [--delay 0.2]

Idempotent: entries are merged; --refresh refetches everything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from database import create_engine_from_settings, session_scope  # noqa: E402
from integrations.yfinance_prices import to_yahoo_symbol  # noqa: E402
from models.security import Stock  # noqa: E402
from runtime.cli import parse_symbols  # noqa: E402
from runtime.db_store import default_shares_path  # noqa: E402
from settings import load_settings  # noqa: E402


def _fetch_one(ticker: str, delay: float) -> dict | None:
    import yfinance as yf

    for _ in range(3):
        try:
            info = yf.Ticker(ticker).fast_info
            shares = info.shares
            cap = info.market_cap
            if shares is None and cap is None:
                return None
            return {"shares": shares, "market_cap": cap}
        except Exception:  # noqa: BLE001 - per-symbol isolation.
            time.sleep(delay * 5)
    return None


def _fetch_etf_assets(stock_id: str, ticker: str) -> dict | None:
    """Slow-info fallback for ETFs: totalAssets stands in for market cap.

    Only for short codes (plain stocks/ETFs); 6+ char codes are auction
    notes and warrants Yahoo never carries.
    """
    import yfinance as yf

    if len(stock_id) > 5:
        return None
    try:
        assets = yf.Ticker(ticker).info.get("totalAssets")
    except Exception:  # noqa: BLE001 - per-symbol isolation.
        return None
    if not isinstance(assets, (int, float)) or not assets > 0:
        return None
    return {"shares": None, "market_cap": float(assets)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache shares outstanding.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--skip-file", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    cache_path = default_shares_path(settings.data.database_url)
    if cache_path is None:
        print("shares cache needs a file database_url", file=sys.stderr)
        return 2
    cache: dict = {}
    if cache_path.is_file():
        # Always merge: a --symbols subset run must never wipe the cache.
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    engine = create_engine_from_settings(settings)
    try:
        with session_scope(engine) as session:
            universe = session.execute(
                select(Stock.stock_id, Stock.market).order_by(Stock.stock_id)
            ).all()
    finally:
        engine.dispose()
    symbols = [(s, m) for s, m in universe if s != "TAIEX"]
    if args.symbols:
        wanted = set(parse_symbols(args.symbols))
        symbols = [(s, m) for s, m in symbols if s in wanted]
    if args.skip_file and Path(args.skip_file).is_file():
        skipped = set(Path(args.skip_file).read_text(encoding="utf-8").split())
        symbols = [(s, m) for s, m in symbols if s not in skipped]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    today = date.today().isoformat()
    done, failed = 0, []
    for stock_id, market in symbols:
        if stock_id in cache and not args.refresh:
            done += 1
            continue
        ticker = to_yahoo_symbol(stock_id, market if market != "UNKNOWN" else "TWSE")
        entry = None
        if market != "TPEX":
            entry = _fetch_one(ticker, args.delay)
        if entry is None:
            entry = _fetch_one(to_yahoo_symbol(stock_id, "TPEX"), args.delay)
        if entry is None:
            entry = _fetch_etf_assets(stock_id, ticker)
        time.sleep(args.delay)
        if entry is None:
            failed.append(stock_id)
            continue
        cache[stock_id] = {**entry, "as_of": today}
        done += 1
        if done % 100 == 0:
            print(f"cached {done}/{len(symbols)} (failed {len(failed)})")
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"shares cache: {cache_path} entries={len(cache)} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
