"""Bulk price base from yfinance (option C foundation).

One bulk download per batch of tickers instead of per-symbol calls:
full-market daily prices in ~15 requests, including the pre-2020 history
Shioaji lacks. Shioaji later overlays blue chips with authoritative
amounts (see tools/sync_shioaji.py). Every upsert is idempotent; symbols
already at/after --end are skipped via one GROUP BY query.

Usage:
    python tools/sync_yfinance.py [--config config.yaml]
        [--start 2015-01-01] [--end 2026-09-22]
        [--symbols 2330,2317] [--limit 50] [--offset 0]
        [--batch 150] [--full]

yfinance carries prices only (no financials/institutional): price-driven
factors work after this; fundamental/chip factors await the FinMind quota
or the single-file bulk import (docs/bulk-import.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.yfinance_prices import fetch_bulk_prices  # noqa: E402
from runtime.cli import parse_symbols  # noqa: E402
from runtime.db_store import build_store  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="yfinance bulk price base.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch", type=int, default=150)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from datetime import date as _date

    from settings import load_settings

    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    end = args.end or _date.today().isoformat()
    if args.start > end:
        print(f"bad range: {args.start}..{end}", file=sys.stderr)
        return 2
    store = build_store(settings)
    if args.symbols:
        symbols = parse_symbols(args.symbols)
    elif args.symbols_file:
        symbols = [
            line.strip()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        symbols = store.load_symbols()
    if not args.full:
        symbols = _resume_filter(settings, symbols, end)
    symbols = symbols[args.offset :]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("no symbols to sync", file=sys.stderr)
        return 1
    market_map = _market_map(settings)
    run_id = f"daily-yfinance-{args.start}-{end}"
    try:
        store.start_run({"run_id": run_id, "job": "daily_update-yfinance", "data_end_date": end})
    except Exception:  # noqa: BLE001 - rerun reuses the run id.
        pass
    total, failed = 0, []
    for offset in range(0, len(symbols), args.batch):
        chunk = symbols[offset : offset + args.batch]
        try:
            frame = fetch_bulk_prices(chunk, args.start, end, market_map, batch=args.batch)
        except Exception as exc:  # noqa: BLE001 - batch isolation.
            print(f"batch @{offset} FAILED: {exc}", file=sys.stderr)
            failed.extend(chunk)
            continue
        batch_failed = frame.attrs.get("failed", [])
        failed.extend(batch_failed)
        total += store.upsert_prices(frame)
        print(
            f"[{min(offset + args.batch, len(symbols))}/{len(symbols)}] "
            f"rows={total} failed={len(failed)}"
        )
        time.sleep(1.0)
    try:
        if failed:
            store.finish_run(run_id, "failed", f"{len(failed)} symbols failed")
        else:
            store.finish_run(run_id, "succeeded")
    except ValueError:
        pass
    if failed:
        skip_path = Path("logs") / "yfinance_skip.txt"
        skip_path.write_text("\n".join(sorted(set(failed))) + "\n", encoding="utf-8")
        print(f"failed list: {skip_path}")
    print(f"done rows={total} failed={len(failed)}")
    return 0 if not failed else 1


def _market_map(settings) -> dict[str, str]:
    """Map stock_id -> TWSE/TPEX for Yahoo ticker suffixes."""
    from sqlalchemy import select

    from database import create_engine_from_settings, session_scope
    from models.security import Stock

    engine = create_engine_from_settings(settings)
    try:
        with session_scope(engine) as session:
            rows = session.execute(select(Stock.stock_id, Stock.market)).all()
    finally:
        engine.dispose()
    return {s: ("TPEX" if m == "TPEX" else "TWSE") for s, m in rows}


def _resume_filter(settings, symbols: list[str], end: str) -> list[str]:
    """Drop symbols whose stored history already reaches ``end``."""
    from sqlalchemy import func, select

    from database import create_engine_from_settings, session_scope
    from models.market import Price

    engine = create_engine_from_settings(settings)
    try:
        with session_scope(engine) as session:
            latest = dict(
                session.execute(
                    select(Price.stock_id, func.max(Price.trade_date)).group_by(Price.stock_id)
                ).all()
            )
    finally:
        engine.dispose()
    kept = [s for s in symbols if (latest.get(s) or "") < end]
    print(f"resume: {len(symbols) - len(kept)} symbols already at {end}")
    return kept


if __name__ == "__main__":
    raise SystemExit(main())
