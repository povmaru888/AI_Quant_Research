"""Sync market history in yearly chunks (bootstrap helper).

The daily job always syncs the full configured window in one call, which is
too large for a first bootstrap (millions of rows per feed). This script
loops over calendar-year windows calling the same sync_market_data service;
every chunk is idempotent, so reruns and interruptions are safe.

Usage:
    python tools/sync_history.py [--config config.yaml]
        [--start 2015-01-01] [--end 2026-09-22] [--chunk-years 1]

Needs FINMIND_TOKEN in the environment and a seeded stocks table
(run tools/seed_stocks.py first: sync needs load_symbols()).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.sync_service import sync_market_data  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402


def _chunks(start: date, end: date, chunk_years: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    cursor = start
    step = timedelta(days=1)
    while cursor <= end:
        chunk_end = min(date(cursor.year + chunk_years - 1, 12, 31), end)
        windows.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + step
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunked historical sync.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--chunk-years", type=int, default=1)
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        if start > end or args.chunk_years < 1:
            raise ValueError("bad range")
        settings = load_settings(args.config)
    except (ValueError, TypeError) as exc:
        print(f"bad arguments: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN: copy .env.example to .env and fill it", file=sys.stderr)
        return 1
    store = build_store(settings)
    symbols = store.load_symbols()
    if not symbols:
        print("no symbols: run tools/seed_stocks.py first", file=sys.stderr)
        return 1
    print(f"syncing {len(symbols)} symbols in {args.chunk_years}y chunks")
    for chunk_start, chunk_end in _chunks(start, end, args.chunk_years):
        run_id = f"daily-bootstrap-{chunk_start}"
        store.start_run(
            {"run_id": run_id, "job": "daily_update-bootstrap", "data_end_date": chunk_end}
        )
        try:
            summary = sync_market_data(
                chunk_start,
                chunk_end,
                settings,
                store,
                run_id,
                token,
                symbols=symbols,
            )
        except Exception as exc:
            store.finish_run(run_id, "failed", str(exc))
            print(f"chunk {chunk_start}..{chunk_end} FAILED: {exc}", file=sys.stderr)
            return 1
        if not summary.ok:
            errors = "; ".join(
                f"{feed.name}: {feed.error}"
                for feed in (summary.prices, summary.financials, summary.institutional)
                if feed.error is not None
            )
            store.finish_run(run_id, "failed", errors)
            print(f"chunk {chunk_start}..{chunk_end} PARTIAL: {errors}", file=sys.stderr)
            return 1
        store.finish_run(run_id, "succeeded")
        print(
            f"chunk {chunk_start}..{chunk_end} ok "
            f"prices={summary.prices.rows} financials={summary.financials.rows} "
            f"institutional={summary.institutional.rows} "
            f"fallback={summary.fallback_used}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
