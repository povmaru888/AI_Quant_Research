"""Per-symbol sync for free ("register") FinMind tokens.

Free tokens reject full-market requests, so this script loops over symbols
calling the same fetcher functions with ``stock_id`` (forwarded as
``data_id``). Float shares are skipped (gated even per-stock). Every
upsert is idempotent, so reruns resume safely; pass --symbols-file with
the failures to retry a subset.

Usage:
    python tools/sync_free.py [--config config.yaml]
        [--start 2015-01-01] [--end 2026-09-22]
        [--feeds prices|financials|institutional|all]
        [--symbols 2330,2317] [--symbols-file failed.txt]
        [--limit 10] [--offset 0] [--delay 0.2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.finmind_fundamentals import (  # noqa: E402
    fetch_financials,
    fetch_institutional,
)
from integrations.finmind_prices import fetch_prices  # noqa: E402
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402

_FEEDS = ("prices", "financials", "institutional")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-symbol free-tier sync.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--feeds", default="all")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args(argv)
    load_dotenv()
    feeds = _FEEDS if args.feeds == "all" else tuple(args.feeds.split(","))
    if any(f not in _FEEDS for f in feeds) or not feeds:
        print(f"bad --feeds: {args.feeds!r}", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN: copy .env.example to .env and fill it", file=sys.stderr)
        return 1
    from datetime import date as _date

    end = args.end or _date.today().isoformat()
    store = build_store(settings)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_file:
        symbols = [
            line.strip()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        symbols = store.load_symbols()
    symbols = symbols[args.offset :]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("no symbols to sync", file=sys.stderr)
        return 1
    run_id = f"daily-free-{args.start}-{end}"
    try:
        store.start_run({"run_id": run_id, "job": "daily_update-free", "data_end_date": end})
    except ValueError:
        pass  # rerun of the same window reuses the run id.
    totals = dict.fromkeys(feeds, 0)
    failed: list[str] = []
    for index, stock_id in enumerate(symbols, 1):
        try:
            if "prices" in feeds:
                frame = fetch_prices(args.start, end, token, stock_id=stock_id)
                totals["prices"] += store.upsert_prices(frame)
            if "financials" in feeds:
                frame = fetch_financials(args.start, end, token, stock_id=stock_id)
                totals["financials"] += store.upsert_financials(frame)
            if "institutional" in feeds:
                frame = fetch_institutional(
                    args.start, end, token, stock_id=stock_id, include_floats=False
                )
                totals["institutional"] += store.upsert_institutional(frame)
        except Exception as exc:  # noqa: BLE001 - per-symbol isolation.
            failed.append(stock_id)
            print(f"[{index}/{len(symbols)}] {stock_id} FAILED: {exc}", file=sys.stderr)
        else:
            if index % 50 == 0 or index == len(symbols):
                print(f"[{index}/{len(symbols)}] ok rows={totals} failed={len(failed)}")
        time.sleep(args.delay)
    try:
        if failed:
            store.finish_run(run_id, "failed", f"{len(failed)} symbols: {failed[:10]}")
        else:
            store.finish_run(run_id, "succeeded")
    except ValueError:
        pass
    print(f"done rows={totals} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
