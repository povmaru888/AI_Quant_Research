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

from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.finmind_fundamentals import (  # noqa: E402
    fetch_financials,
    fetch_institutional,
)
from integrations.finmind_prices import fetch_price_adj, fetch_prices  # noqa: E402
from runtime.cli import parse_symbols  # noqa: E402
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402

_FEEDS = ("prices", "price_adj", "financials", "institutional")


def is_gated(exc: Exception) -> bool:
    """Permanent instrument gate (HTTP 402/403 without a ban notice)."""
    message = str(exc)
    return "HTTP 402" in message or "HTTP 403" in message


def ban_wait_seconds(exc: Exception) -> float | None:
    """Server-requested backoff for an IP ban; None when not a ban."""
    wait = getattr(exc, "retry_after", None)
    if "banned" in str(exc).lower() and isinstance(wait, (int, float)) and wait > 0:
        return float(wait)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-symbol free-tier sync.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--feeds", default="prices,financials,institutional")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-file", default="logs/skip_402.txt")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--gate-burst", type=int, default=8)
    parser.add_argument("--max-sleeps", type=int, default=5)
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
        symbols = parse_symbols(args.symbols)
    elif args.symbols_file:
        symbols = [
            line.strip()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        symbols = store.load_symbols()
    if args.skip_file and Path(args.skip_file).is_file():
        skipped_known = set(Path(args.skip_file).read_text(encoding="utf-8").split())
        symbols = [s for s in symbols if s not in skipped_known]
        print(f"skipping {len(skipped_known)} known-402 symbols")
    symbols = symbols[args.offset :]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("no symbols to sync", file=sys.stderr)
        return 1
    run_id = f"daily-free-{args.start}-{end}"
    try:
        store.start_run({"run_id": run_id, "job": "daily_update-free", "data_end_date": end})
    except (ValueError, IntegrityError):
        pass  # rerun of the same window reuses the run id.
    totals = dict.fromkeys(feeds, 0)
    failed: list[str] = []
    skipped: list[str] = []
    gates_run = 0
    sleeps = 0

    def sync_one(stock_id: str) -> None:
        if "prices" in feeds:
            frame = fetch_prices(args.start, end, token, stock_id=stock_id)
            totals["prices"] += store.upsert_prices(frame)
        if "price_adj" in feeds:
            frame = fetch_price_adj(args.start, end, token, stock_id=stock_id)
            totals["price_adj"] += store.upsert_price_adj(frame)
        if "financials" in feeds:
            frame = fetch_financials(args.start, end, token, stock_id=stock_id)
            totals["financials"] += store.upsert_financials(frame)
        if "institutional" in feeds:
            frame = fetch_institutional(
                args.start, end, token, stock_id=stock_id, include_floats=False
            )
            totals["institutional"] += store.upsert_institutional(frame)

    def abort_run(reason: str) -> int:
        # Never write this run's skip list: entries may be ban victims.
        try:
            store.finish_run(run_id, "failed", reason)
        except ValueError:
            pass
        print(f"aborted rows={totals} failed={failed}", file=sys.stderr)
        return 2

    ban_sleeps = 0
    for index, stock_id in enumerate(symbols, 1):
        try:
            sync_one(stock_id)
        except Exception as exc:  # noqa: BLE001 - per-symbol isolation.
            wait = ban_wait_seconds(exc)
            if wait is not None:
                if ban_sleeps >= 3:
                    return abort_run("aborted: IP ban persists")
                print(f"banned: sleeping {wait + 30:.0f}s then retrying {stock_id}")
                time.sleep(wait + 30)
                ban_sleeps += 1
                try:
                    sync_one(stock_id)
                except Exception as retry_exc:  # noqa: BLE001
                    if ban_wait_seconds(retry_exc) is not None:
                        return abort_run("aborted: IP ban persists")
                    exc = retry_exc
                else:
                    gates_run = 0
                    time.sleep(args.delay)
                    continue
            if is_gated(exc):
                gates_run += 1
                if gates_run >= args.gate_burst:
                    # Burst of gates = maybe throttling: sleep, retry once.
                    if sleeps >= args.max_sleeps:
                        return abort_run("aborted: gate burst persists")
                    print(f"gate burst x{gates_run}: sleeping 60s then retrying")
                    time.sleep(60)
                    sleeps += 1
                    try:
                        sync_one(stock_id)
                    except Exception as retry_exc:  # noqa: BLE001
                        if not is_gated(retry_exc):
                            failed.append(stock_id)
                            print(
                                f"[{index}/{len(symbols)}] {stock_id} FAILED: {retry_exc}",
                                file=sys.stderr,
                            )
                        else:
                            skipped.append(stock_id)
                    gates_run = 0
                    time.sleep(args.delay)
                    continue
                skipped.append(stock_id)
                time.sleep(args.delay)
                continue
            failed.append(stock_id)
            print(f"[{index}/{len(symbols)}] {stock_id} FAILED: {exc}", file=sys.stderr)
        else:
            gates_run = 0
            if index % 50 == 0 or index == len(symbols):
                print(
                    f"[{index}/{len(symbols)}] ok rows={totals} "
                    f"failed={len(failed)} skipped={len(skipped)}"
                )
        time.sleep(args.delay)
    if skipped:
        skip_path = Path("logs") / "skip_402.txt"
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        known = set()
        if skip_path.is_file():
            known = set(skip_path.read_text(encoding="utf-8").split())
        with skip_path.open("a", encoding="utf-8") as handle:
            for stock_id in skipped:
                if stock_id not in known:
                    handle.write(stock_id + "\n")
                    known.add(stock_id)
    try:
        if failed:
            store.finish_run(run_id, "failed", f"{len(failed)} symbols: {failed[:10]}")
        else:
            store.finish_run(run_id, "succeeded")
    except ValueError:
        pass
    print(f"done rows={totals} failed={failed} skipped={len(skipped)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
