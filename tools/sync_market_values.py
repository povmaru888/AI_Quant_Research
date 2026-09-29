"""Backfill daily TaiwanStockMarketValue with resumable checkpoints.

Usage:
    python tools/sync_market_values.py [--config config.yaml]
        [--start 2015-01-01] [--end 2026-09-22] [--delay 0.5]
        [--probe-date 2024-12-31] [--non-trading-day 2016-09-27] [--refresh]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.finmind_market_value import (  # noqa: E402
    FinMindMarketValueError,
    fetch_market_value_day,
)
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402


def _fetch_with_retry(trade_day: str, token: str, retries: int):
    for attempt in range(retries + 1):
        try:
            return fetch_market_value_day(trade_day, token)
        except PermissionError:
            raise
        except FinMindMarketValueError as exc:
            if not exc.temporary:
                raise
            if attempt >= retries:
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = min(60.0, float(retry_after) if retry_after else 1.5 * (2**attempt))
            print(f"[{trade_day}] temporary request error; retry {attempt + 1}/{retries} in {delay:.1f}s")
            time.sleep(delay)
    raise AssertionError("retry loop exhausted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync FinMind daily market values.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--probe-date", default=None)
    parser.add_argument(
        "--non-trading-day",
        action="append",
        default=[],
        help="confirmed closed date found in imported price rows; repeat as needed",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.delay < 0 or args.retries < 0:
        print("--delay and --retries must be non-negative", file=sys.stderr)
        return 2
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN; set it in .env", file=sys.stderr)
        return 1
    store = build_store(settings)
    if args.probe_date:
        try:
            rows = _fetch_with_retry(date.fromisoformat(args.probe_date).isoformat(), token, args.retries)
        except Exception as exc:  # noqa: BLE001
            print(f"FinMind access probe failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"FinMind access probe succeeded: day={args.probe_date} rows={len(rows)} "
            f"schema={','.join(rows.columns)}"
        )
        return 0
    try:
        start = date.fromisoformat(args.start).isoformat()
        end = date.fromisoformat(args.end).isoformat() if args.end else store.load_latest_trade_date()
    except ValueError as exc:
        print(f"invalid date: {exc}", file=sys.stderr)
        return 2
    if not end or start > end:
        print(f"invalid sync interval: {start}..{end}", file=sys.stderr)
        return 2
    days = [day for day in store.trade_days(start, end)]
    if not days:
        print(f"no price trading days in {start}..{end}", file=sys.stderr)
        return 1
    explicit_non_trading = set()
    for value in args.non_trading_day:
        try:
            day = date.fromisoformat(value).isoformat()
        except ValueError as exc:
            print(f"invalid --non-trading-day: {value!r}", file=sys.stderr)
            return 2
        if not start <= day <= end or day not in days:
            print(f"non-trading date {day} is outside the price-date range", file=sys.stderr)
            return 2
        explicit_non_trading.add(day)
    for day in sorted(explicit_non_trading):
        store.mark_market_value_sync_day(day, "non_trading")
    days = [day for day in days if day not in explicit_non_trading]
    completed = store.load_completed_market_value_days(start, end)
    pending = days if args.refresh else [day for day in days if day not in completed]
    print(
        f"market-value sessions: {len(days)} non_trading={len(explicit_non_trading)} "
        f"completed={len(days) - len(pending)} pending={len(pending)}"
    )
    failed: list[str] = []
    total_rows = 0
    for index, trade_day in enumerate(pending, 1):
        try:
            frame = _fetch_with_retry(trade_day, token, args.retries)
            digest = frame.attrs["source_content_hash"]
            source_rows = int(frame.attrs["source_row_count"])
            zero_rows = int(frame.attrs["zero_market_value_rows"])
            count = store.upsert_market_value_day(
                date.fromisoformat(trade_day), frame, source_content_hash=digest
            )
            total_rows += count
            print(
                f"[{index}/{len(pending)}] {trade_day} stored={count} "
                f"source_rows={source_rows} zero_excluded={zero_rows} hash={digest[:12]}",
                flush=True,
            )
        except PermissionError as exc:
            print(f"[{trade_day}] access denied; stopping Phase B sync: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - isolate days and resume.
            failed.append(trade_day)
            store.mark_market_value_sync_day(trade_day, "failed")
            print(f"[{trade_day}] FAILED: {exc}", file=sys.stderr)
        if index < len(pending) and args.delay:
            time.sleep(args.delay)
    print(f"done days={len(pending) - len(failed)}/{len(pending)} rows={total_rows} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
