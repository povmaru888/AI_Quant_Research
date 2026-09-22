"""Backfill daily prices from Shioaji minute kbars (2020-03-02 and later).

Shioaji serves minute kbars (max 30 days per call); this script resamples
to daily bars via integrations.shioaji_prices and upserts idempotently, so
reruns resume safely. Symbols already at/after --end are skipped via one
GROUP BY query. Pace stays under the 10s/50-call quote limit; api.usage()
is checked every 50 symbols and the run aborts before burning the day's
traffic quota.

Usage:
    python tools/sync_shioaji.py [--config config.yaml]
        [--start 2020-03-02] [--end 2026-09-22]
        [--symbols 2330,2317] [--limit 10] [--offset 0]
        [--taiex] [--simulation | --no-simulation] [--delay 0.3]

Needs SHIOAJI_API_KEY/SHIOAJI_SECRET_KEY in the environment (.env).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.shioaji_prices import (  # noqa: E402
    ShioajiError,
    fetch_daily_prices,
    fetch_taiex_daily,
)
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import load_settings  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shioaji price backfill.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2020-03-02")
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--taiex", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument(
        "--full", action="store_true", help="Ignore resume filter and refetch every symbol."
    )
    parser.add_argument(
        "--simulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simulation key (default). Pass --no-simulation for production.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("missing SHIOAJI_API_KEY/SECRET_KEY in .env", file=sys.stderr)
        return 1
    from datetime import date as _date

    end = args.end or _date.today().isoformat()
    if args.start > end:
        print(f"bad range: {args.start}..{end}", file=sys.stderr)
        return 2
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
    if not args.full:
        symbols = _resume_filter(settings, symbols, end)
    symbols = symbols[args.offset :]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols and not args.taiex:
        print("no symbols to sync", file=sys.stderr)
        return 1
    try:
        import shioaji as sj
    except ImportError:
        print("shioaji not installed: pip install shioaji", file=sys.stderr)
        return 1
    api = sj.Shioaji(simulation=args.simulation)
    try:
        api.login(api_key, secret_key)
        usage = api.usage()
        print(
            f"shioaji quota: used {usage.bytes / 1024 / 1024:.1f}MB / "
            f"limit {usage.limit_bytes / 1024 / 1024:.0f}MB "
            f"({'simulation' if args.simulation else 'production'})"
        )
        if usage.remaining_bytes < usage.limit_bytes * 0.1:
            print("quota below 10%: aborting", file=sys.stderr)
            return 1
        run_id = f"daily-shioaji-{args.start}-{end}"
        try:
            store.start_run({"run_id": run_id, "job": "daily_update-shioaji", "data_end_date": end})
        except Exception:  # noqa: BLE001 - rerun reuses the run id.
            pass
        done_rows = _sync_prices(api, store, symbols, args, end)
        taiex_rows = 0
        if args.taiex:
            frame = fetch_taiex_daily(args.start, end, api)
            taiex_rows = store.upsert_prices(frame)
            print(f"TAIEX rows: {taiex_rows}")
        try:
            store.finish_run(run_id, "succeeded")
        except ValueError:
            pass
    except ShioajiError as exc:
        print(f"shioaji failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            api.logout()
        except Exception:  # noqa: BLE001 - logout best effort.
            pass
    print(f"done prices={done_rows} taiex={taiex_rows if args.taiex else 'skipped'}")
    return 0


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


def _sync_prices(api, store, symbols: list[str], args, end: str) -> int:
    done = 0
    for index, stock_id in enumerate(symbols, 1):
        try:
            frame = fetch_daily_prices(stock_id, args.start, end, api)
        except ValueError as exc:
            print(f"[{index}/{len(symbols)}] {stock_id} SKIP: {exc}")
            continue
        except ShioajiError as exc:
            print(f"[{index}/{len(symbols)}] {stock_id} FAILED: {exc}", file=sys.stderr)
            continue
        done += store.upsert_prices(frame)
        if index % 25 == 0 or index == len(symbols):
            print(f"[{index}/{len(symbols)}] rows={done}")
            try:
                usage = api.usage()
            except Exception:  # noqa: BLE001 - usage check best effort.
                usage = None
            if usage is not None and usage.remaining_bytes < usage.limit_bytes * 0.1:
                print("quota below 10%: stopping early", file=sys.stderr)
                break
        time.sleep(args.delay)
    return done


if __name__ == "__main__":
    raise SystemExit(main())
