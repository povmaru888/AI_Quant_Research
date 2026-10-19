"""Backfill FinMind monthly revenue in resumable calendar-year chunks."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from database import create_engine_from_settings, session_scope  # noqa: E402
from integrations.finmind_monthly_revenue import fetch_monthly_revenues  # noqa: E402
from models.security import Stock  # noqa: E402
from repositories.monthly_revenues import upsert_monthly_revenues  # noqa: E402
from repositories.stocks import upsert_stocks  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402


def _month_ranges(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        yield max(start, current), min(end, next_month - timedelta(days=1))
        current = next_month


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        if start > end:
            raise ValueError("start is after end")
    except ValueError as exc:
        parser.error(str(exc))
    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN", file=sys.stderr)
        return 1
    engine = create_engine_from_settings(settings)
    total = 0
    try:
        for chunk_start, chunk_end in _month_ranges(start, end):
            frame = fetch_monthly_revenues(
                chunk_start.isoformat(), chunk_end.isoformat(), token
            )
            if frame.empty:
                print(f"{chunk_start:%Y-%m}: rows=0", flush=True)
                continue
            with session_scope(engine) as session:
                requested = sorted(frame["stock_id"].astype(str).unique())
                existing = set(
                    session.scalars(select(Stock.stock_id).where(Stock.stock_id.in_(requested)))
                )
                missing = [stock_id for stock_id in requested if stock_id not in existing]
                if missing:
                    upsert_stocks(
                        session,
                        pd.DataFrame({"stock_id": missing, "market": "UNKNOWN"}),
                    )
                written = upsert_monthly_revenues(session, frame)
            total += written
            print(f"{chunk_start:%Y-%m}: rows={written} total={total}", flush=True)
    finally:
        engine.dispose()
    print(f"done rows={total} range={start.isoformat()}..{end.isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
