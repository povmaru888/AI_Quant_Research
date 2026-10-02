"""Repair zero-volume/value equity bars for specific days from FinMind."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import session_scope  # noqa: E402
from integrations.finmind_prices import fetch_prices  # noqa: E402
from models.market import Price  # noqa: E402
from repositories import prices as prices_repo  # noqa: E402
from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402


def _repair_day(engine, day: str, token: str) -> dict[str, object]:
    vendor = fetch_prices(day, day, token)
    vendor = vendor.loc[
        vendor["trade_date"].eq(day)
        & vendor["volume"].gt(0)
        & vendor["traded_value"].gt(0)
    ].drop_duplicates(["trade_date", "stock_id"], keep="last")
    if vendor.empty:
        raise RuntimeError(f"FinMind returned no positive liquidity rows for {day}")

    with session_scope(engine) as session:
        existing = session.execute(
            select(Price.stock_id, Price.volume, Price.traded_value).where(
                Price.trade_date == day,
                Price.stock_id != "TAIEX",
            )
        ).all()
        current = pd.DataFrame(existing, columns=["stock_id", "volume", "traded_value"])
        bad = current.loc[
            pd.to_numeric(current["volume"], errors="coerce").fillna(0).le(0)
            | pd.to_numeric(current["traded_value"], errors="coerce").fillna(0).le(0),
            "stock_id",
        ].astype(str)
        repair = vendor.loc[vendor["stock_id"].astype(str).isin(set(bad))].copy()
        written = prices_repo.upsert_prices(session, repair)

    with session_scope(engine) as session:
        remaining = session.execute(
            select(Price.stock_id).where(
                Price.trade_date == day,
                Price.stock_id != "TAIEX",
                (Price.volume <= 0) | (Price.traded_value <= 0),
            )
        ).all()
    return {
        "trade_date": day,
        "finmind_rows": int(len(vendor)),
        "bad_before": int(len(bad)),
        "repaired": int(written),
        "bad_after": int(len(remaining)),
        "unmatched_bad_ids": sorted({str(row[0]) for row in remaining}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.factor_v4.yaml")
    parser.add_argument("--date", action="append", required=True, dest="dates")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN", file=sys.stderr)
        return 1
    engine = get_engine(settings)
    results = [_repair_day(engine, day, token) for day in dict.fromkeys(args.dates)]
    payload = {"source": "FinMind:TaiwanStockPrice", "days": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
