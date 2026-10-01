"""Sync official TWSE/TPEx formal listing dates into the stock master."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import create_engine_from_settings, session_scope  # noqa: E402
from integrations.official_listing_dates import fetch_official_listing_dates  # noqa: E402
from models.security import Stock  # noqa: E402
from repositories import stocks as stocks_repo  # noqa: E402
from settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync official TWSE/TPEx listing dates.")
    parser.add_argument("--config", default="config.factor_v4.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    incoming = fetch_official_listing_dates()
    engine = create_engine_from_settings(settings)
    try:
        with session_scope(engine) as session:
            rows = session.execute(select(Stock.stock_id, Stock.market, Stock.listed_date)).all()
            existing = pd.DataFrame(rows, columns=["stock_id", "market", "listed_date"])
            known = set(existing["stock_id"].astype(str))
            incoming = incoming.loc[incoming["stock_id"].isin(known)].copy()
            old_dates = existing.set_index("stock_id")["listed_date"].to_dict()
            incoming["listed_date"] = [
                min(str(old), new) if isinstance(old, str) and old else new
                for stock_id, new in zip(incoming["stock_id"], incoming["listed_date"], strict=True)
                for old in [old_dates.get(stock_id)]
            ]
            updates = incoming[["stock_id", "listed_date"]].drop_duplicates("stock_id")
            updates["market"] = updates["stock_id"].map(existing.set_index("stock_id")["market"])
            missing = len(known - set(updates["stock_id"]))
            print(
                f"official rows={len(incoming)} matched={len(updates)} unmatched_stocks={missing}"
            )
            if args.dry_run:
                session.rollback()
            else:
                stocks_repo.upsert_stocks(session, updates)
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
