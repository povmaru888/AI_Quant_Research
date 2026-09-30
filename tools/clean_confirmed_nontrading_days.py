"""Remove imported equity bars on confirmed Taiwan market closure dates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import session_scope  # noqa: E402
from models.market import Institutional, Price  # noqa: E402
from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import load_settings  # noqa: E402


CONFIRMED_CLOSED_DAYS = {
    "2015-07-10": "Typhoon Chan-hom; TWSE and TPEx closed",
    "2016-09-27": "Typhoon Megi; TWSE and TPEx closed",
    "2016-09-28": "Typhoon Megi; TWSE and TPEx closed",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.oos2020_stable.yaml")
    parser.add_argument("--apply", action="store_true", help="Delete rows after writing a rollback CSV.")
    parser.add_argument("--backup-dir", default="database/repair_backups")
    args = parser.parse_args(argv)
    load_dotenv()
    settings = load_settings(args.config)
    engine = get_engine(settings)
    days = sorted(CONFIRMED_CLOSED_DAYS)
    with engine.connect() as connection:
        price_rows = connection.execute(
            select(Price).where(Price.trade_date.in_(days)).order_by(Price.trade_date, Price.stock_id)
        ).all()
        inst_rows = connection.execute(
            select(Institutional).where(Institutional.trade_date.in_(days)).order_by(
                Institutional.trade_date, Institutional.stock_id
            )
        ).all()
    prices = pd.DataFrame([row._asdict() for row in price_rows])
    institutions = pd.DataFrame([row._asdict() for row in inst_rows])
    print(
        f"closed days={len(days)} price_rows={len(prices)} institutional_rows={len(institutions)} "
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    if not args.apply:
        return 0

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        day_prices = prices.loc[prices["trade_date"].astype(str).eq(day)] if not prices.empty else prices
        day_institutions = (
            institutions.loc[institutions["trade_date"].astype(str).eq(day)]
            if not institutions.empty
            else institutions
        )
        if not day_prices.empty:
            day_prices.to_csv(
                backup_dir / f"closed_date_prices_{day}.csv",
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
        if not day_institutions.empty:
            day_institutions.to_csv(
                backup_dir / f"closed_date_institutional_{day}.csv",
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
    with session_scope(engine) as session:
        session.execute(delete(Price).where(Price.trade_date.in_(days)))
        session.execute(delete(Institutional).where(Institutional.trade_date.in_(days)))
        from repositories import market_values as market_values_repo

        for day in days:
            market_values_repo.mark_sync_day(session, day, "non_trading")
    summary = {
        "deleted_price_rows": len(prices),
        "deleted_institutional_rows": len(institutions),
        "days": [
            {"trade_date": day, "reason": CONFIRMED_CLOSED_DAYS[day]} for day in days
        ],
        "backup_dir": str(backup_dir),
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    report = Path("reports/confirmed_nontrading_cleanup.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"deleted; rollback CSVs in {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
