"""Repair missing panel-date PIT market values without replacing known rows.

The source order is FinMind daily market value, TPEx's official dated list,
FinMind issued shares times the same-day nominal close, then the approved
previous-day market-value / previous-day close x same-day close estimate.
Dry-run is the default; pass ``--apply`` to persist recoverable rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.market import MarketValue, MarketValueSyncDay, Price  # noqa: E402
from models.security import Stock  # noqa: E402
from repositories.market_values import canonical_day_hash  # noqa: E402
from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402
from repair_saturday_sessions import _fetch_market_values  # noqa: E402


def _load_missing(path: Path) -> dict[str, set[str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read coverage report {path}: {exc}") from exc
    if not str(report.get("feature_version", "")).startswith("factor_adj_pit_v3"):
        raise ValueError("coverage report must come from a factor_adj_pit_v3 check")
    missing = defaultdict(set)
    for row in report.get("missing_details", []):
        day = date.fromisoformat(str(row["signal_date"])).isoformat()
        stock_id = str(row["stock_id"]).strip()
        if stock_id:
            missing[day].add(stock_id)
    return dict(missing)


def _insert_recovered_rows(engine, day: str, rows: pd.DataFrame) -> tuple[int, str]:
    if rows.empty:
        return 0, ""
    records = []
    for row in rows.itertuples(index=False):
        records.append(
            {
                "trade_date": day,
                "stock_id": str(row.stock_id),
                "market_value": float(row.market_value),
                "source": str(getattr(row, "source", "verified PIT market-value recovery")),
            }
        )
    statement = sqlite_insert(MarketValue).values(records).on_conflict_do_nothing(
        index_elements=["trade_date", "stock_id"]
    )
    with engine.begin() as connection:
        result = connection.execute(statement)
        snapshot_rows = connection.execute(
            select(MarketValue.trade_date, MarketValue.stock_id, MarketValue.market_value)
            .where(MarketValue.trade_date == day)
            .order_by(MarketValue.stock_id)
        ).all()
        snapshot = pd.DataFrame(
            snapshot_rows, columns=["trade_date", "stock_id", "market_value"]
        )
        previous_status = connection.execute(
            select(MarketValueSyncDay.status).where(MarketValueSyncDay.trade_date == day)
        ).scalar_one_or_none()
        if previous_status == "non_trading":
            raise ValueError(f"refusing to add market values to confirmed non-trading day {day}")
        next_status = "succeeded" if previous_status == "succeeded" else "partial"
        if previous_status is None:
            connection.execute(
                sqlite_insert(MarketValueSyncDay).values(
                    trade_date=day,
                    status=next_status,
                    row_count=len(snapshot),
                    content_hash=canonical_day_hash(snapshot),
                )
            )
        else:
            connection.execute(
                MarketValueSyncDay.__table__.update()
                .where(MarketValueSyncDay.trade_date == day)
                .values(
                    status=next_status,
                    row_count=len(snapshot),
                    content_hash=canonical_day_hash(snapshot),
                    synced_at=datetime.now(timezone.utc).isoformat(),
                )
            )
    return int(result.rowcount or 0), next_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.oos2020_stable.yaml")
    parser.add_argument(
        "--coverage-report",
        default="reports/pit_market_value_coverage_2015_2019_stable_v2.json",
    )
    parser.add_argument("--report", default="reports/pit_market_value_gap_repair.json")
    parser.add_argument("--apply", action="store_true", help="persist recoverable rows")
    args = parser.parse_args(argv)
    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN; refusing to use fallback sources first", file=sys.stderr)
        return 1
    try:
        missing_by_day = _load_missing(Path(args.coverage_report))
    except Exception as exc:  # noqa: BLE001
        print(f"invalid coverage report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    engine = get_engine(settings)
    with engine.connect() as connection:
        known_ids = set(connection.execute(select(Stock.stock_id)).scalars())
    day_reports: list[dict[str, object]] = []
    total_candidates = total_recoverable = total_written = 0
    for index, (day, original_ids) in enumerate(sorted(missing_by_day.items()), start=1):
        print(f"[{index}/{len(missing_by_day)}] checking {day} candidates={len(original_ids)}", flush=True)
        with engine.connect() as connection:
            current_ids = set(
                connection.execute(
                    select(MarketValue.stock_id).where(
                        MarketValue.trade_date == day,
                        MarketValue.stock_id.in_(original_ids),
                    )
                ).scalars()
            )
            prices = connection.execute(
                select(Price.stock_id, Price.close).where(
                    Price.trade_date == day, Price.stock_id != "TAIEX"
                )
            ).all()
        missing_ids = original_ids - current_ids
        total_candidates += len(missing_ids)
        if not missing_ids:
            continue
        price_frame = pd.DataFrame(prices, columns=["stock_id", "close"])
        try:
            recovered = _fetch_market_values(
                engine,
                day,
                token,
                known_ids,
                known_ids,
                price_frame,
            )
        except PermissionError as exc:
            print(f"[{day}] FinMind access denied; stopping: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - preserve unresolved rows and audit the error.
            day_reports.append(
                {
                    "trade_date": day,
                    "status": "source_error",
                    "candidate_count": len(missing_ids),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        recovered["stock_id"] = recovered["stock_id"].astype(str)
        recovered = recovered.loc[recovered["stock_id"].isin(missing_ids)].copy()
        recovered = recovered.drop_duplicates("stock_id", keep="first")
        total_recoverable += len(recovered)
        written = 0
        status = "dry_run"
        if args.apply and not recovered.empty:
            written, status = _insert_recovered_rows(engine, day, recovered)
            total_written += written
        remaining = sorted(missing_ids - set(recovered["stock_id"]))
        by_source = (
            recovered["source"].value_counts().to_dict()
            if "source" in recovered.columns
            else {}
        )
        day_reports.append(
            {
                "trade_date": day,
                "status": status,
                "candidate_count": len(missing_ids),
                "recoverable_count": len(recovered),
                "written_count": written,
                "remaining_stock_ids": remaining,
                "recovered_by_source": by_source,
                "previous_market_value_fallback_count": recovered.attrs.get(
                    "previous_market_value_fallback_count", 0
                ),
                "market_value_sources_checked": recovered.attrs.get(
                    "market_value_source", ""
                ),
            }
        )
    result = {
        "mode": "apply" if args.apply else "dry-run",
        "coverage_report": str(args.coverage_report),
        "source_order": [
            "FinMind TaiwanStockMarketValue",
            "TPEx official daily market-value list",
            "FinMind TaiwanStockShareholding issued shares x same-day nominal close",
            "previous-day market value / previous-day nominal close x same-day nominal close",
        ],
        "candidate_count": total_candidates,
        "recoverable_count": total_recoverable,
        "written_count": total_written,
        "days": day_reports,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"done candidates={total_candidates} recoverable={total_recoverable} "
        f"written={total_written} report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
