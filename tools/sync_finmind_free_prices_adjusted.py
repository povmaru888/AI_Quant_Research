"""Free-tier FinMind price sync with one request per symbol.

The free FinMind tier gates ``TaiwanStockPriceAdj``.  This command uses the
raw feed's close and spread fields to reconstruct the official reference
price and hence any ex-right/ex-dividend adjustment factor.  Raw and adjusted
OHLC are therefore updated from the same single vendor request.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.finmind import _get
from models.market import Price
from models.sync import DataSyncJob
from repositories import prices as prices_repo
from runtime.db_store import get_engine
from runtime.dotenv import load_dotenv
from settings import get_finmind_token, load_settings

DATASET = "raw_prices_with_local_adjustment"


def job_key(stock_id: str, start: str, end: str) -> str:
    return hashlib.sha256(f"finmind|{DATASET}|{stock_id}|{start}|{end}".encode()).hexdigest()


def standardize(rows: list[dict], stock_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "date": "trade_date",
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
            "Trading_money": "traded_value",
        }
    )
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame = frame.loc[frame["stock_id"].eq(stock_id)].copy()
    numeric = ["open", "high", "low", "close", "volume", "traded_value", "spread"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", *numeric])
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame[["volume", "traded_value"]] >= 0).all(axis=1)
        & frame["high"].ge(frame["low"])
    ]
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["source"] = "finmind"
    return frame.sort_values("trade_date", ignore_index=True)


def event_factors(frame: pd.DataFrame, previous_close: float | None) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    prior = previous_close
    for row in frame.itertuples(index=False):
        if prior is not None and prior > 0:
            reference = float(row.close) - float(row.spread)
            factor = reference / prior
            tolerance = max(1e-7, abs(prior) * 1e-9)
            if reference > 0 and abs(reference - prior) > tolerance:
                if not math.isfinite(factor) or not 0.05 <= factor <= 20:
                    raise ValueError(f"implausible adjustment factor {factor} on {row.trade_date}")
                events.append((str(row.trade_date), factor))
        prior = float(row.close)
    return events


def adjusted_frame(frame: pd.DataFrame, events: list[tuple[str, float]]) -> pd.DataFrame:
    adjusted = frame[["stock_id", "trade_date", "open", "high", "low", "close"]].copy()
    multipliers = pd.Series(1.0, index=adjusted.index)
    for event_day, factor in events:
        multipliers.loc[adjusted["trade_date"].lt(event_day)] *= factor
    for raw, target in (
        ("open", "open_adj"),
        ("high", "high_adj"),
        ("low", "low_adj"),
        ("close", "close_adj"),
    ):
        adjusted[target] = adjusted[raw] * multipliers
    return adjusted[["stock_id", "trade_date", "open_adj", "high_adj", "low_adj", "close_adj"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    start = pd.Timestamp(args.start).date().isoformat()
    end = pd.Timestamp(args.end).date().isoformat()
    if start > end:
        raise ValueError("start date is after end date")
    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        raise RuntimeError("missing FINMIND_TOKEN")
    engine = get_engine(settings)
    symbol_file = Path(args.symbols_file)
    symbols = [
        line.strip()
        for line in symbol_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    totals = {"symbols": 0, "raw": 0, "adjusted": 0, "events": 0, "empty": 0}
    for index, stock_id in enumerate(symbols, 1):
        key = job_key(stock_id, start, end)
        with Session(engine) as session:
            if (
                session.execute(
                    select(DataSyncJob.status).where(DataSyncJob.job_key == key)
                ).scalar_one_or_none()
                == "succeeded"
            ):
                continue
        rows = _get("TaiwanStockPrice", start, end, token, data_id=stock_id)
        frame = standardize(rows, stock_id)
        with Session(engine) as session, session.begin():
            previous = session.execute(
                select(Price.close)
                .where(Price.stock_id == stock_id, Price.trade_date < start)
                .order_by(Price.trade_date.desc())
                .limit(1)
            ).scalar_one_or_none()
            events = event_factors(frame, float(previous) if previous is not None else None)
            for event_day, factor in events:
                session.execute(
                    text(
                        "UPDATE prices SET open_adj=open_adj*:factor, high_adj=high_adj*:factor, "
                        "low_adj=low_adj*:factor, close_adj=close_adj*:factor "
                        "WHERE stock_id=:stock_id AND trade_date<:event_day "
                        "AND open_adj IS NOT NULL AND high_adj IS NOT NULL "
                        "AND low_adj IS NOT NULL AND close_adj IS NOT NULL"
                    ),
                    {"factor": factor, "stock_id": stock_id, "event_day": event_day},
                )
            raw_count = (
                prices_repo.upsert_prices(
                    session,
                    frame[
                        [
                            "stock_id",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "traded_value",
                            "source",
                        ]
                    ],
                )
                if not frame.empty
                else 0
            )
            adj_count = (
                prices_repo.upsert_price_adj(session, adjusted_frame(frame, events))
                if not frame.empty
                else 0
            )
            statement = sqlite_insert(DataSyncJob).values(
                job_key=key,
                source="finmind",
                dataset=DATASET,
                stock_id=stock_id,
                start_date=start,
                end_date=end,
                status="succeeded",
                attempts=1,
                last_error=None,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=["job_key"],
                    set_={"status": "succeeded", "updated_at": statement.excluded.updated_at},
                )
            )
        totals["symbols"] += 1
        totals["raw"] += raw_count
        totals["adjusted"] += adj_count
        totals["events"] += len(events)
        totals["empty"] += int(frame.empty)
        if index % 25 == 0 or index == len(symbols):
            print(index, len(symbols), totals, flush=True)
    print("DONE", totals, flush=True)


if __name__ == "__main__":
    main()
