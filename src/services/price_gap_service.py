"""Plan and execute minimal-request historical price gap repairs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from integrations.finmind import FinMindError
from integrations.finmind_prices import fetch_price_adj, fetch_prices
from integrations.official_prices import fetch_official_price_day
from integrations.request_limiter import default_finmind_limiter
from models import Base
from models.market import Price
from models.security import Stock
from models.sync import DataSyncDay, DataSyncJob
from repositories import prices as prices_repo
from services.price_adjustment_service import calibrate_adjusted_prices

TAIEX_ID = "TAIEX"
MARKETS = ("TWSE", "TPEX")


@dataclass(frozen=True)
class RawGapTask:
    market: str
    trade_date: str
    missing_symbols: tuple[str, ...]


@dataclass(frozen=True)
class PriceGapPlan:
    start: str
    end: str
    raw_tasks: tuple[RawGapTask, ...]
    adjusted_keys: tuple[tuple[str, str], ...]
    trading_days: tuple[str, ...]

    @property
    def official_request_count(self) -> int:
        return len(self.raw_tasks)

    @property
    def adjusted_symbol_count(self) -> int:
        return len({stock_id for stock_id, _ in self.adjusted_keys})

    def to_dict(self, *, include_symbols: bool = True) -> dict:
        tasks = []
        for task in self.raw_tasks:
            item = {
                "market": task.market,
                "trade_date": task.trade_date,
                "missing_count": len(task.missing_symbols),
                "missing_sample": list(task.missing_symbols[:20]),
            }
            if include_symbols:
                item["missing_symbols"] = list(task.missing_symbols)
            tasks.append(item)
        return {
            "start": self.start,
            "end": self.end,
            "trading_day_count": len(self.trading_days),
            "raw_task_count": len(self.raw_tasks),
            "official_request_count": self.official_request_count,
            "raw_missing_row_count": sum(len(task.missing_symbols) for task in self.raw_tasks),
            "adjusted_missing_row_count": len(self.adjusted_keys),
            "adjusted_symbol_count": self.adjusted_symbol_count,
            "raw_tasks": tasks,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_sync_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=[DataSyncDay.__table__, DataSyncJob.__table__])


def _require_range(start: str, end: str) -> tuple[str, str]:
    parsed_start = date.fromisoformat(start).isoformat()
    parsed_end = date.fromisoformat(end).isoformat()
    if parsed_start > parsed_end:
        raise ValueError("start date is after end date")
    return parsed_start, parsed_end


def _stock_master(engine: Engine) -> pd.DataFrame:
    with engine.connect() as connection:
        return pd.read_sql_query(
            select(
                Stock.stock_id,
                Stock.market,
                Stock.listed_date,
                Stock.delisted_date,
            ).where(Stock.market.in_(MARKETS), Stock.stock_id != TAIEX_ID),
            connection,
        )


def _trading_days(engine: Engine, start: str, end: str) -> list[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            select(Price.trade_date)
            .where(
                Price.stock_id == TAIEX_ID,
                Price.trade_date >= start,
                Price.trade_date <= end,
            )
            .distinct()
            .order_by(Price.trade_date)
        ).scalars()
        days = list(rows)
        if days:
            return days
        return list(
            connection.execute(
                select(Price.trade_date)
                .where(Price.trade_date >= start, Price.trade_date <= end)
                .distinct()
                .order_by(Price.trade_date)
            ).scalars()
        )


def _active_ids(
    stocks: pd.DataFrame,
    market: str,
    day: str,
    observation_bounds: dict[str, tuple[str, str]],
    *,
    recent_calendar_days: int = 20,
) -> set[str]:
    rows = stocks.loc[stocks["market"].astype(str).str.upper().eq(market)].copy()
    listed = rows["listed_date"].isna() | rows["listed_date"].astype(str).le(day)
    live = rows["delisted_date"].isna() | rows["delisted_date"].astype(str).ge(day)
    candidates = set(rows.loc[listed & live, "stock_id"].astype(str))
    target = date.fromisoformat(day)
    active: set[str] = set()
    for stock_id in candidates:
        bounds = observation_bounds.get(stock_id)
        if bounds is None:
            continue
        first, last = bounds
        if first <= day <= last:
            active.add(stock_id)
            continue
        if last < day and (target - date.fromisoformat(last)).days <= recent_calendar_days:
            active.add(stock_id)
    return active


def _completed_days(engine: Engine, start: str, end: str) -> set[tuple[str, str]]:
    with engine.connect() as connection:
        try:
            rows = connection.execute(
                select(DataSyncDay.market, DataSyncDay.trade_date).where(
                    DataSyncDay.source == "official",
                    DataSyncDay.dataset == "raw_prices",
                    DataSyncDay.status.in_(("succeeded", "non_trading")),
                    DataSyncDay.trade_date >= start,
                    DataSyncDay.trade_date <= end,
                )
            ).all()
        except Exception:
            return set()
    return {(str(market), str(day)) for market, day in rows}


def _coverage_anomalies(
    observations: pd.DataFrame,
    stocks: pd.DataFrame,
    days: list[str],
    *,
    minimum_ratio: float = 0.95,
) -> set[tuple[str, str]]:
    """Find market-days whose row count materially drops below nearby sessions.

    Security masters retain stale instruments and ordinary suspensions are
    common, so an exact master-vs-price comparison would schedule every day.
    A broad count drop identifies genuine batch gaps while adjusted-column
    checks below still catch isolated incomplete rows already in the table.
    """
    market_by_id = dict(zip(stocks["stock_id"].astype(str), stocks["market"], strict=True))
    counted = observations.loc[:, ["trade_date", "stock_id"]].copy()
    counted["trade_date"] = counted["trade_date"].astype(str)
    counted["stock_id"] = counted["stock_id"].astype(str)
    counted["market"] = counted["stock_id"].map(market_by_id)
    counts = counted.groupby(["market", "trade_date"])["stock_id"].nunique().to_dict()
    anomalies: set[tuple[str, str]] = set()
    for market in MARKETS:
        market_counts = {
            day: int(count)
            for (row_market, day), count in counts.items()
            if str(row_market).upper() == market
        }
        for day in days:
            target = date.fromisoformat(day)
            nearby = [
                count
                for other_day, count in market_counts.items()
                if abs((date.fromisoformat(other_day) - target).days) <= 20
            ]
            current = market_counts.get(day, 0)
            if not nearby:
                if current == 0:
                    anomalies.add((market, day))
                continue
            baseline = float(pd.Series(nearby).median())
            if baseline > 0 and current < baseline * minimum_ratio:
                anomalies.add((market, day))
    return anomalies


def plan_price_gaps(
    engine: Engine,
    start: str,
    end: str,
    *,
    include_raw: bool = True,
    include_adjusted: bool = True,
    resume: bool = True,
) -> PriceGapPlan:
    start, end = _require_range(start, end)
    ensure_sync_schema(engine)
    stocks = _stock_master(engine)
    days = _trading_days(engine, start, end)
    observation_start = (date.fromisoformat(start) - timedelta(days=45)).isoformat()
    observation_end = (date.fromisoformat(end) + timedelta(days=45)).isoformat()
    with engine.connect() as connection:
        observations = pd.read_sql_query(
            select(
                Price.trade_date,
                Price.stock_id,
                Price.open_adj,
                Price.high_adj,
                Price.low_adj,
                Price.close_adj,
            ).where(
                Price.trade_date >= observation_start,
                Price.trade_date <= observation_end,
                Price.stock_id != TAIEX_ID,
            ),
            connection,
        )
    prices = observations.loc[observations["trade_date"].astype(str).between(start, end)].copy()
    bounds_frame = (
        observations.assign(
            stock_id=observations["stock_id"].astype(str),
            trade_date=observations["trade_date"].astype(str),
        )
        .groupby("stock_id")["trade_date"]
        .agg(["min", "max"])
    )
    observation_bounds = {
        str(stock_id): (str(row["min"]), str(row["max"]))
        for stock_id, row in bounds_frame.iterrows()
    }
    coverage_anomalies = _coverage_anomalies(observations, stocks, days)
    price_keys = {
        (str(row.stock_id), str(row.trade_date))
        for row in prices.loc[:, ["stock_id", "trade_date"]].itertuples(index=False)
    }
    completed = _completed_days(engine, start, end) if resume else set()
    raw_tasks: list[RawGapTask] = []
    if include_raw:
        for day in days:
            for market in MARKETS:
                if (market, day) in completed:
                    continue
                if (market, day) not in coverage_anomalies:
                    continue
                expected = _active_ids(stocks, market, day, observation_bounds)
                missing = sorted(
                    stock_id for stock_id in expected if (stock_id, day) not in price_keys
                )
                if missing:
                    raw_tasks.append(RawGapTask(market, day, tuple(missing)))
    adjusted_keys: list[tuple[str, str]] = []
    if include_adjusted and not prices.empty:
        adjusted_columns = ["open_adj", "high_adj", "low_adj", "close_adj"]
        missing_adj = prices[adjusted_columns].isna().any(axis=1) | prices[adjusted_columns].le(
            0
        ).any(axis=1)
        adjusted_keys = sorted(
            set(
                map(
                    tuple,
                    prices.loc[missing_adj, ["stock_id", "trade_date"]].astype(str).to_numpy(),
                )
            )
        )
    return PriceGapPlan(
        start=start,
        end=end,
        raw_tasks=tuple(raw_tasks),
        adjusted_keys=tuple(adjusted_keys),
        trading_days=tuple(days),
    )


def _mark_day(
    session: Session,
    *,
    market: str,
    trade_date: str,
    status: str,
    row_count: int = 0,
    content_hash: str | None = None,
    error: str | None = None,
) -> None:
    statement = sqlite_insert(DataSyncDay).values(
        source="official",
        dataset="raw_prices",
        market=market,
        trade_date=trade_date,
        status=status,
        row_count=row_count,
        content_hash=content_hash,
        error=error,
        synced_at=_utc_now(),
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["source", "dataset", "market", "trade_date"],
            set_={
                "status": statement.excluded.status,
                "row_count": statement.excluded.row_count,
                "content_hash": statement.excluded.content_hash,
                "error": statement.excluded.error,
                "synced_at": statement.excluded.synced_at,
            },
        )
    )


def _insert_only_missing(session: Session, frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    day = str(frame["trade_date"].iloc[0])
    ids = frame["stock_id"].astype(str).tolist()
    existing = {
        (str(stock_id), str(trade_date))
        for trade_date, stock_id in session.execute(
            select(Price.trade_date, Price.stock_id).where(
                Price.trade_date == day, Price.stock_id.in_(ids)
            )
        ).all()
    }
    new_rows = frame.loc[
        [
            (str(stock_id), str(trade_date)) not in existing
            for stock_id, trade_date in frame[["stock_id", "trade_date"]].itertuples(index=False)
        ]
    ]
    return prices_repo.upsert_prices(session, new_rows), len(frame) - len(new_rows)


def _job_key(source: str, dataset: str, stock_id: str, start: str, end: str) -> str:
    raw = f"{source}|{dataset}|{stock_id}|{start}|{end}".encode()
    return hashlib.sha256(raw).hexdigest()


def _mark_job(
    session: Session,
    source: str,
    dataset: str,
    stock_id: str,
    start: str,
    end: str,
    status: str,
    error: str | None = None,
) -> None:
    key = _job_key(source, dataset, stock_id, start, end)
    statement = sqlite_insert(DataSyncJob).values(
        job_key=key,
        source=source,
        dataset=dataset,
        stock_id=stock_id,
        start_date=start,
        end_date=end,
        status=status,
        attempts=1,
        last_error=error,
        updated_at=_utc_now(),
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["job_key"],
            set_={
                "status": statement.excluded.status,
                "attempts": DataSyncJob.attempts + 1,
                "last_error": statement.excluded.last_error,
                "updated_at": statement.excluded.updated_at,
            },
        )
    )


def _job_status(engine: Engine, key: str) -> str | None:
    with engine.connect() as connection:
        return connection.execute(
            select(DataSyncJob.status).where(DataSyncJob.job_key == key)
        ).scalar_one_or_none()


def _finmind_raw_fallback(
    engine: Engine, task: RawGapTask, token: str | None
) -> tuple[int, list[dict]]:
    if not token:
        return 0, [
            {"stock_id": stock_id, "trade_date": task.trade_date, "reason": "missing token"}
            for stock_id in task.missing_symbols
        ]
    written = 0
    unresolved: list[dict] = []
    for stock_id in task.missing_symbols:
        key = _job_key("finmind", "raw_prices", stock_id, task.trade_date, task.trade_date)
        if _job_status(engine, key) == "blocked":
            unresolved.append(
                {
                    "stock_id": stock_id,
                    "trade_date": task.trade_date,
                    "reason": "previous FinMind access denial",
                }
            )
            continue
        try:
            frame = fetch_prices(task.trade_date, task.trade_date, token, stock_id=stock_id)
            frame = frame.loc[
                frame["stock_id"].astype(str).eq(stock_id)
                & frame["trade_date"].astype(str).eq(task.trade_date)
            ]
            if frame.empty:
                continue  # Official and FinMind both absent: likely suspended/non-trading.
            with Session(engine) as session, session.begin():
                count, _ = _insert_only_missing(session, frame)
                _mark_job(
                    session,
                    "finmind",
                    "raw_prices",
                    stock_id,
                    task.trade_date,
                    task.trade_date,
                    "succeeded",
                )
                written += count
        except Exception as exc:  # noqa: BLE001 - preserve the per-symbol repair ledger.
            reason = f"{type(exc).__name__}: {exc}"
            status = "blocked" if "HTTP 40" in reason else "failed"
            with Session(engine) as session, session.begin():
                _mark_job(
                    session,
                    "finmind",
                    "raw_prices",
                    stock_id,
                    task.trade_date,
                    task.trade_date,
                    status,
                    reason,
                )
            unresolved.append(
                {"stock_id": stock_id, "trade_date": task.trade_date, "reason": reason}
            )
    return written, unresolved


def _repair_adjusted_with_finmind(
    engine: Engine, keys: set[tuple[str, str]], token: str | None
) -> tuple[int, list[dict]]:
    if not keys:
        return 0, []
    if not token:
        return 0, [
            {"stock_id": stock_id, "trade_date": day, "reason": "missing token"}
            for stock_id, day in sorted(keys)
        ]
    by_stock: dict[str, list[str]] = defaultdict(list)
    for stock_id, day in keys:
        by_stock[stock_id].append(day)
    written = 0
    unresolved: list[dict] = []
    for stock_id, days in sorted(by_stock.items()):
        target_start, target_end = min(days), max(days)
        fetch_start = (date.fromisoformat(target_start) - timedelta(days=45)).isoformat()
        fetch_end = (date.fromisoformat(target_end) + timedelta(days=45)).isoformat()
        key = _job_key("finmind", "adjusted_prices", stock_id, target_start, target_end)
        if _job_status(engine, key) == "blocked":
            unresolved.extend(
                {
                    "stock_id": stock_id,
                    "trade_date": day,
                    "reason": "previous FinMind access denial",
                }
                for day in days
            )
            continue
        try:
            frame = fetch_price_adj(fetch_start, fetch_end, token, stock_id=stock_id)
            with engine.connect() as connection:
                references = pd.read_sql_query(
                    select(Price.stock_id, Price.trade_date, Price.close_adj).where(
                        Price.stock_id == stock_id,
                        Price.trade_date >= fetch_start,
                        Price.trade_date <= fetch_end,
                        Price.close_adj.is_not(None),
                        Price.close_adj > 0,
                    ),
                    connection,
                )
            frame, calibration = calibrate_adjusted_prices(frame, references)
            if not calibration.get(stock_id, {}).get("accepted", False):
                raise ValueError("FinMind adjusted-price basis could not be calibrated")
            wanted = {(stock_id, day) for day in days}
            frame["stock_id"] = frame["stock_id"].astype(str)
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame = frame.loc[
                pd.MultiIndex.from_frame(frame[["stock_id", "trade_date"]]).isin(wanted)
            ]
            with Session(engine) as session, session.begin():
                count = prices_repo.upsert_price_adj(session, frame)
                _mark_job(
                    session,
                    "finmind",
                    "adjusted_prices",
                    stock_id,
                    target_start,
                    target_end,
                    "succeeded" if count else "partial",
                )
                written += count
            resolved = set(map(tuple, frame[["stock_id", "trade_date"]].to_numpy()))
            unresolved.extend(
                {"stock_id": sid, "trade_date": day, "reason": "provider returned no row"}
                for sid, day in sorted(wanted - resolved)
            )
        except Exception as exc:  # noqa: BLE001 - one ticker must not discard other repairs.
            reason = f"{type(exc).__name__}: {exc}"
            status = (
                "blocked" if isinstance(exc, FinMindError) and "HTTP 40" in reason else "failed"
            )
            with Session(engine) as session, session.begin():
                _mark_job(
                    session,
                    "finmind",
                    "adjusted_prices",
                    stock_id,
                    target_start,
                    target_end,
                    status,
                    reason,
                )
            unresolved.extend(
                {"stock_id": stock_id, "trade_date": day, "reason": reason} for day in days
            )
    return written, unresolved


def execute_price_gap_plan(
    engine: Engine,
    plan: PriceGapPlan,
    *,
    token: str | None,
    cache_dir: str | Path = "data/cache/official_prices",
) -> dict:
    started = perf_counter()
    official_requests = 0
    cache_hits = 0
    raw_written = 0
    adjusted_written = 0
    unresolved: list[dict] = []
    task_reports: list[dict] = []
    for task in plan.raw_tasks:
        try:
            result = fetch_official_price_day(task.market, task.trade_date, cache_dir=cache_dir)
            official_requests += 0 if result.cache_hit else 1
            cache_hits += int(result.cache_hit)
            official_ids = set(result.frame["stock_id"].astype(str))
            wanted = set(task.missing_symbols)
            frame = result.frame.loc[result.frame["stock_id"].astype(str).isin(wanted)]
            with Session(engine) as session, session.begin():
                count, conflicts = _insert_only_missing(session, frame)
                _mark_day(
                    session,
                    market=task.market,
                    trade_date=task.trade_date,
                    status="succeeded",
                    row_count=len(result.frame),
                    content_hash=result.content_hash,
                )
                raw_written += count
            absent = wanted - official_ids
            task_reports.append(
                {
                    "market": task.market,
                    "trade_date": task.trade_date,
                    "official_rows": len(result.frame),
                    "written": count,
                    "already_present": conflicts,
                    "official_absent": len(absent),
                }
            )
        except Exception as exc:  # noqa: BLE001 - exact missing keys go to fallback.
            official_requests += 1
            reason = f"{type(exc).__name__}: {exc}"
            with Session(engine) as session, session.begin():
                _mark_day(
                    session,
                    market=task.market,
                    trade_date=task.trade_date,
                    status="failed",
                    error=reason,
                )
            count, missing = _finmind_raw_fallback(engine, task, token)
            raw_written += count
            unresolved.extend(missing)
    # Raw rows inserted above have NULL adjusted columns and must join the same
    # run instead of waiting for a second invocation.
    refreshed = plan_price_gaps(
        engine,
        plan.start,
        plan.end,
        include_raw=False,
        include_adjusted=True,
        resume=True,
    )
    adjusted_keys = set(plan.adjusted_keys) | set(refreshed.adjusted_keys)
    if adjusted_keys:
        count, missing = _repair_adjusted_with_finmind(engine, adjusted_keys, token)
        adjusted_written += count
        unresolved.extend(missing)
    elapsed = perf_counter() - started
    remaining = plan_price_gaps(
        engine,
        plan.start,
        plan.end,
        include_raw=True,
        include_adjusted=True,
        resume=True,
    )
    affected_months = sorted(
        {task.trade_date[:7] for task in plan.raw_tasks} | {day[:7] for _, day in adjusted_keys}
    )
    return {
        "range": {"start": plan.start, "end": plan.end},
        "elapsed_seconds": elapsed,
        "official_requests": official_requests,
        "official_cache_hits": cache_hits,
        "finmind_requests_in_window": default_finmind_limiter().usage(),
        "raw_rows_written": raw_written,
        "adjusted_rows_written": adjusted_written,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "remaining_raw_tasks": len(remaining.raw_tasks),
        "remaining_adjusted_rows": len(remaining.adjusted_keys),
        "affected_panel_months": affected_months,
        "official_tasks": task_reports,
        "finmind_adjusted_symbol_jobs": len({stock_id for stock_id, _ in adjusted_keys}),
    }


def write_profile(report: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
