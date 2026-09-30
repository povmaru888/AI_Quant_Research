"""Persistence helpers for point-in-time daily market values."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.market import MarketValue, MarketValueSyncDay

_VALUE_KEY = ("trade_date", "stock_id")


def canonical_day_hash(rows: pd.DataFrame) -> str:
    """Hash normalized source values in stable key order."""
    normalized = rows.loc[:, ["trade_date", "stock_id", "market_value"]].copy()
    normalized["trade_date"] = normalized["trade_date"].astype(str)
    normalized["stock_id"] = normalized["stock_id"].astype(str)
    normalized["market_value"] = pd.to_numeric(normalized["market_value"], errors="raise")
    normalized = normalized.sort_values(["trade_date", "stock_id"], kind="mergesort")
    payload = normalized.to_json(orient="records", double_precision=15).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def upsert_market_value_day(
    session: Session,
    trade_day: date,
    rows: pd.DataFrame,
    *,
    source_content_hash: str | None = None,
    source: str = "FinMind:TaiwanStockMarketValue",
) -> int:
    """Atomically upsert one validated full-market day and mark it complete."""
    day = trade_day.isoformat()
    if rows.empty:
        raise ValueError(f"market value day {day} is empty")
    if not {"trade_date", "stock_id", "market_value"}.issubset(rows.columns):
        raise ValueError("market values require trade_date, stock_id and market_value")
    frame = rows.loc[:, ["trade_date", "stock_id", "market_value"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    if frame["trade_date"].ne(day).any():
        raise ValueError(f"market value payload contains dates other than {day}")
    if frame["stock_id"].isna().any() or frame["stock_id"].duplicated().any():
        raise ValueError(f"market value payload has missing or duplicate stock IDs for {day}")
    if frame["market_value"].isna().any() or frame["market_value"].le(0).any():
        raise ValueError(f"market value payload has missing or non-positive values for {day}")
    content_hash = source_content_hash or canonical_day_hash(frame)
    if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
        raise ValueError("source_content_hash must be a lowercase SHA-256 hex digest")
    records = [
        {
            "trade_date": day,
            "stock_id": row.stock_id,
            "market_value": float(row.market_value),
            "source": source,
        }
        for row in frame.itertuples(index=False)
    ]
    existing_marker = session.get(MarketValueSyncDay, day)
    if (
        existing_marker is not None
        and existing_marker.status == "succeeded"
        and existing_marker.content_hash == content_hash
        and existing_marker.row_count == len(records)
    ):
        return len(records)
    # A corrected source response can omit IDs which were present previously.
    # Replace the day's complete snapshot before inserting the new validated
    # payload; both operations and the checkpoint share one transaction.
    session.execute(delete(MarketValue).where(MarketValue.trade_date == day))
    insert_stmt = sqlite_insert(MarketValue)
    statement = insert_stmt.on_conflict_do_update(
        index_elements=list(_VALUE_KEY),
        set_={
            "market_value": insert_stmt.excluded.market_value,
            "source": insert_stmt.excluded.source,
            "updated_at": insert_stmt.excluded.updated_at,
        },
    )
    session.execute(statement, records)
    marker = sqlite_insert(MarketValueSyncDay).values(
        trade_date=day,
        status="succeeded",
        row_count=len(records),
        content_hash=content_hash,
    )
    marker = marker.on_conflict_do_update(
        index_elements=["trade_date"],
        set_={
            "status": marker.excluded.status,
            "row_count": marker.excluded.row_count,
            "content_hash": marker.excluded.content_hash,
            "synced_at": marker.excluded.synced_at,
        },
    )
    session.execute(marker)
    session.flush()
    return len(records)


def upsert_partial_market_value_day(
    session: Session,
    trade_day: date,
    rows: pd.DataFrame,
    *,
    source: str,
) -> int:
    """Store validated known values without claiming a complete daily snapshot."""
    day = trade_day.isoformat()
    if rows.empty:
        raise ValueError(f"partial market value day {day} is empty")
    if not {"trade_date", "stock_id", "market_value"}.issubset(rows.columns):
        raise ValueError("market values require trade_date, stock_id and market_value")
    frame = rows.loc[:, ["trade_date", "stock_id", "market_value"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    if frame["trade_date"].ne(day).any():
        raise ValueError(f"partial market value payload contains dates other than {day}")
    if frame["stock_id"].isna().any() or frame["stock_id"].duplicated().any():
        raise ValueError(f"partial market values have missing or duplicate stock IDs for {day}")
    if frame["market_value"].isna().any() or frame["market_value"].le(0).any():
        raise ValueError(f"partial market values have missing or non-positive values for {day}")
    now = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "trade_date": day,
            "stock_id": row.stock_id,
            "market_value": float(row.market_value),
            "source": source,
            "updated_at": now,
        }
        for row in frame.itertuples(index=False)
    ]
    insert_stmt = sqlite_insert(MarketValue).values(records)
    session.execute(
        insert_stmt.on_conflict_do_update(
            index_elements=list(_VALUE_KEY),
            set_={
                "market_value": insert_stmt.excluded.market_value,
                "source": insert_stmt.excluded.source,
                "updated_at": insert_stmt.excluded.updated_at,
            },
        )
    )
    # Keep verified rows auditable without claiming a complete exchange snapshot.
    marker_stmt = sqlite_insert(MarketValueSyncDay).values(
        trade_date=day,
        status="partial",
        row_count=len(records),
        content_hash=canonical_day_hash(frame),
    )
    session.execute(
        marker_stmt.on_conflict_do_update(
            index_elements=["trade_date"],
            set_={
                "status": marker_stmt.excluded.status,
                "row_count": marker_stmt.excluded.row_count,
                "content_hash": marker_stmt.excluded.content_hash,
                "synced_at": marker_stmt.excluded.synced_at,
            },
        )
    )
    session.flush()
    return len(records)


def mark_sync_day(session: Session, trade_day: str, status: str) -> None:
    """Record a failed or confirmed non-trading day without claiming a snapshot."""
    if status not in {"running", "failed", "non_trading"}:
        raise ValueError(f"invalid market-value checkpoint status: {status!r}")
    if status in {"failed", "non_trading"}:
        session.execute(delete(MarketValue).where(MarketValue.trade_date == trade_day))
    statement = sqlite_insert(MarketValueSyncDay).values(
        trade_date=trade_day,
        status=status,
        row_count=0,
        content_hash=None,
    )
    statement = statement.on_conflict_do_update(
        index_elements=["trade_date"],
        set_={
            "status": statement.excluded.status,
            "row_count": statement.excluded.row_count,
            "content_hash": statement.excluded.content_hash,
            "synced_at": statement.excluded.synced_at,
        },
    )
    session.execute(statement)
    session.flush()


def load_market_value_snapshot(
    session: Session, as_of: date, *, include_partial: bool = False
) -> pd.DataFrame:
    """Load the exact-day market-value rows for a signal date."""
    statuses = ("succeeded", "partial") if include_partial else ("succeeded",)
    rows = session.execute(
        select(MarketValue.stock_id, MarketValue.trade_date, MarketValue.market_value)
        .join(
            MarketValueSyncDay,
            MarketValueSyncDay.trade_date == MarketValue.trade_date,
        )
        .where(MarketValue.trade_date == as_of.isoformat())
        .where(MarketValueSyncDay.status.in_(statuses))
        .order_by(MarketValue.stock_id)
    ).all()
    return pd.DataFrame(rows, columns=["stock_id", "trade_date", "market_value"])


def load_completed_days(session: Session, start: str, end: str) -> set[str]:
    """Return successful snapshots and confirmed closed days in the window."""
    rows = session.execute(
        select(MarketValueSyncDay.trade_date).where(
            MarketValueSyncDay.status.in_(("succeeded", "non_trading")),
            MarketValueSyncDay.trade_date >= start,
            MarketValueSyncDay.trade_date <= end,
        )
    ).scalars()
    return set(rows)
