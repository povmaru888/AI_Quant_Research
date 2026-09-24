"""P1-07: price repository (prices table).

Same conventions as P1-06: takes an active ``Session``, flushes but never
commits; upserts are idempotent via SQLite ON CONFLICT DO UPDATE.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.market import Price
from repositories._frames import to_records

PRICE_COLUMNS = (
    "trade_date",
    "stock_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "traded_value",
    "source",
)

ADJ_COLUMNS = (
    "trade_date",
    "stock_id",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
)

# Historical consumers need both the raw execution quote and its adjusted
# counterpart. PRICE_COLUMNS remains the raw ingestion contract.
PRICE_HISTORY_COLUMNS = (*PRICE_COLUMNS, *ADJ_COLUMNS[2:])

_PRICE_COLUMNS = frozenset(col.name for col in Price.__table__.columns)
_PRICE_KEY = ("trade_date", "stock_id")


def upsert_prices(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update price rows; return the number of rows written."""
    records = to_records(rows, _PRICE_COLUMNS, _PRICE_KEY, "Price")
    if not records:
        return 0
    update_columns = [col for col in rows.columns if col not in _PRICE_KEY]
    statement = sqlite_insert(Price)
    if update_columns:
        statement = statement.on_conflict_do_update(
            index_elements=list(_PRICE_KEY),
            set_={col: statement.excluded[col] for col in update_columns},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=list(_PRICE_KEY))
    session.execute(statement, records)
    session.flush()
    return len(records)


def load_prices(
    session: Session, stock_ids: Collection[str], start: date, end: date
) -> pd.DataFrame:
    """Load raw and adjusted bars in [``start``, ``end``], sorted by id/date."""
    ids = list(stock_ids)
    if not ids:
        return pd.DataFrame(columns=list(PRICE_HISTORY_COLUMNS))
    statement = (
        select(
            Price.trade_date,
            Price.stock_id,
            Price.open,
            Price.high,
            Price.low,
            Price.close,
            Price.volume,
            Price.traded_value,
            Price.source,
            Price.open_adj,
            Price.high_adj,
            Price.low_adj,
            Price.close_adj,
        )
        .where(
            Price.stock_id.in_(ids),
            Price.trade_date >= start.isoformat(),
            Price.trade_date <= end.isoformat(),
        )
        .order_by(Price.stock_id, Price.trade_date)
    )
    result = session.execute(statement).all()
    return pd.DataFrame(result, columns=list(PRICE_HISTORY_COLUMNS))


def upsert_price_adj(session: Session, rows: pd.DataFrame) -> int:
    """Fill adj columns on EXISTING price rows; return rows matched.

    Update-only: adj history predates our raw history, and raw OHLCV is
    NOT NULL, so dates without a raw bar are skipped (counted out).
    """
    records = to_records(rows, _PRICE_COLUMNS, ("trade_date", "stock_id"), "PriceAdj")
    if not records:
        return 0
    result = session.execute(
        text(
            "UPDATE prices SET open_adj = :open_adj, high_adj = :high_adj,"
            " low_adj = :low_adj, close_adj = :close_adj"
            " WHERE trade_date = :trade_date AND stock_id = :stock_id"
        ),
        [
            {
                "trade_date": record["trade_date"],
                "stock_id": record["stock_id"],
                "open_adj": record.get("open_adj"),
                "high_adj": record.get("high_adj"),
                "low_adj": record.get("low_adj"),
                "close_adj": record.get("close_adj"),
            }
            for record in records
        ],
    )
    session.flush()
    return result.rowcount
