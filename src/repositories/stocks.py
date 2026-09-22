"""P1-06: stock master repository (stocks table).

Conventions for all P1-06..P1-11 repositories: functions take an active
``Session`` and never commit; the caller owns the transaction boundary via
``session_scope``. Upserts are idempotent (SQLite ON CONFLICT DO UPDATE).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.security import Stock

_BUSINESS_COLUMNS = (
    "stock_id",
    "stock_name",
    "market",
    "listed_date",
    "delisted_date",
    "industry",
)

_STOCK_COLUMNS = frozenset(col.name for col in Stock.__table__.columns)


def _row_dicts(rows: pd.DataFrame) -> list[dict]:
    unknown = [col for col in rows.columns if col not in _STOCK_COLUMNS]
    if unknown:
        raise ValueError(f"unknown Stock columns: {unknown}")
    records: list[dict] = []
    for record in rows.to_dict(orient="records"):
        cleaned = {key: (None if pd.isna(value) else value) for key, value in record.items()}
        records.append(cleaned)
    return records


def upsert_stocks(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update stock rows; return the number of rows written."""
    if "stock_id" not in rows.columns:
        raise ValueError("upsert_stocks requires a 'stock_id' column")
    records = _row_dicts(rows)
    if not records:
        return 0
    update_columns = [col for col in rows.columns if col != "stock_id"]
    statement = sqlite_insert(Stock)
    if update_columns:
        statement = statement.on_conflict_do_update(
            index_elements=["stock_id"],
            set_={col: statement.excluded[col] for col in update_columns},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=["stock_id"])
    session.execute(statement, records)
    session.flush()
    return len(records)


def get_active_stocks(session: Session, as_of: date) -> pd.DataFrame:
    """Return stocks listed and not yet delisted on ``as_of``."""
    as_of_text = as_of.isoformat()
    statement = (
        select(
            Stock.stock_id,
            Stock.stock_name,
            Stock.market,
            Stock.listed_date,
            Stock.delisted_date,
            Stock.industry,
        )
        .where(
            or_(Stock.listed_date.is_(None), Stock.listed_date <= as_of_text),
            or_(Stock.delisted_date.is_(None), Stock.delisted_date > as_of_text),
        )
        .order_by(Stock.stock_id)
    )
    result = session.execute(statement).all()
    return pd.DataFrame(result, columns=list(_BUSINESS_COLUMNS))
