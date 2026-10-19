"""Idempotent persistence and PIT reads for monthly revenue."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.market import MonthlyRevenue
from repositories._frames import to_records

MONTHLY_REVENUE_COLUMNS = (
    "stock_id",
    "revenue_year",
    "revenue_month",
    "revenue",
    "source_date",
    "create_time",
    "available_date",
    "country",
    "source",
)
_COLUMNS = frozenset(column.name for column in MonthlyRevenue.__table__.columns)
_KEY = ("stock_id", "revenue_year", "revenue_month")


def upsert_monthly_revenues(session: Session, rows: pd.DataFrame) -> int:
    records = to_records(rows, _COLUMNS, _KEY, "MonthlyRevenue")
    if not records:
        return 0
    statement = sqlite_insert(MonthlyRevenue)
    update_columns = [column for column in records[0] if column not in _KEY]
    statement = statement.on_conflict_do_update(
        index_elements=list(_KEY),
        set_={column: statement.excluded[column] for column in update_columns},
    )
    session.execute(statement, records)
    session.flush()
    return len(records)


def load_monthly_revenues(session: Session, as_of: date) -> pd.DataFrame:
    rows = session.execute(
        select(*(getattr(MonthlyRevenue, column) for column in MONTHLY_REVENUE_COLUMNS))
        .where(MonthlyRevenue.available_date <= as_of.isoformat())
        .order_by(
            MonthlyRevenue.stock_id,
            MonthlyRevenue.revenue_year,
            MonthlyRevenue.revenue_month,
        )
    ).all()
    return pd.DataFrame(rows, columns=list(MONTHLY_REVENUE_COLUMNS))
