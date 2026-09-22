"""P1-08: financials and institutional repository.

Same conventions as P1-06/P1-07: takes an active ``Session``, flushes but
never commits; upserts are idempotent via SQLite ON CONFLICT DO UPDATE.
PIT reads follow SDD 6.3 (single window-function query, no N+1).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.market import Financial, Institutional
from repositories._frames import to_records

FINANCIAL_COLUMNS = (
    "stock_id",
    "report_period",
    "announcement_date",
    "available_date",
    "revenue",
    "net_income",
    "equity",
    "assets",
    "operating_income",
    "operating_cash_flow",
    "source",
)

INSTITUTIONAL_COLUMNS = (
    "trade_date",
    "stock_id",
    "foreign_net_buy",
    "trust_net_buy",
    "margin_balance",
    "short_balance",
    "float_shares",
    "source",
)

_FINANCIAL_COLUMNS = frozenset(col.name for col in Financial.__table__.columns)
_FINANCIAL_KEY = ("stock_id", "report_period", "available_date")
_INSTITUTIONAL_COLUMNS = frozenset(col.name for col in Institutional.__table__.columns)
_INSTITUTIONAL_KEY = ("trade_date", "stock_id")

_PIT_SQL = text(
    """
SELECT stock_id, report_period, announcement_date, available_date,
       revenue, net_income, equity, assets,
       operating_income, operating_cash_flow, source
FROM (
    SELECT f.*,
           ROW_NUMBER() OVER (
               PARTITION BY f.stock_id
               ORDER BY f.available_date DESC, f.announcement_date DESC
           ) AS rn
    FROM financials AS f
    WHERE f.available_date <= :as_of
)
WHERE rn = 1
ORDER BY stock_id
"""
)


def _upsert(session: Session, entity, key: tuple[str, ...], records: list[dict]) -> int:
    statement = sqlite_insert(entity)
    update_columns = [col for col in records[0] if col not in key]
    if update_columns:
        statement = statement.on_conflict_do_update(
            index_elements=list(key),
            set_={col: statement.excluded[col] for col in update_columns},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=list(key))
    session.execute(statement, records)
    session.flush()
    return len(records)


def upsert_financials(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update financial rows; return the number of rows written."""
    records = to_records(rows, _FINANCIAL_COLUMNS, _FINANCIAL_KEY, "Financial")
    if not records:
        return 0
    return _upsert(session, Financial, _FINANCIAL_KEY, records)


def upsert_institutional(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update institutional rows; return the number of rows written."""
    records = to_records(rows, _INSTITUTIONAL_COLUMNS, _INSTITUTIONAL_KEY, "Institutional")
    if not records:
        return 0
    return _upsert(session, Institutional, _INSTITUTIONAL_KEY, records)


def load_pit_financials(session: Session, as_of: date) -> pd.DataFrame:
    """Latest available financial row per stock as of ``as_of``."""
    result = session.execute(_PIT_SQL, {"as_of": as_of.isoformat()}).all()
    return pd.DataFrame(result, columns=list(FINANCIAL_COLUMNS))
