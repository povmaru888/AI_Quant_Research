"""P1-10: signals, positions and orders repository.

Same conventions as P1-06..P1-09: takes an active ``Session``, flushes but
never commits. Signals/positions upsert on their composite keys; orders are
immutable events, so a rerun replaces exactly the batch's own ``order_id``s
(delete + insert), which also handles a shrinking batch.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.research import Order, Position, Signal
from repositories._frames import to_records

SIGNAL_COLUMNS = ("signal_date", "stock_id", "run_id", "signal", "rank", "target_weight")

POSITION_COLUMNS = ("position_date", "stock_id", "shares", "weight", "market_value")

ORDER_COLUMNS = (
    "order_id",
    "run_id",
    "signal_date",
    "execution_date",
    "stock_id",
    "side",
    "quantity",
    "open_price",
    "executed_price",
    "notional",
    "broker_fee",
    "transaction_tax",
    "slippage_cost",
    "total_cost",
)

_SIGNAL_COLUMNS = frozenset(col.name for col in Signal.__table__.columns)
_SIGNAL_KEY = ("run_id", "signal_date", "stock_id")
_POSITION_COLUMNS = frozenset(col.name for col in Position.__table__.columns)
_POSITION_KEY = ("position_date", "stock_id")
_ORDER_COLUMNS = frozenset(col.name for col in Order.__table__.columns)


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


def save_signals(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update signal rows; return the number of rows written."""
    records = to_records(rows, _SIGNAL_COLUMNS, _SIGNAL_KEY, "Signal")
    if not records:
        return 0
    return _upsert(session, Signal, _SIGNAL_KEY, records)


def save_positions(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update position rows; return the number of rows written."""
    records = to_records(rows, _POSITION_COLUMNS, _POSITION_KEY, "Position")
    if not records:
        return 0
    return _upsert(session, Position, _POSITION_KEY, records)


def save_orders(session: Session, rows: pd.DataFrame) -> int:
    """Replace this (run_id, signal_date) scope's orders; return rows written.

    A rerun regenerates the whole signal-date batch, so the scope is deleted
    first (this also handles a shrinking batch). The batch must belong to a
    single run_id and signal_date.
    """
    records = to_records(rows, _ORDER_COLUMNS, "order_id", "Order")
    if not records:
        return 0
    order_ids = [record["order_id"] for record in records]
    if any(not order_id or not str(order_id).strip() for order_id in order_ids):
        raise ValueError("save_orders requires non-empty 'order_id' values")
    run_ids = {record["run_id"] for record in records}
    signal_dates = {record["signal_date"] for record in records}
    if len(run_ids) != 1 or len(signal_dates) != 1:
        raise ValueError("save_orders requires a single run_id and signal_date per batch")
    run_id = next(iter(run_ids))
    signal_date = next(iter(signal_dates))
    session.execute(delete(Order).where(Order.run_id == run_id, Order.signal_date == signal_date))
    session.execute(sqlite_insert(Order), records)
    session.flush()
    return len(records)
