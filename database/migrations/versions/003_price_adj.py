"""003: adjusted OHLC columns on prices (TaiwanStockPriceAdj backfill).

Sponsor-only ``TaiwanStockPriceAdj`` carries back-adjusted open/high/low/
/// close (1994-10-01 ~ now); raw columns stay the record of truth and the
adj columns fill alongside them, keyed by the same (trade_date, stock_id)
primary key. Nullable with no CHECK: a NULL adj means "not backfilled",
never "zero". Rows predating our price history (pre-2015) have no raw row
to attach to and are skipped by the upsert.

Uses only stdlib ``sqlite3`` like 001/002; the caller owns the connection.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "003"

_ADJ_COLUMNS: tuple[str, ...] = ("open_adj", "high_adj", "low_adj", "close_adj")


def upgrade(conn: sqlite3.Connection) -> None:
    """Add adj columns (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(prices)").fetchall()}
    for column in _ADJ_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE prices ADD COLUMN {column} REAL")
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Drop adj columns (SQLite rebuild; 001/002 untouched)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(prices)").fetchall()}
    if not any(column in existing for column in _ADJ_COLUMNS):
        return
    conn.execute("ALTER TABLE prices DROP COLUMN open_adj")
    conn.execute("ALTER TABLE prices DROP COLUMN high_adj")
    conn.execute("ALTER TABLE prices DROP COLUMN low_adj")
    conn.execute("ALTER TABLE prices DROP COLUMN close_adj")
    conn.commit()
