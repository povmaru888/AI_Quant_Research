"""004: point-in-time market value and source change tracking.

The source revision is incremented by triggers whenever a source table used
to build research panels changes.  This gives panel cache checks a constant
time data watermark without hashing the full SQLite database on every run.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "004"

_SOURCE_TABLES = ("prices", "financials", "institutional", "stocks")


def upgrade(conn: sqlite3.Connection) -> None:
    """Create market-value storage, supporting indexes and revision triggers."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_values (
            trade_date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            market_value REAL NOT NULL CHECK (market_value > 0),
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, stock_id),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
        );

        CREATE INDEX IF NOT EXISTS idx_market_values_stock_date
            ON market_values (stock_id, trade_date);

        CREATE TABLE IF NOT EXISTS market_value_sync_days (
            trade_date TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'failed', 'non_trading')
            ),
            row_count INTEGER NOT NULL DEFAULT 0 CHECK (row_count >= 0),
            content_hash TEXT,
            synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS source_revision (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            revision INTEGER NOT NULL CHECK (revision >= 0)
        );
        INSERT OR IGNORE INTO source_revision (singleton, revision) VALUES (1, 0);

        CREATE INDEX IF NOT EXISTS idx_institutional_stock_date
            ON institutional (stock_id, trade_date);
        """
    )
    for table in _SOURCE_TABLES:
        for operation in ("INSERT", "UPDATE", "DELETE"):
            trigger = f"trg_source_revision_{table}_{operation.lower()}"
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {trigger} AFTER {operation} ON {table} "
                "BEGIN UPDATE source_revision SET revision = revision + 1 WHERE singleton = 1; END"
            )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Remove revision tracking and market-value objects."""
    for table in (*_SOURCE_TABLES, "market_values"):
        for operation in ("INSERT", "UPDATE", "DELETE"):
            trigger = f"trg_source_revision_{table}_{operation.lower()}"
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP INDEX IF EXISTS idx_institutional_stock_date")
    conn.execute("DROP INDEX IF EXISTS idx_market_values_stock_date")
    conn.execute("DROP TABLE IF EXISTS market_value_sync_days")
    conn.execute("DROP TABLE IF EXISTS market_values")
    conn.execute("DROP TABLE IF EXISTS source_revision")
    conn.commit()
