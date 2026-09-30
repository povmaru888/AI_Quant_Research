"""008: distinguish verified partial market-value snapshots."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "008"


def upgrade(conn: sqlite3.Connection) -> None:
    """Allow partial checkpoints while preserving rows and revision triggers."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='market_value_sync_days'"
    ).fetchone()
    if row is None:
        return
    if "'partial'" not in str(row[0]):
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            CREATE TABLE market_value_sync_days_new (
                trade_date TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (
                    status IN ('running', 'succeeded', 'failed', 'partial', 'non_trading')
                ),
                row_count INTEGER NOT NULL DEFAULT 0 CHECK (row_count >= 0),
                content_hash TEXT,
                synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO market_value_sync_days_new
                (trade_date, status, row_count, content_hash, synced_at)
            SELECT trade_date, status, row_count, content_hash, synced_at
            FROM market_value_sync_days
            """
        )
        conn.execute("DROP TABLE market_value_sync_days")
        conn.execute(
            "ALTER TABLE market_value_sync_days_new RENAME TO market_value_sync_days"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

    for operation in ("INSERT", "UPDATE", "DELETE"):
        trigger = f"trg_source_revision_market_value_sync_days_{operation.lower()}"
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS {trigger} AFTER {operation} "
            "ON market_value_sync_days BEGIN "
            "UPDATE source_revision SET revision = revision + 1 WHERE singleton = 1; END"
        )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Keep the expanded checkpoint schema for safe backward compatibility."""
    conn.commit()
