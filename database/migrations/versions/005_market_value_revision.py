"""005: track daily market-value checkpoints without per-row trigger writes.

Market values arrive in large daily batches. Incrementing the global source
revision once per stock row would add millions of writes during backfill, so
the per-row triggers from migration 004 are replaced by a checkpoint trigger.
SQLite file/WAL metadata remains part of panel source fingerprints as well.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "005"


def upgrade(conn: sqlite3.Connection) -> None:
    """Move market-value change tracking from data rows to daily checkpoints."""
    for operation in ("INSERT", "UPDATE", "DELETE"):
        conn.execute(f"DROP TRIGGER IF EXISTS trg_source_revision_market_values_{operation.lower()}")
        trigger = f"trg_source_revision_market_value_sync_days_{operation.lower()}"
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS {trigger} AFTER {operation} ON market_value_sync_days "
            "BEGIN UPDATE source_revision SET revision = revision + 1 WHERE singleton = 1; END"
        )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Remove checkpoint triggers and restore migration 004 row triggers."""
    for operation in ("INSERT", "UPDATE", "DELETE"):
        conn.execute(f"DROP TRIGGER IF EXISTS trg_source_revision_market_value_sync_days_{operation.lower()}")
        trigger = f"trg_source_revision_market_values_{operation.lower()}"
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS {trigger} AFTER {operation} ON market_values "
            "BEGIN UPDATE source_revision SET revision = revision + 1 WHERE singleton = 1; END"
        )
    conn.commit()
