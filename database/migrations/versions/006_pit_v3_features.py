"""006: add v3 issued-share-normalized feature columns."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "006"

_NEW_COLUMNS = (
    "foreign_net_buy_to_issued_shares",
    "trust_net_buy_to_issued_shares",
)


def upgrade(conn: sqlite3.Connection) -> None:
    """Add nullable v3 columns while preserving the v2 feature schema."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(features)")}
    for name in _NEW_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {name} REAL")
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Keep additive columns on downgrade; SQLite table rebuild is lossy-risky."""
    conn.commit()
