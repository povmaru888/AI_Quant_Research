"""010: point-in-time monthly revenue storage."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "010"

def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS monthly_revenues (
            stock_id TEXT NOT NULL,
            revenue_year INTEGER NOT NULL CHECK (revenue_year >= 1900),
            revenue_month INTEGER NOT NULL CHECK (revenue_month BETWEEN 1 AND 12),
            revenue REAL NOT NULL CHECK (revenue >= 0),
            source_date TEXT NOT NULL,
            create_time TEXT,
            available_date TEXT NOT NULL,
            country TEXT,
            source TEXT NOT NULL,
            PRIMARY KEY (stock_id, revenue_year, revenue_month),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
            CHECK (available_date >= source_date)
        );
        CREATE INDEX IF NOT EXISTS idx_monthly_revenues_available
            ON monthly_revenues (stock_id, available_date);
        CREATE INDEX IF NOT EXISTS idx_monthly_revenues_period
            ON monthly_revenues (revenue_year, revenue_month);
        """
    )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_monthly_revenues_period")
    conn.execute("DROP INDEX IF EXISTS idx_monthly_revenues_available")
    conn.execute("DROP TABLE IF EXISTS monthly_revenues")
    conn.commit()
