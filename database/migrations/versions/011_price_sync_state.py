"""011: checkpoints for gap-driven price synchronization."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "011"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS data_sync_days (
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            market TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'partial', 'failed', 'non_trading')
            ),
            row_count INTEGER NOT NULL DEFAULT 0 CHECK (row_count >= 0),
            content_hash TEXT,
            error TEXT,
            synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, dataset, market, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_data_sync_days_status_date
            ON data_sync_days (status, trade_date);

        CREATE TABLE IF NOT EXISTS data_sync_jobs (
            job_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            market TEXT,
            stock_id TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'running', 'succeeded', 'partial', 'failed', 'blocked')
            ),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            last_error TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_data_sync_jobs_status
            ON data_sync_jobs (status, source, dataset);
        """
    )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_data_sync_jobs_status")
    conn.execute("DROP TABLE IF EXISTS data_sync_jobs")
    conn.execute("DROP INDEX IF EXISTS idx_data_sync_days_status_date")
    conn.execute("DROP TABLE IF EXISTS data_sync_days")
    conn.commit()
