"""009: isolate prediction, signal, and daily portfolio rows by run."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "009"

_TARGET_PRIMARY_KEYS = {
    "predictions": ("run_id", "prediction_date", "stock_id", "model_version"),
    "signals": ("run_id", "signal_date", "stock_id"),
    "portfolio_daily": ("run_id", "trade_date"),
}

_CREATE_SQL = {
    "predictions": """
        CREATE TABLE predictions_run_scoped_new (
            prediction_date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_probability REAL NOT NULL CHECK (
                prediction_probability >= 0 AND prediction_probability <= 1
            ),
            rank INTEGER NOT NULL CHECK (rank > 0),
            PRIMARY KEY (run_id, prediction_date, stock_id, model_version),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """,
    "signals": """
        CREATE TABLE signals_run_scoped_new (
            signal_date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            signal TEXT NOT NULL CHECK (signal IN ('BUY', 'HOLD', 'SELL', 'NONE')),
            rank INTEGER NOT NULL,
            target_weight REAL NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
            PRIMARY KEY (run_id, signal_date, stock_id),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """,
    "portfolio_daily": """
        CREATE TABLE portfolio_daily_run_scoped_new (
            trade_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            nav REAL NOT NULL CHECK (nav >= 0),
            equity_exposure REAL NOT NULL CHECK (
                equity_exposure >= 0 AND equity_exposure <= 1
            ),
            forecast_volatility REAL,
            realized_volatility REAL,
            drawdown REAL,
            taiex_close REAL,
            taiex_ma60 REAL,
            market_regime TEXT NOT NULL,
            PRIMARY KEY (run_id, trade_date),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """,
}

_COLUMNS = {
    "predictions": (
        "prediction_date, stock_id, run_id, model_version, "
        "prediction_probability, rank"
    ),
    "signals": "signal_date, stock_id, run_id, signal, rank, target_weight",
    "portfolio_daily": (
        "trade_date, run_id, nav, equity_exposure, forecast_volatility, "
        "realized_volatility, drawdown, taiex_close, taiex_ma60, market_regime"
    ),
}


def _primary_key(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return tuple(row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5] > 0)


def upgrade(conn: sqlite3.Connection) -> None:
    """Rebuild legacy tables whose primary keys did not include ``run_id``."""
    pending = [
        table
        for table, target in _TARGET_PRIMARY_KEYS.items()
        if _primary_key(conn, table) and _primary_key(conn, table) != target
    ]
    if not pending:
        conn.commit()
        return

    foreign_keys = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table in pending:
            replacement = f"{table}_run_scoped_new"
            conn.execute(_CREATE_SQL[table])
            columns = _COLUMNS[table]
            conn.execute(
                f"INSERT INTO {replacement} ({columns}) SELECT {columns} FROM {table}"
            )
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {replacement} RENAME TO {table}")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_date_rank "
            "ON predictions (prediction_date, rank)"
        )
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"foreign-key violations after migration 009: {violations[:5]}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")


def downgrade(conn: sqlite3.Connection) -> None:
    """Keep run-scoped keys because collapsing them can destroy run history."""
    conn.commit()
