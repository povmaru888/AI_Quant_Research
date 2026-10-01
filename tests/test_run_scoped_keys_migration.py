"""Migration 009 keeps research rows isolated by pipeline run."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "009_run_scoped_research_keys.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("run_scoped_keys_009", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _primary_key(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return tuple(row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5] > 0)


def _legacy_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE stocks (stock_id TEXT PRIMARY KEY);
        CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY);
        INSERT INTO stocks VALUES ('2330');
        INSERT INTO pipeline_runs VALUES ('run-001');
        INSERT INTO pipeline_runs VALUES ('run-002');
        CREATE TABLE predictions (
            prediction_date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_probability REAL NOT NULL CHECK (
                prediction_probability >= 0 AND prediction_probability <= 1
            ),
            rank INTEGER NOT NULL CHECK (rank > 0),
            PRIMARY KEY (prediction_date, stock_id, model_version),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        CREATE INDEX idx_predictions_date_rank ON predictions (prediction_date, rank);
        CREATE TABLE signals (
            signal_date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            signal TEXT NOT NULL CHECK (signal IN ('BUY', 'HOLD', 'SELL', 'NONE')),
            rank INTEGER NOT NULL,
            target_weight REAL NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
            PRIMARY KEY (signal_date, stock_id),
            FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        CREATE TABLE portfolio_daily (
            trade_date TEXT PRIMARY KEY,
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
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        INSERT INTO predictions VALUES ('2020-01-31', '2330', 'run-001', 'xgb', 0.8, 1);
        INSERT INTO signals VALUES ('2020-01-31', '2330', 'run-001', 'BUY', 1, 0.1);
        INSERT INTO portfolio_daily VALUES (
            '2020-02-03', 'run-001', 1000000, 0.9, NULL, NULL, 0, 10000, 9900, 'bull'
        );
        """
    )


def test_upgrade_preserves_rows_and_allows_same_keys_in_another_run(
    temp_db_conn: sqlite3.Connection,
) -> None:
    _legacy_schema(temp_db_conn)
    migration.upgrade(temp_db_conn)

    assert _primary_key(temp_db_conn, "predictions") == (
        "run_id",
        "prediction_date",
        "stock_id",
        "model_version",
    )
    assert _primary_key(temp_db_conn, "signals") == (
        "run_id",
        "signal_date",
        "stock_id",
    )
    assert _primary_key(temp_db_conn, "portfolio_daily") == ("run_id", "trade_date")
    assert temp_db_conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
    assert temp_db_conn.execute("SELECT count(*) FROM signals").fetchone()[0] == 1
    assert temp_db_conn.execute("SELECT count(*) FROM portfolio_daily").fetchone()[0] == 1

    temp_db_conn.execute(
        "INSERT INTO predictions VALUES ('2020-01-31','2330','run-002','xgb',0.4,2)"
    )
    temp_db_conn.execute(
        "INSERT INTO signals VALUES ('2020-01-31','2330','run-002','HOLD',2,0.2)"
    )
    temp_db_conn.execute(
        "INSERT INTO portfolio_daily VALUES "
        "('2020-02-03','run-002',900000,0.5,NULL,NULL,-0.1,10000,9900,'bear')"
    )
    assert temp_db_conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 2
    assert temp_db_conn.execute("SELECT count(*) FROM signals").fetchone()[0] == 2
    assert temp_db_conn.execute("SELECT count(*) FROM portfolio_daily").fetchone()[0] == 2

    migration.upgrade(temp_db_conn)
    assert temp_db_conn.execute("PRAGMA foreign_key_check").fetchall() == []
