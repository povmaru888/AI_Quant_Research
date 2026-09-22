"""P1-01 acceptance: upgrade/downgrade on a fresh SQLite database.

The migration lives outside the ``src`` package layout, so it is loaded by
file path. Full constraint matrices belong to the P1-03..P1-05 ORM tests;
here we prove the DDL itself is complete and enforced.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)

EXPECTED_COLUMNS: dict[str, set[str]] = {
    "stocks": {
        "stock_id",
        "stock_name",
        "market",
        "listed_date",
        "delisted_date",
        "industry",
        "created_at",
        "updated_at",
    },
    "prices": {
        "trade_date",
        "stock_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded_value",
        "source",
    },
    "financials": {
        "stock_id",
        "report_period",
        "announcement_date",
        "available_date",
        "revenue",
        "net_income",
        "equity",
        "assets",
        "operating_income",
        "operating_cash_flow",
        "source",
    },
    "institutional": {
        "trade_date",
        "stock_id",
        "foreign_net_buy",
        "trust_net_buy",
        "margin_balance",
        "short_balance",
        "float_shares",
        "source",
    },
    "pipeline_runs": {
        "run_id",
        "run_time",
        "data_end_date",
        "model_version",
        "feature_version",
        "parameter_version",
        "universe_count",
        "position_count",
        "status",
        "error_message",
    },
    "features": {
        "rebalance_date",
        "stock_id",
        "feature_version",
        "momentum_20d",
        "momentum_60d",
        "momentum_120d",
        "momentum_20d_ex_5d",
        "ma20_ma60_gap",
        "rsi14",
        "price_ma20_gap",
        "volume_ma20_ma60",
        "earnings_yield",
        "book_to_market",
        "sales_yield",
        "dividend_yield",
        "roe",
        "roa",
        "revenue_yoy",
        "operating_income_qoq",
        "accrual_assets",
        "volatility_60d",
        "beta_60d",
        "max_drawdown_120d",
        "turnover_60d",
        "foreign_net_buy_float",
        "trust_net_buy_float",
        "margin_balance_change",
        "close_60d_high",
        "log_market_cap",
        "amihud_illiquidity",
        "short_margin_ratio",
        "operating_margin",
        "revenue_mom",
        "missing_flag",
    },
    "predictions": {
        "prediction_date",
        "stock_id",
        "run_id",
        "model_version",
        "prediction_probability",
        "rank",
    },
    "signals": {"signal_date", "stock_id", "run_id", "signal", "rank", "target_weight"},
    "positions": {"position_date", "stock_id", "shares", "weight", "market_value"},
    "orders": {
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
    },
    "portfolio_daily": {
        "trade_date",
        "run_id",
        "nav",
        "equity_exposure",
        "forecast_volatility",
        "realized_volatility",
        "drawdown",
        "taiex_close",
        "taiex_ma60",
        "market_regime",
    },
}


def _load_migration():
    assert MIGRATION_PATH.is_file(), f"missing {MIGRATION_PATH}"
    spec = importlib.util.spec_from_file_location("initial_schema", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row[0] for row in rows if not row[0].startswith("sqlite_")}


def _seed_parent_rows(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO stocks (stock_id, market) VALUES ('2330', 'TWSE')")
    conn.execute(
        "INSERT INTO pipeline_runs "
        "(run_id, run_time, data_end_date, feature_version, parameter_version, status) "
        "VALUES ('run-001', '2020-01-01', '2019-12-31', 'factor_v1', 'p1', 'started')"
    )


def test_upgrade_creates_all_tables(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    assert set(migration.TABLES) == set(EXPECTED_COLUMNS)
    assert set(migration.TABLES) <= _table_names(temp_db_conn)


def test_upgrade_creates_all_indexes(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    assert set(migration.INDEXES) <= _index_names(temp_db_conn)


def test_schema_columns_match_sdd(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    for table, expected in EXPECTED_COLUMNS.items():
        info = temp_db_conn.execute(f"PRAGMA table_info({table})").fetchall()
        assert {col[1] for col in info} == expected, table


def test_upgrade_idempotent(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    before = _table_names(temp_db_conn)
    migration.upgrade(temp_db_conn)
    assert _table_names(temp_db_conn) == before


def test_downgrade_returns_to_empty(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    migration.downgrade(temp_db_conn)
    assert _table_names(temp_db_conn) == set()
    migration.downgrade(temp_db_conn)
    migration.upgrade(temp_db_conn)
    assert set(migration.TABLES) <= _table_names(temp_db_conn)


def test_price_pk_and_fk(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    _seed_parent_rows(temp_db_conn)
    temp_db_conn.execute(
        "INSERT INTO prices "
        "(trade_date, stock_id, open, high, low, close, volume, traded_value, source) "
        "VALUES ('2020-01-02', '2330', 500, 505, 495, 502, 1000, 502000, 'test')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO prices "
            "(trade_date, stock_id, open, high, low, close, volume, traded_value, source) "
            "VALUES ('2020-01-02', '2330', 500, 505, 495, 502, 1000, 502000, 'test')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO prices "
            "(trade_date, stock_id, open, high, low, close, volume, traded_value, source) "
            "VALUES ('2020-01-02', '9999', 10, 11, 9, 10, 100, 1000, 'test')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO prices "
            "(trade_date, stock_id, open, high, low, close, volume, traded_value, source) "
            "VALUES ('2020-01-03', '2330', 0, 505, 495, 502, 1000, 502000, 'test')"
        )


def test_financials_pit_check(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    _seed_parent_rows(temp_db_conn)
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO financials "
            "(stock_id, report_period, announcement_date, available_date, source) "
            "VALUES ('2330', '2019Q4', '2020-02-01', '2020-01-15', 'test')"
        )


def test_run_status_and_order_timing(temp_db_conn: sqlite3.Connection) -> None:
    migration.upgrade(temp_db_conn)
    _seed_parent_rows(temp_db_conn)
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO pipeline_runs "
            "(run_id, run_time, data_end_date, feature_version, parameter_version, status) "
            "VALUES ('run-002', '2020-01-01', '2019-12-31', 'factor_v1', 'p1', 'done')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        temp_db_conn.execute(
            "INSERT INTO orders "
            "(order_id, run_id, signal_date, execution_date, stock_id, side, quantity, "
            "open_price, executed_price, notional, broker_fee, transaction_tax, "
            "slippage_cost, total_cost) "
            "VALUES ('o-1', 'run-001', '2020-01-31', '2020-01-31', '2330', 'BUY', 10, "
            "500, 500.5, 5000, 7.125, 0, 5, 12.125)"
        )
