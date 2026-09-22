"""P1-01: initial SQLite schema migration (SDD v1.0 section 6.2 baseline).

Single source of DDL for the MVP database. Uses only stdlib ``sqlite3``;
SQLAlchemy engine/session boundaries arrive in P1-02. The caller owns the
connection (including ``PRAGMA foreign_keys = ON``); this module only
executes DDL.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "001"

# Creation order: parents before children (FK dependency order).
TABLES: tuple[str, ...] = (
    "stocks",
    "pipeline_runs",
    "prices",
    "financials",
    "institutional",
    "features",
    "predictions",
    "signals",
    "positions",
    "orders",
    "portfolio_daily",
)

INDEXES: tuple[str, ...] = (
    "idx_prices_stock_date",
    "idx_financials_available",
    "idx_features_date_version",
    "idx_predictions_date_rank",
    "idx_orders_execution_date",
)

# Reverse dependency order for downgrade (children before parents).
_DROP_ORDER: tuple[str, ...] = (
    "portfolio_daily",
    "orders",
    "positions",
    "signals",
    "predictions",
    "features",
    "institutional",
    "financials",
    "prices",
    "pipeline_runs",
    "stocks",
)

INITIAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT,
    market TEXT NOT NULL,
    listed_date TEXT,
    delisted_date TEXT,
    industry TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices (
    trade_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    open REAL NOT NULL CHECK (open > 0),
    high REAL NOT NULL CHECK (high > 0),
    low REAL NOT NULL CHECK (low > 0),
    close REAL NOT NULL CHECK (close > 0),
    volume REAL NOT NULL CHECK (volume >= 0),
    traded_value REAL NOT NULL CHECK (traded_value >= 0),
    source TEXT NOT NULL,
    PRIMARY KEY (trade_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE IF NOT EXISTS financials (
    stock_id TEXT NOT NULL,
    report_period TEXT NOT NULL,
    announcement_date TEXT NOT NULL,
    available_date TEXT NOT NULL,
    revenue REAL,
    net_income REAL,
    equity REAL,
    assets REAL,
    operating_income REAL,
    operating_cash_flow REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (stock_id, report_period, available_date),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    CHECK (available_date >= announcement_date)
);

CREATE TABLE IF NOT EXISTS institutional (
    trade_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    foreign_net_buy REAL,
    trust_net_buy REAL,
    margin_balance REAL,
    short_balance REAL,
    float_shares REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (trade_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    run_time TEXT NOT NULL,
    data_end_date TEXT NOT NULL,
    model_version TEXT,
    feature_version TEXT NOT NULL,
    parameter_version TEXT NOT NULL,
    universe_count INTEGER,
    position_count INTEGER,
    status TEXT NOT NULL CHECK (status IN ('started', 'succeeded', 'failed')),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS features (
    rebalance_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    momentum_20d REAL,
    momentum_60d REAL,
    momentum_120d REAL,
    momentum_20d_ex_5d REAL,
    ma20_ma60_gap REAL,
    rsi14 REAL,
    price_ma20_gap REAL,
    volume_ma20_ma60 REAL,
    earnings_yield REAL,
    book_to_market REAL,
    sales_yield REAL,
    dividend_yield REAL,
    roe REAL,
    roa REAL,
    revenue_yoy REAL,
    operating_income_qoq REAL,
    accrual_assets REAL,
    volatility_60d REAL,
    beta_60d REAL,
    max_drawdown_120d REAL,
    turnover_60d REAL,
    foreign_net_buy_float REAL,
    trust_net_buy_float REAL,
    margin_balance_change REAL,
    close_60d_high REAL,
    log_market_cap REAL,
    amihud_illiquidity REAL,
    short_margin_ratio REAL,
    operating_margin REAL,
    revenue_mom REAL,
    missing_flag INTEGER NOT NULL DEFAULT 0 CHECK (missing_flag IN (0, 1)),
    PRIMARY KEY (rebalance_date, stock_id, feature_version),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE IF NOT EXISTS predictions (
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

CREATE TABLE IF NOT EXISTS signals (
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

CREATE TABLE IF NOT EXISTS positions (
    position_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    shares REAL NOT NULL CHECK (shares >= 0),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    market_value REAL NOT NULL CHECK (market_value >= 0),
    PRIMARY KEY (position_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    execution_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL CHECK (quantity > 0),
    open_price REAL NOT NULL CHECK (open_price > 0),
    executed_price REAL NOT NULL CHECK (executed_price > 0),
    notional REAL NOT NULL CHECK (notional > 0),
    broker_fee REAL NOT NULL CHECK (broker_fee >= 0),
    transaction_tax REAL NOT NULL CHECK (transaction_tax >= 0),
    slippage_cost REAL NOT NULL CHECK (slippage_cost >= 0),
    total_cost REAL NOT NULL CHECK (total_cost >= 0),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id),
    CHECK (execution_date > signal_date)
);

CREATE TABLE IF NOT EXISTS portfolio_daily (
    trade_date TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    nav REAL NOT NULL CHECK (nav >= 0),
    equity_exposure REAL NOT NULL CHECK (equity_exposure >= 0 AND equity_exposure <= 1),
    forecast_volatility REAL,
    realized_volatility REAL,
    drawdown REAL,
    taiex_close REAL,
    taiex_ma60 REAL,
    market_regime TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_prices_stock_date ON prices (stock_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_financials_available ON financials (stock_id, available_date);
CREATE INDEX IF NOT EXISTS idx_features_date_version ON features (rebalance_date, feature_version);
CREATE INDEX IF NOT EXISTS idx_predictions_date_rank ON predictions (prediction_date, rank);
CREATE INDEX IF NOT EXISTS idx_orders_execution_date ON orders (execution_date);
"""


def upgrade(conn: sqlite3.Connection) -> None:
    """Create all SDD baseline tables and indexes (idempotent)."""
    conn.executescript(INITIAL_SCHEMA_SQL)
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Drop all tables, returning the schema to empty (idempotent)."""
    for table in _DROP_ORDER:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
