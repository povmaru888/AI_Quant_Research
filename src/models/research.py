"""P1-05: research and trading ORM (SDD v1.0 section 6.2).

Thin mappings over the P1-01 DDL: CHECK constraints live in the migration,
this module only declares types, keys, nullability and FKs. Traceability
(Prediction/Order -> PipelineRun) is expressed via foreign keys.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class PipelineRun(Base):
    """A research/backtest run log; ``run_id`` is the traceability root."""

    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_time: Mapped[str] = mapped_column(String, nullable=False)
    data_end_date: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    feature_version: Mapped[str] = mapped_column(String, nullable=False)
    parameter_version: Mapped[str] = mapped_column(String, nullable=False)
    universe_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


class Feature(Base):
    """Cross-sectional factor snapshot; key (rebalance_date, stock_id, version)."""

    __tablename__ = "features"

    rebalance_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    feature_version: Mapped[str] = mapped_column(String, primary_key=True)
    momentum_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_20d_ex_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ma20_ma60_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi14: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_ma20_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_ma20_ma60: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    book_to_market: Mapped[float | None] = mapped_column(Float, nullable=True)
    sales_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    roa: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_yoy: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_income_qoq: Mapped[float | None] = mapped_column(Float, nullable=True)
    accrual_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    foreign_net_buy_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_net_buy_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    foreign_net_buy_to_issued_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_net_buy_to_issued_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin_balance_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_60d_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    amihud_illiquidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_margin_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_mom: Mapped[float | None] = mapped_column(Float, nullable=True)
    missing_flag: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))


class Prediction(Base):
    """Model output; key (prediction_date, stock_id, model_version)."""

    __tablename__ = "predictions"

    prediction_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    prediction_probability: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)


class Signal(Base):
    """Target holding signal; key (signal_date, stock_id)."""

    __tablename__ = "signals"

    signal_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=False)
    signal: Mapped[str] = mapped_column(String, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)


class Position(Base):
    """Held position snapshot; key (position_date, stock_id)."""

    __tablename__ = "positions"

    position_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    market_value: Mapped[float] = mapped_column(Float, nullable=False)


class Order(Base):
    """Simulated order with costs; execution must follow the signal."""

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=False)
    signal_date: Mapped[str] = mapped_column(String, nullable=False)
    execution_date: Mapped[str] = mapped_column(String, nullable=False)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    open_price: Mapped[float] = mapped_column(Float, nullable=False)
    executed_price: Mapped[float] = mapped_column(Float, nullable=False)
    notional: Mapped[float] = mapped_column(Float, nullable=False)
    broker_fee: Mapped[float] = mapped_column(Float, nullable=False)
    transaction_tax: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_cost: Mapped[float] = mapped_column(Float, nullable=False)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False)


class PortfolioDaily(Base):
    """Daily portfolio state; key trade_date."""

    __tablename__ = "portfolio_daily"

    trade_date: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=False)
    nav: Mapped[float] = mapped_column(Float, nullable=False)
    equity_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    taiex_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    taiex_ma60: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_regime: Mapped[str] = mapped_column(String, nullable=False)


class RunArtifact(Base):
    """Materialized dashboard payload; key (run_id, kind), JSON document."""

    __tablename__ = "run_artifacts"

    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), primary_key=True)
    kind: Mapped[str] = mapped_column(String, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str | None] = mapped_column(String, nullable=True)
