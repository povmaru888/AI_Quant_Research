"""P1-04: market and fundamental ORM (SDD v1.0 section 6.2).

Thin mappings over the P1-01 DDL: CHECK constraints live in the migration,
this module only declares types, composite keys, nullability and FKs.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class Price(Base):
    """Daily OHLCV bar; composite key (trade_date, stock_id)."""

    __tablename__ = "prices"

    trade_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    traded_value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    open_adj: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_adj: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_adj: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_adj: Mapped[float | None] = mapped_column(Float, nullable=True)


class Financial(Base):
    """Point-in-time financial report; key (stock_id, report_period, available_date)."""

    __tablename__ = "financials"

    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    report_period: Mapped[str] = mapped_column(String, primary_key=True)
    announcement_date: Mapped[str] = mapped_column(String, nullable=False)
    available_date: Mapped[str] = mapped_column(String, primary_key=True)
    revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)


class Institutional(Base):
    """Daily institutional/chip snapshot; composite key (trade_date, stock_id)."""

    __tablename__ = "institutional"

    trade_date: Mapped[str] = mapped_column(String, primary_key=True)
    stock_id: Mapped[str] = mapped_column(ForeignKey("stocks.stock_id"), primary_key=True)
    foreign_net_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_net_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    float_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
