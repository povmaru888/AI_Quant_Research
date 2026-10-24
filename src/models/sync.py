"""Persistent checkpoints for incremental market-data synchronization."""

from __future__ import annotations

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class DataSyncDay(Base):
    __tablename__ = "data_sync_days"

    source: Mapped[str] = mapped_column(String, primary_key=True)
    dataset: Mapped[str] = mapped_column(String, primary_key=True)
    market: Mapped[str] = mapped_column(String, primary_key=True)
    trade_date: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    synced_at: Mapped[str] = mapped_column(String, nullable=False)


class DataSyncJob(Base):
    __tablename__ = "data_sync_jobs"

    job_key: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    market: Mapped[str | None] = mapped_column(String, nullable=True)
    stock_id: Mapped[str | None] = mapped_column(String, nullable=True)
    start_date: Mapped[str] = mapped_column(String, nullable=False)
    end_date: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
