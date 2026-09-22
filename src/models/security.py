"""P1-03: security master ORM (stocks table, SDD v1.0 section 6.2).

Dates are stored as TEXT in YYYY-MM-DD form (SDD 5.3), hence plain string
columns. This is a thin mapping: constraints are enforced by the database.
"""

from __future__ import annotations

from sqlalchemy import String, text
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class Stock(Base):
    """A listed (or delisted) security; ``stock_id`` is the unique key."""

    __tablename__ = "stocks"

    stock_id: Mapped[str] = mapped_column(String, primary_key=True)
    stock_name: Mapped[str | None] = mapped_column(String, nullable=True)
    market: Mapped[str] = mapped_column(String, nullable=False)
    listed_date: Mapped[str | None] = mapped_column(String, nullable=True)
    delisted_date: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
