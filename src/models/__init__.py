"""P1-03+: shared SQLAlchemy declarative base for ORM models."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared base for all domain ORM models (security, market, research)."""
