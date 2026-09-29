"""Helpers for creating a test schema at the current additive migration level."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path


_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "database/migrations/versions"


def apply_pit_v3_feature_migration(conn: sqlite3.Connection) -> None:
    """Bring a 001/003 fixture forward through the current additive schema."""
    versions = (
        "004_market_value.py",
        "005_market_value_revision.py",
        "006_pit_v3_features.py",
        "007_non_trading_checkpoints.py",
    )
    for version in versions:
        path = _MIGRATIONS_DIR / version
        spec = importlib.util.spec_from_file_location(f"migration_{version[:3]}_test", path)
        assert spec is not None and spec.loader is not None
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        migration.upgrade(conn)
