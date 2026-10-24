from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path


def test_price_sync_migration_is_idempotent(tmp_path: Path) -> None:
    path = (
        Path(__file__).resolve().parents[1] / "database/migrations/versions/011_price_sync_state.py"
    )
    spec = importlib.util.spec_from_file_location("migration_011", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with sqlite3.connect(tmp_path / "migration.db") as connection:
        migration.upgrade(connection)
        migration.upgrade(connection)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"data_sync_days", "data_sync_jobs"} <= tables
