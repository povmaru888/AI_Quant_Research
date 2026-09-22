"""P1-02 acceptance: engine wiring and session transaction boundary."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from settings import Settings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _settings_with_url(settings: Settings, url: str) -> Settings:
    return dataclasses.replace(settings, data=dataclasses.replace(settings.data, database_url=url))


def _init_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()


def _pragma_foreign_keys(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("PRAGMA foreign_keys")).scalar()


def test_engine_uses_settings_url(settings: Settings, temp_db_path: Path) -> None:
    engine = create_engine_from_settings(_settings_with_url(settings, f"sqlite:///{temp_db_path}"))
    try:
        assert engine.url.database == str(temp_db_path)
    finally:
        engine.dispose()


def test_commit_on_success(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = create_engine_from_settings(_settings_with_url(settings, f"sqlite:///{temp_db_path}"))
    try:
        with session_scope(engine) as session:
            session.execute(text("INSERT INTO stocks (stock_id, market) VALUES ('2330', 'TWSE')"))
        with session_scope(engine) as session:
            count = session.execute(text("SELECT COUNT(*) FROM stocks")).scalar()
        assert count == 1
    finally:
        engine.dispose()


def test_rollback_on_exception(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = create_engine_from_settings(_settings_with_url(settings, f"sqlite:///{temp_db_path}"))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with session_scope(engine) as session:
                session.execute(
                    text("INSERT INTO stocks (stock_id, market) VALUES ('2330', 'TWSE')")
                )
                raise RuntimeError("boom")
        with session_scope(engine) as session:
            count = session.execute(text("SELECT COUNT(*) FROM stocks")).scalar()
        assert count == 0
    finally:
        engine.dispose()


def test_foreign_keys_enforced(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = create_engine_from_settings(_settings_with_url(settings, f"sqlite:///{temp_db_path}"))
    try:
        assert _pragma_foreign_keys(engine) == 1
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                session.execute(
                    text(
                        "INSERT INTO prices (trade_date, stock_id, open, high, low, "
                        "close, volume, traded_value, source) VALUES ('2020-01-02', "
                        "'9999', 10, 11, 9, 10, 100, 1000, 'test')"
                    )
                )
        # A brand-new pooled connection must also enforce foreign keys.
        engine.dispose()
        assert _pragma_foreign_keys(engine) == 1
    finally:
        engine.dispose()


def test_rejects_non_sqlite_url(settings: Settings) -> None:
    with pytest.raises(ValueError, match="only sqlite"):
        create_engine_from_settings(_settings_with_url(settings, "postgresql://user@localhost/db"))


def test_creates_parent_directory(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "test.db"
    engine = create_engine_from_settings(_settings_with_url(settings, f"sqlite:///{target}"))
    try:
        assert target.parent.is_dir()
    finally:
        engine.dispose()
