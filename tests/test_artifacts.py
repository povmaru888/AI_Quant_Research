"""Run artifacts repository + 002 migration acceptance (temp DB, no network)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from database import create_engine_from_settings, session_scope
from repositories import artifacts as artifacts_repo
from settings import Settings

MIGRATION_001 = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)
MIGRATION_002 = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "002_run_artifacts.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration_002 = _load(MIGRATION_002, "run_artifacts_002")


def test_002_upgrade_downgrade_roundtrip(temp_db_path: Path) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration_002.upgrade(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "run_artifacts" in tables
        migration_002.upgrade(conn)  # idempotent rerun.
        migration_002.downgrade(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "run_artifacts" not in tables
    finally:
        conn.close()


def test_002_kinds_documented() -> None:
    assert set(migration_002.KINDS) == {
        "metrics",
        "factor_ic",
        "monthly_ic",
        "model_explain",
        "sensitivity",
    }


@pytest.fixture()
def session_002(settings: Settings, temp_db_path: Path):
    conn = sqlite3.connect(str(temp_db_path))
    try:
        _load(MIGRATION_001, "initial_schema_art").upgrade(conn)
        migration_002.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    with session_scope(engine) as session:
        session.execute(
            __import__("sqlalchemy").text(
                "INSERT INTO pipeline_runs (run_id, run_time, data_end_date,"
                " feature_version, parameter_version, status)"
                " VALUES ('r1', '2020-03-01T00:00:00+00:00', '2020-02-29',"
                " 'factor_v1', 'p1', 'succeeded')"
            )
        )
        yield session
    engine.dispose()


def test_save_load_roundtrip(session_002) -> None:
    payload = {"cagr": 0.12, "months": ["2020-02"]}
    artifacts_repo.save_artifact(session_002, "r1", "metrics", payload)
    assert artifacts_repo.load_artifact(session_002, "r1", "metrics") == payload
    artifacts_repo.save_artifact(session_002, "r1", "metrics", {"cagr": 0.2})
    assert artifacts_repo.load_artifact(session_002, "r1", "metrics") == {"cagr": 0.2}
    assert artifacts_repo.load_artifact(session_002, "r1", "factor_ic") is None
    assert artifacts_repo.load_artifact(session_002, "nope", "metrics") is None


def test_save_rejects_bad_inputs(session_002) -> None:
    with pytest.raises(ValueError, match="kind"):
        artifacts_repo.save_artifact(session_002, "r1", "nope", {})
    with pytest.raises(ValueError, match="run_id"):
        artifacts_repo.save_artifact(session_002, " ", "metrics", {})
    with pytest.raises(ValueError, match="JSON-serializable"):
        artifacts_repo.save_artifact(session_002, "r1", "metrics", {"v": pd.NA})
    with pytest.raises(ValueError, match="dict or list"):
        artifacts_repo.save_artifact(session_002, "r1", "metrics", "text")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        artifacts_repo.load_artifact(session_002, "r1", "nope")


def test_run_id_fk_enforced(settings: Settings, temp_db_path: Path) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        _load(MIGRATION_001, "initial_schema_art2").upgrade(conn)
        migration_002.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    try:
        with session_scope(engine) as session:
            with pytest.raises(Exception, match="FOREIGN KEY"):
                artifacts_repo.save_artifact(session, "ghost", "metrics", {})
    finally:
        engine.dispose()
