"""P1-11 acceptance: run lifecycle repository."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.research import PipelineRun
from repositories.runs import finish_run, start_run
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


def _engine_for(settings: Settings, path: Path):
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{path}")
    )
    return create_engine_from_settings(db_settings)


def _init_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()


def _metadata(**overrides) -> dict:
    base = {
        "run_id": "run-001",
        "run_time": "2020-01-01T00:00:00+00:00",
        "data_end_date": "2019-12-31",
        "feature_version": "factor_v1",
        "parameter_version": "p1",
    }
    base.update(overrides)
    return base


def test_start_run(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            run = start_run(session, _metadata(universe_count=100))
            assert run.status == "started"
            assert run.run_time == "2020-01-01T00:00:00+00:00"
        with session_scope(engine) as session:
            stored = session.execute(
                select(PipelineRun).where(PipelineRun.run_id == "run-001")
            ).scalar_one()
            assert stored.universe_count == 100
            assert stored.data_end_date == "2019-12-31"
    finally:
        engine.dispose()


def test_start_run_defaults_time(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        metadata = _metadata()
        del metadata["run_time"]
        with session_scope(engine) as session:
            run = start_run(session, metadata)
            assert run.run_time is not None and len(run.run_time) > 0
    finally:
        engine.dispose()


def test_finish_succeeded(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            start_run(session, _metadata())
        with session_scope(engine) as session:
            run = finish_run(session, "run-001", "succeeded")
            assert run.status == "succeeded"
            assert run.error_message is None
        with session_scope(engine) as session:
            stored = session.execute(
                select(PipelineRun).where(PipelineRun.run_id == "run-001")
            ).scalar_one()
            assert stored.status == "succeeded"
    finally:
        engine.dispose()


def test_finish_failed_keeps_error(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            start_run(session, _metadata())
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="non-empty error"):
                finish_run(session, "run-001", "failed")
        with session_scope(engine) as session:
            run = finish_run(session, "run-001", "failed", error="covariance missing")
            assert run.error_message == "covariance missing"
        with session_scope(engine) as session:
            stored = session.execute(
                select(PipelineRun).where(PipelineRun.run_id == "run-001")
            ).scalar_one()
            assert stored.status == "failed"
    finally:
        engine.dispose()


def test_illegal_transitions(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown run_id"):
                finish_run(session, "run-999", "succeeded")
        with session_scope(engine) as session:
            start_run(session, _metadata())
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="invalid terminal status"):
                finish_run(session, "run-001", "done")
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="non-empty 'run_id'"):
                start_run(session, _metadata(run_id="  "))
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                start_run(session, _metadata())
            session.rollback()
        with session_scope(engine) as session:
            finish_run(session, "run-001", "succeeded")
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="already 'succeeded'"):
                finish_run(session, "run-001", "failed", error="late")
    finally:
        engine.dispose()


def test_start_validation(settings: Settings, temp_db_path: Path) -> None:
    _init_schema(temp_db_path)
    engine = _engine_for(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            metadata = _metadata()
            del metadata["data_end_date"]
            with pytest.raises(ValueError, match="missing run metadata keys"):
                start_run(session, metadata)
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown run metadata keys"):
                start_run(session, _metadata(nope=1))
    finally:
        engine.dispose()
