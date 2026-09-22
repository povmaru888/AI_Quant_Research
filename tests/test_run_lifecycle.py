"""P5-02 acceptance: run lifecycle on a real SQLite database."""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select

from database import create_engine_from_settings, session_scope
from models.research import PipelineRun
from observability.run_lifecycle import managed_run, summarize_error
from settings import Settings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_p502", MIGRATION_PATH)
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


@pytest.fixture()
def engine(settings: Settings, temp_db_path: Path):
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()
    engine = _engine_for(settings, temp_db_path)
    try:
        yield engine
    finally:
        engine.dispose()


def _metadata(run_id: str, **overrides) -> dict:
    base = {
        "run_id": run_id,
        "data_end_date": "2020-02-29",
        "feature_version": "factor_v1",
        "parameter_version": "p1",
    }
    base.update(overrides)
    return base


def _status(engine, run_id: str) -> tuple[str, str | None]:
    with session_scope(engine) as session:
        run = session.execute(select(PipelineRun).where(PipelineRun.run_id == run_id)).scalar_one()
        return run.status, run.error_message


def test_managed_run_success(engine) -> None:
    with session_scope(engine) as session:
        with managed_run(session, _metadata("daily-2020-02-29")) as run:
            assert run.run_id == "daily-2020-02-29"
    assert _status(engine, "daily-2020-02-29") == ("succeeded", None)


def test_managed_run_failure_records_summary(engine) -> None:
    with session_scope(engine) as session:
        with (
            pytest.raises(RuntimeError, match="half dataset"),
            managed_run(session, _metadata("monthly-2020-02-29")),
        ):
            raise RuntimeError("half dataset")
    status, error = _status(engine, "monthly-2020-02-29")
    assert status == "failed"
    assert error == "RuntimeError: half dataset"


def test_managed_run_reraises_and_terminal_runs_stay_closed(engine) -> None:
    with session_scope(engine) as session:
        with (
            pytest.raises(ValueError, match="boom"),
            managed_run(session, _metadata("research-2020-02")),
        ):
            raise ValueError("boom")
    # Terminal runs are immutable: closing a failed run again must fail loudly.
    from repositories.runs import finish_run

    with session_scope(engine) as session:
        with pytest.raises(ValueError, match="already"):
            finish_run(session, "research-2020-02", "succeeded")
    assert _status(engine, "research-2020-02")[0] == "failed"


def test_managed_run_logs_bracket(caplog: pytest.LogCaptureFixture, engine) -> None:
    logger = logging.getLogger("p502-probe")
    with session_scope(engine) as session:
        with (
            caplog.at_level(logging.INFO, logger="p502-probe"),
            managed_run(session, _metadata("daily-log-probe"), logger=logger),
        ):
            pass
    messages = [record.getMessage() for record in caplog.records]
    assert "run started" in messages
    assert "run finished" in messages


def test_bad_metadata_rejected(engine) -> None:
    with session_scope(engine) as session:
        with pytest.raises(ValueError, match="run_id"):
            with managed_run(session, {"data_end_date": "2020-02-29"}):
                pass
        with pytest.raises(ValueError, match="must be a dict"):
            with managed_run(session, []):  # type: ignore[arg-type]
                pass


def test_summarize_error_single_line() -> None:
    assert summarize_error(RuntimeError("a\nb")) == "RuntimeError: a b"
    long = summarize_error(RuntimeError("x" * 600))
    assert len(long) <= 500 + len("\u2026[truncated]")
    assert long.endswith("[truncated]")
