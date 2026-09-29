"""P5-04 acceptance: system health query on a real SQLite database."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from database import create_engine_from_settings, session_scope
from models.market import Price
from models.research import Feature
from tests.migration_utils import apply_pit_v3_feature_migration
from models.security import Stock
from observability.health import get_system_health
from repositories.runs import finish_run, start_run
from settings import Settings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)
MIGRATION_003_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "003_price_adj.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_p504", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _load_migration_003():
    spec = importlib.util.spec_from_file_location("price_adj_003", MIGRATION_003_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration_003 = _load_migration_003()


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
        migration_003.upgrade(conn)
        apply_pit_v3_feature_migration(conn)
    finally:
        conn.close()
    engine = _engine_for(settings, temp_db_path)
    try:
        yield engine
    finally:
        engine.dispose()


def _metadata(run_id: str, run_time: str) -> dict:
    return {
        "run_id": run_id,
        "run_time": run_time,
        "data_end_date": "2020-02-29",
        "feature_version": "factor_v1",
        "parameter_version": "p1",
    }


def _seed(engine) -> None:
    with session_scope(engine) as session:
        session.add(Stock(stock_id="2330", stock_name="t", market="TW"))
    with session_scope(engine) as session:
        start_run(session, _metadata("run-old", "2020-03-01T00:00:00+00:00"))
        finish_run(session, "run-old", "succeeded")
        start_run(session, _metadata("run-new", "2020-03-02T00:00:00+00:00"))
        finish_run(session, "run-new", "failed", "covariance missing")
    with session_scope(engine) as session:
        session.add(
            Price(
                trade_date="2020-03-03",
                stock_id="2330",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                open_adj=100.0,
                high_adj=101.0,
                low_adj=99.0,
                close_adj=100.0,
                volume=1000.0,
                traded_value=100000.0,
                source="test",
            )
        )
        factor_names = [
            col.name
            for col in Feature.__table__.columns
            if col.name not in ("rebalance_date", "stock_id", "feature_version", "missing_flag")
        ]
        values = {
            "rebalance_date": "2020-02-29",
            "stock_id": "2330",
            "feature_version": "factor_v1",
            "missing_flag": 0,
        }
        values.update({name: 0.1 for name in factor_names})
        session.add(Feature(**values))


def test_health_snapshot(engine) -> None:
    _seed(engine)
    with session_scope(engine) as session:
        health = get_system_health(session, today="2020-03-06")
    assert health["latest_success"] == {
        "run_id": "run-old",
        "data_end_date": "2020-02-29",
        "run_time": "2020-03-01T00:00:00+00:00",
    }
    assert health["latest_failure"]["run_id"] == "run-new"
    assert health["latest_failure"]["error"] == "covariance missing"
    assert health["data_end_date"] == "2020-02-29"
    assert health["feature_coverage"] == 1.0
    assert health["data_lag_days"] == 3
    assert health["raw_price_date"] == "2020-03-03"
    assert health["adjusted_price_date"] == "2020-03-03"
    assert health["raw_data_lag_days"] == 3
    assert health["adjusted_price_coverage_latest_raw_day"] == 1.0
    dumped = json.dumps(health)
    assert "TOKEN" not in dumped and "token" not in dumped.replace("latest", "")


def test_empty_database_degrades_to_none(engine) -> None:
    with session_scope(engine) as session:
        health = get_system_health(session, today="2020-03-06")
    assert health == {
        "latest_success": None,
        "latest_failure": None,
        "data_end_date": None,
        "feature_coverage": None,
        "raw_price_date": None,
        "adjusted_price_date": None,
        "raw_data_lag_days": None,
        "data_lag_days": None,
        "adjusted_price_coverage_latest_raw_day": None,
    }


def test_raw_update_does_not_hide_adjusted_price_lag(engine) -> None:
    _seed(engine)
    with session_scope(engine) as session:
        session.add(
            Price(
                trade_date="2020-03-05",
                stock_id="2330",
                open=102.0,
                high=103.0,
                low=101.0,
                close=102.0,
                volume=1000.0,
                traded_value=102000.0,
                source="test",
            )
        )
    with session_scope(engine) as session:
        health = get_system_health(session, today="2020-03-06")
    assert health["raw_price_date"] == "2020-03-05"
    assert health["adjusted_price_date"] == "2020-03-03"
    assert health["raw_data_lag_days"] == 1
    assert health["data_lag_days"] == 3
    assert health["adjusted_price_coverage_latest_raw_day"] == 0.0


def test_bad_today_rejected(engine) -> None:
    with session_scope(engine) as session:
        with pytest.raises(ValueError, match="today"):
            get_system_health(session, today="tomorrow")
