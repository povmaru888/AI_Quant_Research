"""P1-09 acceptance: research repository save and prediction load."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from database import create_engine_from_settings, session_scope
from models.research import Feature, PipelineRun
from repositories.research import (
    PREDICTION_COLUMNS,
    load_predictions,
    save_features,
    save_predictions,
)
from repositories.stocks import upsert_stocks
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


def _seeded_engine(settings: Settings, path: Path):
    _init_schema(path)
    engine = _engine_for(settings, path)
    with session_scope(engine) as session:
        upsert_stocks(
            session,
            pd.DataFrame(
                [
                    {"stock_id": "2330", "market": "TWSE"},
                    {"stock_id": "0050", "market": "TWSE"},
                    {"stock_id": "2317", "market": "TWSE"},
                ]
            ),
        )
        session.add(
            PipelineRun(
                run_id="run-001",
                run_time="2020-01-01",
                data_end_date="2019-12-31",
                feature_version="factor_v1",
                parameter_version="p1",
                status="started",
            )
        )
    return engine


def _feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rebalance_date": "2020-01-31",
                "stock_id": "2330",
                "feature_version": "factor_v1",
                "momentum_20d": 0.05,
                "volatility_60d": 0.2,
                "missing_flag": 0,
            },
            {
                "rebalance_date": "2020-01-31",
                "stock_id": "0050",
                "feature_version": "factor_v1",
                "momentum_20d": -0.02,
                "volatility_60d": 0.15,
                "missing_flag": 0,
            },
        ]
    )


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_date": "2020-01-31",
                "stock_id": "2330",
                "run_id": "run-001",
                "model_version": "xgb_v1",
                "prediction_probability": 0.8,
                "rank": 1,
            },
            {
                "prediction_date": "2020-01-31",
                "stock_id": "0050",
                "run_id": "run-001",
                "model_version": "xgb_v1",
                "prediction_probability": 0.3,
                "rank": 2,
            },
        ]
    )


def _table_count(engine, table: str) -> int:
    with session_scope(engine) as session:
        return session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def test_save_features_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = _feature_rows()
        with session_scope(engine) as session:
            assert save_features(session, rows) == 2
        with session_scope(engine) as session:
            assert save_features(session, rows) == 2
        assert _table_count(engine, "features") == 2

        changed = rows.copy()
        changed.loc[0, "momentum_20d"] = 0.09
        with session_scope(engine) as session:
            save_features(session, changed)
        assert _table_count(engine, "features") == 2
        with session_scope(engine) as session:
            value = session.execute(
                select(Feature.momentum_20d).where(
                    Feature.rebalance_date == "2020-01-31",
                    Feature.stock_id == "2330",
                )
            ).scalar_one()
        assert value == 0.09
    finally:
        engine.dispose()


def test_save_features_missing_flag_default(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = pd.DataFrame(
            [
                {
                    "rebalance_date": "2020-01-31",
                    "stock_id": "2317",
                    "feature_version": "factor_v1",
                    "momentum_20d": 0.01,
                }
            ]
        )
        with session_scope(engine) as session:
            assert save_features(session, rows) == 1
        with session_scope(engine) as session:
            flag = session.execute(
                select(Feature.missing_flag).where(Feature.stock_id == "2317")
            ).scalar_one()
        assert flag == 0
    finally:
        engine.dispose()


def test_save_predictions_idempotent(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = _prediction_rows()
        with session_scope(engine) as session:
            assert save_predictions(session, rows) == 2
        with session_scope(engine) as session:
            assert save_predictions(session, rows) == 2
        assert _table_count(engine, "predictions") == 2

        changed = rows.copy()
        changed.loc[1, "prediction_probability"] = 0.95
        changed.loc[1, "rank"] = 1
        changed.loc[0, "rank"] = 2
        with session_scope(engine) as session:
            save_predictions(session, changed)
        assert _table_count(engine, "predictions") == 2
    finally:
        engine.dispose()


def test_load_predictions_rank_order(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        rows = pd.DataFrame(
            [
                {
                    "prediction_date": "2020-01-31",
                    "stock_id": "0050",
                    "run_id": "run-001",
                    "model_version": "xgb_v1",
                    "prediction_probability": 0.3,
                    "rank": 3,
                },
                {
                    "prediction_date": "2020-01-31",
                    "stock_id": "2317",
                    "run_id": "run-001",
                    "model_version": "xgb_v1",
                    "prediction_probability": 0.5,
                    "rank": 2,
                },
                {
                    "prediction_date": "2020-01-31",
                    "stock_id": "2330",
                    "run_id": "run-001",
                    "model_version": "xgb_v1",
                    "prediction_probability": 0.8,
                    "rank": 1,
                },
            ]
        )
        with session_scope(engine) as session:
            save_predictions(session, rows)
        with session_scope(engine) as session:
            loaded = load_predictions(session, date(2020, 1, 31), "xgb_v1")
        assert list(loaded.columns) == list(PREDICTION_COLUMNS)
        assert list(loaded["stock_id"]) == ["2330", "2317", "0050"]
        assert list(loaded["rank"]) == [1, 2, 3]
        with session_scope(engine) as session:
            missing = load_predictions(session, date(2020, 2, 29), "xgb_v1")
            assert missing.empty
            assert list(missing.columns) == list(PREDICTION_COLUMNS)
            assert load_predictions(session, date(2020, 1, 31), "other").empty
        with pytest.raises(ValueError, match="non-empty model_version"):
            with session_scope(engine) as session:
                load_predictions(session, date(2020, 1, 31), "  ")
    finally:
        engine.dispose()


def test_validation_and_fk(settings: Settings, temp_db_path: Path) -> None:
    engine = _seeded_engine(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="requires columns"):
                save_features(session, pd.DataFrame([{"stock_id": "2330"}]))
        with session_scope(engine) as session:
            with pytest.raises(ValueError, match="unknown Prediction columns"):
                save_predictions(
                    session,
                    pd.DataFrame(
                        [
                            {
                                "prediction_date": "2020-01-31",
                                "stock_id": "2330",
                                "run_id": "run-001",
                                "model_version": "xgb_v1",
                                "nope": 1,
                            }
                        ]
                    ),
                )
        with session_scope(engine) as session:
            empty = pd.DataFrame(
                {
                    "rebalance_date": pd.Series(dtype=str),
                    "stock_id": pd.Series(dtype=str),
                    "feature_version": pd.Series(dtype=str),
                }
            )
            assert save_features(session, empty) == 0
        orphan = pd.DataFrame(
            [
                {
                    "prediction_date": "2020-01-31",
                    "stock_id": "2330",
                    "run_id": "run-999",
                    "model_version": "xgb_v1",
                    "prediction_probability": 0.8,
                    "rank": 1,
                }
            ]
        )
        with session_scope(engine) as session:
            with pytest.raises(IntegrityError):
                save_predictions(session, orphan)
    finally:
        engine.dispose()
