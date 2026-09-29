"""P5-08 acceptance: release checklist coverage and MVP fixture walkthrough."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

from database import create_engine_from_settings, session_scope
from models.market import Price
from models.research import Feature
from tests.migration_utils import apply_pit_v3_feature_migration
from models.security import Stock
from observability.data_quality import audit_data_quality
from observability.health import get_system_health
from report import export_reports
from repositories.runs import finish_run, start_run
from settings import Settings

CHECKLIST_PATH = Path(__file__).resolve().parents[1] / "docs" / "release-checklist.md"
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

REQUIRED_SECTIONS = ("資料", "研究", "回測", "Dashboard", "安全", "文件")
REQUIRED_TOPICS = (
    "Point-in-Time",
    "Survivorship",
    "OOS",
    "敏感度",
    "次日開盤",
    "Top 15",
    "succeeded",
    "SHAP",
    "secret",
    "Runbook",
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_p508", MIGRATION_PATH)
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


def test_checklist_covers_sdd_gates() -> None:
    assert CHECKLIST_PATH.is_file(), f"missing {CHECKLIST_PATH}"
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"checklist missing section: {section}"
    for topic in REQUIRED_TOPICS:
        assert topic in text, f"checklist missing topic: {topic}"
    assert "- [ ]" in text, "checklist items must be checkable"
    assert "簽核" in text, "checklist needs a sign-off line"


class _ReportStore:
    """Minimal MVP store over one successful run."""

    def __init__(self, summary: dict, performance, factor_ic) -> None:
        self._summary = summary
        self._performance = performance
        self._factor_ic = factor_ic

    def get_run_status(self, run_id: str) -> str:
        assert run_id == self._summary["run_id"]
        return "succeeded"

    def load_run_summary(self, run_id: str) -> dict:
        assert run_id == self._summary["run_id"]
        return self._summary

    def load_performance(self, run_id: str):
        return self._performance

    def load_factor_ic(self, run_id: str):
        return self._factor_ic


def test_mvp_fixture_completes_checklist(
    settings: Settings, temp_db_path: Path, tmp_path: Path
) -> None:
    """Walk every checklist evidence source on one MVP fixture run."""
    import pandas as pd

    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
        migration_003.upgrade(conn)
        apply_pit_v3_feature_migration(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    try:
        with session_scope(engine) as session:
            session.add(Stock(stock_id="2330", stock_name="t", market="TW"))
        with session_scope(engine) as session:
            session.add(
                Price(
                    trade_date="2020-02-29",
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
            start_run(
                session,
                {
                    "run_id": "mvp-2020-02",
                    "run_time": "2020-03-01T00:00:00+00:00",
                    "data_end_date": "2020-02-29",
                    "feature_version": "factor_v1",
                    "parameter_version": "params_abc",
                },
            )
            finish_run(session, "mvp-2020-02", "succeeded")
        with session_scope(engine) as session:
            # 1. 資料門檻：稽核通過。
            assert audit_data_quality("2020-02-29", session).passed is True
            # 5. 健康門檻：最近成功 run 可見。
            health = get_system_health(session, today="2020-03-01")
            assert health["latest_success"]["run_id"] == "mvp-2020-02"
            assert health["feature_coverage"] == 1.0
        # 2/3. 研究＋回測產物：四檔報表齊。
        store = _ReportStore(
            {
                "run_id": "mvp-2020-02",
                "data_end_date": "2020-02-29",
                "feature_version": "factor_v1",
                "model_version": "xgb_202002",
                "parameter_version": "params_abc",
                "metrics": {"cagr": 0.1},
                "oos_months": ["2020-02"],
            },
            pd.DataFrame({"date": ["2020-02-29"], "nav": [1.0]}),
            pd.DataFrame({"factor": ["momentum_20d"], "ic": [0.05]}),
        )
        paths = export_reports("mvp-2020-02", tmp_path / "release", store)
        for path in (
            paths.performance_csv,
            paths.factor_ic_csv,
            paths.run_summary_json,
            paths.report_pdf,
        ):
            assert path.is_file()
    finally:
        engine.dispose()
