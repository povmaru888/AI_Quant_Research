"""P5-07 acceptance: runbook coverage, secret hygiene, fixture drill."""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import sqlite3
from pathlib import Path

import pytest

from database import create_engine_from_settings, session_scope
from models.market import Price
from models.security import Stock
from observability.data_quality import assert_no_blockers, audit_data_quality
from settings import Settings

RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "docs" / "runbook.md"
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

REQUIRED_SCENARIOS = ("資料失敗", "PIT 違規", "模型失敗", "協方差不足", "報表失敗")


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_p507", MIGRATION_PATH)
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


def test_runbook_covers_all_scenarios() -> None:
    assert RUNBOOK_PATH.is_file(), f"missing {RUNBOOK_PATH}"
    text = RUNBOOK_PATH.read_text(encoding="utf-8")
    for scenario in REQUIRED_SCENARIOS:
        assert scenario in text, f"runbook missing scenario: {scenario}"
    for keyword in ("觀察", "判斷", "重跑", "停止"):
        assert keyword in text, f"runbook missing step keyword: {keyword}"
    assert "daily_update" in text
    assert "monthly_rebalance" in text


def test_runbook_has_no_secrets() -> None:
    text = RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "secrets." not in text
    for name in ("FINMIND_TOKEN", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"):
        assert name not in text
    # No assignment-shaped suspected values (KEY=long-token).
    assert not re.search(r"=\s*[A-Za-z0-9_\-]{16,}", text)


def test_fixture_drill_follows_runbook(settings: Settings, temp_db_path: Path) -> None:
    """A failing fixture walks the runbook path: audit -> block -> stop."""
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
        migration_003.upgrade(conn)
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
            # Invalid bar: low > high -> blocker per runbook section 1/2.
            session.add(
                Price(
                    trade_date="2020-02-03",
                    stock_id="2330",
                    open=50.0,
                    high=51.0,
                    low=52.0,
                    close=50.0,
                    volume=500.0,
                    traded_value=25000.0,
                    source="test",
                )
            )
        with session_scope(engine) as session:
            report = audit_data_quality("2020-02-03", session)
        assert report.passed is False
        # Runbook says: blockers stop the signal job.
        with pytest.raises(ValueError, match="blockers"):
            assert_no_blockers(report)
    finally:
        engine.dispose()
