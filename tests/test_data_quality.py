"""P5-03 acceptance: data quality audit on a real SQLite database."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from database import create_engine_from_settings, session_scope
from models.market import Financial, Institutional, Price
from models.research import Feature
from models.security import Stock
from observability.data_quality import (
    assert_no_blockers,
    audit_data_quality,
)
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
    spec = importlib.util.spec_from_file_location("initial_schema_p503", MIGRATION_PATH)
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

AS_OF = "2020-02-05"
DAYS = ["2020-02-03", "2020-02-04", "2020-02-05"]


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
    finally:
        conn.close()
    engine = _engine_for(settings, temp_db_path)
    try:
        yield engine
    finally:
        engine.dispose()


def _seed_parents(engine, stock_ids: tuple[str, ...] = ("2330", "2317")) -> None:
    """Commit parent rows first: same-transaction parent/child ORM inserts
    hit FK ordering trouble (repo-wide pattern seeds parents separately)."""
    with session_scope(engine) as session:
        session.add_all(
            [
                Stock(
                    stock_id=stock_id,
                    stock_name="t",
                    market="TW",
                    listed_date="2010-01-01",
                    delisted_date=None,
                    industry="semi",
                )
                for stock_id in stock_ids
            ]
        )


def _seed_dirty(engine) -> None:
    _seed_parents(engine)
    with session_scope(engine) as session:
        for day in DAYS:
            session.add(
                Price(
                    trade_date=day,
                    stock_id="2330",
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1000.0,
                    traded_value=100000.0,
                    source="test",
                )
            )
        # 2317 missing 2020-02-04 entirely; 2020-02-05 violates low > high.
        session.add(
            Price(
                trade_date="2020-02-03",
                stock_id="2317",
                open=50.0,
                high=51.0,
                low=49.0,
                close=50.0,
                volume=500.0,
                traded_value=25000.0,
                source="test",
            )
        )
        session.add(
            Price(
                trade_date="2020-02-05",
                stock_id="2317",
                open=50.0,
                high=51.0,
                low=52.0,
                close=50.0,
                volume=500.0,
                traded_value=25000.0,
                source="test",
            )
        )
        session.add(
            Institutional(
                trade_date="2020-02-03",
                stock_id="2330",
                foreign_net_buy=1.0,
                trust_net_buy=0.0,
                margin_balance=None,
                short_balance=None,
                float_shares=None,
                source="test",
            )
        )
        session.add(
            Financial(
                stock_id="2330",
                report_period="2019Q4",
                announcement_date="2020-01-15",
                available_date="2020-01-16",
                revenue=1.0,
                net_income=None,
                equity=None,
                assets=None,
                operating_income=None,
                operating_cash_flow=None,
                source="test",
            )
        )
        # NOTE: an available_date < announcement_date row cannot be seeded:
        # the migration CHECK blocks it at the SQLite level, so the
        # financial_pit_violation probe stays defense-in-depth (absence is
        # asserted below). The malformed-period warning path is exercised.
        session.add(
            Financial(
                stock_id="2317",
                report_period="FY2019",
                announcement_date="2020-01-15",
                available_date="2020-01-16",
                revenue=1.0,
                net_income=None,
                equity=None,
                assets=None,
                operating_income=None,
                operating_cash_flow=None,
                source="test",
            )
        )
        factor_names = [
            col.name
            for col in Feature.__table__.columns
            if col.name not in ("rebalance_date", "stock_id", "feature_version", "missing_flag")
        ]
        for stock_id in ("2330", "2317"):
            values = {
                "rebalance_date": "2020-01-31",
                "stock_id": stock_id,
                "feature_version": "factor_v1",
                "missing_flag": 1,
            }
            values.update({name: None for name in factor_names})
            session.add(Feature(**values))


def test_audit_reports_all_five_checks(engine) -> None:
    _seed_dirty(engine)
    with session_scope(engine) as session:
        report = audit_data_quality(AS_OF, session)
    assert report.as_of == AS_OF
    assert report.passed is False
    by_check = {}
    for issue in report.issues:
        by_check.setdefault(issue.check, []).append(issue)
    assert by_check["missing_prices"][0].severity == "warning"
    assert by_check["invalid_prices"][0].severity == "blocker"
    assert "2317" in by_check["invalid_prices"][0].detail
    assert by_check["adjusted_price_stale"][0].severity == "blocker"
    assert by_check["missing_adjusted_prices"][0].severity == "warning"
    # PIT time-travel rows are rejected by the DDL CHECK; the probe must
    # report absence rather than fire on seedable data.
    assert "financial_pit_violation" not in by_check
    assert by_check["financial_report_period"][0].severity == "warning"
    assert "FY2019" in by_check["financial_report_period"][0].detail
    assert by_check["feature_coverage"][0].severity == "warning"
    assert "duplicate_keys" not in by_check
    with pytest.raises(ValueError, match="blockers"):
        assert_no_blockers(report)


def test_clean_database_passes(engine) -> None:
    _seed_parents(engine, ("2330",))
    with session_scope(engine) as session:
        for day in DAYS:
            session.add(
                Price(
                    trade_date=day,
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
            "rebalance_date": "2020-01-31",
            "stock_id": "2330",
            "feature_version": "factor_v1",
            "missing_flag": 0,
        }
        values.update({name: 0.1 for name in factor_names})
        session.add(Feature(**values))
    with session_scope(engine) as session:
        report = audit_data_quality(AS_OF, session)
    assert report.passed is True
    assert report.issues == ()
    assert_no_blockers(report)


def test_adjusted_gap_warns_and_stale_latest_blocks(engine) -> None:
    _seed_parents(engine, ("2330",))
    with session_scope(engine) as session:
        for day in DAYS:
            session.add(
                Price(
                    trade_date=day,
                    stock_id="2330",
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    open_adj=100.0 if day == DAYS[0] else None,
                    high_adj=101.0 if day == DAYS[0] else None,
                    low_adj=99.0 if day == DAYS[0] else None,
                    close_adj=100.0 if day == DAYS[0] else None,
                    volume=1000.0,
                    traded_value=100000.0,
                    source="test",
                )
            )
    with session_scope(engine) as session:
        report = audit_data_quality(AS_OF, session)
    checks = {issue.check: issue for issue in report.issues}
    assert checks["adjusted_price_stale"].severity == "blocker"
    assert "2020-02-03" in checks["adjusted_price_stale"].detail
    assert checks["missing_adjusted_prices"].count == 2


def test_bad_as_of_rejected(engine) -> None:
    with session_scope(engine) as session:
        with pytest.raises(ValueError, match="as_of"):
            audit_data_quality("not-a-date", session)
        with pytest.raises(ValueError, match="as_of"):
            audit_data_quality(None, session)  # type: ignore[arg-type]
