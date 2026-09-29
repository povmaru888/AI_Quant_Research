"""PIT market-value ETL and checkpoint acceptance tests."""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from integrations.finmind_market_value import FinMindMarketValueError, fetch_market_value_day
from models.market import MarketValue, MarketValueSyncDay
from models.security import Stock  # noqa: F401 - register the FK target table.
from repositories.market_values import (
    canonical_day_hash,
    load_completed_days,
    load_market_value_snapshot,
    mark_sync_day,
    upsert_market_value_day,
)


def _load_migration(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _migrate(conn: sqlite3.Connection) -> None:
    migration_dir = Path(__file__).resolve().parents[1] / "database/migrations/versions"
    for path in sorted(migration_dir.glob("[0-9]*.py")):
        _load_migration(path).upgrade(conn)


def test_non_trading_migration_upgrades_existing_checkpoint_schema() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "database/migrations/versions/007_non_trading_checkpoints.py"
    )
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE source_revision (
                singleton INTEGER PRIMARY KEY CHECK (singleton=1),
                revision INTEGER NOT NULL
            );
            INSERT INTO source_revision(singleton, revision) VALUES (1, 0);
            CREATE TABLE market_value_sync_days (
                trade_date TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
                row_count INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TRIGGER trg_source_revision_market_value_sync_days_insert
                AFTER INSERT ON market_value_sync_days BEGIN
                    UPDATE source_revision SET revision=revision+1 WHERE singleton=1;
                END;
            INSERT INTO market_value_sync_days(trade_date, status) VALUES ('2024-12-31', 'failed');
            """
        )
        migration = _load_migration(migration_path)
        migration.upgrade(conn)
        conn.execute(
            "INSERT INTO market_value_sync_days(trade_date, status) VALUES (?, ?)",
            ("2024-12-01", "non_trading"),
        )
        assert conn.execute(
            "SELECT status FROM market_value_sync_days WHERE trade_date='2024-12-31'"
        ).fetchone() == ("failed",)
        assert conn.execute(
            "SELECT status FROM market_value_sync_days WHERE trade_date='2024-12-01'"
        ).fetchone() == ("non_trading",)
    finally:
        conn.close()


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self.payload


def test_finmind_fetch_excludes_zero_market_value_and_hashes_raw_rows() -> None:
    raw = [
        {"date": "2024-12-31", "stock_id": "2330", "market_value": 1234},
        {"date": "2024-12-31", "stock_id": "00625K", "market_value": 0},
    ]
    calls: list[dict] = []

    def requester(url: str, **kwargs) -> _Response:
        calls.append({"url": url, **kwargs})
        return _Response({"status": 200, "data": raw})

    frame = fetch_market_value_day("2024-12-31", "test-token", requester=requester)
    assert frame["stock_id"].tolist() == ["2330"]
    assert frame.attrs["zero_market_value_rows"] == 1
    assert frame.attrs["source_row_count"] == 2
    expected_raw = pd.DataFrame(
        [
            {"trade_date": "2024-12-31", "stock_id": "2330", "market_value": 1234},
            {"trade_date": "2024-12-31", "stock_id": "00625K", "market_value": 0},
        ]
    )
    assert frame.attrs["source_content_hash"] == canonical_day_hash(expected_raw)
    assert calls[0]["params"] == {
        "dataset": "TaiwanStockMarketValue",
        "start_date": "2024-12-31",
    }
    assert calls[0]["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.parametrize("value", [-1, None, "not-a-number"])
def test_finmind_rejects_negative_or_missing_market_values(value) -> None:
    response = _Response(
        {"status": 200, "data": [{"date": "2024-12-31", "stock_id": "2330", "market_value": value}]}
    )
    with pytest.raises(FinMindMarketValueError, match="missing/negative"):
        fetch_market_value_day("2024-12-31", "test-token", requester=lambda *_a, **_k: response)


def test_market_value_daily_upsert_replaces_corrected_snapshot_atomically(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn)
    conn.execute("INSERT INTO stocks (stock_id, market) VALUES ('2330', 'TWSE')")
    conn.execute("INSERT INTO stocks (stock_id, market) VALUES ('2317', 'TWSE')")
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db_path}")
    day = date(2024, 12, 31)
    original = pd.DataFrame(
        [
            {"trade_date": day.isoformat(), "stock_id": "2330", "market_value": 100.0},
            {"trade_date": day.isoformat(), "stock_id": "2317", "market_value": 200.0},
        ]
    )
    corrected = original.iloc[[0]].assign(market_value=110.0)
    try:
        with Session(engine) as session:
            assert upsert_market_value_day(session, day, original) == 2
            session.commit()
        with Session(engine) as session:
            assert upsert_market_value_day(
                session, day, original, source_content_hash=canonical_day_hash(original)
            ) == 2
            session.commit()
            rows = session.execute(
                select(MarketValue.stock_id, MarketValue.market_value).where(
                    MarketValue.trade_date == day.isoformat()
                ).order_by(MarketValue.stock_id)
            ).all()
            assert rows == [("2317", 200.0), ("2330", 100.0)]
        corrected_hash = "a" * 64
        with Session(engine) as session:
            assert upsert_market_value_day(
                session, day, corrected, source_content_hash=corrected_hash
            ) == 1
            session.commit()
            rows = session.execute(
                select(MarketValue.stock_id, MarketValue.market_value).where(
                    MarketValue.trade_date == day.isoformat()
                )
            ).all()
            assert rows == [("2330", 110.0)]
            marker = session.get(MarketValueSyncDay, day.isoformat())
            assert marker is not None
            assert marker.status == "succeeded"
            assert marker.row_count == 1
            assert marker.content_hash == corrected_hash
            mark_sync_day(session, day.isoformat(), "failed")
            session.commit()
            assert load_market_value_snapshot(session, day).empty
            mark_sync_day(session, day.isoformat(), "non_trading")
            session.commit()
            assert load_market_value_snapshot(session, day).empty
            assert load_completed_days(session, "2024-12-01", "2024-12-31") == {
                day.isoformat()
            }
            non_trading_marker = session.get(MarketValueSyncDay, day.isoformat())
            assert non_trading_marker is not None
            assert non_trading_marker.status == "non_trading"
            assert non_trading_marker.row_count == 0
    finally:
        engine.dispose()


def test_market_value_rows_do_not_trigger_one_revision_write_per_security(temp_db_conn) -> None:
    migrations = Path(__file__).resolve().parents[1] / "database/migrations/versions"
    for path in sorted(migrations.glob("[0-9]*.py")):
        _load_migration(path).upgrade(temp_db_conn)
    temp_db_conn.executemany(
        "INSERT INTO stocks (stock_id, market) VALUES (?, 'TWSE')",
        [(f"S{index}",) for index in range(10)],
    )
    before = temp_db_conn.execute(
        "SELECT revision FROM source_revision WHERE singleton=1"
    ).fetchone()[0]
    temp_db_conn.executemany(
        "INSERT INTO market_values (trade_date, stock_id, market_value, source) VALUES (?, ?, ?, ?)",
        [("2024-12-31", f"S{index}", 100.0 + index, "test") for index in range(10)],
    )
    after_market_rows = temp_db_conn.execute(
        "SELECT revision FROM source_revision WHERE singleton=1"
    ).fetchone()[0]
    assert after_market_rows == before
    temp_db_conn.execute(
        "INSERT INTO market_value_sync_days (trade_date, status, row_count) VALUES (?, ?, ?)",
        ("2024-12-31", "succeeded", 10),
    )
    after_checkpoint = temp_db_conn.execute(
        "SELECT revision FROM source_revision WHERE singleton=1"
    ).fetchone()[0]
    assert after_checkpoint == after_market_rows + 1
