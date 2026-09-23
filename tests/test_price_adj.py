"""PriceAdj acceptance: fetch mapping, 003 migration, update-only upsert."""

from __future__ import annotations

import dataclasses
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from database import create_engine_from_settings, session_scope
from integrations.finmind_prices import ADJ_COLUMNS, fetch_price_adj
from models.market import Price
from repositories import prices as prices_repo
from runtime.db_store import DbStore
from settings import Settings

MIGRATION_001 = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)
MIGRATION_003 = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "003_price_adj.py"
)

TOKEN = "secret-token-xyz"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration_003 = _load(MIGRATION_003, "price_adj_003")


class _StubResponse:
    def __init__(self, payload: object) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _ok(data: list[dict]) -> _StubResponse:
    return _StubResponse({"status": 200, "msg": "success", "data": data})


def test_003_upgrade_downgrade(temp_db_path: Path) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        _load(MIGRATION_001, "initial_schema_adj").upgrade(conn)
        migration_003.upgrade(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(prices)").fetchall()}
        assert {"open_adj", "high_adj", "low_adj", "close_adj"} <= cols
        migration_003.upgrade(conn)  # idempotent.
        migration_003.downgrade(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(prices)").fetchall()}
        assert not ({"open_adj", "high_adj", "low_adj", "close_adj"} & cols)
    finally:
        conn.close()


def _adj_row(**overrides) -> dict:
    row = {
        "stock_id": "2330",
        "date": "2020-04-06",
        "Trading_Volume": 1000,
        "Trading_money": 250000,
        "open": "242.06",
        "max": "244.28",
        "min": "239.40",
        "close": "244.28",
        "spread": 3.5,
    }
    row.update(overrides)
    return row


def _requester(data: list[dict], seen: dict):
    def requester(url, params=None, headers=None, timeout=None, **kwargs):
        seen.update(params)
        return _ok(data)

    return requester


def test_fetch_price_adj_mapping_and_data_id() -> None:
    seen: dict = {}
    frame = fetch_price_adj(
        "2020-04-01",
        "2020-04-12",
        TOKEN,
        requester=_requester([_adj_row()], seen),
        stock_id="2330",
    )
    assert list(frame.columns) == list(ADJ_COLUMNS)
    assert seen["dataset"] == "TaiwanStockPriceAdj"
    assert seen["data_id"] == "2330"
    row = frame.iloc[0].to_dict()
    assert row["trade_date"] == "2020-04-06"
    assert row["close_adj"] == 244.28
    assert row["open_adj"] == 242.06


def test_fetch_price_adj_drops_bad_rows() -> None:
    seen: dict = {}
    rows = [
        _adj_row(),
        _adj_row(stock_id="zero", close="0"),
        _adj_row(stock_id="inverted", max="10", min="20"),
    ]
    frame = fetch_price_adj("2020-04-01", "2020-04-12", TOKEN, requester=_requester(rows, seen))
    assert frame["stock_id"].tolist() == ["2330"]
    assert "data_id" not in seen


def _db(settings: Settings, temp_db_path: Path):
    conn = sqlite3.connect(str(temp_db_path))
    try:
        _load(MIGRATION_001, "initial_schema_adj2").upgrade(conn)
        migration_003.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    return engine, db_settings


def test_upsert_price_adj_fills_and_skips(settings: Settings, temp_db_path: Path) -> None:
    from repositories import stocks as stocks_repo

    engine, _ = _db(settings, temp_db_path)
    try:
        with session_scope(engine) as session:
            stocks_repo.upsert_stocks(
                session, pd.DataFrame({"stock_id": ["2330"], "market": ["TWSE"]})
            )
            prices_repo.upsert_prices(
                session,
                pd.DataFrame(
                    {
                        "stock_id": ["2330"],
                        "trade_date": ["2020-04-06"],
                        "open": [273.0],
                        "high": [275.5],
                        "low": [270.0],
                        "close": [275.5],
                        "volume": [1000.0],
                        "traded_value": [275500.0],
                        "source": ["test"],
                    }
                ),
            )
            adj = pd.DataFrame(
                {
                    "stock_id": ["2330", "2330", "9999"],
                    "trade_date": ["2020-04-06", "2020-04-07", "2020-04-06"],
                    "open_adj": [242.06, 250.0, 1.0],
                    "high_adj": [244.28, 252.0, 2.0],
                    "low_adj": [239.40, 248.0, 0.5],
                    "close_adj": [244.28, 251.0, 1.5],
                }
            )
            assert prices_repo.upsert_price_adj(session, adj) == 1
            assert prices_repo.upsert_price_adj(session, adj.iloc[0:0]) == 0
            got = session.execute(select(Price.close_adj).where(Price.stock_id == "2330")).scalar()
            assert got == 244.28
    finally:
        engine.dispose()


def test_store_upsert_price_adj(settings: Settings, temp_db_path: Path) -> None:
    engine, db_settings = _db(settings, temp_db_path)
    try:
        store = DbStore(engine, db_settings)
        assert store.upsert_price_adj(pd.DataFrame()) == 0
    finally:
        engine.dispose()
