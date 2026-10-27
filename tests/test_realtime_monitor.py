from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from realtime_monitor.engine import FundEngine, LiveQuote
from realtime_monitor.intraday import (
    current_session_point,
    intraday_payload,
    intraday_point,
    merge_intraday_payload,
)
from realtime_monitor.monitor_store import MonitorStore
from realtime_monitor.oos_bridge import load_fund_seed, load_order_history
from realtime_monitor.shioaji_feed import DatabaseQuoteFeed, TwseIndexQuoteFeed
from realtime_monitor.trading_session import TAIPEI, TradingCalendar


def _research_database(path: Path) -> None:
    payload = {
        "equity_curve": [
            {"date": "2026-01-02", "nav": 30_000_000.0},
            {"date": "2026-03-31", "nav": 30_100_000.0},
        ],
        "monthly_returns": [{"month": "2026-03", "return": 0.01}],
        "metrics": {"sharpe": 1.2, "max_drawdown": -0.04, "turnover": 3.0},
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, status TEXT);
            CREATE TABLE run_artifacts (run_id TEXT, kind TEXT, payload_json TEXT);
            CREATE TABLE stocks (stock_id TEXT PRIMARY KEY, stock_name TEXT);
            CREATE TABLE prices (stock_id TEXT, trade_date TEXT, close REAL);
            CREATE TABLE signals (
                run_id TEXT, signal_date TEXT, stock_id TEXT, signal TEXT,
                rank INTEGER, target_weight REAL
            );
            CREATE TABLE predictions (
                run_id TEXT, prediction_date TEXT, stock_id TEXT,
                prediction_probability REAL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY, run_id TEXT, signal_date TEXT,
                execution_date TEXT, stock_id TEXT, side TEXT, quantity INTEGER,
                open_price REAL, executed_price REAL, notional REAL,
                broker_fee REAL, transaction_tax REAL, slippage_cost REAL,
                total_cost REAL
            );
            """
        )
        connection.execute("INSERT INTO pipeline_runs VALUES ('run', 'succeeded')")
        connection.execute(
            "INSERT INTO run_artifacts VALUES ('run', 'metrics', ?)",
            (json.dumps(payload),),
        )
        connection.execute("INSERT INTO stocks VALUES ('2330', '台積電')")
        connection.execute(
            "INSERT INTO signals VALUES ('run', '2026-03-31', '2330', 'HOLD', 1, .25)"
        )
        connection.execute("INSERT INTO predictions VALUES ('run', '2026-03-31', '2330', .9)")
        connection.execute(
            "INSERT INTO orders VALUES "
            "(1, 'run', '2026-03-31', '2026-04-01', '2330', 'BUY', 10000, "
            "500, 500, 5000000, 7125, 0, 2500, 9625)"
        )
        connection.execute("INSERT INTO prices VALUES ('2330', '2026-03-31', 500)")
        for day in range(1, 61):
            connection.execute(
                "INSERT INTO prices VALUES ('TAIEX', ?, ?)",
                (f"2026-01-{day:02d}", 20_000 + day),
            )


def test_seed_bridge_and_live_valuation(tmp_path: Path) -> None:
    database = tmp_path / "research.db"
    _research_database(database)

    seed = load_fund_seed(database, "run")
    assert seed.anchor_nav == 30_100_000
    assert seed.anchor_cash == 25_100_000
    assert seed.holdings[0].shares == 10_000
    assert seed.taiex_ma60 == pytest.approx(sum(20_000 + i for i in range(1, 61)) / 60)
    assert load_order_history(database, "run")[0]["stock_name"] == "台積電"

    engine = FundEngine(seed)
    engine.update_quote(LiveQuote("2330", 510, "2026-04-02T09:01:00+08:00", "test", reference=500))
    snapshot = engine.snapshot()
    assert snapshot["nav"] == 30_200_000
    assert snapshot["pnl_since_anchor"] == 100_000
    assert snapshot["holdings"][0]["day_change"] == pytest.approx(0.02)

    engine.replace_seed(replace(seed, signal_date="2026-04-30"))
    assert engine.snapshot()["signal_date"] == "2026-04-30"


def test_quote_order_compares_mixed_timezones_by_instant(tmp_path: Path) -> None:
    database = tmp_path / "research.db"
    _research_database(database)
    engine = FundEngine(load_fund_seed(database, "run"))
    engine.update_quote(
        LiveQuote("TAIEX", 23_000, "2026-10-27T02:11:20+00:00", "snapshot")
    )
    engine.update_quote(
        LiveQuote("TAIEX", 23_100, "2026-10-27T10:12:00", "stream")
    )
    engine.update_quote(
        LiveQuote("TAIEX", 22_900, "2026-10-27T02:10:00+00:00", "late-snapshot")
    )
    market = engine.snapshot()["market"]
    assert market["price"] == 23_100
    assert market["source"] == "stream"


def test_database_feed_and_monitor_store(tmp_path: Path) -> None:
    database = tmp_path / "research.db"
    _research_database(database)
    quotes = []
    message = DatabaseQuoteFeed(database, quotes.append).start(["2330"])
    assert "stored closes" in message
    assert {quote.code for quote in quotes} == {"2330", "TAIEX"}

    seed = load_fund_seed(database, "run")
    snapshot = FundEngine(seed).snapshot()
    store = MonitorStore(tmp_path / "monitor.db")
    store.save_snapshot("run", snapshot)
    store.log_event("warning", "test", "message")
    assert store.recent_snapshots("run")[0]["nav"] == snapshot["nav"]


def test_twse_index_payload_maps_to_live_quote() -> None:
    quote = TwseIndexQuoteFeed.quote_from_payload(
        {
            "msgArray": [
                {
                    "d": "20261027",
                    "t": "11:02:00",
                    "z": "45777.22",
                    "y": "46041.43",
                    "o": "46211.20",
                    "h": "46585.45",
                    "l": "45676.67",
                    "m": "7280042",
                }
            ]
        }
    )
    assert quote.code == "TAIEX"
    assert quote.price == pytest.approx(45777.22)
    assert quote.reference == pytest.approx(46041.43)
    assert quote.timestamp == "2026-10-27T11:02:00+08:00"
    assert quote.source == "twse-mis"


def test_intraday_session_mapping_and_deduplication() -> None:
    friday_close = {
        "as_of": "2026-10-23T13:31:00+08:00",
        "quote_revision": "2026-10-23T13:31:00+08:00",
        "nav": 31_000_000.0,
        "market": {"price": 23_000.0, "timestamp": "2026-10-23T13:31:00+08:00"},
    }
    saturday_heartbeat = {
        **friday_close,
        "as_of": "2026-10-24T10:00:00+08:00",
        "quote_revision": "2026-10-24T10:00:00+08:00",
        "nav": 31_100_000.0,
    }

    assert intraday_point(friday_close)["time"] == "2026-10-23T13:30:00+08:00"
    assert intraday_point(saturday_heartbeat)["session_date"] == "2026-10-23"
    payload = intraday_payload([friday_close, saturday_heartbeat])
    assert payload["session_date"] == "2026-10-23"
    assert payload["points"] == [intraday_point(saturday_heartbeat)]

    overridden = intraday_point(saturday_heartbeat, session_date_override="2026-10-22")
    assert overridden["time"] == "2026-10-22T13:30:00+08:00"
    merged = merge_intraday_payload(
        {
            "session_date": "2026-10-22",
            "points": [
                {
                    "session_date": "2026-10-22",
                    "time": "2026-10-22T09:01:00+08:00",
                    "nav": 30_000_000.0,
                    "taiex": 22_900.0,
                    "quote_revision": "2026-10-22T09:01:00+08:00",
                }
            ],
        },
        saturday_heartbeat,
    )
    assert len(merged["points"]) == 1
    assert merged["points"][-1]["session_date"] == "2026-10-22"
    assert current_session_point(saturday_heartbeat, "2026-10-22") is None


def test_current_point_uses_market_timestamp_not_newer_stock_revision() -> None:
    snapshot = {
        "nav": 31_000_000.0,
        "as_of": "2026-10-27T10:30:00+08:00",
        "quote_revision": "2026-10-27T10:30:00+08:00",
        "market": {
            "price": 23_000.0,
            "timestamp": "2026-10-24T13:30:00+08:00",
        },
    }
    assert current_session_point(snapshot, "2026-10-27") is None


def test_trading_session_boundaries_and_holiday(tmp_path: Path) -> None:
    store = MonitorStore(tmp_path / "monitor.db")
    calendar = TradingCalendar(store)
    calendar._holidays[2026] = {"2026-10-26"}  # noqa: SLF001
    calendar._loaded.add(2026)  # noqa: SLF001

    from datetime import datetime

    state = calendar.state(datetime(2026, 10, 27, 8, 44, 59, tzinfo=TAIPEI))
    assert (state.phase, state.target_date) == ("previous_close", "2026-10-23")
    assert calendar.state(datetime(2026, 10, 27, 8, 45, tzinfo=TAIPEI)).phase == "clearing"
    assert calendar.state(datetime(2026, 10, 27, 9, 0, tzinfo=TAIPEI)).phase == "live"
    assert calendar.state(datetime(2026, 10, 27, 13, 30, tzinfo=TAIPEI)).phase == "closed"
    holiday = calendar.state(datetime(2026, 10, 26, 10, 0, tzinfo=TAIPEI))
    assert (holiday.phase, holiday.target_date) == ("non_trading", "2026-10-23")


def test_intraday_session_cache_round_trip(tmp_path: Path) -> None:
    store = MonitorStore(tmp_path / "monitor.db")
    payload = {"session_date": "2026-10-27", "points": [{"time": "09:00"}]}
    store.save_intraday_session("fund", payload)
    assert store.load_intraday_session("fund", "2026-10-27") == payload
