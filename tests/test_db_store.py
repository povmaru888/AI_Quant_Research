"""Runtime DbStore acceptance: protocols against a temp database, no network."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from contracts import PortfolioTarget
from database import create_engine_from_settings
from runtime.db_store import (
    INITIAL_CAPITAL,
    DbStore,
    build_store,
    default_shares_path,
    load_shares_cache,
)
from settings import Settings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)

AS_OF = "2020-02-05"


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_dbstore", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


@pytest.fixture()
def store(settings: Settings, temp_db_path: Path) -> Iterator[DbStore]:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    shares = {
        "2330": {"shares": 25_000_000_000.0, "market_cap": None, "as_of": "2026-09-22"},
        "0050": {"shares": None, "market_cap": 500_000_000_000.0, "as_of": "2026-09-22"},
    }
    yield DbStore(engine, db_settings, shares_outstanding=shares)
    engine.dispose()


def _seed_market(store: DbStore, sample_prices: pd.DataFrame) -> None:
    from repositories import stocks as stocks_repo

    with store._scope() as session:  # noqa: SLF001 - test-only seeding hook.
        stocks_repo.upsert_stocks(
            session,
            pd.DataFrame(
                {
                    "stock_id": ["2330", "0050"],
                    "stock_name": ["t1", "t2"],
                    "market": ["TWSE", "TWSE"],
                    "listed_date": ["2010-01-01", "2010-01-01"],
                }
            ),
        )
    store.upsert_prices(sample_prices)
    store.upsert_institutional(
        pd.DataFrame(
            {
                "stock_id": ["2330", "0050"],
                "trade_date": [AS_OF, AS_OF],
                "float_shares": [25_000_000_000.0, 5_000_000_000.0],
                "source": ["test", "test"],
            }
        )
    )


def test_start_run_defaults_and_status(store: DbStore, settings: Settings) -> None:
    store.start_run({"run_id": "daily-2020-02-05", "job": "daily_update", "data_end_date": AS_OF})
    assert store._run_id == "daily-2020-02-05"  # noqa: SLF001
    assert store._as_of == AS_OF  # noqa: SLF001
    summary = store.load_run_summary("daily-2020-02-05")
    assert summary["feature_version"] == settings.features.feature_version
    assert summary["parameter_version"] == "job:daily_update"
    assert summary["model_version"] == "xgb_202002"
    store.finish_run("daily-2020-02-05", "succeeded")
    assert store.get_run_status("daily-2020-02-05") == "succeeded"
    assert store.list_runs(status="succeeded") == [
        {"run_id": "daily-2020-02-05", "status": "succeeded"}
    ]
    with pytest.raises(ValueError, match="unknown run"):
        store.get_run_status("nope")


def test_upsert_prices_creates_missing_parents(store: DbStore) -> None:
    frame = pd.DataFrame(
        {
            "stock_id": ["9999"],
            "trade_date": [AS_OF],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.0],
            "volume": [1000.0],
            "traded_value": [10000.0],
            "source": ["test"],
        }
    )
    assert store.upsert_prices(frame) == 1
    assert store.upsert_prices(frame) == 1  # idempotent rerun.
    assert store.upsert_prices(frame.iloc[0:0]) == 0
    assert "9999" in store.load_symbols()


def test_symbols_and_latest_trade_date(store: DbStore, sample_prices: pd.DataFrame) -> None:
    assert store.load_latest_trade_date() is None
    _seed_market(store, sample_prices)
    assert store.load_latest_trade_date() == AS_OF
    assert store.load_symbols() == ["0050", "2330"]


def test_research_loads_and_saves(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run(
        {
            "run_id": "rebalance-2020-02-05",
            "data_end_date": AS_OF,
            "feature_version": "factor_v1",
            "parameter_version": "p1",
        }
    )
    stocks = store.load_stocks()
    assert set(stocks["stock_id"]) == {"2330", "0050"}
    assert (stocks["market_cap"] > 0).all()
    assert (stocks["flags"] == "").all()
    history = store.load_institutional_history()
    assert set(history.columns) >= {"stock_id", "trade_date", "margin_balance"}
    snapshot = store.load_institutional_snapshot()
    assert len(snapshot) == 2
    returns = store.load_returns()
    assert set(returns.columns) == {"stock_id", "trade_date", "log_return"}
    assert not returns.empty

    preds = pd.DataFrame(
        {
            "stock_id": ["2330", "0050"],
            "prediction_probability": [0.7, 0.4],
            "rank": [1, 2],
            "prediction_date": [AS_OF, AS_OF],
        }
    )
    assert store.save_predictions(preds, "xgb_202002") == 2
    target = PortfolioTarget(
        run_id="rebalance-2020-02-05",
        signal_date=AS_OF,
        top_n=15,
        actions={"2330": "BUY", "0050": "NONE"},
        weights={"2330": 0.1},
        cash_weight=0.9,
        equity_exposure=0.1,
    )
    store.save_target_holdings(target)
    assert store.load_previous_positions()["stock_id"].tolist() == ["2330"]


def _execution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "r|2020-02-06|2330|BUY",
                "run_id": "r",
                "signal_date": "2020-02-05",
                "execution_date": "2020-02-06",
                "stock_id": "2330",
                "side": "BUY",
                "target_weight": 0.1,
                "target_shares": 100,
                "executed_price": 500.5,
                "broker_fee": 71.32125,
                "transaction_tax": 0.0,
                "slippage_cost": 50.05,
                "total_cost": 121.37125,
            },
            {
                "order_id": "r|2020-02-06|0050|SELL",
                "run_id": "r",
                "signal_date": "2020-02-05",
                "execution_date": "2020-02-06",
                "stock_id": "0050",
                "side": "SELL",
                "target_weight": 0.0,
                "target_shares": 0,
                "executed_price": 139.86,
                "broker_fee": 9.965025,
                "transaction_tax": 20.979,
                "slippage_cost": 6.993,
                "total_cost": 37.937025,
            },
        ]
    )


def test_save_orders_derives_exact_quantities(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run({"run_id": "r", "job": "test", "data_end_date": AS_OF})
    assert store.save_orders(_execution_frame()) == 2
    with store._scope() as session:  # noqa: SLF001
        from sqlalchemy import select

        from models.research import Order

        rows = session.execute(
            select(
                Order.stock_id,
                Order.quantity,
                Order.open_price,
                Order.executed_price,
                Order.notional,
            ).order_by(Order.stock_id)
        ).all()
    by_stock = {s: (q, o, x, n) for s, q, o, x, n in rows}
    assert by_stock["2330"][0] == 100
    assert by_stock["0050"][0] == 50
    assert by_stock["2330"][1] == pytest.approx(500.0)
    assert by_stock["0050"][1] == pytest.approx(140.0)
    assert by_stock["2330"][3] == pytest.approx(100 * 500.5)
    assert store.save_orders(_execution_frame().iloc[0:0]) == 0


def test_verify_month_end_and_replace(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    assert store.verify_month_end(date(2020, 1, 31)) is True
    assert store.verify_month_end(date(2020, 1, 30)) is False
    store.start_run({"run_id": "r", "job": "test", "data_end_date": AS_OF})
    frame = _execution_frame()
    assert store.replace_orders("r", "2020-02-05", frame) == 2
    shrunk = frame.iloc[[0]].copy()
    assert store.replace_orders("r", "2020-02-05", shrunk) == 1
    with store._scope() as session:  # noqa: SLF001
        from sqlalchemy import func, select

        from models.research import Order

        count = session.execute(select(func.count()).where(Order.run_id == "r")).scalar()
    assert count == 1


def test_dashboard_reads(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run(
        {
            "run_id": "rebalance-2020-02-05",
            "data_end_date": AS_OF,
            "feature_version": "factor_v1",
            "parameter_version": "p1",
        }
    )
    store.finish_run("rebalance-2020-02-05", "succeeded")
    summary = store.load_run_summary("rebalance-2020-02-05")
    assert summary["metrics"] == {}
    assert summary["oos_months"] == []
    holdings = store.load_holdings("rebalance-2020-02-05", AS_OF)
    assert list(holdings.columns) == [
        "stock_id",
        "rank",
        "prediction_probability",
        "weight",
        "volatility_60d",
        "beta_60d",
    ]
    risk = store.load_risk("rebalance-2020-02-05")
    assert set(risk) == {
        "equity_exposure",
        "predicted_volatility",
        "realized_volatility",
        "max_drawdown",
        "turnover",
        "market_regime",
        "exposure_cap",
    }
    assert risk["exposure_cap"] == 1.0
    comparison = store.load_comparison("rebalance-2020-02-05")
    assert list(comparison.columns) == ["scenario"]
    factor_ic = store.load_factor_ic("rebalance-2020-02-05")
    assert list(factor_ic.columns) == ["factor", "ic"]
    model_data = store.load_model_data("rebalance-2020-02-05")
    assert set(model_data) == {"shap_top", "feature_importance", "monthly_ic", "prediction_dist"}


def test_load_performance_replays_orders(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run({"run_id": "r", "job": "test", "data_end_date": "2020-02-04"})
    frame = _execution_frame()
    frame["signal_date"] = "2020-02-04"
    frame["execution_date"] = AS_OF
    assert store.save_orders(frame) == 2
    performance = store.load_performance("r")
    assert list(performance.columns) == ["date", "nav"]
    assert not performance.empty
    assert performance["nav"].iloc[0] == pytest.approx(INITIAL_CAPITAL)


def test_guards_without_run_or_asof(store: DbStore) -> None:
    with pytest.raises(ValueError, match="no run"):
        store.save_predictions(pd.DataFrame(), "xgb_202002")
    with pytest.raises(ValueError, match="no as_of"):
        store.load_stocks()
    with pytest.raises(ValueError, match="initial_capital"):
        DbStore(
            store._engine,
            store._settings,  # noqa: SLF001
            initial_capital=0.0,
        )


def test_market_cap_shares_and_etf_fallback(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run({"run_id": "r", "job": "test", "data_end_date": AS_OF})
    stocks = store.load_stocks().set_index("stock_id")
    assert stocks.loc["2330", "market_cap"] == pytest.approx(524.0 * 25_000_000_000.0)
    assert stocks.loc["0050", "market_cap"] == pytest.approx(500_000_000_000.0)


def test_market_cap_missing_without_cache(
    settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame
) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    engine = create_engine_from_settings(db_settings)
    try:
        bare = DbStore(engine, db_settings)
        _seed_market(bare, sample_prices)
        bare.start_run({"run_id": "r", "job": "test", "data_end_date": AS_OF})
        caps = bare.load_stocks()["market_cap"]
        assert caps.isna().all()
    finally:
        engine.dispose()


def test_load_shares_cache_roundtrip(tmp_path: Path) -> None:
    assert load_shares_cache(None) == {}
    assert load_shares_cache(tmp_path / "missing.json") == {}
    path = tmp_path / "shares.json"
    path.write_text(json.dumps({"2330": {"shares": 1.0}}), encoding="utf-8")
    assert load_shares_cache(path) == {"2330": {"shares": 1.0}}
    assert default_shares_path("sqlite:///database/quant.db") == Path("database/shares.json")
    assert default_shares_path("sqlite:///:memory:") is None


def test_build_store_factory(settings: Settings, temp_db_path: Path) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
    finally:
        conn.close()
    db_settings = dataclasses.replace(
        settings, data=dataclasses.replace(settings.data, database_url=f"sqlite:///{temp_db_path}")
    )
    factory_store = build_store(db_settings)
    try:
        assert isinstance(factory_store, DbStore)
        assert factory_store.load_portfolio_value() == INITIAL_CAPITAL
    finally:
        factory_store._engine.dispose()  # noqa: SLF001
