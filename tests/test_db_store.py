"""Runtime DbStore acceptance: protocols against a temp database, no network."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts import PortfolioTarget
from database import create_engine_from_settings
from repositories.market_values import upsert_partial_market_value_day
from runtime.db_store import (
    INITIAL_CAPITAL,
    DbStore,
    build_store,
    default_shares_path,
    load_shares_cache,
)
from settings import Settings
from tests.migration_utils import apply_pit_v3_feature_migration

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)
MIGRATION_003_PATH = MIGRATION_PATH.with_name("003_price_adj.py")

AS_OF = "2020-02-05"


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema_dbstore", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _load_adj_migration():
    spec = importlib.util.spec_from_file_location("price_adj_dbstore", MIGRATION_003_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adj_migration = _load_adj_migration()


@pytest.fixture()
def store(settings: Settings, temp_db_path: Path) -> Iterator[DbStore]:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
        adj_migration.upgrade(conn)
        apply_pit_v3_feature_migration(conn)
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
                    "stock_name": ["台積電", "元大台灣50"],
                    "market": ["TWSE", "TWSE"],
                    "listed_date": ["2010-01-01", "2010-01-01"],
                }
            ),
        )
    bars = sample_prices.copy()
    for column in ("open", "high", "low", "close"):
        bars[f"{column}_adj"] = bars[column]
    store.upsert_prices(bars)
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
    assert summary["model_version"] == f"xgb_202002_{settings.features.feature_version}"
    store.finish_run("daily-2020-02-05", "succeeded")
    assert store.get_run_status("daily-2020-02-05") == "succeeded"
    assert store.list_runs(status="succeeded") == [
        {
            "run_id": "daily-2020-02-05",
            "status": "succeeded",
            "parameter_version": "job:daily_update",
            "feature_version": settings.features.feature_version,
        }
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
    assert store.trade_days() == []
    _seed_market(store, sample_prices)
    assert store.load_latest_trade_date() == AS_OF
    assert store.load_symbols() == ["0050", "2330"]
    assert store.trade_days()[-1] == AS_OF
    assert store.trade_days("2020-02-01", "2020-02-05")[-1] == AS_OF


def test_adjusted_price_reads_and_daily_coverage(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    assert store.load_adjusted_coverage(AS_OF) == (2, 2)
    bar = sample_prices.loc[
        (sample_prices["stock_id"] == "2330") & (sample_prices["trade_date"] == AS_OF)
    ].iloc[0]
    adjusted = {
        "stock_id": "2330",
        "trade_date": AS_OF,
        **{f"{column}_adj": float(bar[column]) / 2 for column in ("open", "high", "low", "close")},
    }
    store.upsert_price_adj(pd.DataFrame([adjusted]))
    loaded = store.load_prices()
    changed = loaded.loc[(loaded["stock_id"] == "2330") & (loaded["trade_date"] == AS_OF)].iloc[0]
    assert changed["close"] == bar["close"]
    assert changed["close_adj"] == pytest.approx(float(bar["close"]) / 2)
    returns = store.load_returns(["2330"])
    previous = sample_prices.loc[
        (sample_prices["stock_id"] == "2330") & (sample_prices["trade_date"] < AS_OF), "close"
    ].iloc[-1]
    actual = returns.loc[returns["trade_date"] == AS_OF, "log_return"].iloc[0]
    assert actual == pytest.approx(np.log((float(bar["close"]) / 2) / float(previous)))

    store.upsert_price_adj(
        pd.DataFrame(
            [
                {
                    "stock_id": "0050",
                    "trade_date": AS_OF,
                    **{
                        f"{column}_adj": float("nan") for column in ("open", "high", "low", "close")
                    },
                }
            ]
        )
    )
    assert store.load_adjusted_coverage(date.fromisoformat(AS_OF)) == (2, 1)


def test_adjusted_returns_do_not_bridge_missing_day(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    days = sorted(sample_prices["trade_date"].unique())
    missing_day, following_day = days[-2:]
    store.upsert_price_adj(
        pd.DataFrame(
            [
                {
                    "stock_id": "0050",
                    "trade_date": missing_day,
                    **{
                        f"{column}_adj": float("nan") for column in ("open", "high", "low", "close")
                    },
                }
            ]
        )
    )
    returns = store.load_returns(["0050"])
    assert missing_day not in set(returns["trade_date"])
    assert following_day not in set(returns["trade_date"])


def test_research_loads_and_saves(
    store: DbStore, sample_prices: pd.DataFrame, settings: Settings
) -> None:
    _seed_market(store, sample_prices)
    store.start_run(
        {
            "run_id": "rebalance-2020-02-05",
            "data_end_date": AS_OF,
            "feature_version": settings.features.feature_version,
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
    assert set(store.load_returns(["2330"])["stock_id"]) == {"2330"}
    assert store.load_returns([]).empty

    preds = pd.DataFrame(
        {
            "stock_id": ["2330", "0050"],
            # Service-side name; the store maps it to prediction_probability.
            "probability": [0.7, 0.4],
            "rank": [1, 2],
            "prediction_date": [AS_OF, AS_OF],
        }
    )
    assert store.save_predictions(preds, f"xgb_202002_{settings.features.feature_version}") == 2
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
    holdings = store.load_holdings("rebalance-2020-02-05", AS_OF)
    assert holdings["stock_id"].tolist() == ["2330"]
    assert holdings["volatility_60d"].isna().all()  # features never persisted.
    assert holdings.set_index("stock_id").loc["2330", "weight"] == pytest.approx(0.1)


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
        "stock_name",
        "rank",
        "prediction_probability",
        "weight",
        "volatility_60d",
        "beta_60d",
    ]
    # No features table rows: holdings still resolve (vol/beta NaN).
    assert holdings["volatility_60d"].isna().all()
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


def test_load_holdings_uses_run_feature_version(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    from models.research import Feature, Signal

    _seed_market(store, sample_prices)
    run_id = "rebalance-versioned"
    store.start_run(
        {
            "run_id": run_id,
            "data_end_date": AS_OF,
            "feature_version": "factor_adj_v2",
            "parameter_version": "p1",
        }
    )
    with store._scope() as session:  # noqa: SLF001 - test-only seeding hook.
        session.add_all(
            [
                Signal(
                    signal_date=AS_OF,
                    stock_id=stock_id,
                    run_id=run_id,
                    signal="BUY",
                    rank=rank,
                    target_weight=0.1,
                )
                for rank, stock_id in enumerate(("2330", "0050"), start=1)
            ]
        )
        session.add_all(
            [
                Feature(
                    rebalance_date=AS_OF,
                    stock_id="2330",
                    feature_version="factor_v1",
                    volatility_60d=9.0,
                    beta_60d=9.0,
                ),
                Feature(
                    rebalance_date=AS_OF,
                    stock_id="0050",
                    feature_version="factor_v1",
                    volatility_60d=8.0,
                    beta_60d=8.0,
                ),
                Feature(
                    rebalance_date=AS_OF,
                    stock_id="2330",
                    feature_version="factor_adj_v2",
                    volatility_60d=0.2,
                    beta_60d=1.1,
                ),
            ]
        )

    holdings = store.load_holdings(run_id, AS_OF).set_index("stock_id")
    assert len(holdings) == 2
    assert holdings.loc["2330", "volatility_60d"] == pytest.approx(0.2)
    assert holdings.loc["2330", "beta_60d"] == pytest.approx(1.1)
    assert pd.isna(holdings.loc["0050", "volatility_60d"])
    assert pd.isna(holdings.loc["0050", "beta_60d"])


def test_load_holdings_scopes_signal_prediction_and_features_to_selected_month(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    from models.research import Feature, Prediction, Signal

    _seed_market(store, sample_prices)
    run_id = "oos-multimonth"
    store.start_run(
        {
            "run_id": run_id,
            "data_end_date": "2024-02-29",
            "feature_version": "factor_adj_v2",
            "parameter_version": "p1",
        }
    )
    summary = store.load_run_summary(run_id)
    model_version = summary["model_version"]
    with store._scope() as session:  # noqa: SLF001 - seed multi-month OOS rows.
        session.add_all(
            [
                Signal(
                    signal_date="2024-01-31",
                    stock_id="2330",
                    run_id=run_id,
                    signal="BUY",
                    rank=1,
                    target_weight=0.5,
                ),
                Signal(
                    signal_date="2024-01-31",
                    stock_id="0050",
                    run_id=run_id,
                    signal="NONE",
                    rank=2,
                    target_weight=0.0,
                ),
                Signal(
                    signal_date="2024-02-29",
                    stock_id="2330",
                    run_id=run_id,
                    signal="NONE",
                    rank=2,
                    target_weight=0.0,
                ),
                Signal(
                    signal_date="2024-02-29",
                    stock_id="0050",
                    run_id=run_id,
                    signal="BUY",
                    rank=1,
                    target_weight=0.25,
                ),
            ]
        )
        session.add_all(
            [
                Prediction(
                    prediction_date="2024-01-31",
                    stock_id="2330",
                    run_id=run_id,
                    model_version=model_version,
                    prediction_probability=0.8,
                    rank=1,
                ),
                Prediction(
                    prediction_date="2024-01-31",
                    stock_id="0050",
                    run_id=run_id,
                    model_version=model_version,
                    prediction_probability=0.4,
                    rank=2,
                ),
                Prediction(
                    prediction_date="2024-02-29",
                    stock_id="2330",
                    run_id=run_id,
                    model_version=model_version,
                    prediction_probability=0.5,
                    rank=2,
                ),
                Prediction(
                    prediction_date="2024-02-29",
                    stock_id="0050",
                    run_id=run_id,
                    model_version=model_version,
                    prediction_probability=0.9,
                    rank=1,
                ),
            ]
        )
        session.add_all(
            [
                Feature(
                    rebalance_date="2024-01-31",
                    stock_id="2330",
                    feature_version="factor_adj_v2",
                    volatility_60d=0.11,
                    beta_60d=0.21,
                ),
                Feature(
                    rebalance_date="2024-01-31",
                    stock_id="0050",
                    feature_version="factor_adj_v2",
                    volatility_60d=0.22,
                    beta_60d=0.32,
                ),
                Feature(
                    rebalance_date="2024-02-29",
                    stock_id="2330",
                    feature_version="factor_adj_v2",
                    volatility_60d=0.44,
                    beta_60d=0.54,
                ),
                Feature(
                    rebalance_date="2024-02-29",
                    stock_id="0050",
                    feature_version="factor_adj_v2",
                    volatility_60d=0.55,
                    beta_60d=0.65,
                ),
            ]
        )

    assert store.list_holding_dates(run_id) == ["2024-01-31", "2024-02-29"]
    january = store.load_holdings(run_id, "2024-01")
    february = store.load_holdings(run_id, "2024-02")
    assert january["stock_id"].tolist() == ["2330"]
    assert january["stock_name"].tolist() == ["台積電"]
    assert january["rank"].tolist() == [1]
    assert january["prediction_probability"].tolist() == [pytest.approx(0.8)]
    assert january["volatility_60d"].tolist() == [pytest.approx(0.11)]
    assert february["stock_id"].tolist() == ["0050"]
    assert february["stock_name"].tolist() == ["元大台灣50"]
    assert february["rank"].tolist() == [1]
    assert february["prediction_probability"].tolist() == [pytest.approx(0.9)]
    assert february["volatility_60d"].tolist() == [pytest.approx(0.55)]
    assert store.load_holdings(run_id, "2024-03").empty


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


def test_load_performance_prefers_bounded_materialized_curve(
    store: DbStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        store,
        "_artifact",
        lambda _run_id, _kind: {
            "equity_curve": [
                {"date": "2024-01-02", "nav": 100.0},
                {"date": "2024-01-31", "nav": 110.0},
            ]
        },
    )

    def _unexpected_replay(*_args, **_kwargs):
        raise AssertionError("materialized performance must not replay beyond its OOS horizon")

    monkeypatch.setattr(store, "replay_backtest", _unexpected_replay)

    performance = store.load_performance("oos")

    assert performance.to_dict("records") == [
        {"date": "2024-01-02", "nav": 100.0},
        {"date": "2024-01-31", "nav": 110.0},
    ]


def test_replay_backtest_excludes_orders_after_oos_signal_end(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    store.start_run({"run_id": "oos", "job": "test", "data_end_date": "2020-01-31"})
    opens = sample_prices.set_index(["trade_date", "stock_id"])["open"]
    with store._scope() as session:  # noqa: SLF001 - test-only ledger setup.
        from models.research import Order

        session.add_all(
            [
                Order(
                    order_id="buy",
                    run_id="oos",
                    signal_date="2020-01-31",
                    execution_date="2020-02-03",
                    stock_id="2330",
                    side="BUY",
                    quantity=100,
                    open_price=float(opens.loc[("2020-02-03", "2330")]),
                    executed_price=float(opens.loc[("2020-02-03", "2330")]),
                    notional=50_000.0,
                    broker_fee=0.0,
                    transaction_tax=0.0,
                    slippage_cost=0.0,
                    total_cost=0.0,
                ),
                Order(
                    order_id="terminal-liquidation",
                    run_id="oos",
                    signal_date="2020-02-03",
                    execution_date="2020-02-04",
                    stock_id="2330",
                    side="SELL",
                    quantity=100,
                    open_price=float(opens.loc[("2020-02-04", "2330")]),
                    executed_price=float(opens.loc[("2020-02-04", "2330")]),
                    notional=50_000.0,
                    broker_fee=0.0,
                    transaction_tax=0.0,
                    slippage_cost=0.0,
                    total_cost=0.0,
                ),
            ]
        )

    full = store.replay_backtest("oos")
    bounded = store.replay_backtest("oos", signal_end_date="2020-01-31")

    assert full.orders["side"].tolist() == ["BUY", "SELL"]
    assert bounded.orders["side"].tolist() == ["BUY"]


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


def test_market_cap_uses_latest_close_before_run_date(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    first_day = sample_prices["trade_date"].min()
    store.start_run({"run_id": "early-cap", "job": "test", "data_end_date": first_day})
    first_close = sample_prices.loc[
        (sample_prices["stock_id"] == "2330") & (sample_prices["trade_date"] == first_day),
        "close",
    ].iloc[0]
    caps = store.load_stocks().set_index("stock_id")["market_cap"]
    assert caps.loc["2330"] == pytest.approx(first_close * 25_000_000_000.0)


def test_market_cap_missing_without_cache(
    settings: Settings, temp_db_path: Path, sample_prices: pd.DataFrame
) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
        adj_migration.upgrade(conn)
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


def test_stable_db_store_reads_partial_pit_market_values(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    stable_settings = dataclasses.replace(
        store._settings,
        features=dataclasses.replace(
            store._settings.features, feature_version="factor_adj_pit_v3_stable"
        ),
    )
    regular_pit_settings = dataclasses.replace(
        stable_settings,
        features=dataclasses.replace(
            stable_settings.features, feature_version="factor_adj_pit_v3"
        ),
    )
    stable_store = DbStore(store._engine, stable_settings)
    regular_pit_store = DbStore(store._engine, regular_pit_settings)
    with stable_store._scope() as session:
        upsert_partial_market_value_day(
            session,
            date.fromisoformat(AS_OF),
            pd.DataFrame(
                {"trade_date": [AS_OF], "stock_id": ["2330"], "market_value": [1234.0]}
            ),
            source="test:verified-partial",
        )

    stable_store.start_run({"run_id": "stable-partial", "job": "test", "data_end_date": AS_OF})
    regular_pit_store.start_run(
        {"run_id": "regular-pit-partial", "job": "test", "data_end_date": AS_OF}
    )
    stable_caps = stable_store.load_stocks().set_index("stock_id")["market_cap"]
    regular_pit_caps = regular_pit_store.load_stocks().set_index("stock_id")["market_cap"]

    assert stable_caps.loc["2330"] == pytest.approx(1234.0)
    assert pd.isna(regular_pit_caps.loc["2330"])
    assert stable_store.load_market_value_snapshot()["market_value"].tolist() == [1234.0]
    assert regular_pit_store.load_market_value_snapshot().empty


def test_load_shares_cache_roundtrip(tmp_path: Path) -> None:
    assert load_shares_cache(None) == {}
    assert load_shares_cache(tmp_path / "missing.json") == {}
    path = tmp_path / "shares.json"
    path.write_text(json.dumps({"2330": {"shares": 1.0}}), encoding="utf-8")
    assert load_shares_cache(path) == {"2330": {"shares": 1.0}}
    assert default_shares_path("sqlite:///database/quant.db") == Path("database/shares.json")
    assert default_shares_path("sqlite:///:memory:") is None


def test_bind_attaches_to_existing_run(store) -> None:
    store.start_run({"run_id": "r", "job": "test", "data_end_date": "2020-02-05"})
    store.finish_run("r", "succeeded")
    fresh = DbStore(store._engine, store._settings)
    with pytest.raises(ValueError, match="no run"):
        fresh.save_predictions(pd.DataFrame({"stock_id": ["2330"]}), "m")
    fresh.bind("r", "2020-02-05")
    assert fresh.save_orders(pd.DataFrame()) == 0
    with pytest.raises(ValueError, match="unknown run"):
        fresh.bind("ghost")
    with pytest.raises(ValueError, match="run_id"):
        fresh.bind("  ")


def test_replay_liquidates_full_sell_to_flat(store: DbStore, sample_prices: pd.DataFrame) -> None:
    _seed_market(store, sample_prices)
    store.start_run({"run_id": "r", "job": "test", "data_end_date": AS_OF})
    buy = pd.DataFrame(
        [
            {
                "order_id": "r|2020-02-04|2330|BUY",
                "run_id": "r",
                "signal_date": "2020-02-03",
                "execution_date": "2020-02-04",
                "stock_id": "2330",
                "side": "BUY",
                "target_weight": 0.1,
                "target_shares": 100,
                "executed_price": 500.5,
                "broker_fee": 71.32125,
                "transaction_tax": 0.0,
                "slippage_cost": 50.05,
                "total_cost": 121.37125,
            }
        ]
    )
    sell = pd.DataFrame(
        [
            {
                "order_id": "r|2020-02-05|2330|SELL",
                "run_id": "r",
                "signal_date": "2020-02-04",
                "execution_date": "2020-02-05",
                "stock_id": "2330",
                "side": "SELL",
                "target_weight": 0.0,
                "target_shares": 0,
                "executed_price": 500.0,
                "broker_fee": 71.25,
                "transaction_tax": 150.0,
                "slippage_cost": 50.0,
                "total_cost": 271.25,
            }
        ]
    )
    assert store.save_orders(buy) == 1
    assert store.save_orders(sell) == 1
    result = store.replay_backtest("r")
    # Executed prices already include slippage; the NAV subtracts the price
    # move plus fees/tax, without charging the slippage fields twice.
    assert result.nav.iloc[-1] == pytest.approx(
        INITIAL_CAPITAL - 71.32125 - 71.25 - 150.0 - 50.0
    )
    tail = result.nav.loc[result.nav.index.astype(str) >= "2020-02-05"]
    assert (tail == tail.iloc[0]).all()


def test_build_store_factory(settings: Settings, temp_db_path: Path) -> None:
    conn = sqlite3.connect(str(temp_db_path))
    try:
        migration.upgrade(conn)
        adj_migration.upgrade(conn)
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


def test_save_target_holdings_ranks_unscored_sell_last(
    store: DbStore, sample_prices: pd.DataFrame
) -> None:
    _seed_market(store, sample_prices)
    store.start_run(
        {
            "run_id": "rebalance-2020-02-05",
            "data_end_date": AS_OF,
            "feature_version": "factor_v1",
            "parameter_version": "p1",
        }
    )
    preds = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "probability": [0.9],
            "rank": [1],
            "prediction_date": [AS_OF],
        }
    )
    assert store.save_predictions(preds, "xgb_202002_factor_v1") == 1
    from repositories import stocks as stocks_repo

    with store._scope() as session:  # noqa: SLF001
        stocks_repo.upsert_stocks(session, pd.DataFrame({"stock_id": ["9999"], "market": ["TWSE"]}))
    target = PortfolioTarget(
        run_id="rebalance-2020-02-05",
        signal_date=AS_OF,
        top_n=15,
        actions={"2330": "BUY", "9999": "SELL"},
        weights={"2330": 0.1},
        cash_weight=0.9,
        equity_exposure=0.1,
    )
    store.save_target_holdings(target, model_version="xgb_202002_factor_v1")
    with store._scope() as session:  # noqa: SLF001
        from sqlalchemy import select

        from models.research import Signal

        rank = session.execute(
            select(Signal.rank).where(
                Signal.run_id == "rebalance-2020-02-05", Signal.stock_id == "9999"
            )
        ).scalar_one()
    assert rank == 2
