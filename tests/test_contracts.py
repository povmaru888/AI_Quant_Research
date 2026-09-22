"""P2-01 acceptance: cross-module domain contracts."""

from __future__ import annotations

import pytest

from contracts import (
    BacktestResult,
    FeatureSet,
    ModelArtifact,
    PortfolioTarget,
    UniverseEntry,
    UniverseSnapshot,
)

pd = pytest.importorskip("pandas")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["2330", "2317"],
            "momentum_20d": [0.05, -0.02],
            "volatility_60d": [0.2, 0.3],
            "missing_flag": [0, 1],
        }
    )


def test_universe_snapshot_happy_path() -> None:
    snapshot = UniverseSnapshot(
        run_id="run-001",
        as_of="2019-12-31",
        entries=[
            UniverseEntry(stock_id="2330", included=True, reason="pass"),
            UniverseEntry(stock_id="9999", included=False, reason="price<=10"),
        ],
    )
    assert snapshot.included_ids == ("2330",)
    assert snapshot.excluded_ids == ("9999",)


def test_universe_snapshot_rejects_bad_traceability() -> None:
    entry = UniverseEntry(stock_id="2330", included=True, reason="pass")
    with pytest.raises(ValueError, match="run_id"):
        UniverseSnapshot(run_id="  ", as_of="2019-12-31", entries=[entry])
    with pytest.raises(ValueError, match="as_of"):
        UniverseSnapshot(run_id="run-001", as_of="2019-13-01", entries=[entry])
    with pytest.raises(ValueError, match="duplicate"):
        UniverseSnapshot(
            run_id="run-001",
            as_of="2019-12-31",
            entries=[entry, UniverseEntry("2330", False, "cap")],
        )
    with pytest.raises(ValueError, match="reason"):
        UniverseEntry(stock_id="2330", included=True, reason=" ")


def test_feature_set_happy_path() -> None:
    features = FeatureSet(
        run_id="run-001",
        as_of="2019-12-31",
        feature_version="factor_v1",
        frame=_frame(),
        feature_columns=["momentum_20d", "volatility_60d"],
        coverage={"momentum_20d": 1.0, "volatility_60d": 0.5},
    )
    assert features.feature_columns == ("momentum_20d", "volatility_60d")


def test_feature_set_rejects_bad_frame_and_coverage() -> None:
    with pytest.raises(ValueError, match="not in frame"):
        FeatureSet(
            run_id="run-001",
            as_of="2019-12-31",
            feature_version="factor_v1",
            frame=_frame(),
            feature_columns=["nope"],
            coverage={"nope": 1.0},
        )
    with pytest.raises(ValueError, match="coverage"):
        FeatureSet(
            run_id="run-001",
            as_of="2019-12-31",
            feature_version="factor_v1",
            frame=_frame(),
            feature_columns=["momentum_20d", "volatility_60d"],
            coverage={"momentum_20d": 1.0},
        )
    with pytest.raises(ValueError, match="coverage"):
        FeatureSet(
            run_id="run-001",
            as_of="2019-12-31",
            feature_version="factor_v1",
            frame=_frame(),
            feature_columns=["momentum_20d", "volatility_60d"],
            coverage={"momentum_20d": 1.0, "volatility_60d": 1.5},
        )
    with pytest.raises(ValueError, match="stock_id"):
        FeatureSet(
            run_id="run-001",
            as_of="2019-12-31",
            feature_version="factor_v1",
            frame=_frame().drop(columns=["stock_id"]),
            feature_columns=["momentum_20d"],
            coverage={"momentum_20d": 1.0},
        )


def test_model_artifact_happy_path() -> None:
    artifact = ModelArtifact(
        run_id="run-001",
        model_version="xgb_v1",
        feature_version="factor_v1",
        parameter_version="p1",
        feature_columns=["momentum_20d"],
        best_params={"max_depth": 4},
        validation_rank_ic=0.05,
    )
    assert artifact.validation_rank_ic == pytest.approx(0.05)


def test_model_artifact_rejects_bad_values() -> None:
    kwargs: dict = {
        "run_id": "run-001",
        "model_version": "xgb_v1",
        "feature_version": "factor_v1",
        "parameter_version": "p1",
        "feature_columns": ["momentum_20d"],
        "best_params": {},
        "validation_rank_ic": 0.05,
    }
    with pytest.raises(ValueError, match="finite"):
        ModelArtifact(**{**kwargs, "validation_rank_ic": float("nan")})
    with pytest.raises(ValueError, match="model_version"):
        ModelArtifact(**{**kwargs, "model_version": ""})
    with pytest.raises(ValueError, match="feature_columns"):
        ModelArtifact(**{**kwargs, "feature_columns": []})


def test_portfolio_target_happy_path() -> None:
    target = PortfolioTarget(
        run_id="run-001",
        signal_date="2019-12-31",
        top_n=15,
        actions={"2330": "BUY", "2317": "SELL", "2454": "NONE"},
        weights={"2330": 0.6},
        cash_weight=0.4,
        equity_exposure=0.6,
    )
    assert target.top_n == 15


def test_portfolio_target_rejects_bad_weights_and_top_n() -> None:
    base: dict = {
        "run_id": "run-001",
        "signal_date": "2019-12-31",
        "top_n": 15,
        "actions": {"2330": "BUY"},
        "weights": {"2330": 0.6},
        "cash_weight": 0.4,
        "equity_exposure": 0.6,
    }
    with pytest.raises(ValueError, match="weights"):
        PortfolioTarget(**{**base, "weights": {"2330": -0.1}})
    with pytest.raises(ValueError, match="equity_exposure"):
        PortfolioTarget(**{**base, "weights": {"2330": 0.5}})
    with pytest.raises(ValueError, match="top_n"):
        PortfolioTarget(**{**base, "top_n": -15})
    with pytest.raises(ValueError, match="top_n"):
        PortfolioTarget(**{**base, "top_n": True})
    with pytest.raises(ValueError, match="action"):
        PortfolioTarget(**{**base, "actions": {"2330": "YOLO"}, "weights": {"2330": 0.6}})
    with pytest.raises(ValueError, match="BUY/HOLD"):
        PortfolioTarget(**{**base, "actions": {"2330": "SELL"}})


def test_backtest_result_happy_path() -> None:
    result = BacktestResult(
        run_id="run-001",
        start_date="2020-01-01",
        end_date="2020-12-31",
        initial_cash=1_000_000.0,
        nav=pd.Series([1_000_000.0, 1_010_000.0]),
        orders=pd.DataFrame(
            {
                "order_id": ["run-001|20200102|2330|BUY"],
                "executed_price": [300.0],
                "total_cost": [500.0],
            }
        ),
        total_cost=500.0,
    )
    assert result.total_cost == pytest.approx(500.0)


def test_backtest_result_rejects_bad_values() -> None:
    kwargs: dict = {
        "run_id": "run-001",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "initial_cash": 1_000_000.0,
        "nav": pd.Series([1_000_000.0, 1_010_000.0]),
        "orders": pd.DataFrame(columns=["order_id", "executed_price", "total_cost"]),
        "total_cost": 0.0,
    }
    with pytest.raises(ValueError, match="negative"):
        BacktestResult(**{**kwargs, "nav": pd.Series([1.0, -1.0])})
    with pytest.raises(ValueError, match="NaN"):
        BacktestResult(**{**kwargs, "nav": pd.Series([1.0, float("nan")])})
    with pytest.raises(ValueError, match="start_date"):
        BacktestResult(**{**kwargs, "start_date": "2021-01-01"})
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestResult(**{**kwargs, "initial_cash": 0.0})
    with pytest.raises(ValueError, match="columns"):
        BacktestResult(**{**kwargs, "orders": pd.DataFrame({"order_id": ["x"]})})
