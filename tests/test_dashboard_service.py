"""P4-02 acceptance: dashboard read service (fake store)."""

from __future__ import annotations

import pandas as pd
import pytest

from services.dashboard_service import (
    HOLDING_COLUMNS,
    get_comparison,
    get_holding_months,
    get_holdings,
    get_model_data,
    get_overview,
    get_risk,
)


class FakeStore:
    def __init__(self, status: str = "succeeded") -> None:
        self._status = status

    def get_run_status(self, run_id: str) -> str:
        if run_id == "missing":
            raise KeyError(run_id)
        return self._status

    def load_run_summary(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "data_end_date": "2020-02-29",
            "feature_version": "v1",
            "model_version": "xgb_202002",
            "parameter_version": "params_abc",
            "metrics": {"cagr": 0.12, "sharpe": float("nan")},
            "oos_months": ["2020-02"],
            "equity_curve": [{"date": "2020-02-03", "nav": 1.0}],
            "monthly_returns": [{"month": "2020-02", "return": 0.01}],
            "orm_entity": object(),
        }

    def list_holding_dates(self, run_id: str) -> list[str]:
        assert run_id
        return ["2024-01-31", "2024-02-29", "2024-02-29"]

    def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
        assert run_id and isinstance(as_of, str)
        frame = pd.DataFrame(
            {
                "stock_id": ["2330", "2317"],
                "stock_name": ["台積電", "鴻海"],
                "rank": [1, 2],
                "prediction_probability": [0.7, 0.6],
                "weight": [0.2, 0.15],
                "volatility_60d": [0.25, 0.3],
                "beta_60d": [1.1, 0.9],
                "orm_extra": [object(), object()],
            }
        )
        return frame.set_index("stock_id")

    def load_model_data(self, run_id: str) -> dict:
        return {
            "shap_top": [{"feature": "momentum_20d", "value": 0.05}],
            "feature_importance": [{"feature": "momentum_20d", "gain": 12.0}],
            "monthly_ic": [{"month": "2020-02", "ic": 0.08}],
            "prediction_dist": [0.1, 0.5, 0.7],
            "booster": object(),
        }

    def load_risk(self, run_id: str) -> dict:
        return {
            "equity_exposure": 1.0,
            "predicted_volatility": 0.18,
            "realized_volatility": float("nan"),
            "max_drawdown": -0.1,
            "turnover": 0.5,
            "market_regime": "above_ma60",
            "exposure_cap": 1.0,
        }

    def load_comparison(self, run_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scenario": ["baseline", "ablation_momentum"],
                "cagr_before": [0.12, 0.10],
                "cagr_after": [0.09, 0.07],
            },
            index=[5, 6],
        )


def test_get_overview_shape_and_nan() -> None:
    payload = get_overview("run-001", FakeStore())
    assert payload["run_id"] == "run-001"
    assert payload["metrics"] == {"cagr": 0.12, "sharpe": None}
    assert payload["oos_months"] == ["2020-02"]
    assert payload["equity_curve"] == [{"date": "2020-02-03", "nav": 1.0}]
    assert payload["monthly_returns"][0]["month"] == "2020-02"
    assert "orm_entity" not in payload


def test_get_holdings_columns_and_index() -> None:
    frame = get_holdings("run-001", "", FakeStore())
    assert list(frame.columns) == list(HOLDING_COLUMNS)
    assert list(frame.index) == [0, 1]
    assert frame["weight"].tolist() == [0.2, 0.15]


def test_get_holding_months_deduplicates_dates() -> None:
    assert get_holding_months("run-001", FakeStore()) == ["2024-01", "2024-02"]


def test_get_holdings_accepts_month_and_rejects_bad_month() -> None:
    class MonthStore(FakeStore):
        def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
            assert as_of == "2024-02"
            return super().load_holdings(run_id, as_of)

    assert len(get_holdings("run-001", "2024-02", MonthStore())) == 2
    with pytest.raises(ValueError, match="YYYY-MM"):
        get_holdings("run-001", "2024-13", FakeStore())


def test_get_model_and_risk() -> None:
    model = get_model_data("run-001", FakeStore())
    assert model["shap_top"][0]["feature"] == "momentum_20d"
    assert "booster" not in model
    risk = get_risk("run-001", FakeStore())
    assert risk["market_regime"] == "above_ma60"
    assert risk["realized_volatility"] is None


def test_get_comparison() -> None:
    frame = get_comparison("run-001", FakeStore())
    assert frame["scenario"].tolist() == ["baseline", "ablation_momentum"]
    assert "cagr_before" in frame.columns and "cagr_after" in frame.columns
    assert list(frame.index) == [0, 1]


def test_incomplete_run_rejected_everywhere() -> None:
    store = FakeStore(status="failed")
    with pytest.raises(ValueError, match="not completed"):
        get_overview("run-001", store)
    with pytest.raises(ValueError, match="not completed"):
        get_holdings("run-001", "", store)
    with pytest.raises(ValueError, match="not completed"):
        get_holding_months("run-001", store)
    with pytest.raises(ValueError, match="not completed"):
        get_model_data("run-001", store)
    with pytest.raises(ValueError, match="not completed"):
        get_risk("run-001", store)
    with pytest.raises(ValueError, match="not completed"):
        get_comparison("run-001", store)
    with pytest.raises(ValueError, match="unknown run"):
        get_overview("missing", FakeStore())
    with pytest.raises(ValueError, match="run_id"):
        get_overview("  ", FakeStore())


def test_bad_payloads_rejected() -> None:
    class BadSummary(FakeStore):
        def load_run_summary(self, run_id: str) -> dict:
            return {"run_id": run_id}

    with pytest.raises(ValueError, match="missing"):
        get_overview("run-001", BadSummary())

    class BadHoldings(FakeStore):
        def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
            return pd.DataFrame({"stock_id": ["2330"]})

    with pytest.raises(ValueError, match="missing columns"):
        get_holdings("run-001", "", BadHoldings())

    class BadComparison(FakeStore):
        def load_comparison(self, run_id: str) -> pd.DataFrame:
            return pd.DataFrame({"cagr": [0.1]})

    with pytest.raises(ValueError, match="scenario"):
        get_comparison("run-001", BadComparison())

    with pytest.raises(ValueError, match="as_of"):
        get_holdings("run-001", None, FakeStore())  # type: ignore[arg-type]
