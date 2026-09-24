"""P2-12 acceptance: weight and market risk service."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

from contracts import PortfolioTarget
from services.risk_service import RiskFailure, apply_risk_controls

SIGNAL = date(2019, 12, 31)
rng = np.random.default_rng(5)


def _wide_settings(settings):
    """Same settings with the single-name cap lifted to isolate weighting logic."""
    portfolio = dataclasses.replace(settings.portfolio, max_individual_weight=1.0)
    return dataclasses.replace(settings, portfolio=portfolio)


def _returns(spec: dict[str, np.ndarray], end: str = "2019-12-31") -> pd.DataFrame:
    days = max(len(v) for v in spec.values())
    dates = pd.bdate_range(end=end, periods=days).strftime("%Y-%m-%d")
    rows = []
    for stock_id, rets in spec.items():
        for trade_date, value in zip(dates[-len(rets) :], rets, strict=True):
            rows.append({"stock_id": stock_id, "trade_date": trade_date, "log_return": value})
    return pd.DataFrame(rows)


def _taiex(start: float, stop: float, days: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range(end="2019-12-31", periods=days).strftime("%Y-%m-%d")
    return pd.DataFrame({"trade_date": dates, "close": np.linspace(start, stop, days)})


def _target(actions: dict[str, str]) -> PortfolioTarget:
    active = [s for s, action in actions.items() if action in ("BUY", "HOLD")]
    weight = 1.0 / len(active)
    return PortfolioTarget(
        run_id="r",
        signal_date="2019-12-31",
        top_n=15,
        actions=dict(actions),
        weights={s: weight for s in active},
        cash_weight=0.0,
        equity_exposure=1.0,
    )


def test_equal_volatility_equal_weights(settings) -> None:
    settings = _wide_settings(settings)
    rets = rng.normal(0, 0.005, 60)
    returns = _returns({"A": rets.copy(), "B": rets.copy()})
    target = apply_risk_controls(
        _target({"A": "BUY", "B": "HOLD"}), returns, _taiex(100, 160), SIGNAL, settings
    )
    assert target.weights["A"] == pytest.approx(target.weights["B"])
    assert target.weights["A"] == pytest.approx(0.5)
    assert target.equity_exposure == pytest.approx(1.0)


def test_high_volatility_gets_less(settings) -> None:
    settings = _wide_settings(settings)
    returns = _returns({"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.03, 60)})
    target = apply_risk_controls(
        _target({"A": "BUY", "B": "BUY"}), returns, _taiex(100, 160), SIGNAL, settings
    )
    ratio = target.weights["A"] / target.weights["B"]
    assert 2.0 < ratio < 4.0


def test_individual_cap_with_redistribution(settings) -> None:
    spec = {"STAR": rng.normal(0, 0.0005, 60)}
    for i in range(11):
        spec[f"S{i:02d}"] = rng.normal(0, 0.02, 60)
    returns = _returns(spec)
    actions = {"STAR": "BUY", **{f"S{i:02d}": "BUY" for i in range(11)}}
    target = apply_risk_controls(_target(actions), returns, _taiex(100, 160), SIGNAL, settings)
    assert max(target.weights.values()) <= 0.10 + 1e-9
    assert target.weights["STAR"] == pytest.approx(0.10)
    assert sum(target.weights.values()) == pytest.approx(target.equity_exposure)


def test_taiex_below_ma60_caps_exposure(settings) -> None:
    settings = _wide_settings(settings)
    rets = rng.normal(0, 0.005, 60)
    returns = _returns({"A": rets.copy(), "B": rets.copy()})
    target = apply_risk_controls(
        _target({"A": "BUY", "B": "HOLD"}), returns, _taiex(160, 100), SIGNAL, settings
    )
    assert target.equity_exposure == pytest.approx(0.5)
    assert sum(target.weights.values()) == pytest.approx(0.5)
    assert target.cash_weight == pytest.approx(0.5)


def test_infeasible_cap_parks_residual_in_cash(settings) -> None:
    rets = rng.normal(0, 0.005, 60)
    returns = _returns({"A": rets.copy(), "B": rets.copy()})
    target = apply_risk_controls(
        _target({"A": "BUY", "B": "HOLD"}), returns, _taiex(100, 160), SIGNAL, settings
    )
    assert max(target.weights.values()) <= 0.10 + 1e-9
    assert sum(target.weights.values()) == pytest.approx(target.equity_exposure)
    assert target.cash_weight == pytest.approx(1.0 - target.equity_exposure)


def test_unmeasurable_stock_flips_to_sell(settings) -> None:
    returns = _returns({"A": rng.normal(0, 0.01, 60), "C": rng.normal(0, 0.01, 10)})
    target = apply_risk_controls(
        _target({"A": "BUY", "C": "HOLD"}), returns, _taiex(100, 160), SIGNAL, settings
    )
    assert target.actions["C"] == "SELL"
    assert set(target.weights) == {"A"}


def test_irrelevant_and_future_returns_do_not_change_risk(settings) -> None:
    settings = _wide_settings(settings)
    sample = rng.normal(0, 0.005, 60)
    base = _returns({"A": sample.copy(), "B": sample.copy()})
    extra = _returns({"OTHER": rng.normal(0, 0.02, 60)})
    extra = pd.concat(
        [extra, pd.DataFrame([{"stock_id": "A", "trade_date": "2020-01-02", "log_return": 1.0}])],
        ignore_index=True,
    )
    target = _target({"A": "BUY", "B": "HOLD"})
    expected = apply_risk_controls(target, base, _taiex(100, 160), SIGNAL, settings)
    actual = apply_risk_controls(
        target, pd.concat([base, extra], ignore_index=True), _taiex(100, 160), SIGNAL, settings
    )
    assert actual.weights == expected.weights
    assert actual.equity_exposure == expected.equity_exposure


def test_insufficient_overlap_raises_risk_failure(settings) -> None:
    early = _returns({"A": rng.normal(0, 0.01, 60)}, end="2019-10-31")
    late = _returns({"B": rng.normal(0, 0.01, 60)}, end="2019-12-31")
    returns = pd.concat([early, late], ignore_index=True)
    with pytest.raises(RiskFailure):
        apply_risk_controls(
            _target({"A": "BUY", "B": "BUY"}),
            returns,
            _taiex(100, 160),
            SIGNAL,
            settings,
        )
    assert issubclass(RiskFailure, ValueError)
