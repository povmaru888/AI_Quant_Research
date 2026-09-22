"""P2-11 acceptance: rank-buffer target holdings service."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from contracts import PortfolioTarget
from services.portfolio_service import build_target_holdings

SIGNAL = date(2019, 12, 31)


def _predictions(ranks: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": list(ranks),
            "probability": [1.0 / r for r in ranks.values()],
            "rank": list(ranks.values()),
        }
    )


def _positions(held: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"stock_id": held})


def test_build_target_holdings_four_states(settings) -> None:
    predictions = _predictions({"A": 5, "B": 20, "C": 40, "D": 50})
    target = build_target_holdings(predictions, _positions(["B", "C"]), settings, "r", SIGNAL)
    assert isinstance(target, PortfolioTarget)
    assert target.actions == {"A": "BUY", "B": "HOLD", "C": "SELL", "D": "NONE"}
    assert target.weights == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}
    assert target.equity_exposure == pytest.approx(1.0)
    assert target.cash_weight == pytest.approx(0.0)


def test_build_target_holdings_boundaries(settings) -> None:
    predictions = _predictions({"N15": 15, "N16": 16, "H30": 30, "H31": 31})
    target = build_target_holdings(
        predictions, _positions(["H30", "H31", "N15"]), settings, "r", SIGNAL
    )
    assert target.actions["N15"] == "HOLD"  # held before: HOLD, not BUY
    assert target.actions["N16"] == "NONE"
    assert target.actions["H30"] == "HOLD"
    assert target.actions["H31"] == "SELL"


def test_build_target_holdings_missing_prediction_sells(settings) -> None:
    predictions = _predictions({"A": 1})
    target = build_target_holdings(predictions, _positions(["GONE"]), settings, "r", SIGNAL)
    assert target.actions["GONE"] == "SELL"
    assert target.actions["A"] == "BUY"


def test_build_target_holdings_first_period(settings) -> None:
    predictions = _predictions({"A": 1, "B": 100})
    target = build_target_holdings(
        predictions, pd.DataFrame({"stock_id": []}), settings, "r", SIGNAL
    )
    assert target.actions == {"A": "BUY", "B": "NONE"}
    assert target.weights == {"A": pytest.approx(1.0)}


def test_build_target_holdings_all_cash(settings) -> None:
    predictions = _predictions({"A": 100})
    target = build_target_holdings(
        predictions, pd.DataFrame({"stock_id": []}), settings, "r", SIGNAL
    )
    assert target.actions == {"A": "NONE"}
    assert target.weights == {}
    assert target.equity_exposure == pytest.approx(0.0)
    assert target.cash_weight == pytest.approx(1.0)


def test_build_target_holdings_rejects_bad_inputs(settings) -> None:
    predictions = _predictions({"A": 1})
    with pytest.raises(ValueError, match="signal_date"):
        build_target_holdings(predictions, _positions([]), settings, "r", "2019-12-31")
    with pytest.raises(ValueError, match="missing columns"):
        build_target_holdings(
            predictions.drop(columns=["rank"]), _positions([]), settings, "r", SIGNAL
        )
    doubled = pd.concat([predictions, predictions], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_target_holdings(doubled, _positions([]), settings, "r", SIGNAL)
