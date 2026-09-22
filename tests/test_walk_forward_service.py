"""P2-10 acceptance: walk-forward service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import FeatureSet
from services.walk_forward_service import build_folds, run_walk_forward
from services.xgb_service import train_xgb

rng = np.random.default_rng(23)


def _months(n: int, start_year: int = 2013) -> list[str]:
    out = []
    year, month = start_year, 1
    for _ in range(n):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def _feature_set(month: str, columns: tuple[str, ...] = ("f1", "f2")) -> FeatureSet:
    n = 10
    data: dict[str, object] = {
        "stock_id": [f"S{i:02d}" for i in range(n)],
        "missing_flag": np.zeros(n, dtype=int),
    }
    for column in columns:
        data[column] = rng.normal(size=n)
    return FeatureSet(
        run_id="run-001",
        as_of=f"{month}-15",
        feature_version="factor_v1",
        frame=pd.DataFrame(data),
        feature_columns=columns,
        coverage={c: 1.0 for c in columns},
    )


def _labels(n: int = 10) -> pd.Series:
    return pd.Series(rng.integers(0, 2, size=n))


def _spy_calls():
    calls: list[dict] = []

    def spy(train_x, train_y, valid_x, valid_y, *_args, **_kwargs):
        calls.append({"train_rows": len(train_x), "valid_rows": len(valid_x)})
        artifact, booster = train_xgb(
            train_x, train_y, valid_x, valid_y, {"n_estimators": 5}, "v", "f", "p", "r"
        )
        return artifact, booster, pd.DataFrame()

    return calls, spy


def test_build_folds_two_folds_disjoint(settings) -> None:
    months = _months(84)
    folds = build_folds(months, settings)
    assert len(folds) == 2
    for fold in folds:
        assert len(fold.train_months) == 48
        assert len(fold.valid_months) == 12
        assert len(fold.test_months) == 12
        assert not (set(fold.train_months) & set(fold.valid_months) & set(fold.test_months))
        assert not (set(fold.train_months) & set(fold.valid_months))
        assert not (set(fold.valid_months) & set(fold.test_months))
        assert not (set(fold.train_months) & set(fold.test_months))
    assert folds[0].test_months[0] == "2018-01"
    assert folds[1].train_months[0] == "2014-01"


def test_build_folds_insufficient_months_empty(settings) -> None:
    assert build_folds(_months(60), settings) == []
    with pytest.raises(ValueError, match="invalid month"):
        build_folds(["2019-13"], settings)


def test_run_walk_forward_purge_and_oos(settings) -> None:
    months = _months(84)
    features = {m: _feature_set(m) for m in months}
    labels = {m: _labels() for m in months}
    calls, spy = _spy_calls()
    results = run_walk_forward(
        features, labels, settings, "xgb", "factor_v1", "p1", "run-001", optimize=spy
    )
    assert len(results) == 2
    # Purge drops the last training month: 47 x 10 rows, not 48 x 10.
    assert calls[0] == {"train_rows": 470, "valid_rows": 120}
    result = results[0]
    assert len(result.oos_predictions) == 120
    assert set(result.oos_predictions["prediction_date"].unique()) == set(
        pd.Series([features[m].as_of for m in result.fold.test_months]).unique()
    )
    for _, group in result.oos_predictions.groupby("prediction_date"):
        assert group["rank"].is_unique
    assert list(result.oos_predictions.columns)[:3] == ["stock_id", "probability", "rank"]


def test_run_walk_forward_no_common_columns(settings) -> None:
    months = _months(84)
    features = {
        m: _feature_set(m, ("f1", "f2") if i % 2 else ("f3", "f4")) for i, m in enumerate(months)
    }
    labels = {m: _labels() for m in months}
    _, spy = _spy_calls()
    with pytest.raises(ValueError, match="no common feature columns"):
        run_walk_forward(features, labels, settings, "x", "f", "p", "r", optimize=spy)
