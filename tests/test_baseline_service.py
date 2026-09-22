"""P2-07 acceptance: baseline model service."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from contracts import FeatureSet
from services.baseline_service import BASELINE_COLUMNS, score_baselines


def _features(extra_drop: tuple[str, ...] = ()) -> FeatureSet:
    n = 20
    rng = np.random.default_rng(7)
    data: dict[str, object] = {
        "stock_id": [f"S{i:02d}" for i in range(n)],
        "missing_flag": np.zeros(n, dtype=int),
        "momentum_20d": np.linspace(-2, 2, n),
        "momentum_60d": np.linspace(-1, 1, n),
        "volatility_60d": rng.normal(size=n),
    }
    frame = pd.DataFrame(data).drop(columns=list(extra_drop))
    candidates = ("momentum_20d", "momentum_60d", "volatility_60d")
    columns = tuple(c for c in candidates if c not in extra_drop)
    return FeatureSet(
        run_id="run-001",
        as_of="2019-12-31",
        feature_version="factor_v1",
        frame=frame,
        feature_columns=columns,
        coverage={c: 1.0 for c in columns},
    )


def _unique_ranks(scores: pd.DataFrame) -> None:
    for column in BASELINE_COLUMNS:
        valid = scores[column].dropna()
        if valid.empty:
            continue
        assert valid.rank(method="first").is_unique, column


def test_score_baselines_shape_and_values() -> None:
    features = _features()
    scores = score_baselines(features)
    assert list(scores.columns) == ["stock_id", *BASELINE_COLUMNS]
    assert len(scores) == len(features.frame)
    assert scores["momentum_20d"].tolist() == pytest.approx(features.frame["momentum_20d"].tolist())
    expected_blend = features.frame["momentum_20d"] + features.frame["momentum_60d"]
    assert scores["momentum_20d_60d"].tolist() == pytest.approx(expected_blend.tolist())
    expected_equal = features.frame[list(features.feature_columns)].mean(axis=1)
    assert scores["equal_weight_rank"].tolist() == pytest.approx(expected_equal.tolist())
    assert scores["logistic_regression"].isna().all()
    _unique_ranks(scores)


def test_score_baselines_logistic_ordered_with_labels() -> None:
    features = _features()
    frame = features.frame
    labels = pd.Series(
        (frame["momentum_20d"] > 0).astype(int).tolist(),
        index=frame["stock_id"],
        name="2019-12-31",
    )
    scores = score_baselines(features, labels)
    top = scores.loc[scores["logistic_regression"].idxmax(), "stock_id"]
    bottom = scores.loc[scores["logistic_regression"].idxmin(), "stock_id"]
    assert frame.set_index("stock_id").loc[top, "momentum_20d"] > 0
    assert frame.set_index("stock_id").loc[bottom, "momentum_20d"] < 0
    _unique_ranks(scores)


def test_score_baselines_missing_leg_yields_nan_blend() -> None:
    features = _features(extra_drop=("momentum_60d",))
    scores = score_baselines(features)
    assert scores["momentum_20d_60d"].isna().all()
    assert scores["equal_weight_rank"].notna().all()
    assert scores["momentum_20d"].notna().all()


def test_score_baselines_rejects_bad_inputs() -> None:
    features = _features()
    with pytest.raises(ValueError, match="momentum_20d"):
        bad = FeatureSet(
            run_id="r",
            as_of="2019-12-31",
            feature_version="factor_v1",
            frame=features.frame.drop(columns=["momentum_20d"]),
            feature_columns=("momentum_60d",),
            coverage={"momentum_60d": 1.0},
            missing_flag_column="missing_flag",
        )
        score_baselines(bad)
    with pytest.raises(ValueError, match="unknown stock"):
        score_baselines(features, pd.Series([1], index=["ZZZ"]))
    with pytest.raises(ValueError, match="0/1"):
        score_baselines(
            features,
            pd.Series([2] * len(features.frame), index=features.frame["stock_id"]),
        )
    _ = date(2019, 12, 31)
