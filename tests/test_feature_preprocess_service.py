"""P2-05 acceptance: cross-sectional preprocessing service."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from contracts import FeatureSet
from services.feature_preprocess_service import preprocess_features
from services.feature_service import FACTOR_COLUMNS

AS_OF = date(2019, 12, 31)


def _raw(n: int = 80, seed: int = 42, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "stock_id": [f"S{i:04d}" for i in range(n)],
        "missing_flag": np.zeros(n, dtype=int),
    }
    for column in FACTOR_COLUMNS:
        data[column] = rng.normal(size=n)
    data.update(overrides)
    return pd.DataFrame(data)


def test_preprocess_returns_feature_set(settings) -> None:
    result = preprocess_features(_raw(), settings, "run-001", AS_OF)
    assert isinstance(result, FeatureSet)
    assert result.as_of == "2019-12-31"
    assert result.feature_version == settings.features.feature_version
    assert set(result.coverage) == set(result.feature_columns)
    assert result.frame["stock_id"].tolist() == [f"S{i:04d}" for i in range(80)]
    assert result.frame[list(result.feature_columns)].isna().sum().sum() == 0


def test_preprocess_median_fill_and_coverage(settings) -> None:
    raw = _raw()
    raw.loc[0, "momentum_20d"] = np.nan
    raw["dividend_yield"] = np.nan
    result = preprocess_features(raw, settings, "run-001", AS_OF)
    assert result.coverage["momentum_20d"] == pytest.approx(79 / 80)
    assert "dividend_yield" not in result.feature_columns


def test_preprocess_winsorize_collapses_outliers(settings) -> None:
    raw = _raw(n=200)
    raw["momentum_20d"] = np.linspace(0, 1, 200)
    raw.loc[198, "momentum_20d"] = 1e6
    raw.loc[199, "momentum_20d"] = 1e9
    result = preprocess_features(raw, settings, "run-001", AS_OF)
    col = result.frame["momentum_20d"]
    assert col.iloc[198] == pytest.approx(col.iloc[199])


def test_preprocess_dedup_correlation(settings) -> None:
    result = preprocess_features(_raw(), settings, "run-001", AS_OF)
    kept = list(result.feature_columns)
    assert len(kept) >= 2
    corr = result.frame[kept].corr().abs().to_numpy()
    off_diag = corr[~np.eye(len(kept), dtype=bool)]
    assert off_diag.max() <= settings.features.correlation_threshold + 1e-9


def test_preprocess_rejects_bad_inputs(settings) -> None:
    with pytest.raises(ValueError, match="run_id"):
        preprocess_features(_raw(), settings, " ", AS_OF)
    with pytest.raises(ValueError, match="missing columns"):
        preprocess_features(_raw().drop(columns=["momentum_20d"]), settings, "r", AS_OF)
    raw = _raw()
    raw[list(FACTOR_COLUMNS)] = np.nan
    with pytest.raises(ValueError, match="all-NaN"):
        preprocess_features(raw, settings, "r", AS_OF)
