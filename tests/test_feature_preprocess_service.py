"""P2-05 acceptance: cross-sectional preprocessing service."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from contracts import FeatureSet
from services.feature_preprocess_service import preprocess_features
from services.feature_service import FACTOR_COLUMNS, FACTOR_COLUMNS_PIT_V3, FACTOR_COLUMNS_V4

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


def test_stable_schema_keeps_factors_and_missing_indicators(settings) -> None:
    stable_settings = replace(
        settings,
        features=replace(settings.features, feature_version="factor_adj_pit_v3_stable"),
    )
    raw = _raw().rename(
        columns={
            "foreign_net_buy_float": "foreign_net_buy_to_issued_shares",
            "trust_net_buy_float": "trust_net_buy_to_issued_shares",
        }
    )
    raw.loc[0, "momentum_20d"] = np.nan
    raw["dividend_yield"] = np.nan

    result = preprocess_features(raw, stable_settings, "run-stable", AS_OF)

    assert len(result.feature_columns) == 2 * len(FACTOR_COLUMNS_PIT_V3)
    assert set(FACTOR_COLUMNS_PIT_V3).issubset(result.frame.columns)
    assert result.frame["momentum_20d__missing"].iloc[0] == 1.0
    assert result.frame["momentum_20d__missing"].iloc[1] == 0.0
    assert result.frame["dividend_yield__missing"].eq(1.0).all()
    assert result.frame[list(result.feature_columns)].isna().sum().sum() == 0
    assert result.coverage["momentum_20d"] == pytest.approx(79 / 80)


def test_stable_schema_does_not_correlate_deduplicate(settings) -> None:
    stable_settings = replace(
        settings,
        features=replace(settings.features, feature_version="factor_adj_pit_v3_stable"),
    )
    raw = _raw().rename(
        columns={
            "foreign_net_buy_float": "foreign_net_buy_to_issued_shares",
            "trust_net_buy_float": "trust_net_buy_to_issued_shares",
        }
    )
    raw["price_ma20_gap"] = raw["momentum_20d"]

    result = preprocess_features(raw, stable_settings, "run-stable", AS_OF)

    assert "momentum_20d" in result.feature_columns
    assert "price_ma20_gap" in result.feature_columns


def test_factor_v4_has_base_factors_only_and_median_fills(settings) -> None:
    v4_settings = replace(
        settings, features=replace(settings.features, feature_version="factor_v4")
    )
    raw = _raw().rename(
        columns={
            "foreign_net_buy_float": "foreign_net_buy_to_issued_shares",
            "trust_net_buy_float": "trust_net_buy_to_issued_shares",
        }
    )
    raw.loc[0, "momentum_20d"] = np.nan

    result = preprocess_features(raw, v4_settings, "run-v4", AS_OF)

    assert result.feature_columns == FACTOR_COLUMNS_V4
    assert "missing_flag" not in result.frame.columns
    assert not any(column.endswith("__missing") for column in result.frame.columns)
    assert result.frame[list(FACTOR_COLUMNS_V4)].isna().sum().sum() == 0
    assert result.coverage["momentum_20d"] == pytest.approx(79 / 80)


def test_factor_v4_rejects_whole_month_source_outage(settings) -> None:
    v4_settings = replace(
        settings, features=replace(settings.features, feature_version="factor_v4")
    )
    raw = _raw().rename(
        columns={
            "foreign_net_buy_float": "foreign_net_buy_to_issued_shares",
            "trust_net_buy_float": "trust_net_buy_to_issued_shares",
        }
    )
    raw["beta_60d"] = np.nan
    with pytest.raises(ValueError, match="source outage.*beta_60d"):
        preprocess_features(raw, v4_settings, "run-v4", AS_OF)
