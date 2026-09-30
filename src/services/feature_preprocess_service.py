"""P2-05: cross-sectional preprocessing service (SDD 9.1, 9.3).

Single-month pipeline: median fill -> winsorize -> rank -> standardize
-> correlation dedup. Everything is computed within the input month, so
no future distribution can leak in.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from contracts import FeatureSet
from services.feature_service import (
    FACTOR_COLUMNS,
    FACTOR_COLUMNS_PIT_V3,
    is_pit_v3_feature_version,
    uses_stable_feature_schema,
)
from settings import Settings


def preprocess_features(
    raw: pd.DataFrame,
    settings: Settings,
    run_id: str,
    as_of: date,
) -> FeatureSet:
    """Preprocess one month of raw factors into a model-ready ``FeatureSet``."""
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("invalid raw: must be a non-empty DataFrame")
    factor_columns = (
        FACTOR_COLUMNS_PIT_V3
        if is_pit_v3_feature_version(settings.features.feature_version)
        else FACTOR_COLUMNS
    )
    missing = [c for c in ("stock_id", "missing_flag", *factor_columns) if c not in raw.columns]
    if missing:
        raise ValueError(f"invalid raw: missing columns {missing}")

    features = settings.features
    work = raw[["stock_id", "missing_flag", *factor_columns]].copy()
    for column in factor_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work[list(factor_columns)] = work[list(factor_columns)].replace([np.inf, -np.inf], np.nan)

    coverage = {column: float(work[column].notna().mean()) for column in factor_columns}
    if uses_stable_feature_schema(settings.features.feature_version):
        return _preprocess_stable(work, factor_columns, coverage, settings, run_id, as_of)

    usable = [c for c in factor_columns if coverage[c] > 0]
    if not usable:
        raise ValueError("invalid raw: every factor column is all-NaN")

    filled = work[usable].apply(lambda s: s.fillna(s.median()))
    lower = filled.quantile(features.winsor_lower_quantile)
    upper = filled.quantile(features.winsor_upper_quantile)
    winsored = filled.clip(lower=lower, upper=upper, axis=1)
    ranked = winsored.rank(pct=True)
    means = ranked.mean()
    stds = ranked.std(ddof=1).replace(0, np.nan)
    standardized = ((ranked - means) / stds).fillna(0.0)

    kept = _dedup(list(usable), standardized, features.correlation_threshold)
    frame = pd.concat(
        [work[["stock_id", "missing_flag"]].reset_index(drop=True), standardized[kept]],
        axis=1,
    )
    return FeatureSet(
        run_id=run_id,
        as_of=as_of.isoformat(),
        feature_version=features.feature_version,
        frame=frame,
        feature_columns=tuple(kept),
        coverage={column: coverage[column] for column in kept},
    )


def _preprocess_stable(
    work: pd.DataFrame,
    factor_columns: tuple[str, ...],
    raw_coverage: dict[str, float],
    settings: Settings,
    run_id: str,
    as_of: date,
) -> FeatureSet:
    """Keep every candidate factor and expose row-level source missingness."""
    features = settings.features
    raw_values = work.loc[:, factor_columns]
    missing_masks = raw_values.isna()
    usable = [column for column in factor_columns if raw_coverage[column] > 0]
    standardized = pd.DataFrame(0.0, index=work.index, columns=factor_columns)
    if usable:
        filled = raw_values.loc[:, usable].apply(lambda series: series.fillna(series.median()))
        lower = filled.quantile(features.winsor_lower_quantile)
        upper = filled.quantile(features.winsor_upper_quantile)
        winsored = filled.clip(lower=lower, upper=upper, axis=1)
        ranked = winsored.rank(pct=True)
        stds = ranked.std(ddof=1).replace(0, np.nan)
        standardized.loc[:, usable] = ((ranked - ranked.mean()) / stds).fillna(0.0)

    indicators = missing_masks.astype("float64").rename(
        columns={column: f"{column}__missing" for column in factor_columns}
    )
    model_columns = [*factor_columns, *indicators.columns]
    frame = pd.concat(
        [
            work[["stock_id", "missing_flag"]].reset_index(drop=True),
            standardized.reset_index(drop=True),
            indicators.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_coverage = dict(raw_coverage)
    feature_coverage.update(
        {f"{column}__missing": 1.0 - raw_coverage[column] for column in factor_columns}
    )
    return FeatureSet(
        run_id=run_id,
        as_of=as_of.isoformat(),
        feature_version=features.feature_version,
        frame=frame,
        feature_columns=tuple(model_columns),
        coverage=feature_coverage,
    )


def _dedup(columns: list[str], frame: pd.DataFrame, threshold: float) -> list[str]:
    """Greedily keep the first of any pair with |corr| above threshold."""
    if len(columns) < 2:
        return columns
    corr = frame[columns].corr().abs()
    dropped: set[str] = set()
    for i, first in enumerate(columns):
        if first in dropped:
            continue
        for second in columns[i + 1 :]:
            if second not in dropped and corr.loc[first, second] > threshold:
                dropped.add(second)
    return [c for c in columns if c not in dropped]
