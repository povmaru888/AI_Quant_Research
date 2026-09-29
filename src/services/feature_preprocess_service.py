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
from services.feature_service import FACTOR_COLUMNS, FACTOR_COLUMNS_PIT_V3
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
        if settings.features.feature_version == "factor_adj_pit_v3"
        else FACTOR_COLUMNS
    )
    missing = [c for c in ("stock_id", "missing_flag", *factor_columns) if c not in raw.columns]
    if missing:
        raise ValueError(f"invalid raw: missing columns {missing}")

    features = settings.features
    work = raw[["stock_id", "missing_flag", *factor_columns]].copy()
    for column in factor_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    coverage = {column: float(work[column].notna().mean()) for column in factor_columns}
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
