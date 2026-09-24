"""P2-07: baseline model service (SDD 10.3).

Four reference scores for one month of processed features: single
momentum, blended momentum, equal-weight factor rank, and logistic
regression. Baselines are robust to dropped columns: the blended score
goes NaN only when a leg is missing, while the equal-weight score
averages whatever columns survived preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from contracts import FeatureSet

BASELINE_COLUMNS: tuple[str, ...] = (
    "momentum_20d",
    "momentum_20d_60d",
    "equal_weight_rank",
    "logistic_regression",
)


def score_baselines(features: FeatureSet, labels: pd.Series | None = None) -> pd.DataFrame:
    """Score one month; ``labels`` (indexed by stock_id) enables the logit leg."""
    frame = features.frame
    if "momentum_20d" not in frame.columns:
        raise ValueError("invalid features: missing 'momentum_20d' column")
    kept = [c for c in features.feature_columns if c in frame.columns]
    if not kept:
        raise ValueError("invalid features: no feature columns in frame")
    if labels is not None:
        unknown = set(labels.index) - set(frame["stock_id"])
        if unknown:
            raise ValueError(f"invalid labels: unknown stock ids {sorted(unknown)[:5]}")
        if set(labels.unique()) - {0, 1}:
            raise ValueError("invalid labels: must contain only 0/1")

    out = pd.DataFrame({"stock_id": frame["stock_id"].tolist()})
    out["momentum_20d"] = pd.to_numeric(frame["momentum_20d"], errors="coerce").to_numpy()
    if "momentum_60d" in frame.columns:
        out["momentum_20d_60d"] = (
            out["momentum_20d"] + pd.to_numeric(frame["momentum_60d"], errors="coerce").to_numpy()
        )
    else:
        out["momentum_20d_60d"] = np.nan
    out["equal_weight_rank"] = (
        frame[kept].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    )
    out["logistic_regression"] = _logit_scores(frame[kept], frame["stock_id"], labels)
    return out


def _logit_scores(
    matrix: pd.DataFrame, stock_ids: pd.Series, labels: pd.Series | None
) -> np.ndarray:
    if labels is None:
        return np.full(len(matrix), np.nan)
    aligned = labels.reindex(stock_ids.tolist())
    if aligned.isna().any() or aligned.nunique() < 2:
        return np.full(len(matrix), np.nan)
    values = matrix.to_numpy(dtype=float)
    model = LogisticRegression(max_iter=1000)
    model.fit(values, aligned.to_numpy(dtype=int))
    return model.predict_proba(values)[:, 1]
