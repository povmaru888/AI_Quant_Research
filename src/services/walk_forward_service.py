"""P2-10: walk-forward service (SDD section 11).

Month-keyed folds: Train 48 months -> Validation 12 months -> Purge
(~20 trading days, approximated by dropping the last training month) ->
Test 12 months. Whole months always sit in a single split (SDD section
2, principle 2). Test months only ever reach ``predict_xgb``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
from xgboost import XGBClassifier

from contracts import FeatureSet, ModelArtifact
from services.optimization_service import optimize_xgb
from services.xgb_service import predict_xgb
from settings import Settings

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_TRAIN_MONTHS = 48
_VALID_MONTHS = 12
_TEST_MONTHS = 12
_PURGE_MONTHS = 1


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, keyed by YYYY-MM months."""

    train_months: tuple[str, ...]
    valid_months: tuple[str, ...]
    test_months: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class WalkForwardResult:
    """Per-fold outcome: tuned model plus test-month OOS predictions."""

    fold: Fold
    artifact: ModelArtifact
    booster: XGBClassifier
    oos_predictions: pd.DataFrame


def _require_month(value: str) -> str:
    if not isinstance(value, str) or not _MONTH_RE.match(value):
        raise ValueError(f"invalid month: {value!r}, expected YYYY-MM")
    return value


def build_folds(months: Sequence[str], settings: Settings) -> list[Fold]:
    """Slide 48/12/12 month windows over the sorted month axis."""
    ordered = sorted({_require_month(m) for m in months})
    train_years = settings.validation.train_years
    valid_years = settings.validation.validation_years
    test_years = settings.validation.test_years
    train_span = train_years * 12
    valid_span = valid_years * 12
    test_span = test_years * 12
    folds: list[Fold] = []
    step = test_span
    start = 0
    while start + train_span + valid_span + test_span <= len(ordered):
        train_end = start + train_span
        valid_end = train_end + valid_span
        test_end = valid_end + test_span
        folds.append(
            Fold(
                train_months=tuple(ordered[start:train_end]),
                valid_months=tuple(ordered[train_end:valid_end]),
                test_months=tuple(ordered[valid_end:test_end]),
            )
        )
        start += step
    return folds


def _common_columns(
    months: Sequence[str], features_by_month: dict[str, FeatureSet]
) -> tuple[str, ...]:
    """Intersection of feature columns across months, in first-month order."""
    ordered = [features_by_month[m].feature_columns for m in months]
    first = list(ordered[0])
    rest = [set(other) for other in ordered[1:]]
    common = tuple(c for c in first if all(c in other for other in rest))
    if not common:
        raise ValueError(f"no common feature columns across months {list(months)[:3]}...")
    return common


def run_walk_forward(
    features_by_month: dict[str, FeatureSet],
    labels_by_month: dict[str, pd.Series],
    settings: Settings,
    model_version: str,
    feature_version: str,
    parameter_version: str,
    run_id: str,
    n_trials: int | None = None,
    optimize: Callable[..., tuple[ModelArtifact, XGBClassifier, pd.DataFrame]] = optimize_xgb,
) -> list[WalkForwardResult]:
    """Tune per fold on train/valid, predict test months, concatenate OOS."""
    months = sorted(set(features_by_month) & set(labels_by_month))
    folds = build_folds(months, settings)
    results: list[WalkForwardResult] = []
    for index, fold in enumerate(folds):
        # Purge: drop the training month adjacent to validation (~20 trading days).
        purged_train = fold.train_months[:-_PURGE_MONTHS] if fold.train_months else ()
        span = (*purged_train, *fold.valid_months, *fold.test_months)
        common = _common_columns(span, features_by_month)
        train_x = pd.concat(
            [features_by_month[m].frame[list(common)] for m in purged_train],
            ignore_index=True,
        )
        train_y = pd.concat([labels_by_month[m] for m in purged_train], ignore_index=True)
        valid_x = pd.concat(
            [features_by_month[m].frame[list(common)] for m in fold.valid_months],
            ignore_index=True,
        )
        valid_y = pd.concat([labels_by_month[m] for m in fold.valid_months], ignore_index=True)
        artifact, booster, _ = optimize(
            train_x,
            train_y,
            valid_x,
            valid_y,
            settings,
            f"{model_version}_f{index}",
            feature_version,
            parameter_version,
            run_id,
            n_trials=n_trials,
        )
        oos_parts = []
        for month in fold.test_months:
            month_features = features_by_month[month]
            scored = predict_xgb(
                artifact,
                month_features.frame,
                booster,
            )
            scored["prediction_date"] = month_features.as_of
            oos_parts.append(scored)
        oos = pd.concat(oos_parts, ignore_index=True)
        results.append(
            WalkForwardResult(fold=fold, artifact=artifact, booster=booster, oos_predictions=oos)
        )
    return results
