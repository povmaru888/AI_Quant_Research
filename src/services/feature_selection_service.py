"""Training-only feature eligibility for fixed-schema panels."""

from __future__ import annotations

from math import ceil
from typing import Mapping

import pandas as pd


DEFAULT_ROW_COVERAGE = 0.80
DEFAULT_MONTH_FRACTION = 0.90
DEFAULT_MINIMUM_FACTORS = 8


def select_stable_training_features(
    panels: Mapping[str, dict],
    months: list[str],
    *,
    row_coverage_threshold: float = DEFAULT_ROW_COVERAGE,
    month_fraction_threshold: float = DEFAULT_MONTH_FRACTION,
    minimum_factors: int = DEFAULT_MINIMUM_FACTORS,
) -> tuple[list[str], dict[str, dict[str, float | int | bool]]]:
    """Select factors using training panels only and keep useful missing flags.

    A base factor is eligible when at least ``month_fraction_threshold`` of
    training months have ``row_coverage_threshold`` or better coverage. An
    eligible factor's missingness flag is included only if training data
    actually contains missing observations for it.
    """
    if not months:
        raise ValueError("training months must not be empty")
    if not 0.0 < row_coverage_threshold <= 1.0:
        raise ValueError("row_coverage_threshold must be in (0, 1]")
    if not 0.0 < month_fraction_threshold <= 1.0:
        raise ValueError("month_fraction_threshold must be in (0, 1]")
    if minimum_factors < 1:
        raise ValueError("minimum_factors must be positive")

    first = panels[months[0]]
    schema = [column for column in first.get("feature_columns", []) if not column.endswith("__missing")]
    if not schema:
        raise ValueError(f"{months[0]} has no base factor schema")
    required_months = ceil(len(months) * month_fraction_threshold)
    diagnostics: dict[str, dict[str, float | int | bool]] = {}
    eligible: list[str] = []
    for factor in schema:
        monthly_coverage: list[float] = []
        for month in months:
            panel = panels[month]
            coverage = panel.get("feature_coverage", {}).get(factor)
            if coverage is None:
                raise ValueError(f"{month} has no raw feature coverage for {factor}")
            value = float(coverage)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{month} has invalid feature coverage for {factor}: {value}")
            if factor not in panel["frame"].columns:
                raise ValueError(f"{month} fixed schema is missing factor {factor}")
            indicator = f"{factor}__missing"
            if indicator not in panel["frame"].columns:
                raise ValueError(f"{month} fixed schema is missing indicator {indicator}")
            monthly_coverage.append(value)
        qualified_months = sum(value >= row_coverage_threshold for value in monthly_coverage)
        is_eligible = qualified_months >= required_months
        diagnostics[factor] = {
            "eligible": is_eligible,
            "qualified_months": qualified_months,
            "training_months": len(months),
            "required_months": required_months,
            "mean_row_coverage": sum(monthly_coverage) / len(monthly_coverage),
            "min_row_coverage": min(monthly_coverage),
        }
        if is_eligible:
            eligible.append(factor)

    if len(eligible) < minimum_factors:
        raise ValueError(
            f"only {len(eligible)} factors meet training coverage; "
            f"minimum is {minimum_factors}"
        )

    selected = list(eligible)
    for factor in eligible:
        indicator = f"{factor}__missing"
        has_missing = any(
            float(panels[month]["feature_coverage"][factor]) < 1.0 for month in months
        )
        diagnostics[factor]["missing_indicator_included"] = has_missing
        if has_missing:
            selected.append(indicator)
    return selected, diagnostics
