from __future__ import annotations

import pandas as pd
import pytest

from services.feature_selection_service import select_stable_training_features


def _panel(coverage: float, factors: list[str]) -> dict:
    columns = [*factors, *(f"{factor}__missing" for factor in factors)]
    frame = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    return {
        "feature_columns": columns,
        "feature_coverage": {factor: coverage for factor in factors},
        "frame": frame,
    }


def test_selects_features_using_only_training_month_coverage() -> None:
    factors = [f"f{i}" for i in range(9)]
    months = [f"2020-{month:02d}" for month in range(1, 11)]
    panels = {month: _panel(1.0, factors) for month in months}
    panels[months[0]]["feature_coverage"]["f0"] = 0.79
    panels[months[1]]["feature_coverage"]["f0"] = 0.79
    panels[months[2]]["feature_coverage"]["f0"] = 0.79

    columns, diagnostics = select_stable_training_features(panels, months)

    assert "f0" not in columns
    assert all(factor in columns for factor in factors[1:])
    assert diagnostics["f0"]["qualified_months"] == 7
    assert diagnostics["f1"]["eligible"] is True


def test_rejects_training_period_with_too_few_stable_factors() -> None:
    factors = [f"f{i}" for i in range(8)]
    months = ["2020-01", "2020-02"]
    panels = {month: _panel(0.0, factors) for month in months}

    with pytest.raises(ValueError, match="minimum is 8"):
        select_stable_training_features(panels, months)
