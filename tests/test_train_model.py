from __future__ import annotations

import pytest

from tools.train_model import _apply_feature_exclusions


def test_apply_feature_exclusions_preserves_order() -> None:
    assert _apply_feature_exclusions(
        ["momentum_120d", "beta_60d", "revenue_yoy"], ["beta_60d"]
    ) == ["momentum_120d", "revenue_yoy"]


def test_apply_feature_exclusions_rejects_unknown_feature() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        _apply_feature_exclusions(["momentum_120d"], ["beta_60d"])


def test_apply_feature_exclusions_rejects_empty_schema() -> None:
    with pytest.raises(ValueError, match="every model feature"):
        _apply_feature_exclusions(["beta_60d"], ["beta_60d"])
