from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRanker

from services.ranking_service import _validate_groups, train_xgb_ranker
from services.xgb_service import predict_xgb


def test_rank_pairwise_uses_month_groups_and_predicts_scores() -> None:
    rng = np.random.default_rng(7)
    train_x = pd.DataFrame({"f": rng.normal(size=24)})
    train_y = pd.Series(([0, 0, 0, 1, 1, 1] * 4), dtype=int)
    valid_x = pd.DataFrame({"f": rng.normal(size=12)})
    valid_y = pd.Series(([0, 0, 0, 1, 1, 1] * 2), dtype=int)
    artifact, booster = train_xgb_ranker(
        train_x, train_y, [6] * 4, valid_x, valid_y, [6] * 2,
        {"n_estimators": 5}, "ranker", "factor_v4", "p", "r",
    )
    assert isinstance(booster, XGBRanker)
    features = valid_x.copy()
    features.insert(0, "stock_id", [f"S{i}" for i in range(len(features))])
    scored = predict_xgb(artifact, features, booster)
    assert scored["rank"].is_unique
    assert np.isfinite(scored["probability"]).all()
    assert scored["probability"].between(0.0, 1.0).all()


def test_rank_pairwise_rejects_invalid_group_total() -> None:
    with pytest.raises(ValueError, match="must sum"):
        _validate_groups("groups", [2, 2], 5)
