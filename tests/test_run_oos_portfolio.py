from __future__ import annotations

import pandas as pd
import pytest

from tools.run_oos_portfolio import _smooth_percentile_scores


def _scores(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["A", "B", "C"],
            "probability": values,
            "rank": [1, 2, 3],
        }
    )


def test_percentile_smoothing_blends_only_stocks_seen_last_month() -> None:
    first, state = _smooth_percentile_scores(_scores([0.9, 0.5, 0.1]), {}, 0.6)
    assert first.set_index("stock_id")["probability"].to_dict() == pytest.approx(
        {"A": 1.0, "B": 2 / 3, "C": 1 / 3}
    )
    second_input = pd.DataFrame(
        {"stock_id": ["A", "B", "D"], "probability": [0.1, 0.5, 0.9], "rank": [3, 2, 1]}
    )
    second, state = _smooth_percentile_scores(second_input, state, 0.6)
    values = second.set_index("stock_id")["probability"]
    assert values["A"] == pytest.approx(0.6 * (1 / 3) + 0.4 * 1.0)
    assert values["B"] == pytest.approx(2 / 3)
    assert values["D"] == pytest.approx(1.0)
    assert set(state) == {"A", "B", "D"}


def test_alpha_one_preserves_original_scores_and_ranking() -> None:
    original = _scores([0.9, 0.5, 0.1])
    smoothed, _ = _smooth_percentile_scores(original, {"A": 0.1}, 1.0)
    pd.testing.assert_frame_equal(smoothed, original)


def test_smoothing_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        _smooth_percentile_scores(_scores([0.9, 0.5, 0.1]), {}, 0.0)
