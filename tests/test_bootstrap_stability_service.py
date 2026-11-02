from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.bootstrap_stability_service import (
    MonthBlock,
    calculate_path_metrics,
    extract_month_blocks,
    sample_month_indices,
    simulate_paths,
    summarize_simulations,
)


def _blocks(count: int = 72) -> list[MonthBlock]:
    return [
        MonthBlock(
            source_index=index,
            run_id=f"run-{index // 12}",
            month=f"month-{index:02d}",
            start_date="2020-01-01",
            end_date="2020-01-02",
            daily_returns=np.array([0.01 + index / 100_000, -0.002]),
            artifact_monthly_return=(1.01 + index / 100_000) * 0.998 - 1,
            artifact_validation_return=(1.01 + index / 100_000) * 0.998 - 1,
        )
        for index in range(count)
    ]


def test_extract_month_blocks_includes_first_day_against_initial_capital() -> None:
    blocks = extract_month_blocks(
        run_id="run",
        equity_curve=[
            {"date": "2020-02-03", "nav": 101.0},
            {"date": "2020-02-04", "nav": 99.99},
            {"date": "2020-03-02", "nav": 102.0},
        ],
        monthly_returns=[
            {"month": "2020-02", "return": -0.01},
            {"month": "2020-03", "return": 0.0},
        ],
        expected_months=["2020-02", "2020-03"],
        initial_capital=100.0,
    )
    assert blocks[0].daily_returns.tolist() == pytest.approx([0.01, -0.01])
    assert blocks[0].compounded_return == pytest.approx(-0.0001)
    assert blocks[0].artifact_validation_return == pytest.approx(-0.01)
    assert blocks[1].daily_returns.tolist() == pytest.approx([102.0 / 99.99 - 1.0])
    assert blocks[1].artifact_validation_return == pytest.approx(0.0)


def test_calculate_path_metrics_compounds_and_includes_starting_nav_in_mdd() -> None:
    metrics = calculate_path_metrics([0.10, -0.20, 0.05])
    assert metrics["cumulative_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1.0)
    assert metrics["max_drawdown"] == pytest.approx(-0.20)
    assert metrics["trading_days"] == 3


def test_sampling_is_reproducible_and_circular_blocks_are_contiguous() -> None:
    first = sample_month_indices(72, 5000, "monthly_iid", seed=42)
    second = sample_month_indices(72, 5000, "monthly_iid", seed=42)
    assert np.array_equal(first, second)
    assert first.shape == (5000, 12)
    circular = sample_month_indices(72, 5000, "circular_3m", seed=42)
    assert circular.shape == (5000, 12)
    triples = circular.reshape(-1, 4, 3)
    assert np.all((triples[:, :, 1] - triples[:, :, 0]) % 72 == 1)
    assert np.all((triples[:, :, 2] - triples[:, :, 1]) % 72 == 1)


def test_simulation_records_twelve_slots_and_summary_convergence_sizes() -> None:
    paths, indices = simulate_paths(_blocks(), 5000, "monthly_iid", seed=42)
    assert len(paths) == 5000
    assert len(indices) == 60_000
    assert indices.groupby("simulation_id")["slot"].count().eq(12).all()
    second_paths, _ = simulate_paths(_blocks(), 5000, "circular_3m", seed=42)
    summary = summarize_simulations(pd.concat([paths, second_paths], ignore_index=True))
    assert set(summary["sample_size"]) == {500, 2000, 5000}
    assert set(summary["method"]) == {"monthly_iid", "circular_3m"}
    assert (summary.loc[summary["metric"] != "event_probabilities", "valid_count"] > 0).all()
