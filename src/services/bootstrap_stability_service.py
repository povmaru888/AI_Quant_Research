"""Bootstrap stability analysis for monthly blocks of daily portfolio returns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
BootstrapMethod = Literal["monthly_iid", "circular_3m"]


@dataclass(frozen=True)
class MonthBlock:
    source_index: int
    run_id: str
    month: str
    start_date: str
    end_date: str
    daily_returns: np.ndarray
    artifact_monthly_return: float
    artifact_validation_return: float
    known_data_issue: str | None = None

    @property
    def compounded_return(self) -> float:
        return float(np.prod(1.0 + self.daily_returns) - 1.0)


def extract_month_blocks(
    run_id: str,
    equity_curve: Sequence[dict],
    monthly_returns: Sequence[dict],
    expected_months: Sequence[str],
    initial_capital: float,
    known_issue_months: dict[str, str] | None = None,
) -> list[MonthBlock]:
    """Convert one independently seeded OOS NAV curve into monthly return blocks."""
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    curve = pd.DataFrame(equity_curve)
    if curve.empty or not {"date", "nav"}.issubset(curve.columns):
        raise ValueError(f"invalid equity curve: {run_id}")
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve["nav"] = pd.to_numeric(curve["nav"], errors="coerce")
    if curve[["date", "nav"]].isna().any().any() or (curve["nav"] <= 0).any():
        raise ValueError(f"invalid equity curve values: {run_id}")
    if curve["date"].duplicated().any() or not curve["date"].is_monotonic_increasing:
        raise ValueError(f"equity curve dates must be unique and increasing: {run_id}")

    prior = pd.concat(
        [pd.Series([float(initial_capital)]), curve["nav"].iloc[:-1].reset_index(drop=True)],
        ignore_index=True,
    )
    curve["daily_return"] = curve["nav"].reset_index(drop=True) / prior - 1.0
    curve["month"] = curve["date"].dt.strftime("%Y-%m")
    actual_months = curve["month"].drop_duplicates().tolist()
    if actual_months != list(expected_months):
        raise ValueError(
            f"OOS month mismatch for {run_id}: actual={actual_months}, "
            f"expected={list(expected_months)}"
        )
    reported = {
        str(row["month"]): float(row["return"])
        for row in monthly_returns
        if "month" in row and "return" in row
    }
    if set(reported) != set(expected_months):
        raise ValueError(f"monthly return coverage mismatch: {run_id}")

    blocks: list[MonthBlock] = []
    issues = known_issue_months or {}
    for month, frame in curve.groupby("month", sort=False):
        values = frame["daily_return"].to_numpy(dtype=float)
        # The persisted dashboard monthly-return series is last NAV / first
        # NAV - 1 within every calendar month, so it omits each month's first
        # trading-day return. Validate that convention, while retaining the
        # complete return vector for bootstrap analysis.
        comparable_values = values[1:]
        artifact_validation_return = float(np.prod(1.0 + comparable_values) - 1.0)
        if not np.isclose(
            artifact_validation_return, reported[month], rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                f"monthly return mismatch for {run_id} {month}: "
                f"derived={artifact_validation_return}, artifact={reported[month]}"
            )
        blocks.append(
            MonthBlock(
                source_index=len(blocks),
                run_id=run_id,
                month=month,
                start_date=frame["date"].iloc[0].date().isoformat(),
                end_date=frame["date"].iloc[-1].date().isoformat(),
                daily_returns=values,
                artifact_monthly_return=reported[month],
                artifact_validation_return=artifact_validation_return,
                known_data_issue=issues.get(month),
            )
        )
    return blocks


def calculate_path_metrics(daily_returns: Sequence[float]) -> dict[str, float]:
    """Calculate metrics with the same Sharpe/Sortino convention as the dashboard."""
    returns = np.asarray(daily_returns, dtype=float)
    if returns.size < 2 or not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError("daily_returns need 2+ finite values greater than -1")
    nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    years = returns.size / TRADING_DAYS_PER_YEAR
    cumulative_return = float(nav[-1] - 1.0)
    cagr = float(nav[-1] ** (1.0 / years) - 1.0)
    standard_deviation = float(returns.std(ddof=1))
    downside = returns[returns < 0]
    downside_deviation = float(downside.std(ddof=1)) if downside.size >= 2 else np.nan
    return {
        "trading_days": int(returns.size),
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "sharpe": (
            float(returns.mean() / standard_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
            if standard_deviation > 0
            else np.nan
        ),
        "sortino": (
            float(returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(downside_deviation) and downside_deviation > 0
            else np.nan
        ),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
    }


def sample_month_indices(
    month_count: int,
    simulations: int,
    method: BootstrapMethod,
    seed: int,
) -> np.ndarray:
    """Return one 12-month source-index row per simulated year."""
    if month_count < 12 or simulations <= 0:
        raise ValueError("month_count must be >=12 and simulations must be positive")
    rng = np.random.default_rng(seed)
    if method == "monthly_iid":
        return rng.integers(0, month_count, size=(simulations, 12))
    if method == "circular_3m":
        starts = rng.integers(0, month_count, size=(simulations, 4))
        offsets = np.arange(3)
        return ((starts[:, :, None] + offsets) % month_count).reshape(simulations, 12)
    raise ValueError(f"unsupported bootstrap method: {method}")


def simulate_paths(
    blocks: Sequence[MonthBlock],
    simulations: int,
    method: BootstrapMethod,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate annual paths and return path metrics plus auditable sampled indices."""
    if [block.source_index for block in blocks] != list(range(len(blocks))):
        raise ValueError("month blocks must have contiguous source_index values")
    indices = sample_month_indices(len(blocks), simulations, method, seed)
    metric_rows: list[dict] = []
    index_rows: list[dict] = []
    for simulation_id, selected in enumerate(indices, start=1):
        path = np.concatenate([blocks[int(index)].daily_returns for index in selected])
        metric_rows.append(
            {
                "method": method,
                "simulation_id": simulation_id,
                "seed": seed,
                **calculate_path_metrics(path),
            }
        )
        for slot, source_index in enumerate(selected, start=1):
            block = blocks[int(source_index)]
            index_rows.append(
                {
                    "method": method,
                    "simulation_id": simulation_id,
                    "slot": slot,
                    "source_index": int(source_index),
                    "source_run_id": block.run_id,
                    "source_month": block.month,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(index_rows)


def summarize_simulations(paths: pd.DataFrame) -> pd.DataFrame:
    """Build convergence-aware distribution summaries for 500, 2000, and 5000 paths."""
    rows: list[dict] = []
    metric_names = ("cumulative_return", "cagr", "sharpe", "sortino", "max_drawdown")
    for method, method_frame in paths.groupby("method", sort=False):
        for sample_size in (500, 2000, 5000):
            frame = method_frame.loc[method_frame["simulation_id"] <= sample_size]
            if len(frame) != sample_size:
                raise ValueError(f"missing simulations for {method} at {sample_size}")
            for metric in metric_names:
                values = pd.to_numeric(frame[metric], errors="coerce").dropna()
                rows.append(
                    {
                        "method": method,
                        "sample_size": sample_size,
                        "metric": metric,
                        "valid_count": len(values),
                        "na_count": len(frame) - len(values),
                        "mean": values.mean(),
                        "median": values.median(),
                        "std": values.std(ddof=1),
                        "p05": values.quantile(0.05),
                        "p25": values.quantile(0.25),
                        "p75": values.quantile(0.75),
                        "p95": values.quantile(0.95),
                    }
                )
            rows.append(
                {
                    "method": method,
                    "sample_size": sample_size,
                    "metric": "event_probabilities",
                    "valid_count": len(frame),
                    "na_count": 0,
                    "prob_cumulative_return_lt_0": (frame["cumulative_return"] < 0).mean(),
                    "prob_sharpe_lt_0": (frame["sharpe"] < 0).mean(),
                    "prob_mdd_le_minus_10pct": (frame["max_drawdown"] <= -0.10).mean(),
                    "prob_mdd_le_minus_20pct": (frame["max_drawdown"] <= -0.20).mean(),
                    "prob_mdd_le_minus_30pct": (frame["max_drawdown"] <= -0.30).mean(),
                }
            )
    return pd.DataFrame(rows)
