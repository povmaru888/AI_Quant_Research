"""Model-selection diagnostics based on each stock's label horizon."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from services.xgb_service import rank_ic

SELECTION_METRICS_SCHEMA_VERSION = 2
SELECTION_TOP_N = 15
SELECTION_HORIZON_TRADING_DAYS = 20

_PREDICTION_COLUMNS = (
    "prediction_date",
    "stock_id",
    "prediction_probability",
    "rank",
)
_PRICE_COLUMNS = ("stock_id", "trade_date", "close_adj")
_SUMMARY_FIELDS = {
    "mean_continuous_rank_ic": "continuous_rank_ic",
    "mean_top15_actual_return": "top15_actual_return",
    "mean_top15_excess_return": "top15_excess_return",
    "mean_top_bottom_spread": "top_bottom_spread",
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _monthly_average(rows: list[dict], field: str) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(field))) is not None]
    return float(np.mean(values)) if values else None


def _positive_month_ratio(rows: list[dict], field: str) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(field))) is not None]
    if not values:
        return None
    return float(sum(value > 0 for value in values) / len(values))


def build_selection_metrics(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_n: int = SELECTION_TOP_N,
    horizon: int = SELECTION_HORIZON_TRADING_DAYS,
) -> dict:
    """Build versioned monthly selection metrics using exact t and t+h bars.

    Eligible stocks have a prediction and positive, finite adjusted closes on
    the exact signal date and the stock's own ``horizon``-th subsequent price
    row. Missing intermediate closes remain in the sequence and do not shorten
    the horizon.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("invalid predictions: must be a DataFrame")
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("invalid prices: must be a DataFrame")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(f"invalid top_n: {top_n!r}")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"invalid horizon: {horizon!r}")

    missing_predictions = [name for name in _PREDICTION_COLUMNS if name not in predictions]
    if missing_predictions:
        raise ValueError(f"invalid predictions: missing columns {missing_predictions}")
    missing_prices = [name for name in _PRICE_COLUMNS if name not in prices]
    if missing_prices:
        raise ValueError(f"invalid prices: missing columns {missing_prices}")

    pred = predictions.loc[:, list(_PREDICTION_COLUMNS)].copy()
    pred["prediction_date"] = pred["prediction_date"].astype(str)
    pred["stock_id"] = pred["stock_id"].astype(str)
    pred["prediction_probability"] = pd.to_numeric(
        pred["prediction_probability"], errors="coerce"
    )
    pred["rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    if pred.duplicated(["prediction_date", "stock_id"]).any():
        raise ValueError("invalid predictions: duplicate prediction_date and stock_id")

    px = prices.loc[:, list(_PRICE_COLUMNS)].copy()
    px["stock_id"] = px["stock_id"].astype(str)
    px["trade_date"] = px["trade_date"].astype(str)
    px["close_adj"] = pd.to_numeric(px["close_adj"], errors="coerce")
    px.loc[px["close_adj"] <= 0, "close_adj"] = np.nan
    px = px.sort_values(["stock_id", "trade_date"], kind="mergesort")
    history = {
        stock_id: (
            rows["trade_date"].to_numpy(dtype=str),
            rows["close_adj"].to_numpy(dtype=float, na_value=np.nan),
        )
        for stock_id, rows in px.groupby("stock_id", sort=False)
    }

    monthly_rows: list[dict] = []
    for signal_date, month_predictions in pred.groupby("prediction_date", sort=True):
        returns: dict[str, float] = {}
        eligible_predictions = []
        for row in month_predictions.itertuples(index=False):
            stock_id = str(row.stock_id)
            stock_history = history.get(stock_id)
            if stock_history is None:
                continue
            dates, closes = stock_history
            start_idx = int(np.searchsorted(dates, signal_date, side="left"))
            end_idx = start_idx + horizon
            if start_idx >= len(dates) or dates[start_idx] != signal_date or end_idx >= len(dates):
                continue
            start, end = closes[start_idx], closes[end_idx]
            probability, model_rank = _finite(row.prediction_probability), _finite(row.rank)
            if (
                probability is None
                or model_rank is None
                or not np.isfinite(start)
                or not np.isfinite(end)
                or start <= 0
                or end <= 0
            ):
                continue
            forward_return = _finite(end / start - 1.0)
            if forward_return is None:
                continue
            returns[stock_id] = forward_return
            eligible_predictions.append(
                {
                    "stock_id": stock_id,
                    "prediction_probability": probability,
                    "rank": model_rank,
                }
            )

        eligible = pd.DataFrame(eligible_predictions)
        eligible_count = len(eligible)
        if eligible_count:
            eligible = eligible.sort_values(
                ["rank", "prediction_probability", "stock_id"],
                ascending=[True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            scores = eligible.set_index("stock_id")["prediction_probability"]
            forward_returns = pd.Series(returns, dtype=float)
            continuous_ic = _finite(rank_ic(scores, forward_returns))
            universe_return = float(np.mean(list(returns.values())))
        else:
            continuous_ic = None
            universe_return = None

        row_payload = {
            "signal_date": str(signal_date),
            "month": str(signal_date)[:7],
            "continuous_rank_ic": continuous_ic,
            "eligible_count": int(eligible_count),
            "top15_actual_return": None,
            "universe_actual_return": universe_return,
            "top15_excess_return": None,
            "bottom15_actual_return": None,
            "top_bottom_spread": None,
        }
        if eligible_count >= 2 * top_n:
            top = eligible.head(top_n)["stock_id"].tolist()
            bottom = eligible.tail(top_n)["stock_id"].tolist()
            top_return = float(np.mean([returns[stock_id] for stock_id in top]))
            bottom_return = float(np.mean([returns[stock_id] for stock_id in bottom]))
            row_payload.update(
                {
                    "top15_actual_return": top_return,
                    "top15_excess_return": top_return - universe_return,
                    "bottom15_actual_return": bottom_return,
                    "top_bottom_spread": top_return - bottom_return,
                }
            )
        monthly_rows.append(row_payload)

    summary = {
        key: _monthly_average(monthly_rows, field)
        for key, field in _SUMMARY_FIELDS.items()
    }
    summary["ic_positive_month_ratio"] = _positive_month_ratio(
        monthly_rows, "continuous_rank_ic"
    )
    summary["top15_excess_positive_month_ratio"] = _positive_month_ratio(
        monthly_rows, "top15_excess_return"
    )
    return {
        "schema_version": SELECTION_METRICS_SCHEMA_VERSION,
        "definition": {
            "top_n": top_n,
            "horizon_trading_days": horizon,
            "price_field": "close_adj",
            "return_type": "simple",
            "signal_price_rule": "exact_signal_date",
            "horizon_price_rule": "stock_trade_row_offset",
        },
        "summary": summary,
        "monthly": monthly_rows,
    }


def validate_selection_metrics(
    payload: dict,
    *,
    minimum_eligible: int = 2 * SELECTION_TOP_N,
) -> list[str]:
    """Return preflight errors; an empty list means the payload is batch-ready."""
    if not isinstance(payload, dict) or payload.get("schema_version") != SELECTION_METRICS_SCHEMA_VERSION:
        return ["unsupported or invalid selection metrics payload"]
    monthly = payload.get("monthly")
    if not isinstance(monthly, list) or not monthly:
        return ["no monthly selection metrics"]
    errors = []
    for row in monthly:
        if not isinstance(row, dict):
            errors.append("invalid monthly row")
            continue
        signal_date = row.get("signal_date", "unknown")
        count = row.get("eligible_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum_eligible:
            errors.append(f"{signal_date}: eligible_count={count!r} (< {minimum_eligible})")
            continue
        required = (
            "continuous_rank_ic",
            "top15_actual_return",
            "universe_actual_return",
            "top15_excess_return",
            "bottom15_actual_return",
            "top_bottom_spread",
        )
        missing = [field for field in required if _finite(row.get(field)) is None]
        if missing:
            errors.append(f"{signal_date}: missing finite metrics {missing}")
    return errors
