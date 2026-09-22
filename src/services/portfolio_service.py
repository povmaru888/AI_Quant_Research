"""P2-11: rank-buffer target holdings service (SDD 12.1).

Entry rank <= top_n -> BUY; previous holding with rank <= hold
threshold -> HOLD; previous holding ranked above it (or missing from
predictions, e.g. delisted) -> SELL; the rest -> NONE. One action per
stock per signal date. Weights here are equal among BUY/HOLD names at
full exposure; P2-12 reshapes them under risk controls.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from contracts import PortfolioTarget
from settings import Settings


def build_target_holdings(
    predictions: pd.DataFrame,
    previous_positions: pd.DataFrame,
    settings: Settings,
    run_id: str,
    signal_date: date,
) -> PortfolioTarget:
    """Decide BUY/HOLD/SELL/NONE for one signal date."""
    if not isinstance(signal_date, date):
        raise ValueError(f"invalid signal_date: must be a date, got {signal_date!r}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    for name, frame, columns in (
        ("predictions", predictions, ("stock_id", "probability", "rank")),
        ("previous_positions", previous_positions, ("stock_id",)),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"invalid {name}: must be a DataFrame")
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"invalid {name}: missing columns {missing}")
    if predictions["stock_id"].duplicated().any():
        raise ValueError("invalid predictions: duplicate stock_id")

    top_n = settings.portfolio.top_n
    hold_threshold = settings.portfolio.hold_rank_threshold
    ranks = predictions.set_index("stock_id")["rank"].apply(float).to_dict()
    previous = previous_positions["stock_id"].astype(str).tolist()

    actions: dict[str, str] = {}
    for stock_id in predictions["stock_id"].astype(str):
        rank = ranks[stock_id]
        if stock_id in previous:
            actions[stock_id] = "HOLD" if rank <= hold_threshold else "SELL"
        else:
            actions[stock_id] = "BUY" if rank <= top_n else "NONE"
    for stock_id in previous:
        if stock_id not in actions:
            actions[stock_id] = "SELL"

    active = sorted(s for s, action in actions.items() if action in ("BUY", "HOLD"))
    if active:
        weight = 1.0 / len(active)
        weights = {s: weight for s in active}
        exposure, cash = 1.0, 0.0
    else:
        weights, exposure, cash = {}, 0.0, 1.0
    return PortfolioTarget(
        run_id=run_id,
        signal_date=signal_date.isoformat(),
        top_n=top_n,
        actions=actions,
        weights=weights,
        cash_weight=cash,
        equity_exposure=exposure,
    )
