"""Calibrate Yahoo adjusted prices onto the existing FinMind price basis."""

from __future__ import annotations

import numpy as np
import pandas as pd


ADJUSTED_COLUMNS = ("open_adj", "high_adj", "low_adj", "close_adj")


def calibrate_adjusted_prices(
    yahoo_prices: pd.DataFrame,
    reference_closes: pd.DataFrame,
    *,
    max_relative_spread: float = 0.001,
    minimum_anchors: int = 3,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int | bool]]]:
    """Apply a stable per-stock scale from overlapping adjusted closes.

    Yahoo and FinMind can use different absolute adjustment bases while
    preserving the same adjusted return series. Only stocks whose overlap
    ratios stay within ``max_relative_spread`` are returned.
    """
    required_yahoo = {"stock_id", "trade_date", *ADJUSTED_COLUMNS}
    required_reference = {"stock_id", "trade_date", "close_adj"}
    if not required_yahoo.issubset(yahoo_prices.columns):
        raise ValueError(f"Yahoo adjusted prices missing {sorted(required_yahoo - set(yahoo_prices.columns))}")
    if not required_reference.issubset(reference_closes.columns):
        raise ValueError(
            f"reference adjusted closes missing {sorted(required_reference - set(reference_closes.columns))}"
        )
    if max_relative_spread < 0 or minimum_anchors < 1:
        raise ValueError("invalid calibration thresholds")

    yahoo = yahoo_prices.loc[:, ["stock_id", "trade_date", *ADJUSTED_COLUMNS]].copy()
    reference = reference_closes.loc[:, ["stock_id", "trade_date", "close_adj"]].copy()
    for frame in (yahoo, reference):
        frame["stock_id"] = frame["stock_id"].astype(str)
        frame["trade_date"] = frame["trade_date"].astype(str)
    yahoo["close_adj"] = pd.to_numeric(yahoo["close_adj"], errors="coerce")
    reference["close_adj"] = pd.to_numeric(reference["close_adj"], errors="coerce")
    overlap = yahoo[["stock_id", "trade_date", "close_adj"]].merge(
        reference, on=["stock_id", "trade_date"], suffixes=("_yahoo", "_reference"),
        validate="one_to_one",
    )
    overlap["scale"] = overlap["close_adj_reference"] / overlap["close_adj_yahoo"]
    overlap = overlap.loc[np.isfinite(overlap["scale"]) & overlap["scale"].gt(0)]

    scales: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float | int | bool]] = {}
    for stock_id, group in overlap.groupby("stock_id", sort=False):
        ratios = group["scale"].to_numpy(dtype=float)
        scale = float(np.median(ratios))
        spread = float((np.quantile(ratios, 0.99) - np.quantile(ratios, 0.01)) / scale)
        accepted = len(ratios) >= minimum_anchors and spread <= max_relative_spread
        diagnostics[str(stock_id)] = {
            "accepted": accepted,
            "anchor_count": len(ratios),
            "scale": scale,
            "relative_p99_p01_spread": spread,
        }
        if accepted:
            scales[str(stock_id)] = scale

    adjusted = yahoo.loc[yahoo["stock_id"].isin(scales)].copy()
    adjusted["scale"] = adjusted["stock_id"].map(scales)
    adjusted.loc[:, list(ADJUSTED_COLUMNS)] = adjusted.loc[:, list(ADJUSTED_COLUMNS)].multiply(
        adjusted["scale"], axis=0
    )
    adjusted = adjusted.drop(columns="scale")
    return adjusted.reset_index(drop=True), diagnostics
