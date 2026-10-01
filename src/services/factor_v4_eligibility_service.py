"""Complete-case universe gates for the factor_v4 research contract."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

from contracts import UniverseEntry, UniverseSnapshot
from settings import Settings

_TAIEX_ID = "TAIEX"


def filter_to_official_sessions(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop vendor pseudo-bars on dates absent from the TAIEX calendar."""
    market_mask = prices["stock_id"].astype(str).eq(_TAIEX_ID)
    traded = pd.to_numeric(prices.get("traded_value"), errors="coerce")
    active_days = set(prices.loc[~market_mask & traded.gt(0), "trade_date"].astype(str))
    calendar = set(prices.loc[market_mask, "trade_date"].astype(str)) & active_days
    if not calendar:
        raise ValueError("factor_v4 requires a non-empty TAIEX trading calendar")
    equity_mask = prices["trade_date"].astype(str).isin(calendar)
    return prices.loc[market_mask | equity_mask].copy().reset_index(drop=True)


def labelable_ids_from_prices(
    prices: pd.DataFrame, stock_ids: tuple[str, ...] | list[str], as_of: date, horizon: int
) -> set[str]:
    """Return stocks with valid adjusted prices at exact t and own-row t+horizon."""
    as_of_text = as_of.isoformat()
    targets = {str(stock_id) for stock_id in stock_ids}
    out: set[str] = set()
    for stock_id, group in prices.loc[
        prices["stock_id"].astype(str).isin(targets),
        ["stock_id", "trade_date", "close_adj"],
    ].groupby("stock_id", sort=False):
        ordered = group.sort_values("trade_date")
        days = ordered["trade_date"].astype(str).to_numpy()
        start = int(np.searchsorted(days, as_of_text, side="left"))
        ahead = start + horizon
        if start >= len(days) or days[start] != as_of_text or ahead >= len(days):
            continue
        values = pd.to_numeric(ordered["close_adj"], errors="coerce").to_numpy(dtype=float)
        if (
            np.isfinite(values[start])
            and values[start] > 0
            and np.isfinite(values[ahead])
            and values[ahead] > 0
        ):
            out.add(str(stock_id))
    return out


def filter_factor_v4_universe(
    universe: UniverseSnapshot,
    stocks: pd.DataFrame,
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    as_of: date,
    settings: Settings,
    *,
    labelable_ids: set[str] | None = None,
) -> tuple[UniverseSnapshot, dict[str, object]]:
    """Apply v4 source-completeness gates before any label ranking.

    ``labelable_ids`` is supplied by historical panel construction.  Live
    inference omits it because future prices do not exist yet.
    """
    as_of_text = as_of.isoformat()
    stock_rows = stocks.copy()
    stock_rows["stock_id"] = stock_rows["stock_id"].astype(str)
    stock_info = stock_rows.drop_duplicates("stock_id", keep="last").set_index("stock_id")

    calendar = np.asarray(
        sorted(
            prices.loc[
                prices["stock_id"].astype(str).eq(_TAIEX_ID)
                & prices["trade_date"].astype(str).le(as_of_text),
                "trade_date",
            ]
            .astype(str)
            .unique()
        ),
        dtype="U10",
    )
    price_groups = {
        str(stock_id): group.sort_values("trade_date")
        for stock_id, group in prices.loc[
            prices["stock_id"].astype(str).ne(_TAIEX_ID)
            & prices["trade_date"].astype(str).le(as_of_text)
        ].groupby("stock_id", sort=False)
    }
    eligible_financials = financials.loc[
        financials["available_date"].astype(str).le(as_of_text)
    ].copy()
    sort_columns = ["stock_id", "available_date"]
    if "report_period" in eligible_financials.columns:
        sort_columns.append("report_period")
    eligible_financials = eligible_financials.sort_values(sort_columns)
    latest_financials = {
        str(stock_id): group.iloc[-1]
        for stock_id, group in eligible_financials.groupby("stock_id", sort=False)
    }

    new_entries: list[UniverseEntry] = []
    for entry in universe.entries:
        reason = None if entry.included else entry.reason
        if reason is None:
            reason = _factor_v4_reason(
                entry.stock_id,
                stock_info,
                calendar,
                price_groups,
                latest_financials,
                as_of_text,
                settings,
                labelable_ids,
            )
        new_entries.append(
            UniverseEntry(
                stock_id=entry.stock_id,
                included=reason is None,
                reason=reason or "pass",
            )
        )

    filtered = UniverseSnapshot(run_id=universe.run_id, as_of=universe.as_of, entries=new_entries)
    excluded: dict[str, list[str]] = defaultdict(list)
    for entry in filtered.entries:
        if not entry.included:
            excluded[_reason_category(entry.reason)].append(entry.stock_id)
    report = {
        "initial_count": len(universe.included_ids),
        "eligible_count": len(filtered.included_ids),
        "excluded_counts": dict(
            sorted(
                Counter(
                    _reason_category(entry.reason)
                    for entry in filtered.entries
                    if not entry.included
                ).items()
            )
        ),
        "excluded_ids": {reason: sorted(ids) for reason, ids in sorted(excluded.items())},
    }
    return filtered, report


def _reason_category(reason: str) -> str:
    if reason.startswith("market_cap:") and reason != "market_cap:missing":
        return "market_cap:below_threshold"
    if reason.startswith("avg_traded_value:insufficient_history:"):
        return "avg_traded_value:insufficient_history"
    if reason.startswith("avg_traded_value:"):
        return "avg_traded_value:below_threshold"
    if reason.startswith("price:"):
        return "price:below_threshold"
    if reason.startswith("listing_age:insufficient:"):
        return "listing_age:insufficient"
    if reason.startswith("listing_age:insufficient_calendar:"):
        return "listing_age:insufficient_calendar"
    if reason.startswith("adjusted_price:insufficient_history:"):
        return "adjusted_price:insufficient_history"
    return reason


def _factor_v4_reason(
    stock_id: str,
    stocks: pd.DataFrame,
    calendar: np.ndarray,
    price_groups: dict[str, pd.DataFrame],
    latest_financials: dict[str, pd.Series],
    as_of: str,
    settings: Settings,
    labelable_ids: set[str] | None,
) -> str | None:
    if stock_id not in stocks.index:
        return "listing_age:missing_date"
    listed_date = stocks.loc[stock_id].get("listed_date")
    if not isinstance(listed_date, str) or not listed_date:
        return "listing_age:missing_date"
    try:
        date.fromisoformat(listed_date)
    except ValueError:
        return "listing_age:invalid_date"
    min_age = settings.universe.min_listing_age_trading_days
    if min_age:
        if len(calendar) == 0:
            return "listing_age:missing_calendar"
        start = int(np.searchsorted(calendar, listed_date, side="left"))
        age = len(calendar) - start
        if listed_date < str(calendar[0]) and age < min_age:
            return f"listing_age:insufficient_calendar:{age}"
        if age < min_age:
            return f"listing_age:insufficient:{age}"

    required_rows = settings.features.required_adjusted_price_rows
    history = price_groups.get(stock_id)
    count = 0 if history is None else len(history)
    if history is None or count < required_rows:
        return f"adjusted_price:insufficient_history:{count}"
    window = history.tail(required_rows)
    for column in ("close_adj", "high_adj"):
        values = pd.to_numeric(window[column], errors="coerce").to_numpy(dtype=float)
        if len(values) != required_rows or not np.all(np.isfinite(values)) or np.any(values <= 0):
            return f"adjusted_price:invalid_{column}"
    if str(window.iloc[-1]["trade_date"]) != as_of:
        return "adjusted_price:no_signal_date"

    latest = latest_financials.get(stock_id)
    if latest is None:
        return "financials:missing_pit"
    for field in settings.features.required_financial_fields:
        try:
            value = float(latest.get(field))
        except (TypeError, ValueError):
            return f"financials:missing_{field}"
        if not np.isfinite(value):
            return f"financials:missing_{field}"

    if labelable_ids is not None and stock_id not in labelable_ids:
        return "label:missing_t_plus_horizon"
    return None
