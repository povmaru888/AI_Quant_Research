"""P2-02: tradable universe service (SDD section 8).

Builds the historical tradable universe for one rebalance date. Every
filter of the SDD chain maps to one gate below, in order; the first gate
a stock fails becomes its machine-readable ``reason``. No thresholds are
hard-coded: all come from ``Settings.universe``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from contracts import UniverseEntry, UniverseSnapshot
from settings import Settings

_HISTORY_WINDOW = 20


def _iso(value: date) -> str:
    return value.isoformat()


def build_universe(
    prices: pd.DataFrame,
    stocks: pd.DataFrame,
    as_of: date,
    settings: Settings,
    run_id: str,
) -> UniverseSnapshot:
    """Filter the historical tradable universe as of ``as_of``.

    ``prices`` needs ``stock_id, trade_date, close, traded_value`` columns;
    ``stocks`` needs ``stock_id, listed_date, delisted_date`` plus an
    optional ``flags`` column (comma-separated, e.g. ``"KY"``) and an
    optional ``market_cap`` column. Missing ``market_cap`` fails loud:
    the stock is excluded with reason ``market_cap:missing`` instead of
    silently skipping the gate (SDD section 16).
    """
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    for name, frame, columns in (
        ("prices", prices, ("stock_id", "trade_date", "close", "traded_value")),
        ("stocks", stocks, ("stock_id", "listed_date", "delisted_date")),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"invalid {name}: must be a DataFrame")
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"invalid {name}: missing columns {missing}")

    universe = settings.universe
    as_of_str = _iso(as_of)
    has_flags = "flags" in stocks.columns
    has_market_cap = "market_cap" in stocks.columns
    excluded_flags = {f.strip().upper() for f in universe.excluded_flags}

    listed = stocks.set_index("stock_id")
    day_prices = prices.loc[prices["trade_date"] == as_of_str].set_index("stock_id")
    history = prices.loc[prices["trade_date"] <= as_of_str].sort_values("trade_date")
    recent_values = {
        stock_id: values.tail(_HISTORY_WINDOW)
        for stock_id, values in history.groupby("stock_id", sort=False)["traded_value"]
    }

    entries: list[UniverseEntry] = []
    for stock_id, info in listed.iterrows():
        reason = _screen(
            str(stock_id),
            info,
            day_prices,
            recent_values,
            has_flags,
            has_market_cap,
            excluded_flags,
            universe.min_price_twd,
            universe.min_market_cap_twd,
            universe.min_avg_traded_value_20d_twd,
            as_of_str,
        )
        entries.append(
            UniverseEntry(stock_id=str(stock_id), included=reason is None, reason=reason or "pass")
        )
    return UniverseSnapshot(run_id=run_id, as_of=as_of_str, entries=entries)


def _screen(
    stock_id: str,
    info: pd.Series,
    day_prices: pd.DataFrame,
    recent_values: dict[str, pd.Series],
    has_flags: bool,
    has_market_cap: bool,
    excluded_flags: set[str],
    min_price: float,
    min_market_cap: float,
    min_avg_traded_value: float,
    as_of_str: str,
) -> str | None:
    """Return the exclusion reason, or ``None`` when the stock passes."""
    listed_date = info.get("listed_date")
    if isinstance(listed_date, str) and listed_date and listed_date > as_of_str:
        return f"not_listed:{listed_date}"
    delisted_date = info.get("delisted_date")
    if isinstance(delisted_date, str) and delisted_date and delisted_date <= as_of_str:
        return f"delisted:{delisted_date}"
    if stock_id not in day_prices.index:
        return "no_price"
    close = day_prices.loc[stock_id, "close"]
    if isinstance(close, pd.Series):
        close = close.iloc[-1]
    if pd.isna(close) or float(close) <= 0:
        return "no_price"
    if float(close) <= min_price:
        return f"price:{float(close):.2f}"
    if not has_market_cap or pd.isna(info.get("market_cap")):
        return "market_cap:missing"
    if float(info["market_cap"]) <= min_market_cap:
        return f"market_cap:{float(info['market_cap']):.0f}"
    past = recent_values.get(stock_id)
    count = 0 if past is None else len(past)
    if count < _HISTORY_WINDOW:
        return f"avg_traded_value:insufficient_history:{count}"
    avg_value = float(past.mean())
    if avg_value <= min_avg_traded_value:
        return f"avg_traded_value:{avg_value:.0f}"
    if has_flags:
        raw_flags = info.get("flags")
        own = (
            {f.strip().upper() for f in str(raw_flags).split(",") if f.strip()}
            if isinstance(raw_flags, str) and raw_flags.strip()
            else set()
        )
        hit = sorted(own & excluded_flags)
        if hit:
            return f"flag:{','.join(hit)}"
    return None
