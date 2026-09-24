"""P2-14: backtest service (SDD 13.3).

Orders record real, unadjusted execution prices and share counts. For
historical total-return replay, a buy converts those shares to adjusted
units using that day's raw/adjusted open ratio. Adjusted closes value
the units, and sells release the same fraction of adjusted units as raw
shares. This is a reinvested-distribution approximation: it preserves
economic value across ex-rights/ex-dividend dates without pretending
that adjusted prices were executable market quotes. The original orders
and their broker fees, taxes, and slippage remain auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import BacktestResult
from settings import Settings

_REQUIRED_ORDER_COLUMNS = (
    "order_id",
    "signal_date",
    "execution_date",
    "stock_id",
    "side",
    "target_shares",
    "executed_price",
    "total_cost",
)


def run_backtest(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    initial_cash: float,
    settings: Settings,
    run_id: str,
) -> BacktestResult:
    """Replay orders into a daily NAV series wrapped as ``BacktestResult``."""
    _ = settings  # costs already materialized in orders; kept for signature parity.
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    if not isinstance(initial_cash, (int, float)) or not initial_cash > 0:
        raise ValueError(f"invalid initial_cash: {initial_cash!r}")
    if not isinstance(orders, pd.DataFrame):
        raise ValueError("invalid orders: must be a DataFrame")
    if not orders.empty:
        missing = [c for c in _REQUIRED_ORDER_COLUMNS if c not in orders.columns]
        if missing:
            raise ValueError(f"invalid orders: missing columns {missing}")
        if (orders["execution_date"] <= orders["signal_date"]).any():
            raise ValueError("invalid orders: execution_date must exceed signal_date")
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("invalid prices: must be a DataFrame")
    missing = [
        c
        for c in ("stock_id", "trade_date", "open", "open_adj", "close_adj")
        if c not in prices.columns
    ]
    if missing:
        raise ValueError(f"invalid prices: missing columns {missing}")
    calendar = sorted(prices["trade_date"].dropna().unique().tolist())
    if not calendar:
        raise ValueError("invalid prices: no dated prices")

    ledger = orders.sort_values(["execution_date", "order_id"]) if not orders.empty else orders
    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    adjusted_units: dict[str, float] = {}
    charged = 0.0
    fills = ledger.to_dict("records") if not ledger.empty else []

    # Only traded symbols need quotes. Keep the full market calendar so
    # index-only days still carry the last adjusted close forward.
    if fills:
        traded_symbols = {str(fill["stock_id"]) for fill in fills}
        traded_quotes = prices.loc[
            prices["stock_id"].astype(str).isin(traded_symbols),
            ["trade_date", "stock_id", "open", "open_adj", "close_adj"],
        ].copy()
        traded_quotes["stock_id"] = traded_quotes["stock_id"].astype(str)
        quote_rows = iter(
            traded_quotes.sort_values("trade_date").itertuples(index=False, name=None)
        )
    else:
        quote_rows = iter(())
    next_quote = next(quote_rows, None)

    nav_points: list[tuple[str, float]] = []
    fill_cursor = 0
    ordered_fills = sorted(fills, key=lambda r: (str(r["execution_date"]), str(r["order_id"])))
    last_close: dict[str, float] = {}
    for day in calendar:
        today_quotes: dict[str, tuple[object, object, object]] = {}
        while next_quote is not None and next_quote[0] == day:
            _, stock_id, raw_open, adj_open, adj_close = next_quote
            if stock_id in today_quotes:
                raise ValueError(f"invalid prices: duplicate bar for {stock_id} on {day}")
            today_quotes[stock_id] = (raw_open, adj_open, adj_close)
            if _positive_price(adj_close):
                last_close[stock_id] = float(adj_close)
            next_quote = next(quote_rows, None)

        while fill_cursor < len(ordered_fills) and str(
            ordered_fills[fill_cursor]["execution_date"]
        ) <= str(day):
            fill = ordered_fills[fill_cursor]
            stock_id = str(fill["stock_id"])
            execution_day = str(fill["execution_date"])
            if execution_day != str(day) or stock_id not in today_quotes:
                raise ValueError(
                    f"invalid prices: missing execution bar for {stock_id} on {execution_day}"
                )
            raw_open, adj_open, _ = today_quotes[stock_id]
            if not _positive_price(raw_open) or not _positive_price(adj_open):
                raise ValueError(
                    f"invalid prices: missing raw/adjusted open for {stock_id} on {day}"
                )
            cash, charged = _apply_fill(
                fill, holdings, adjusted_units, cash, charged,
                float(raw_open), float(adj_open),
            )
            fill_cursor += 1

        for stock_id, raw_adj_quotes in today_quotes.items():
            if holdings.get(stock_id, 0) > 0 and not _positive_price(raw_adj_quotes[2]):
                raise ValueError(f"invalid prices: missing close_adj for {stock_id} on {day}")
        missing_close = [s for s, shares in holdings.items() if shares > 0 and s not in last_close]
        if missing_close:
            raise ValueError(f"invalid prices: no close_adj for held stocks {missing_close}")
        # A missing bar carries the last adjusted close forward; a present
        # but null adjusted close fails instead of silently using raw close.
        equity = sum(
            units * last_close[stock_id]
            for stock_id, units in adjusted_units.items()
            if holdings.get(stock_id, 0) > 0
        )
        nav_points.append((day, cash + equity))

    if fill_cursor < len(ordered_fills):
        fill = ordered_fills[fill_cursor]
        raise ValueError(
            f"invalid prices: no execution date {fill['execution_date']} for {fill['stock_id']}"
        )

    nav = pd.Series(
        [value for _, value in nav_points],
        index=pd.Index([day for day, _ in nav_points], name="trade_date"),
        name="nav",
    )
    return BacktestResult(
        run_id=run_id,
        start_date=calendar[0],
        end_date=calendar[-1],
        initial_cash=float(initial_cash),
        nav=nav,
        orders=orders.reset_index(drop=True),
        total_cost=charged,
    )


def _apply_fill(
    fill: dict,
    holdings: dict[str, int],
    adjusted_units: dict[str, float],
    cash: float,
    charged: float,
    raw_open: float,
    adj_open: float,
) -> tuple[float, float]:
    stock_id = str(fill["stock_id"])
    side = str(fill["side"])
    price = float(fill["executed_price"])
    cost = float(fill["total_cost"])
    # Slippage is already reflected in ``executed_price``.  Keep it in the
    # reported total cost, but do not deduct it from cash a second time.
    slippage_cost = float(fill.get("slippage_cost", 0.0) or 0.0)
    cash_cost = max(cost - slippage_cost, 0.0)
    if side == "BUY":
        shares = int(fill["target_shares"]) - holdings.get(stock_id, 0)
        if shares <= 0:
            return cash, charged
        outlay = shares * price + cash_cost
        if outlay > cash:
            return cash, charged  # skipped whole: no partial fills.
        holdings[stock_id] = holdings.get(stock_id, 0) + shares
        adjusted_units[stock_id] = (
            adjusted_units.get(stock_id, 0.0) + shares * raw_open / adj_open
        )
        return cash - outlay, charged + cost
    if side == "SELL":
        held = holdings.get(stock_id, 0)
        goal = int(fill["target_shares"])
        shares = held - goal if goal > 0 else held
        if shares <= 0:
            return cash, charged
        # Release the same fraction of historical adjusted units as real
        # shares. The difference from the raw sale proceeds is the adjusted
        # series' reinvested corporate-action return, not an extra fee.
        units_sold = adjusted_units[stock_id] * shares / held
        economic_proceeds = units_sold * price * adj_open / raw_open
        holdings[stock_id] = held - shares
        adjusted_units[stock_id] -= units_sold
        return cash + economic_proceeds - cash_cost, charged + cost
    raise ValueError(f"invalid side: {side!r}")


def _positive_price(value: object) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(price) and price > 0
