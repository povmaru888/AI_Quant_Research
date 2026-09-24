"""P2-14: backtest service (SDD 13.3).

Cash-plus-holdings ledger replayed from auditable orders, valued at
daily closes. Costs are taken from the orders themselves (already split
by P2-13), so nothing is double-counted. A VectorBT cross-check point:
``Portfolio.from_orders`` with ``fees=0``/``slippage=0`` on the same
orders and closes must reproduce this NAV; the pandas ledger stays the
record of truth for auditability and version-risk reasons.
"""

from __future__ import annotations

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
    missing = [c for c in ("stock_id", "trade_date", "close") if c not in prices.columns]
    if missing:
        raise ValueError(f"invalid prices: missing columns {missing}")

    close_values = pd.to_numeric(prices["close"], errors="coerce")
    valid_close = close_values.notna()
    calendar = sorted(prices.loc[valid_close, "trade_date"].unique().tolist())
    if not calendar:
        raise ValueError("invalid prices: no dated closes")

    ledger = orders.sort_values(["execution_date", "order_id"]) if not orders.empty else orders
    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    charged = 0.0
    fills = ledger.to_dict("records") if not ledger.empty else []

    # Only symbols that can enter the ledger need daily close updates. Keep
    # the full calendar so index-only days still carry holdings forward.
    if fills:
        traded_symbols = {str(fill["stock_id"]) for fill in fills}
        held_close = valid_close & prices["stock_id"].astype(str).isin(traded_symbols)
        close_rows = iter(
            pd.DataFrame(
                {
                    "trade_date": prices.loc[held_close, "trade_date"],
                    "stock_id": prices.loc[held_close, "stock_id"],
                    "close": close_values.loc[held_close],
                }
            )
            .sort_values("trade_date")
            .itertuples(index=False, name=None)
        )
    else:
        close_rows = iter(())
    next_close = next(close_rows, None)

    nav_points: list[tuple[str, float]] = []
    fill_cursor = 0
    ordered_fills = sorted(fills, key=lambda r: (str(r["execution_date"]), str(r["order_id"])))
    last_close: dict[str, float] = {}
    for day in calendar:
        while fill_cursor < len(ordered_fills) and str(
            ordered_fills[fill_cursor]["execution_date"]
        ) <= str(day):
            cash, charged = _apply_fill(ordered_fills[fill_cursor], holdings, cash, charged)
            fill_cursor += 1
        while next_close is not None and next_close[0] == day:
            _, stock_id, price = next_close
            last_close[str(stock_id)] = float(price)
            next_close = next(close_rows, None)
        # Missing bar carries the last close forward: a data gap must never
        # read as a worthless position (it once zeroed whole portfolios on
        # index-only calendar days). Never-seen stocks stay unvalued.
        equity = sum(
            shares * last_close[stock_id]
            for stock_id, shares in holdings.items()
            if shares > 0 and stock_id in last_close
        )
        nav_points.append((day, cash + equity))

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
    fill: dict, holdings: dict[str, int], cash: float, charged: float
) -> tuple[float, float]:
    stock_id = str(fill["stock_id"])
    side = str(fill["side"])
    price = float(fill["executed_price"])
    cost = float(fill["total_cost"])
    if side == "BUY":
        shares = int(fill["target_shares"]) - holdings.get(stock_id, 0)
        if shares <= 0:
            return cash, charged
        outlay = shares * price + cost
        if outlay > cash:
            return cash, charged  # skipped whole: no partial fills.
        holdings[stock_id] = holdings.get(stock_id, 0) + shares
        return cash - outlay, charged + cost
    if side == "SELL":
        held = holdings.get(stock_id, 0)
        goal = int(fill["target_shares"])
        shares = held - goal if goal > 0 else held
        if shares <= 0:
            return cash, charged
        holdings[stock_id] = held - shares
        return cash + shares * price - cost, charged + cost
    raise ValueError(f"invalid side: {side!r}")
