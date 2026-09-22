"""P2-13: execution and cost service (SDD 13.1, 13.2).

Signal at month-end close, execution at next trading day open. Costs
follow the SDD table: 0.1425% broker fee both sides, 0.3% tax on sells,
per-side slippage on the open. Output columns match what P1-10
``save_orders`` persists.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from contracts import PortfolioTarget
from settings import ExecutionSettings, Settings

ORDER_COLUMNS: tuple[str, ...] = (
    "order_id",
    "run_id",
    "signal_date",
    "execution_date",
    "stock_id",
    "side",
    "target_weight",
    "target_shares",
    "executed_price",
    "broker_fee",
    "transaction_tax",
    "slippage_cost",
    "total_cost",
)


def executed_price(open_price: float, side: str, slippage: float) -> float:
    """Slide the open against the trader: up on BUY, down on SELL."""
    if side == "BUY":
        return open_price * (1 + slippage)
    if side == "SELL":
        return open_price * (1 - slippage)
    raise ValueError(side)


def transaction_cost(notional: float, side: str, cfg: ExecutionSettings) -> dict[str, float]:
    """Split one order's cost into fee, tax, and slippage legs."""
    if side not in ("BUY", "SELL"):
        raise ValueError(side)
    broker_fee = notional * cfg.broker_fee_rate
    transaction_tax = notional * cfg.sell_tax_rate if side == "SELL" else 0.0
    slippage_cost = notional * cfg.slippage_rate_per_side
    return {
        "broker_fee": broker_fee,
        "transaction_tax": transaction_tax,
        "slippage_cost": slippage_cost,
        "total_cost": broker_fee + transaction_tax + slippage_cost,
    }


def create_orders(
    target: PortfolioTarget,
    signal_date: date,
    next_open: pd.DataFrame,
    current_holdings: pd.DataFrame,
    portfolio_value: float,
    settings: Settings,
    run_id: str,
) -> pd.DataFrame:
    """Turn target weights into next-open executable orders."""
    if not isinstance(signal_date, date):
        raise ValueError(f"invalid signal_date: must be a date, got {signal_date!r}")
    if signal_date.isoformat() != target.signal_date:
        raise ValueError("signal_date does not match target.signal_date")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    if not portfolio_value > 0:
        raise ValueError(f"invalid portfolio_value: {portfolio_value!r}")
    for name, frame, columns in (
        ("next_open", next_open, ("stock_id", "trade_date", "open")),
        ("current_holdings", current_holdings, ("stock_id", "shares")),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"invalid {name}: must be a DataFrame")
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"invalid {name}: missing columns {missing}")
    execution_dates = next_open["trade_date"].unique()
    if len(execution_dates) != 1:
        raise ValueError("invalid next_open: must hold a single execution date")
    execution_date = str(execution_dates[0])
    if execution_date <= target.signal_date:
        raise ValueError("invalid next_open: execution_date must exceed signal_date")

    opens = next_open.set_index("stock_id")["open"].apply(pd.to_numeric, errors="coerce")
    held = (
        current_holdings.set_index("stock_id")["shares"].apply(pd.to_numeric, errors="coerce")
    ).to_dict()

    cfg = settings.execution
    orders: list[dict] = []
    for stock_id, action in target.actions.items():
        if action == "NONE":
            continue
        if action in ("BUY", "HOLD"):
            weight = target.weights.get(stock_id, 0.0)
            if stock_id not in opens.index or not opens.loc[stock_id] > 0:
                continue
            target_shares = int(round(weight * portfolio_value / float(opens.loc[stock_id])))
            if action == "BUY":
                if target_shares <= 0:
                    continue
                side, shares = "BUY", target_shares
            else:
                current = int(held.get(stock_id, 0))
                diff = target_shares - current
                if diff == 0:
                    continue
                side, shares = ("BUY", diff) if diff > 0 else ("SELL", -diff)
        else:  # SELL: liquidate the full current holding.
            current = int(held.get(stock_id, 0))
            if current <= 0 or stock_id not in opens.index or not opens.loc[stock_id] > 0:
                continue
            side = "SELL"
            shares = current
            weight = 0.0
            target_shares = 0
        price = executed_price(float(opens.loc[stock_id]), side, cfg.slippage_rate_per_side)
        notional = shares * price
        costs = transaction_cost(notional, side, cfg)
        orders.append(
            {
                "order_id": f"{run_id}|{execution_date}|{stock_id}|{side}",
                "run_id": run_id,
                "signal_date": target.signal_date,
                "execution_date": execution_date,
                "stock_id": stock_id,
                "side": side,
                "target_weight": weight,
                "target_shares": target_shares,
                "executed_price": price,
                **costs,
            }
        )
    return pd.DataFrame(orders, columns=[*ORDER_COLUMNS])
