"""Append a liquidate-everything batch to a run (bounds backtest replays).

A run whose positions stay open forever replays to the latest calendar
date, where delisted names or adjusted-price gaps fail the strict
backtest. Liquidating at a chosen signal date closes the ledger: later
data gaps cannot touch the run. Uses a fresh (run_id, signal_date) scope
so earlier signal months are never overwritten.

Usage:
    python tools/liquidate_run.py --run-id oos-2024-b2
        [--signal-date 2025-01-02] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import PortfolioTarget  # noqa: E402
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.execution_service import create_orders  # noqa: E402
from settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Liquidate a run's positions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    store = build_store(settings)
    store.bind(args.run_id)

    from sqlalchemy import select

    from models.research import Order

    with store._scope() as session:  # noqa: SLF001
        rows = session.execute(
            select(
                Order.order_id,
                Order.execution_date,
                Order.stock_id,
                Order.side,
                Order.quantity,
                Order.signal_date,
            )
            .where(Order.run_id == args.run_id)
            .order_by(Order.execution_date, Order.order_id)
        ).all()
    if not rows:
        print("run has no orders", file=sys.stderr)
        return 1
    last_signal = max(r[5] for r in rows)
    prices = store.load_prices()

    if args.signal_date:
        signal_str = args.signal_date
    else:
        later = sorted({d for d in prices["trade_date"].unique() if d > last_signal})
        signal_str = later[0] if later else last_signal
    as_of = date.fromisoformat(signal_str)

    positions: dict[str, int] = {}
    for _order_id, _exec_date, sid, side, qty, _ in sorted(rows, key=lambda r: (r[1], r[0])):
        # Order.quantity is executed shares: BUY adds, SELL removes.
        if side == "BUY":
            positions[str(sid)] = positions.get(str(sid), 0) + int(qty)
        elif side == "SELL":
            positions[str(sid)] = positions.get(str(sid), 0) - int(qty)
    held = {s: q for s, q in positions.items() if q > 0}
    if not held:
        print("nothing to liquidate")
        return 0
    print(f"liquidating {len(held)} positions at signal {signal_str}")

    exec_days = sorted({d for d in prices["trade_date"].unique() if d > signal_str})
    execution = None
    day_frame = None
    for day in exec_days:
        frame = prices.loc[prices["trade_date"] == day, ["stock_id", "trade_date", "open"]]
        frame = frame.loc[frame["stock_id"].isin(held)]
        if len(frame) == len(held) and (frame["open"] > 0).all():
            execution, day_frame = day, frame
            break
    if execution is None:
        print("no execution day covers all holdings", file=sys.stderr)
        return 1
    target = PortfolioTarget(
        run_id=args.run_id,
        signal_date=signal_str,
        top_n=settings.portfolio.top_n,
        actions={s: "SELL" for s in held},
        weights={},
        cash_weight=1.0,
        equity_exposure=0.0,
    )
    current = pd.DataFrame({"stock_id": list(held), "shares": list(held.values())})
    orders = create_orders(target, as_of, day_frame, current, 1.0, settings, args.run_id)
    store.bind(args.run_id, signal_str)
    n_saved = store.save_orders(orders)
    print(f"saved {n_saved} liquidation orders for {execution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
