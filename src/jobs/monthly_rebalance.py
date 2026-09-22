"""P3-06: monthly rebalance CLI job (SDD 14.3).

Validates the signal date is a month-end trading day, runs the research
pipeline, then persists orders atomically: ``replace_orders`` deletes
the (run_id, signal_date) scope and inserts in one transaction, so a
failure never leaves a half-written order set (P1-10 semantics).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python -m src.jobs.*`

import pandas as pd

from contracts import BacktestResult
from services.research_service import ResearchStore, run_research
from settings import Settings, load_settings


class RebalanceStore(ResearchStore, Protocol):
    """Research store plus month-end verification and atomic order replace."""

    def verify_month_end(self, signal_date: date) -> bool: ...
    def replace_orders(self, run_id: str, signal_date: str, orders: pd.DataFrame) -> int: ...


def run_monthly_rebalance(
    signal_date: date,
    settings: Settings,
    store: RebalanceStore,
    run_id: str | None = None,
    research_fn=run_research,
    n_trials: int | None = None,
) -> dict:
    """Run one monthly rebalance; return the order summary."""
    if not isinstance(signal_date, date):
        raise ValueError(f"invalid signal_date: must be a date, got {signal_date!r}")
    if not store.verify_month_end(signal_date):
        raise ValueError(f"not a month-end trading day: {signal_date.isoformat()}")
    job_run_id = run_id or f"rebalance-{signal_date.isoformat()}"
    result: BacktestResult = research_fn(signal_date, settings, store, job_run_id, n_trials)
    orders = result.orders
    if not orders.empty:
        execution_dates = orders["execution_date"].unique().tolist()
        if len(execution_dates) != 1 or execution_dates[0] <= signal_date.isoformat():
            raise RuntimeError(f"orders span unexpected execution dates: {execution_dates}")
        execution_date: str | None = str(execution_dates[0])
    else:
        execution_date = None
    stored = store.replace_orders(job_run_id, signal_date.isoformat(), orders)
    return {
        "run_id": job_run_id,
        "signal_date": signal_date.isoformat(),
        "execution_date": execution_date,
        "orders": stored,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monthly rebalance signal job.")
    parser.add_argument("--signal-date", default=None, help="YYYY-MM-DD month-end trading day.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser


def main(argv: list[str] | None = None, store_factory=None, research_fn=None) -> int:
    """CLI entry; 0 on success, 1 on failure, 2 on bad arguments."""
    args = build_parser().parse_args(argv)
    try:
        signal_date = date.fromisoformat(args.signal_date) if args.signal_date else None
    except ValueError:
        print(f"invalid --signal-date: {args.signal_date!r}, expected YYYY-MM-DD", file=sys.stderr)
        return 2
    if signal_date is None:
        print("missing required --signal-date", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    try:
        factory = store_factory or _build_store
        store = factory(settings)
    except NotImplementedError as exc:
        print(f"monthly rebalance unavailable: {exc}", file=sys.stderr)
        return 1
    try:
        result = run_monthly_rebalance(
            signal_date, settings, store, research_fn=research_fn or run_research
        )
    except ValueError as exc:
        print(f"monthly rebalance rejected: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"monthly rebalance failed: {exc}", file=sys.stderr)
        return 1
    print(f"run_id={result['run_id']} signal={result['signal_date']} orders={result['orders']}")
    return 0


def _build_store(settings: Settings) -> RebalanceStore:
    """Assemble the database-backed store (Phase 5 wiring point)."""
    raise NotImplementedError("database-backed RebalanceStore lands in Phase 5")


if __name__ == "__main__":
    raise SystemExit(main())
