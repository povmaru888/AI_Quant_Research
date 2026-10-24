"""Read-only bridge from the research database into a live fund seed."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeedHolding:
    stock_id: str
    stock_name: str
    shares: int
    rank: int
    score: float | None
    target_weight: float
    action: str
    anchor_price: float
    anchor_price_date: str
    volatility_60d: float | None = None
    beta_60d: float | None = None


@dataclass(frozen=True)
class FundSeed:
    run_id: str
    initial_capital: float
    anchor_date: str
    anchor_nav: float
    anchor_cash: float
    signal_date: str
    execution_date: str
    holdings: tuple[SeedHolding, ...]
    history: tuple[dict, ...]
    monthly_returns: tuple[dict, ...]
    benchmark_history: tuple[dict, ...]
    metrics: dict
    taiex_ma60: float | None


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def load_fund_seed(
    database_path: str | Path,
    run_id: str,
    *,
    initial_capital: float = 30_000_000.0,
) -> FundSeed:
    """Reconstruct the final nominal book and anchor it to materialized OOS NAV."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    with _connect(database_path) as connection:
        run = connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None or run["status"] != "succeeded":
            raise ValueError(f"run is unavailable or incomplete: {run_id}")
        artifact = connection.execute(
            "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = 'metrics'",
            (run_id,),
        ).fetchone()
        if artifact is None:
            raise ValueError(f"metrics artifact is missing: {run_id}")
        payload = json.loads(artifact["payload_json"])
        history = tuple(payload.get("equity_curve") or ())
        if not history:
            raise ValueError(f"equity curve is empty: {run_id}")
        anchor_date = str(history[-1]["date"])
        anchor_nav = float(history[-1]["nav"])
        signal_date = str(
            connection.execute(
                "SELECT MAX(signal_date) FROM signals WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
        execution_date = str(
            connection.execute(
                "SELECT MAX(execution_date) FROM orders WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
        order_rows = connection.execute(
            "SELECT stock_id, side, quantity FROM orders WHERE run_id = ? "
            "ORDER BY execution_date, order_id",
            (run_id,),
        ).fetchall()
        shares: dict[str, int] = {}
        for row in order_rows:
            delta = int(row["quantity"]) * (1 if row["side"] == "BUY" else -1)
            shares[str(row["stock_id"])] = shares.get(str(row["stock_id"]), 0) + delta
        shares = {stock_id: value for stock_id, value in shares.items() if value > 0}
        holding_rows = connection.execute(
            """
            SELECT s.stock_id, COALESCE(st.stock_name, s.stock_id) AS stock_name,
                   s.signal, s.rank, s.target_weight, p.prediction_probability
            FROM signals s
            LEFT JOIN stocks st ON st.stock_id = s.stock_id
            LEFT JOIN predictions p
              ON p.run_id = s.run_id AND p.prediction_date = s.signal_date
             AND p.stock_id = s.stock_id
            WHERE s.run_id = ? AND s.signal_date = ? AND s.target_weight > 0
            ORDER BY s.rank, s.stock_id
            """,
            (run_id, signal_date),
        ).fetchall()
        holdings: list[SeedHolding] = []
        for row in holding_rows:
            stock_id = str(row["stock_id"])
            if shares.get(stock_id, 0) <= 0:
                continue
            price = connection.execute(
                "SELECT trade_date, close FROM prices WHERE stock_id = ? "
                "AND trade_date <= ? AND close > 0 ORDER BY trade_date DESC LIMIT 1",
                (stock_id, anchor_date),
            ).fetchone()
            if price is None:
                raise ValueError(f"anchor price is missing for {stock_id} on {anchor_date}")
            holdings.append(
                SeedHolding(
                    stock_id=stock_id,
                    stock_name=str(row["stock_name"]),
                    shares=shares[stock_id],
                    rank=int(row["rank"]),
                    score=(
                        float(row["prediction_probability"])
                        if row["prediction_probability"] is not None
                        else None
                    ),
                    target_weight=float(row["target_weight"]),
                    action=str(row["signal"]),
                    anchor_price=float(price["close"]),
                    anchor_price_date=str(price["trade_date"]),
                )
            )
        if set(shares) != {holding.stock_id for holding in holdings}:
            missing = sorted(set(shares) - {holding.stock_id for holding in holdings})
            raise ValueError(f"current shares do not match final positive signals: {missing}")
        equity = sum(item.shares * item.anchor_price for item in holdings)
        anchor_cash = anchor_nav - equity
        if anchor_cash < -1e-6:
            raise ValueError("anchor holdings exceed the materialized NAV")
        benchmark_rows = connection.execute(
            "SELECT trade_date, close FROM prices WHERE stock_id = 'TAIEX' "
            "AND trade_date BETWEEN ? AND ? AND close > 0 ORDER BY trade_date",
            (str(history[0]["date"]), anchor_date),
        ).fetchall()
        if benchmark_rows:
            base = float(benchmark_rows[0]["close"])
            benchmark = tuple(
                {
                    "date": str(row["trade_date"]),
                    "nav": initial_capital * float(row["close"]) / base,
                }
                for row in benchmark_rows
            )
        else:
            benchmark = ()
        ma60_rows = connection.execute(
            "SELECT close FROM prices WHERE stock_id = 'TAIEX' "
            "AND trade_date <= ? AND close > 0 ORDER BY trade_date DESC LIMIT 60",
            (anchor_date,),
        ).fetchall()
        taiex_ma60 = (
            sum(float(row["close"]) for row in ma60_rows) / 60 if len(ma60_rows) == 60 else None
        )
    return FundSeed(
        run_id=run_id,
        initial_capital=float(initial_capital),
        anchor_date=anchor_date,
        anchor_nav=anchor_nav,
        anchor_cash=anchor_cash,
        signal_date=signal_date,
        execution_date=execution_date,
        holdings=tuple(holdings),
        history=history,
        monthly_returns=tuple(payload.get("monthly_returns") or ()),
        benchmark_history=benchmark,
        metrics=dict(payload.get("metrics") or {}),
        taiex_ma60=taiex_ma60,
    )


def load_holding_history(database_path: str | Path, run_id: str) -> list[dict]:
    """Return positive target holdings by signal date for the history page."""
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT s.signal_date, s.stock_id, COALESCE(st.stock_name, s.stock_id) stock_name,
                   s.signal, s.rank, s.target_weight
            FROM signals s LEFT JOIN stocks st ON st.stock_id = s.stock_id
            WHERE s.run_id = ? AND s.target_weight > 0
            ORDER BY s.signal_date DESC, s.rank, s.stock_id
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_order_history(database_path: str | Path, run_id: str) -> list[dict]:
    """Return executed simulated orders for operational review."""
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT o.execution_date, o.stock_id,
                   COALESCE(st.stock_name, o.stock_id) AS stock_name,
                   o.side, o.quantity, o.executed_price, o.notional,
                   o.broker_fee, o.transaction_tax, o.slippage_cost, o.total_cost
            FROM orders o LEFT JOIN stocks st ON st.stock_id = o.stock_id
            WHERE o.run_id = ?
            ORDER BY o.execution_date DESC, o.order_id DESC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]
