"""Compare inverse-volatility and equal-weight OOS run families."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(connection: sqlite3.Connection, run_id: str) -> dict[str, float]:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = 'metrics'",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing metrics artifact: {run_id}")
    return json.loads(row[0])["metrics"]


def _run_stats(connection: sqlite3.Connection, run_id: str) -> dict[str, float]:
    costs = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost), 0) FROM orders WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    signals = pd.read_sql_query(
        "SELECT signal_date, target_weight FROM signals "
        "WHERE run_id = ? AND target_weight > 0",
        connection,
        params=(run_id,),
    )
    if signals.empty:
        raise ValueError(f"missing signals: {run_id}")
    monthly = signals.groupby("signal_date")["target_weight"]
    return {
        "order_count": int(costs[0]),
        "total_cost": float(costs[1]),
        "average_holdings": float(monthly.count().mean()),
        "average_equity_exposure": float(monthly.sum().mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="database/quant.db")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    connection = sqlite3.connect(args.database)
    try:
        for year in range(args.start_year, args.end_year + 1):
            run_ids = {
                "inverse_volatility": f"oos-{year}-factor-v4-pseudohuber-NOrefit",
                "equal": f"oos-{year}-factor-v4-pseudohuber-equalweight-NOrefit",
            }
            data: dict[str, dict[str, float]] = {}
            for method, run_id in run_ids.items():
                data[method] = _metrics(connection, run_id) | _run_stats(connection, run_id)
            row: dict[str, object] = {"oos_year": year}
            fields = (
                "cagr",
                "sharpe",
                "sortino",
                "max_drawdown",
                "realized_volatility",
                "turnover",
                "total_cost",
                "order_count",
                "average_holdings",
                "average_equity_exposure",
                "rank_ic",
            )
            for field in fields:
                inverse = data["inverse_volatility"][field]
                equal = data["equal"][field]
                row[f"inverse_{field}"] = inverse
                row[f"equal_{field}"] = equal
                row[f"delta_equal_minus_inverse_{field}"] = equal - inverse
            rows.append(row)
    finally:
        connection.close()

    annual = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output, index=False, encoding="utf-8-sig")

    summaries = []
    for method in ("inverse", "equal"):
        cagr = annual[f"{method}_cagr"]
        summaries.append(
            {
                "method": "inverse_volatility" if method == "inverse" else "equal",
                "mean_annual_cagr": cagr.mean(),
                "compounded_2020_2024_return": float(np.prod(1.0 + cagr) - 1.0),
                "positive_year_ratio": float((cagr > 0).mean()),
                "mean_sharpe": annual[f"{method}_sharpe"].mean(),
                "mean_sortino": annual[f"{method}_sortino"].mean(),
                "worst_max_drawdown": annual[f"{method}_max_drawdown"].min(),
                "mean_realized_volatility": annual[f"{method}_realized_volatility"].mean(),
                "mean_turnover": annual[f"{method}_turnover"].mean(),
                "total_transaction_cost": annual[f"{method}_total_cost"].sum(),
                "mean_equity_exposure": annual[f"{method}_average_equity_exposure"].mean(),
                "mean_rank_ic": annual[f"{method}_rank_ic"].mean(),
            }
        )
    pd.DataFrame(summaries).to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    print(annual.to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
