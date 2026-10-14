"""Compare the controlled OOS-2024 month-end vs midmonth-augmented experiment."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def _artifact(connection: sqlite3.Connection, run_id: str, kind: str) -> dict:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = ?",
        (run_id, kind),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing {kind} artifact: {run_id}")
    payload = json.loads(row[0])
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {kind} artifact: {run_id}")
    return payload


def _run_stats(connection: sqlite3.Connection, run_id: str) -> dict[str, float | int]:
    order_count, total_cost = connection.execute(
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
        "order_count": int(order_count),
        "total_transaction_cost": float(total_cost),
        "average_holdings": float(monthly.count().mean()),
        "average_equity_exposure": float(monthly.sum().mean()),
    }


def _summary_row(
    connection: sqlite3.Connection,
    variant: str,
    run_id: str,
    model_dir: Path,
) -> dict[str, object]:
    metrics = _artifact(connection, run_id, "metrics").get("metrics", {})
    selection = _artifact(connection, run_id, "monthly_ic")
    selection_summary = selection.get("summary", {})
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    return {
        "variant": variant,
        "run_id": run_id,
        "model_dir": str(model_dir),
        "validation_rank_ic": report.get("validation_rank_ic"),
        "validation_top_decile_spread": report.get("validation_top_decile_spread"),
        "train_rows": report.get("train_rows"),
        "train_snapshot_count": report.get("train_snapshot_count", 47),
        "month_end_train_rows": report.get("month_end_train_rows", report.get("train_rows")),
        "extra_train_rows": report.get("extra_train_rows", 0),
        **{
            field: metrics.get(field)
            for field in (
                "cagr",
                "sharpe",
                "sortino",
                "max_drawdown",
                "realized_volatility",
                "turnover",
            )
        },
        **_run_stats(connection, run_id),
        "continuous_rank_ic": selection_summary.get("mean_continuous_rank_ic"),
        "ic_positive_month_ratio": selection_summary.get("ic_positive_month_ratio"),
        "top15_actual_return": selection_summary.get("mean_top15_actual_return"),
        "top15_excess_return": selection_summary.get("mean_top15_excess_return"),
        "top15_excess_positive_month_ratio": selection_summary.get(
            "top15_excess_positive_month_ratio"
        ),
        "top_bottom_spread": selection_summary.get("mean_top_bottom_spread"),
    }


def _monthly_comparison(
    connection: sqlite3.Connection, baseline_run: str, augmented_run: str
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for variant, run_id in (("baseline", baseline_run), ("augmented", augmented_run)):
        payload = _artifact(connection, run_id, "monthly_ic")
        frame = pd.DataFrame(payload.get("monthly", []))
        if frame.empty:
            raise ValueError(f"missing monthly selection metrics: {run_id}")
        frames[variant] = frame
    fields = [
        "signal_date",
        "eligible_count",
        "continuous_rank_ic",
        "top15_actual_return",
        "top15_excess_return",
        "top_bottom_spread",
    ]
    output = frames["baseline"][fields].merge(
        frames["augmented"][fields],
        on="signal_date",
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    for field in fields[1:]:
        output[f"{field}_delta"] = (
            output[f"{field}_augmented"] - output[f"{field}_baseline"]
        )
    predictions: dict[str, pd.DataFrame] = {}
    for variant, run_id in (("baseline", baseline_run), ("augmented", augmented_run)):
        predictions[variant] = pd.read_sql_query(
            "SELECT prediction_date, stock_id, rank FROM predictions WHERE run_id = ?",
            connection,
            params=(run_id,),
        )
    merged = predictions["baseline"].merge(
        predictions["augmented"],
        on=["prediction_date", "stock_id"],
        suffixes=("_baseline", "_augmented"),
        validate="one_to_one",
    )
    similarities = []
    for signal_date, group in merged.groupby("prediction_date"):
        top_baseline = set(group.nsmallest(15, "rank_baseline")["stock_id"])
        top_augmented = set(group.nsmallest(15, "rank_augmented")["stock_id"])
        similarities.append(
            {
                "signal_date": signal_date,
                "prediction_rank_spearman": group["rank_baseline"].corr(
                    group["rank_augmented"], method="spearman"
                ),
                "top15_overlap_ratio": len(top_baseline & top_augmented) / 15.0,
            }
        )
    return output.merge(pd.DataFrame(similarities), on="signal_date", how="left")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="database/quant.db")
    parser.add_argument(
        "--baseline-run", default="oos-2024-factor-v4-pseudohuber-NOrefit"
    )
    parser.add_argument(
        "--augmented-run",
        default="oos-2024-factor-v4-pseudohuber-midmonth-augmented-NOrefit",
    )
    parser.add_argument(
        "--baseline-model",
        default="models/model_oos2024_validation_2019_2022_v4_pseudohuber_NOrefit",
    )
    parser.add_argument(
        "--augmented-model",
        default=(
            "models/model_oos2024_validation_2019_2022_v4_"
            "pseudohuber_midmonth_augmented_NOrefit"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="output/research/midmonth_augmented_vs_monthend_oos2024.csv",
    )
    parser.add_argument(
        "--monthly-output",
        default="output/research/midmonth_augmented_vs_monthend_oos2024_monthly.csv",
    )
    args = parser.parse_args(argv)
    with sqlite3.connect(args.database) as connection:
        summary = pd.DataFrame(
            [
                _summary_row(
                    connection, "month_end_only", args.baseline_run, Path(args.baseline_model)
                ),
                _summary_row(
                    connection,
                    "month_end_plus_midmonth",
                    args.augmented_run,
                    Path(args.augmented_model),
                ),
            ]
        )
        for field in (
            "validation_rank_ic",
            "validation_top_decile_spread",
            "cagr",
            "sharpe",
            "sortino",
            "max_drawdown",
            "realized_volatility",
            "turnover",
            "total_transaction_cost",
            "continuous_rank_ic",
            "ic_positive_month_ratio",
            "top15_actual_return",
            "top15_excess_return",
            "top15_excess_positive_month_ratio",
            "top_bottom_spread",
        ):
            baseline = pd.to_numeric(summary.loc[0, field], errors="coerce")
            summary[f"{field}_delta_vs_baseline"] = pd.to_numeric(
                summary[field], errors="coerce"
            ) - baseline
        monthly = _monthly_comparison(connection, args.baseline_run, args.augmented_run)
    for destination, frame in (
        (Path(args.summary_output), summary),
        (Path(args.monthly_output), monthly),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, encoding="utf-8-sig", lineterminator="\n")
        print(f"saved {destination} ({len(frame)} rows)")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
