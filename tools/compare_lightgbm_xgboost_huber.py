"""Build reproducible 2020-2024 LightGBM-vs-XGBoost Huber comparisons."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

FAMILIES = ("xgboost_pseudohuber", "lightgbm_huber")


def _run_id(year: int, family: str) -> str:
    stem = "pseudohuber" if family == "xgboost_pseudohuber" else "lightgbm-huber"
    return f"oos-{year}-factor-v4-{stem}-NOrefit"


def _model_dir(root: Path, year: int, family: str) -> Path:
    start = year - 5
    end = year - 2
    stem = "pseudohuber" if family == "xgboost_pseudohuber" else "lightgbm_huber"
    return root / f"model_oos{year}_validation_{start}_{end}_v4_{stem}_NOrefit"


def _artifact(connection: sqlite3.Connection, run_id: str, kind: str) -> object:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = ?",
        (run_id, kind),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing {kind} artifact: {run_id}")
    return json.loads(row[0])


def _positive_ratio(rows: list[dict], field: str) -> float:
    values = pd.to_numeric(pd.Series([row.get(field) for row in rows]), errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else float("nan")


def _run_stats(connection: sqlite3.Connection, run_id: str) -> dict[str, float]:
    order_count, total_cost = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost), 0) FROM orders WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    signals = pd.read_sql_query(
        "SELECT signal_date, target_weight FROM signals WHERE run_id = ? AND target_weight > 0",
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


def _top_features(payload: object, key: str, value: str) -> str:
    rows = payload.get(key, []) if isinstance(payload, dict) else []
    cleaned = [
        {"feature": row.get("feature"), value: row.get(value)}
        for row in rows[:10]
        if isinstance(row, dict)
    ]
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def _annual_row(
    connection: sqlite3.Connection,
    year: int,
    family: str,
    models_root: Path,
    profiles_root: Path,
) -> dict[str, object]:
    run_id = _run_id(year, family)
    metrics_payload = _artifact(connection, run_id, "metrics")
    selection = _artifact(connection, run_id, "monthly_ic")
    explain = _artifact(connection, run_id, "model_explain")
    if not isinstance(metrics_payload, dict) or not isinstance(selection, dict):
        raise ValueError(f"invalid artifacts: {run_id}")
    metrics = metrics_payload.get("metrics", {})
    selection_summary = selection.get("summary", {})
    monthly = selection.get("monthly", [])
    model_dir = _model_dir(models_root, year, family)
    with (model_dir / "report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    model_path = model_dir / str(
        report.get("model_file") or ("booster.txt" if family == "lightgbm_huber" else "booster.ubj")
    )
    profile_path = profiles_root / f"{run_id}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    inference_seconds = profile.get("stages_seconds", {}).get("predictions")
    return {
        "oos_year": year,
        "family": family,
        "run_id": run_id,
        "validation_rank_ic": report.get("validation_rank_ic"),
        "validation_top_decile_actual_return_spread": report.get("validation_top_decile_spread"),
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
        "top_bottom_positive_month_ratio": _positive_ratio(monthly, "top_bottom_spread"),
        "training_seconds": report.get("training_seconds"),
        "inference_seconds": inference_seconds,
        "model_file_size_bytes": report.get("model_file_size_bytes") or model_path.stat().st_size,
        "shap_top10": _top_features(explain, "shap_top", "value"),
        "gain_top10": _top_features(explain, "feature_importance", "gain"),
    }


def _monthly_rows(connection: sqlite3.Connection, year: int) -> list[dict[str, object]]:
    by_family: dict[str, dict[str, dict]] = {}
    for family in FAMILIES:
        payload = _artifact(connection, _run_id(year, family), "monthly_ic")
        rows = payload.get("monthly", []) if isinstance(payload, dict) else []
        by_family[family] = {str(row["signal_date"]): row for row in rows}
    xgb_run = _run_id(year, FAMILIES[0])
    lgb_run = _run_id(year, FAMILIES[1])
    xgb_predictions = pd.read_sql_query(
        "SELECT prediction_date, stock_id, prediction_probability, rank FROM predictions "
        "WHERE run_id = ?",
        connection,
        params=(xgb_run,),
    )
    lgb_predictions = pd.read_sql_query(
        "SELECT prediction_date, stock_id, prediction_probability, rank FROM predictions "
        "WHERE run_id = ?",
        connection,
        params=(lgb_run,),
    )
    merged = xgb_predictions.merge(
        lgb_predictions,
        on=["prediction_date", "stock_id"],
        suffixes=("_xgb", "_lgb"),
        validate="one_to_one",
    )
    similarities: dict[str, dict[str, float]] = {}
    for signal_date, frame in merged.groupby("prediction_date"):
        correlation = frame["rank_xgb"].corr(frame["rank_lgb"], method="spearman")
        top_xgb = set(frame.nsmallest(15, "rank_xgb")["stock_id"])
        top_lgb = set(frame.nsmallest(15, "rank_lgb")["stock_id"])
        similarities[str(signal_date)] = {
            "prediction_rank_spearman": float(correlation),
            "top15_overlap_ratio": len(top_xgb & top_lgb) / 15.0,
        }
    fields = (
        "eligible_count",
        "continuous_rank_ic",
        "top15_actual_return",
        "top15_excess_return",
        "top_bottom_spread",
    )
    output = []
    all_dates = sorted(set(by_family[FAMILIES[0]]) | set(by_family[FAMILIES[1]]))
    for signal_date in all_dates:
        row: dict[str, object] = {"oos_year": year, "signal_date": signal_date}
        for family, prefix in ((FAMILIES[0], "xgboost"), (FAMILIES[1], "lightgbm")):
            source = by_family[family].get(signal_date, {})
            for field in fields:
                row[f"{prefix}_{field}"] = source.get(field)
        row.update(similarities.get(signal_date, {}))
        output.append(row)
    return output


def _stitched_metrics(connection: sqlite3.Connection, family: str, years: range) -> dict:
    returns = []
    for year in years:
        payload = _artifact(connection, _run_id(year, family), "metrics")
        curve = pd.DataFrame(payload.get("equity_curve", [])) if isinstance(payload, dict) else None
        if curve is None or curve.empty:
            raise ValueError(f"missing equity curve: {_run_id(year, family)}")
        values = pd.to_numeric(curve["nav"], errors="coerce")
        returns.append(values.pct_change().dropna())
    daily = pd.concat(returns, ignore_index=True)
    wealth = (1.0 + daily).cumprod()
    years_count = len(daily) / 252.0
    cagr = float(wealth.iloc[-1] ** (1.0 / years_count) - 1.0)
    standard_deviation = float(daily.std(ddof=1))
    downside = daily.loc[daily < 0]
    downside_deviation = float(downside.std(ddof=1)) if len(downside) >= 2 else float("nan")
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "stitched_cagr": cagr,
        "stitched_sharpe": float(daily.mean() / standard_deviation * math.sqrt(252)),
        "stitched_sortino": float(daily.mean() / downside_deviation * math.sqrt(252)),
        "stitched_max_drawdown": float(drawdown.min()),
        "stitched_trading_days": len(daily),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="database/quant.db")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--profiles-root", default="output/research/profiles")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--annual-output",
        default="lightgbm_huber_vs_xgboost_pseudohuber_2020_2024.csv",
    )
    parser.add_argument(
        "--monthly-output",
        default="lightgbm_huber_vs_xgboost_pseudohuber_monthly_2020_2024.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="lightgbm_huber_vs_xgboost_pseudohuber_summary_2020_2024.csv",
    )
    args = parser.parse_args(argv)
    years = range(args.start_year, args.end_year + 1)
    connection = sqlite3.connect(args.database)
    try:
        annual = pd.DataFrame(
            [
                _annual_row(
                    connection,
                    year,
                    family,
                    Path(args.models_root),
                    Path(args.profiles_root),
                )
                for year in years
                for family in FAMILIES
            ]
        )
        monthly = pd.DataFrame([row for year in years for row in _monthly_rows(connection, year)])
        numeric = annual.select_dtypes(include="number").columns.difference(["oos_year"])
        summaries = []
        for family in FAMILIES:
            subset = annual.loc[annual["family"] == family]
            row: dict[str, object] = {"family": family, "oos_years": len(subset)}
            row.update({f"mean_{field}": float(subset[field].mean()) for field in numeric})
            row["positive_oos_year_ratio"] = float((subset["cagr"] > 0).mean())
            row["total_transaction_cost"] = float(subset["total_transaction_cost"].sum())
            row.update(_stitched_metrics(connection, family, years))
            summaries.append(row)
        summary = pd.DataFrame(summaries)
    finally:
        connection.close()

    for destination, frame in (
        (Path(args.annual_output), annual),
        (Path(args.monthly_output), monthly),
        (Path(args.summary_output), summary),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, encoding="utf-8-sig", lineterminator="\n")
        print(f"saved {destination} ({len(frame)} rows)")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
