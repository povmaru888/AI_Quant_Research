"""Compare unweighted and 4:1 weighted XGBoost binary models across OOS 2020-2025."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "research"
YEARS = range(2020, 2026)
VARIANTS = {
    "unweighted": "binary-unweighted",
    "classweight4_fixedparams": "binary-classweight4-fixedparams",
    "classweight4_optuna": "binary-classweight4-optuna",
}


def _run_id(year: int, variant: str) -> str:
    return f"oos-{year}-factor-v4-{VARIANTS[variant]}-NOrefit"


def _model_dir(year: int, variant: str) -> Path:
    start, end = year - 5, year - 2
    suffix = {
        "unweighted": "binary_unweighted",
        "classweight4_fixedparams": "binary_classweight4_fixedparams",
        "classweight4_optuna": "binary_classweight4_optuna",
    }[variant]
    return ROOT / "models" / f"model_oos{year}_validation_{start}_{end}_v4_{suffix}_NOrefit"


def _artifact(connection: sqlite3.Connection, run_id: str, kind: str) -> dict:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = ?",
        (run_id, kind),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing {kind} artifact: {run_id}")
    return json.loads(row[0])


def _positive_ratio(rows: list[dict], field: str) -> float | None:
    values = pd.to_numeric(pd.Series([row.get(field) for row in rows]), errors="coerce").dropna()
    return float((values > 0).mean()) if not values.empty else None


def _run_stats(connection: sqlite3.Connection, run_id: str) -> dict[str, float | int]:
    count, cost = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost), 0) FROM orders WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    signals = pd.read_sql_query(
        "SELECT signal_date, target_weight FROM signals WHERE run_id = ? AND target_weight > 0",
        connection,
        params=(run_id,),
    )
    grouped = signals.groupby("signal_date")["target_weight"]
    return {
        "order_count": int(count),
        "total_transaction_cost": float(cost),
        "average_holdings": float(grouped.count().mean()),
        "average_equity_exposure": float(grouped.sum().mean()),
    }


def _annual_row(connection: sqlite3.Connection, year: int, variant: str) -> dict:
    run_id = _run_id(year, variant)
    report = json.loads((_model_dir(year, variant) / "report.json").read_text(encoding="utf-8"))
    metrics = _artifact(connection, run_id, "metrics").get("metrics", {})
    selection = _artifact(connection, run_id, "monthly_ic")
    summary = selection.get("summary", {})
    monthly = selection.get("monthly", [])
    weighting = report.get("class_weighting", {})
    return {
        "oos_year": year,
        "variant": variant,
        "run_id": run_id,
        "positive_class_weight": weighting.get("positive_class_weight"),
        "parameter_source": weighting.get("parameter_source"),
        "validation_rank_ic": report.get("validation_rank_ic"),
        "validation_top_decile_spread": report.get("validation_top_decile_spread"),
        "validation_roc_auc": report.get("validation_roc_auc"),
        "validation_log_loss": report.get("validation_log_loss"),
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
        "continuous_rank_ic": summary.get("mean_continuous_rank_ic"),
        "ic_positive_month_ratio": summary.get("ic_positive_month_ratio"),
        "top15_actual_return": summary.get("mean_top15_actual_return"),
        "top15_excess_return": summary.get("mean_top15_excess_return"),
        "top15_excess_positive_month_ratio": summary.get(
            "top15_excess_positive_month_ratio"
        ),
        "top_bottom_spread": summary.get("mean_top_bottom_spread"),
        "top_bottom_positive_month_ratio": _positive_ratio(monthly, "top_bottom_spread"),
    }


def _monthly_rows(connection: sqlite3.Connection, year: int) -> list[dict]:
    baseline = pd.read_sql_query(
        "SELECT prediction_date, stock_id, rank FROM predictions WHERE run_id = ?",
        connection,
        params=(_run_id(year, "unweighted"),),
    )
    output: list[dict] = []
    for variant in VARIANTS:
        run_id = _run_id(year, variant)
        payload = _artifact(connection, run_id, "monthly_ic")
        predictions = pd.read_sql_query(
            "SELECT prediction_date, stock_id, rank FROM predictions WHERE run_id = ?",
            connection,
            params=(run_id,),
        )
        merged = baseline.merge(
            predictions,
            on=["prediction_date", "stock_id"],
            suffixes=("_baseline", "_variant"),
            validate="one_to_one",
        )
        similarities = {}
        for signal_date, frame in merged.groupby("prediction_date"):
            top_base = set(frame.nsmallest(15, "rank_baseline")["stock_id"])
            top_variant = set(frame.nsmallest(15, "rank_variant")["stock_id"])
            similarities[str(signal_date)] = {
                "prediction_rank_spearman_vs_unweighted": frame["rank_baseline"].corr(
                    frame["rank_variant"], method="spearman"
                ),
                "top15_overlap_ratio_vs_unweighted": len(top_base & top_variant) / 15.0,
            }
        for source in payload.get("monthly", []):
            row = {"oos_year": year, "variant": variant, "run_id": run_id, **source}
            row.update(similarities.get(str(source.get("signal_date")), {}))
            output.append(row)
    return output


def _stitched_metrics(connection: sqlite3.Connection, variant: str) -> dict[str, float]:
    returns = []
    for year in YEARS:
        payload = _artifact(connection, _run_id(year, variant), "metrics")
        curve = pd.DataFrame(payload["equity_curve"])
        returns.append(pd.to_numeric(curve["nav"], errors="coerce").pct_change().dropna())
    daily = pd.concat(returns, ignore_index=True)
    wealth = (1.0 + daily).cumprod()
    years = len(daily) / 252.0
    downside = daily.loc[daily < 0]
    return {
        "stitched_cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "stitched_sharpe": float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)),
        "stitched_sortino": float(daily.mean() / downside.std(ddof=1) * math.sqrt(252)),
        "stitched_max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
    }


def _summary(connection: sqlite3.Connection, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in annual.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                **_stitched_metrics(connection, variant),
                "mean_annual_cagr": float(group["cagr"].mean()),
                "mean_annual_sharpe": float(group["sharpe"].mean()),
                "mean_continuous_rank_ic": float(group["continuous_rank_ic"].mean()),
                "mean_top15_excess_return": float(group["top15_excess_return"].mean()),
                "mean_top_bottom_spread": float(group["top_bottom_spread"].mean()),
                "mean_turnover": float(group["turnover"].mean()),
                "positive_cagr_years": int((group["cagr"] > 0).sum()),
                "best_sharpe_years": 0,
            }
        )
    summary = pd.DataFrame(rows)
    winners = annual.loc[annual.groupby("oos_year")["sharpe"].idxmax(), "variant"].value_counts()
    summary["best_sharpe_years"] = summary["variant"].map(winners).fillna(0).astype(int)
    return summary


def _markdown(annual: pd.DataFrame, summary: pd.DataFrame) -> str:
    winner = summary.loc[summary["stitched_sharpe"].idxmax()]
    base = summary.loc[summary["variant"] == "unweighted"].iloc[0]
    fixed = summary.loc[summary["variant"] == "classweight4_fixedparams"].iloc[0]
    tuned = summary.loc[summary["variant"] == "classweight4_optuna"].iloc[0]
    lines = [
        "# XGBoost Binary Class Weight 4:1 — OOS 2020–2025",
        "",
        "## Protocol",
        "",
        "- Each year uses 47 training months, one purge month, and the prior year for validation.",
        "- Factor v4; no refit; Top15/Hold30; alpha 1.0; inverse-volatility weighting.",
        "- Class weights affect training loss only; validation and OOS metrics remain unweighted.",
        "",
        "## Six-year stitched results",
        "",
        "| Variant | CAGR | Sharpe | Sortino | MDD | Mean turnover | Best Sharpe years |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.stitched_cagr:.2%} | {row.stitched_sharpe:.3f} | "
            f"{row.stitched_sortino:.3f} | {row.stitched_max_drawdown:.2%} | "
            f"{row.mean_turnover:.2f} | {row.best_sharpe_years} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        f"- Highest stitched Sharpe: **{winner['variant']}** ({winner['stitched_sharpe']:.3f}).",
        (
            "- Fixed 4:1 weighting changed stitched CAGR by "
            f"{fixed['stitched_cagr'] - base['stitched_cagr']:+.2%} and Sharpe by "
            f"{fixed['stitched_sharpe'] - base['stitched_sharpe']:+.3f} versus unweighted."
        ),
        (
            "- Weighted Optuna changed stitched CAGR by "
            f"{tuned['stitched_cagr'] - base['stitched_cagr']:+.2%} and Sharpe by "
            f"{tuned['stitched_sharpe'] - base['stitched_sharpe']:+.3f} versus unweighted."
        ),
        (
            "- The fixed-parameter arm isolates class weighting; the Optuna arm also includes "
            "parameter reselection."
        ),
        (
            "- Continuous Rank IC and Top15 excess return remain the checks against a "
            "portfolio-only result."
        ),
        (
            "- 2025-08-01 contains only 323 local price rows, so July-end execution has "
            "incomplete coverage."
        ),
        "",
        "## Annual audit",
        "",
    ]
    for year in YEARS:
        lines.append(f"### {year}")
        for row in annual.loc[annual["oos_year"] == year].itertuples(index=False):
            lines.append(
                f"- {row.variant}: CAGR {row.cagr:.2%}, Sharpe {row.sharpe:.3f}, "
                f"MDD {row.max_drawdown:.2%}, turnover {row.turnover:.2f}."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ROOT / "database" / "quant.db") as connection:
        annual = pd.DataFrame(
            [_annual_row(connection, year, variant) for year in YEARS for variant in VARIANTS]
        )
        monthly = pd.DataFrame(
            [row for year in YEARS for row in _monthly_rows(connection, year)]
        )
        summary = _summary(connection, annual)
    for name, frame in (
        ("binary_classweight4_oos2020_2025_annual.csv", annual),
        ("binary_classweight4_oos2020_2025_monthly.csv", monthly),
        ("binary_classweight4_oos2020_2025_summary.csv", summary),
    ):
        frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig", lineterminator="\n")
    (OUTPUT / "binary_classweight4_oos2020_2025_comparison.md").write_text(
        _markdown(annual, summary), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
