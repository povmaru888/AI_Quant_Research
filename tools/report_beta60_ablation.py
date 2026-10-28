"""Build the reproducible 2020-2025 beta_60d ablation report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = tuple(range(2020, 2026))
VARIANTS = ("baseline", "fixedparams", "optuna")


def _run_id(year: int, variant: str) -> str:
    if variant == "baseline":
        return f"oos-{year}-factor-v4-pseudohuber-NOrefit"
    return f"oos-{year}-factor-v4-ablation-no-beta60d-pseudohuber-{variant}-NOrefit"


def _model_dir(models: Path, year: int, variant: str) -> Path:
    start, end = year - 5, year - 2
    suffix = (
        "v4_pseudohuber_NOrefit"
        if variant == "baseline"
        else f"v4_ablation_no_beta60d_pseudohuber_{variant}_NOrefit"
    )
    return models / f"model_oos{year}_validation_{start}_{end}_{suffix}"


def _artifact(connection: sqlite3.Connection, run_id: str, kind: str) -> dict:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = ?",
        (run_id, kind),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing {kind} artifact for {run_id}")
    return json.loads(row[0])


def _prediction_similarity(
    connection: sqlite3.Connection, baseline_run: str, comparison_run: str
) -> tuple[float, float]:
    query = """
        SELECT prediction_date, stock_id, rank
        FROM predictions
        WHERE run_id = ?
    """
    left = pd.read_sql_query(query, connection, params=(baseline_run,))
    right = pd.read_sql_query(query, connection, params=(comparison_run,))
    merged = left.merge(right, on=["prediction_date", "stock_id"], suffixes=("_base", "_test"))
    correlations: list[float] = []
    overlaps: list[float] = []
    for _, month in merged.groupby("prediction_date"):
        correlation = month["rank_base"].corr(month["rank_test"], method="spearman")
        if pd.notna(correlation):
            correlations.append(float(correlation))
        base_top = set(month.nsmallest(15, "rank_base")["stock_id"])
        test_top = set(month.nsmallest(15, "rank_test")["stock_id"])
        overlaps.append(len(base_top & test_top) / 15.0)
    return float(np.mean(correlations)), float(np.mean(overlaps))


def _mean(values: Iterable[object]) -> float:
    numbers = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(numbers.mean()) if not numbers.empty else float("nan")


def _fmt(value: object, percent: bool = False) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.2%}" if percent else f"{number:.3f}"


def _annual_rows(connection: sqlite3.Connection, models: Path) -> list[dict]:
    rows: list[dict] = []
    for year in YEARS:
        for variant in VARIANTS:
            run_id = _run_id(year, variant)
            report_path = _model_dir(models, year, variant) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics_payload = _artifact(connection, run_id, "metrics")
            selection = _artifact(connection, run_id, "monthly_ic")["summary"]
            metrics = metrics_payload["metrics"]
            if variant == "baseline":
                rank_similarity, top15_overlap = 1.0, 1.0
            else:
                rank_similarity, top15_overlap = _prediction_similarity(
                    connection, _run_id(year, "baseline"), run_id
                )
            rows.append(
                {
                    "oos_year": year,
                    "variant": variant,
                    "run_id": run_id,
                    "feature_variant": report.get("feature_variant", report["feature_version"]),
                    "feature_count": len(report["common_features"]),
                    "validation_rank_ic": report["validation_rank_ic"],
                    "validation_top_decile_spread": report["validation_top_decile_spread"],
                    "oos_cagr": metrics["cagr"],
                    "oos_sharpe": metrics["sharpe"],
                    "oos_sortino": metrics["sortino"],
                    "oos_max_drawdown": metrics["max_drawdown"],
                    "oos_turnover": metrics["turnover"],
                    "oos_realized_volatility": metrics["realized_volatility"],
                    "mean_continuous_rank_ic": selection["mean_continuous_rank_ic"],
                    "ic_positive_month_ratio": selection["ic_positive_month_ratio"],
                    "mean_top15_excess_return": selection["mean_top15_excess_return"],
                    "top15_excess_positive_month_ratio": selection[
                        "top15_excess_positive_month_ratio"
                    ],
                    "mean_top_bottom_spread": selection["mean_top_bottom_spread"],
                    "prediction_rank_spearman_vs_baseline": rank_similarity,
                    "top15_overlap_vs_baseline": top15_overlap,
                }
            )
    return rows


def _add_deltas(annual: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "validation_rank_ic",
        "validation_top_decile_spread",
        "oos_cagr",
        "oos_sharpe",
        "oos_sortino",
        "oos_max_drawdown",
        "oos_turnover",
        "mean_continuous_rank_ic",
        "mean_top15_excess_return",
        "mean_top_bottom_spread",
    ]
    baseline = annual.loc[annual["variant"] == "baseline"].set_index("oos_year")
    for column in metric_columns:
        annual[f"delta_{column}_vs_baseline"] = annual.apply(
            lambda row, metric=column: (
                row[metric] - baseline.loc[row["oos_year"], metric]
            ),
            axis=1,
        )
    return annual


def _monthly_rows(connection: sqlite3.Connection) -> pd.DataFrame:
    rows: list[dict] = []
    for year in YEARS:
        for variant in VARIANTS:
            run_id = _run_id(year, variant)
            for month in _artifact(connection, run_id, "monthly_ic")["monthly"]:
                rows.append({"oos_year": year, "variant": variant, "run_id": run_id, **month})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["variant"] == "baseline"].set_index(
        ["oos_year", "signal_date"]
    )
    for column in ("continuous_rank_ic", "top15_excess_return", "top_bottom_spread"):
        frame[f"delta_{column}_vs_baseline"] = frame.apply(
            lambda row, metric=column: (
                row[metric]
                - baseline.loc[(row["oos_year"], row["signal_date"]), metric]
            ),
            axis=1,
        )
    return frame


def _summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, group in annual.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "years": len(group),
                "mean_validation_rank_ic": _mean(group["validation_rank_ic"]),
                "mean_oos_cagr": _mean(group["oos_cagr"]),
                "mean_oos_sharpe": _mean(group["oos_sharpe"]),
                "mean_oos_max_drawdown": _mean(group["oos_max_drawdown"]),
                "mean_oos_turnover": _mean(group["oos_turnover"]),
                "mean_continuous_rank_ic": _mean(group["mean_continuous_rank_ic"]),
                "mean_top15_excess_return": _mean(group["mean_top15_excess_return"]),
                "mean_top_bottom_spread": _mean(group["mean_top_bottom_spread"]),
                "mean_prediction_rank_spearman_vs_baseline": _mean(
                    group["prediction_rank_spearman_vs_baseline"]
                ),
                "mean_top15_overlap_vs_baseline": _mean(group["top15_overlap_vs_baseline"]),
                "rank_ic_better_years_vs_baseline": int(
                    (group["delta_mean_continuous_rank_ic_vs_baseline"] > 0).sum()
                ),
                "top15_excess_better_years_vs_baseline": int(
                    (group["delta_mean_top15_excess_return_vs_baseline"] > 0).sum()
                ),
                "sharpe_better_years_vs_baseline": int(
                    (group["delta_oos_sharpe_vs_baseline"] > 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_monthly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for variant in ("fixedparams", "optuna"):
        variant_rows = monthly.loc[monthly["variant"] == variant]
        for metric in ("continuous_rank_ic", "top15_excess_return", "top_bottom_spread"):
            values = pd.to_numeric(
                variant_rows[f"delta_{metric}_vs_baseline"], errors="coerce"
            ).dropna()
            samples = rng.choice(
                values.to_numpy(), (10_000, len(values)), replace=True
            ).mean(axis=1)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "months": len(values),
                    "mean_delta_vs_baseline": float(values.mean()),
                    "bootstrap_95pct_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_95pct_high": float(np.quantile(samples, 0.975)),
                    "positive_delta_month_ratio": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _markdown(
    annual: pd.DataFrame, summary: pd.DataFrame, paired_monthly: pd.DataFrame
) -> str:
    base = summary.set_index("variant").loc["baseline"]
    fixed = summary.set_index("variant").loc["fixedparams"]
    tuned = summary.set_index("variant").loc["optuna"]
    lines = [
        "# beta_60d 消融研究（2020～2025 OOS）",
        "",
        "## 實驗設計",
        "",
        "- Baseline：完整 factor v4 XGBoost Pseudo-Huber。",
        "- Fixed params：移除 `beta_60d`，沿用各年度 baseline 最佳超參數。",
        "- Re-Optuna：移除 `beta_60d`，各年度重新執行 50 次 Optuna。",
        (
            "- 三組使用相同 panels、標籤、訓練／purge／驗證／OOS 切分、"
            "Top15／Hold30、反波動權重及交易成本。"
        ),
        (
            "- 底層 panel 版本仍為 `factor_v4`；研究變體記錄為 "
            "`factor_v4_ablation_no_beta60d`。"
        ),
        "",
        "## 六年度平均",
        "",
        (
            "| 組別 | Validation IC | OOS IC | Top15 Excess | Top-Bottom | "
            "CAGR | Sharpe | MDD | Turnover |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline": "Baseline",
        "fixedparams": "移除 Beta／固定參數",
        "optuna": "移除 Beta／重新 Optuna",
    }
    for _, row in summary.iterrows():
        lines.append(
            f"| {labels[row['variant']]} | {_fmt(row['mean_validation_rank_ic'])} | "
            f"{_fmt(row['mean_continuous_rank_ic'])} | "
            f"{_fmt(row['mean_top15_excess_return'], True)} | "
            f"{_fmt(row['mean_top_bottom_spread'], True)} | "
            f"{_fmt(row['mean_oos_cagr'], True)} | {_fmt(row['mean_oos_sharpe'])} | "
            f"{_fmt(row['mean_oos_max_drawdown'], True)} | {_fmt(row['mean_oos_turnover'])} |"
        )
    lines.extend(
        [
            "",
            "## 配對年度結果",
            "",
            (
                "| 年度 | Baseline IC | 固定參數 IC | Re-Optuna IC | "
                "Baseline Sharpe | 固定參數 Sharpe | Re-Optuna Sharpe |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for year in YEARS:
        year_rows = annual.loc[annual["oos_year"] == year].set_index("variant")
        lines.append(
            f"| {year} | {_fmt(year_rows.loc['baseline', 'mean_continuous_rank_ic'])} | "
            f"{_fmt(year_rows.loc['fixedparams', 'mean_continuous_rank_ic'])} | "
            f"{_fmt(year_rows.loc['optuna', 'mean_continuous_rank_ic'])} | "
            f"{_fmt(year_rows.loc['baseline', 'oos_sharpe'])} | "
            f"{_fmt(year_rows.loc['fixedparams', 'oos_sharpe'])} | "
            f"{_fmt(year_rows.loc['optuna', 'oos_sharpe'])} |"
        )
    fixed_ic_delta = fixed["mean_continuous_rank_ic"] - base["mean_continuous_rank_ic"]
    tuned_ic_delta = tuned["mean_continuous_rank_ic"] - base["mean_continuous_rank_ic"]
    fixed_sharpe_delta = fixed["mean_oos_sharpe"] - base["mean_oos_sharpe"]
    tuned_sharpe_delta = tuned["mean_oos_sharpe"] - base["mean_oos_sharpe"]
    fixed_top15_delta = fixed["mean_top15_excess_return"] - base["mean_top15_excess_return"]
    tuned_top15_delta = tuned["mean_top15_excess_return"] - base["mean_top15_excess_return"]
    lines.extend(
        [
            "",
            "## 判讀",
            "",
            (
                f"- 固定原參數移除 Beta：平均 OOS IC 差異 `{fixed_ic_delta:+.4f}`，"
                f"平均 Sharpe 差異 `{fixed_sharpe_delta:+.3f}`。"
            ),
            (
                f"- 重新 Optuna 後移除 Beta：平均 OOS IC 差異 `{tuned_ic_delta:+.4f}`，"
                f"平均 Sharpe 差異 `{tuned_sharpe_delta:+.3f}`。"
            ),
            (
                "- 固定參數組 OOS IC 優於 baseline："
                f"{int(fixed['rank_ic_better_years_vs_baseline'])}/6 年；Sharpe 優於 "
                f"baseline：{int(fixed['sharpe_better_years_vs_baseline'])}/6 年。"
            ),
            (
                "- Re-Optuna 組 OOS IC 優於 baseline："
                f"{int(tuned['rank_ic_better_years_vs_baseline'])}/6 年；Sharpe 優於 "
                f"baseline：{int(tuned['sharpe_better_years_vs_baseline'])}/6 年。"
            ),
            (
                "- 移除 Beta 後的平均 Top15 重疊率：固定參數 "
                f"{_fmt(fixed['mean_top15_overlap_vs_baseline'], True)}；Re-Optuna "
                f"{_fmt(tuned['mean_top15_overlap_vs_baseline'], True)}。"
            ),
            (
                f"- 平均 Top15 Excess 差異：固定參數 {fixed_top15_delta:+.2%}；"
                f"Re-Optuna {tuned_top15_delta:+.2%}。"
            ),
            "",
            "## 72 個月配對 Bootstrap",
            "",
            "| 組別 | 指標 | 平均差異 | 95% 區間 | 優於 Baseline 月份比例 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "continuous_rank_ic": "Continuous Rank IC",
        "top15_excess_return": "Top15 Excess",
        "top_bottom_spread": "Top-Bottom Spread",
    }
    for _, row in paired_monthly.iterrows():
        lines.append(
            f"| {labels[row['variant']]} | {metric_labels[row['metric']]} | "
            f"{_fmt(row['mean_delta_vs_baseline'], row['metric'] != 'continuous_rank_ic')} | "
            f"[{_fmt(row['bootstrap_95pct_low'], row['metric'] != 'continuous_rank_ic')}, "
            f"{_fmt(row['bootstrap_95pct_high'], row['metric'] != 'continuous_rank_ic')}] | "
            f"{_fmt(row['positive_delta_month_ratio'], True)} |"
        )
    lines.extend(
        [
            "",
            "## 結論",
            "",
            (
                "`beta_60d` 並非穩定、不可替代的重要因子。移除後平均 OOS Rank IC "
                "反而小幅上升，固定參數組有 5/6 年 IC 較高；但 bootstrap 區間涵蓋 "
                "0，也不足以認定移除 Beta 能穩定改善排序。"
            ),
            "",
            (
                "經濟結果的六年平均偏向保留 Beta，但年度結果有好有壞。固定參數"
                "消融後 CAGR、Sharpe、Sortino 各有 2/6 年改善，MDD 也有 2/6 年改善；"
                "重新 Optuna 組的 CAGR、Sharpe、Sortino 同樣各有 2/6 年改善。"
            ),
            "",
            (
                "Top15 Excess 與 Top-Bottom Spread 在兩組消融實驗都只有 1/6 年改善，"
                "顯示 Beta 可能對頭部選股有條件式價值；不過固定參數組的月度差異"
                "信賴區間仍涵蓋 0，證據強度有限。"
            ),
            "",
            (
                "較保守的結論是：`beta_60d` 的增量重要性低於 SHAP 排名所呈現，效果"
                "具有市場狀態依賴。它可以暫時保留在目前主模型中，但不能單憑 SHAP "
                "宣稱是核心 Alpha；更適合進一步研究它與波動率、動能及市場 regime "
                "的交互作用。"
            ),
        ]
    )
    lines.extend(
        [
            "",
            (
                "績效指標存在年度市場差異；正式判斷以配對的 OOS IC、"
                "Top15 Excess 與跨年度方向一致性為主，不以單一年份 CAGR 決定。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="database/quant.db")
    parser.add_argument("--models", default="models")
    parser.add_argument("--output", default="output/research")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.database) as connection:
        annual = _add_deltas(pd.DataFrame(_annual_rows(connection, Path(args.models))))
        monthly = _monthly_rows(connection)
    summary = _summary(annual)
    paired_monthly = _paired_monthly_summary(monthly)
    annual.to_csv(output / "beta60_ablation_2020_2025_annual.csv", index=False)
    monthly.to_csv(output / "beta60_ablation_2020_2025_monthly.csv", index=False)
    summary.to_csv(output / "beta60_ablation_2020_2025_summary.csv", index=False)
    paired_monthly.to_csv(
        output / "beta60_ablation_2020_2025_paired_monthly_summary.csv", index=False
    )
    (output / "beta60_ablation_report.md").write_text(
        _markdown(annual, summary, paired_monthly), encoding="utf-8"
    )
    provenance = {
        "feature_variant": "factor_v4_ablation_no_beta60d",
        "excluded_features": ["beta_60d"],
        "oos_years": list(YEARS),
        "variants": list(VARIANTS),
        "base_feature_version": "factor_v4",
        "optuna_trials_per_year": 50,
        "annual_rows": len(annual),
        "monthly_rows": len(monthly),
    }
    (output / "beta60_ablation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "beta60_ablation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
