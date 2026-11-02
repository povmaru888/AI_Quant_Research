"""Analyze one-year performance stability from Pseudo-Huber OOS monthly blocks."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.bootstrap_stability_service import (  # noqa: E402,I001
    MonthBlock,
    calculate_path_metrics,
    extract_month_blocks,
    simulate_paths,
    summarize_simulations,
)


RUN_TEMPLATE = "oos-{year}-factor-v4-pseudohuber-NOrefit"
INITIAL_CAPITAL = 30_000_000.0
METHOD_LABELS = {
    "monthly_iid": "單月 Bootstrap",
    "circular_3m": "三個月 Circular Block Bootstrap",
}


def _artifact(connection: sqlite3.Connection, run_id: str) -> dict:
    row = connection.execute(
        "SELECT payload_json FROM run_artifacts WHERE run_id = ? AND kind = 'metrics'",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"missing metrics artifact: {run_id}")
    payload = json.loads(row[0])
    if not isinstance(payload, dict):
        raise ValueError(f"invalid metrics artifact: {run_id}")
    return payload


def _known_data_issue(connection: sqlite3.Connection) -> tuple[str | None, dict[str, object]]:
    price_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM prices WHERE trade_date = '2025-08-01'"
        ).fetchone()[0]
    )
    run_id = RUN_TEMPLATE.format(year=2025)
    order_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM orders WHERE run_id = ? AND signal_date = '2025-07-31'",
            (run_id,),
        ).fetchone()[0]
    )
    detail = {"price_rows_2025_08_01": price_count, "orders_for_2025_07_31": order_count}
    if price_count < 1_000 or order_count == 0:
        return (
            "2025-08-01 價格僅有 "
            f"{price_count} 筆，且 2025-07-31 訊號訂單數為 {order_count}；"
            "該月沿用舊持股，並非完整執行預定換倉。",
            detail,
        )
    return None, detail


def _load_blocks(
    connection: sqlite3.Connection, start_year: int, end_year: int
) -> tuple[list[MonthBlock], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    issue, issue_audit = _known_data_issue(connection)
    all_blocks: list[MonthBlock] = []
    actual_rows: list[dict] = []
    audit_rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        run_id = RUN_TEMPLATE.format(year=year)
        payload = _artifact(connection, run_id)
        blocks = extract_month_blocks(
            run_id=run_id,
            equity_curve=payload.get("equity_curve", []),
            monthly_returns=payload.get("monthly_returns", []),
            expected_months=payload.get("oos_months", []),
            initial_capital=INITIAL_CAPITAL,
            known_issue_months={"2025-08": issue} if issue else None,
        )
        if len(blocks) != 12:
            raise ValueError(f"expected 12 OOS months for {run_id}, got {len(blocks)}")
        blocks = [
            dataclasses.replace(block, source_index=len(all_blocks) + offset)
            for offset, block in enumerate(blocks)
        ]
        all_blocks.extend(blocks)
        annual_returns = np.concatenate([block.daily_returns for block in blocks])
        calculated = calculate_path_metrics(annual_returns)
        dashboard = payload.get("metrics", {})
        actual_rows.append(
            {
                "oos_year": year,
                "run_id": run_id,
                **calculated,
                **{
                    f"dashboard_{metric}": dashboard.get(metric)
                    for metric in ("cagr", "sharpe", "sortino", "max_drawdown")
                },
                **{
                    f"delta_vs_dashboard_{metric}": calculated[metric] - dashboard.get(metric)
                    for metric in ("cagr", "sharpe", "sortino", "max_drawdown")
                },
            }
        )
        for block in blocks:
            audit_rows.append(
                {
                    "source_index": block.source_index,
                    "run_id": block.run_id,
                    "month": block.month,
                    "start_date": block.start_date,
                    "end_date": block.end_date,
                    "trading_days": len(block.daily_returns),
                    "derived_monthly_return": block.compounded_return,
                    "artifact_monthly_return": block.artifact_monthly_return,
                    "artifact_validation_return": block.artifact_validation_return,
                    "artifact_validation_absolute_difference": abs(
                        block.artifact_validation_return - block.artifact_monthly_return
                    ),
                    "first_day_return_included": float(block.daily_returns[0]),
                    "known_data_issue": block.known_data_issue,
                }
            )
    expected_count = (end_year - start_year + 1) * 12
    if len(all_blocks) != expected_count:
        raise ValueError(f"expected {expected_count} monthly blocks, got {len(all_blocks)}")
    return (
        all_blocks,
        pd.DataFrame(actual_rows),
        pd.DataFrame(audit_rows),
        issue_audit,
    )


def _write_distribution_chart(
    paths: pd.DataFrame, actual: pd.DataFrame, destination: Path
) -> None:
    metrics = (
        ("cagr", "CAGR", ".1%"),
        ("sharpe", "Sharpe", ".2f"),
        ("sortino", "Sortino", ".2f"),
        ("max_drawdown", "MDD", ".1%"),
    )
    colors = {"monthly_iid": "#35d0ba", "circular_3m": "#8d7cf6"}
    figure = make_subplots(rows=2, cols=2, subplot_titles=[item[1] for item in metrics])
    for position, (metric, _, hover_format) in enumerate(metrics):
        row, column = position // 2 + 1, position % 2 + 1
        for method in METHOD_LABELS:
            values = paths.loc[paths["method"] == method, metric]
            figure.add_trace(
                go.Histogram(
                    x=values,
                    name=METHOD_LABELS[method],
                    marker_color=colors[method],
                    opacity=0.58,
                    histnorm="probability",
                    nbinsx=55,
                    hovertemplate=(
                        f"{metric}: %{{x:{hover_format}}}<br>"
                        "比例: %{y:.2%}<extra></extra>"
                    ),
                    legendgroup=method,
                    showlegend=position == 0,
                ),
                row=row,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=actual[metric],
                y=np.zeros(len(actual)),
                mode="markers",
                marker={"symbol": "line-ns", "size": 16, "color": "#ffbd59", "line_width": 3},
                text=[f"OOS {year}" for year in actual["oos_year"]],
                name="實際年度",
                hovertemplate=f"%{{text}}<br>{metric}: %{{x:{hover_format}}}<extra></extra>",
                legendgroup="actual",
                showlegend=position == 0,
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Pseudo-Huber OOS 2020–2025：模擬一年績效分布",
        template="plotly_dark",
        barmode="overlay",
        height=820,
        margin={"l": 55, "r": 35, "t": 85, "b": 50},
        paper_bgcolor="#090d14",
        plot_bgcolor="#101722",
        legend={"orientation": "h", "y": 1.06, "x": 0.5, "xanchor": "center"},
    )
    figure.write_html(destination, include_plotlyjs=True, full_html=True)


def _format_value(metric: str, value: float) -> str:
    if metric in {"cumulative_return", "cagr", "max_drawdown"}:
        return f"{value:.2%}"
    return f"{value:.3f}"


def _write_report(
    destination: Path,
    paths: pd.DataFrame,
    summary: pd.DataFrame,
    actual: pd.DataFrame,
    audit: pd.DataFrame,
    issue_audit: dict[str, object],
) -> None:
    final = summary.loc[summary["sample_size"] == 5000]
    metrics = ("cagr", "sharpe", "sortino", "max_drawdown")
    lines = [
        "# Pseudo-Huber OOS 2020～2025 Bootstrap 穩定度分析",
        "",
        (
            "本報告以 72 個歷史持有月份重抽模擬一年績效。分布代表歷史行情月份"
            "可重複出現時的情境範圍，不是未來績效保證或六年平均績效的信賴區間。"
        ),
        "",
        "## 5,000 次模擬結果",
        "",
        "| 方法 | 指標 | 平均 | 中位數 | P5 | P25 | P75 | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_LABELS:
        for metric in metrics:
            row = final.loc[(final["method"] == method) & (final["metric"] == metric)].iloc[0]
            values = [_format_value(metric, float(row[field])) for field in (
                "mean", "median", "p05", "p25", "p75", "p95"
            )]
            lines.append(f"| {METHOD_LABELS[method]} | {metric} | " + " | ".join(values) + " |")

    lines += ["", "## 風險事件機率", ""]
    for method in METHOD_LABELS:
        row = final.loc[
            (final["method"] == method) & (final["metric"] == "event_probabilities")
        ].iloc[0]
        lines.append(
            f"- {METHOD_LABELS[method]}：一年累積報酬為負 "
            f"{row['prob_cumulative_return_lt_0']:.1%}；"
            f"Sharpe < 0 為 {row['prob_sharpe_lt_0']:.1%}；MDD 達 10%／20%／30% 分別為 "
            f"{row['prob_mdd_le_minus_10pct']:.1%}／{row['prob_mdd_le_minus_20pct']:.1%}／"
            f"{row['prob_mdd_le_minus_30pct']:.1%}。"
        )

    lines += [
        "",
        "## 實際六個 OOS 年度",
        "",
        "| OOS | CAGR | Sharpe | Sortino | MDD | 交易日 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in actual.itertuples(index=False):
        lines.append(
            f"| {row.oos_year} | {row.cagr:.2%} | {row.sharpe:.3f} | {row.sortino:.3f} | "
            f"{row.max_drawdown:.2%} | {row.trading_days} |"
        )

    convergence = []
    for method in METHOD_LABELS:
        for metric in metrics:
            small = summary.loc[
                (summary["method"] == method)
                & (summary["sample_size"] == 2000)
                & (summary["metric"] == metric)
            ].iloc[0]
            large = final.loc[(final["method"] == method) & (final["metric"] == metric)].iloc[0]
            convergence.append(
                max(
                    abs(float(large[field]) - float(small[field]))
                    for field in ("p05", "median", "p95")
                )
            )
    issue_rows = audit.loc[audit["known_data_issue"].notna()]
    lines += [
        "",
        "## 驗證與解讀限制",
        "",
        (
            "- 72 個來源月份均通過每日複利與 artifact 月報酬核對；最大絕對差為 "
            f"{audit['artifact_validation_absolute_difference'].max():.3e}。"
        ),
        (
            "- 現有 artifact 月報酬採每月最後 NAV ÷ 每月第一 NAV − 1，因此漏掉"
            "每個月第一個交易日；本分析已從每日 NAV 補回並保留於月份稽核 CSV。"
        ),
        (
            "- 2,000 次增加至 5,000 次時，兩種方法四項指標的 P5／中位數／P95 "
            f"最大變動為 {max(convergence):.4f}。"
        ),
        "- Sortino 沿用儀表板公式：每日平均報酬 ÷ 負報酬日樣本標準差 × √252，無風險利率為 0。",
        "- 模擬重用已實現且已扣成本的日報酬；抽樣接縫不重新換倉，也不額外扣交易成本。",
        "- 三個月 circular block 在 72 個月末端會接回開頭，該接縫是抽樣設計，不代表真實時間連續。",
        "- 六年樣本有限且模型經過研究選擇；Bootstrap 無法消除選模偏誤，也無法創造歷史未見的危機。",
    ]
    if not issue_rows.empty:
        lines.append(
            "- 資料品質警示："
            + str(issue_rows.iloc[0]["known_data_issue"])
            + f" 資料庫稽核值：{issue_audit}。因此 2025-08 的報酬代表現有已物化"
            "策略路徑，並非完整執行預定換倉後的反事實績效。"
        )
    dashboard_deltas = actual[
        [f"delta_vs_dashboard_{metric}" for metric in ("cagr", "sharpe", "sortino", "max_drawdown")]
    ].abs().max()
    lines.append(
        "- 本報告把年度首日相對三千萬元的報酬納入，並以實際日報酬筆數年化；"
        "與儀表板既有計算的最大絕對差為："
        + "、".join(
            f"{name.removeprefix('delta_vs_dashboard_')}={value:.6f}"
            for name, value in dashboard_deltas.items()
        )
        + "。"
    )
    lines += [
        "",
        "## 產出檔案",
        "",
        "- `pseudohuber_bootstrap_paths_2020_2025.csv`：10,000 條路徑指標。",
        "- `pseudohuber_bootstrap_summary_2020_2025.csv`：500／2,000／5,000 次收斂摘要。",
        "- `pseudohuber_bootstrap_months_2020_2025.csv`：72 個來源月份稽核。",
        "- `pseudohuber_bootstrap_sample_indices_2020_2025.csv`：每條路徑的月份抽樣索引。",
        "- `pseudohuber_bootstrap_actual_years_2020_2025.csv`：實際年度同公式指標。",
        "- `pseudohuber_bootstrap_distributions_2020_2025.html`：互動式分布圖。",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="database/quant.db")
    parser.add_argument("--output", default="output/research")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.simulations != 5000:
        parser.error("this preregistered analysis requires exactly 5000 simulations per method")

    database = Path(args.database).resolve()
    uri = f"file:{database.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        blocks, actual, month_audit, issue_audit = _load_blocks(
            connection, args.start_year, args.end_year
        )
    monthly_paths, monthly_indices = simulate_paths(
        blocks, args.simulations, "monthly_iid", seed=args.seed
    )
    block_paths, block_indices = simulate_paths(
        blocks, args.simulations, "circular_3m", seed=args.seed
    )
    paths = pd.concat([monthly_paths, block_paths], ignore_index=True)
    sample_indices = pd.concat([monthly_indices, block_indices], ignore_index=True)
    summary = summarize_simulations(paths)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "pseudohuber_bootstrap"
    outputs = {
        "paths": output / f"{prefix}_paths_2020_2025.csv",
        "summary": output / f"{prefix}_summary_2020_2025.csv",
        "months": output / f"{prefix}_months_2020_2025.csv",
        "indices": output / f"{prefix}_sample_indices_2020_2025.csv",
        "actual": output / f"{prefix}_actual_years_2020_2025.csv",
        "chart": output / f"{prefix}_distributions_2020_2025.html",
        "report": output / f"{prefix}_report_2020_2025.md",
    }
    paths.to_csv(outputs["paths"], index=False, encoding="utf-8-sig", lineterminator="\n")
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig", lineterminator="\n")
    month_audit.to_csv(outputs["months"], index=False, encoding="utf-8-sig", lineterminator="\n")
    sample_indices.to_csv(
        outputs["indices"], index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    actual.to_csv(outputs["actual"], index=False, encoding="utf-8-sig", lineterminator="\n")
    _write_distribution_chart(paths, actual, outputs["chart"])
    _write_report(outputs["report"], paths, summary, actual, month_audit, issue_audit)
    print(f"validated monthly blocks: {len(blocks)}")
    print(f"simulated paths: {len(paths)}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
