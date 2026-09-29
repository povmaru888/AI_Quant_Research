"""Compare v3 panels, model OOS scores, and materialized portfolio metrics."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import load_settings  # noqa: E402


def _months(start: str, end: str) -> list[str]:
    result: list[str] = []
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _read_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _panel_summary(directory: Path, months: list[str]) -> dict:
    universe: dict[str, int] = {}
    distributions: dict[str, list[np.ndarray]] = {}
    versions: set[str] = set()
    for month in months:
        path = directory / f"{month}.pkl"
        with path.open("rb") as handle:
            panel = pickle.load(handle)
        frame = panel["frame"]
        universe[month] = len(panel["universe"])
        versions.add(str(panel["feature_version"]))
        for column in frame.columns:
            if column in ("stock_id", "missing_flag"):
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                distributions.setdefault(str(column), []).append(finite)

    summarized: dict[str, dict] = {}
    for feature, chunks in sorted(distributions.items()):
        values = np.concatenate(chunks)
        summarized[feature] = {
            "n": int(len(values)),
            "mean": _finite(np.mean(values)),
            "std": _finite(np.std(values, ddof=1)) if len(values) > 1 else None,
            "p01": _finite(np.quantile(values, 0.01)),
            "p50": _finite(np.quantile(values, 0.50)),
            "p99": _finite(np.quantile(values, 0.99)),
        }
    counts = list(universe.values())
    return {
        "feature_versions": sorted(versions),
        "universe_by_month": universe,
        "universe_count_summary": {
            "months": len(counts),
            "mean": _finite(np.mean(counts)) if counts else None,
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
        },
        "feature_distributions": summarized,
    }


def _deltas(old: dict, new: dict) -> dict:
    keys = sorted(set(old) | set(new))
    result: dict[str, dict] = {}
    for key in keys:
        old_value, new_value = _finite(old.get(key)), _finite(new.get(key))
        result[key] = {
            "old": old_value,
            "new": new_value,
            "delta": (new_value - old_value) if old_value is not None and new_value is not None else None,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.pit_v3.yaml")
    parser.add_argument("--old-panels", default="data/panels")
    parser.add_argument("--new-panels", default="data/panels_pit_v3")
    parser.add_argument("--old-model-report", default="models/model_b2/report.json")
    parser.add_argument("--new-model-report", default="models/model_b2_pit_v3/report.json")
    parser.add_argument("--old-run-id", default="oos-2024-b2")
    parser.add_argument("--new-run-id", default="oos-2024-b2-pit-v3")
    parser.add_argument("--start", default="2019-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--out", default="reports/pit_v3_comparison.json")
    args = parser.parse_args(argv)
    load_dotenv()
    settings = load_settings(args.config)
    months = _months(args.start, args.end)
    old_panels = _panel_summary(Path(args.old_panels), months)
    new_panels = _panel_summary(Path(args.new_panels), months)
    old_model = _read_report(Path(args.old_model_report))
    new_model = _read_report(Path(args.new_model_report))

    old_features = old_panels["feature_distributions"]
    new_features = new_panels["feature_distributions"]
    feature_comparison = {}
    for feature in sorted(set(old_features) | set(new_features)):
        left, right = old_features.get(feature), new_features.get(feature)
        feature_comparison[feature] = {
            "old": left,
            "new": right,
            "mean_delta": (
                _finite(right["mean"] - left["mean"])
                if left is not None and right is not None
                else None
            ),
        }

    store = build_store(settings)
    old_metrics_artifact = store._artifact(args.old_run_id, "metrics")
    new_metrics_artifact = store._artifact(args.new_run_id, "metrics")
    old_metrics = (old_metrics_artifact or {}).get("metrics", {})
    new_metrics = (new_metrics_artifact or {}).get("metrics", {})
    portfolio_keys = ("cagr", "sharpe", "max_drawdown", "turnover", "realized_volatility", "rank_ic", "icir")
    old_portfolio = {key: old_metrics.get(key) for key in portfolio_keys}
    new_portfolio = {key: new_metrics.get(key) for key in portfolio_keys}
    result = {
        "period": [args.start, args.end],
        "data_quality_note": (
            "factor_adj_pit_v3 uses exact-day market_value only; old shares.json values are not used."
        ),
        "universe": {
            "old_mean": old_panels["universe_count_summary"],
            "new_mean": new_panels["universe_count_summary"],
            "monthly_counts": {
                month: {
                    "old": old_panels["universe_by_month"][month],
                    "new": new_panels["universe_by_month"][month],
                }
                for month in months
            },
        },
        "feature_distributions": feature_comparison,
        "model_oos": {
            "rank_ic": _deltas(
                {"value": old_model.get("oos_rank_ic")},
                {"value": new_model.get("oos_rank_ic")},
            )["value"],
            "top_decile_spread": _deltas(
                {"value": old_model.get("oos_top_decile_spread")},
                {"value": new_model.get("oos_top_decile_spread")},
            )["value"],
            "monthly_rank_ic": {
                month: {
                    "old": _finite(old_model.get("oos_monthly_ic", {}).get(month)),
                    "new": _finite(new_model.get("oos_monthly_ic", {}).get(month)),
                }
                for month in _months("2024-01", "2024-12")
            },
        },
        "portfolio_oos": {
            "available": old_metrics_artifact is not None and new_metrics_artifact is not None,
            "metrics": _deltas(old_portfolio, new_portfolio),
            "old_run_id": args.old_run_id,
            "new_run_id": args.new_run_id,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "universe_mean_old": old_panels["universe_count_summary"]["mean"],
        "universe_mean_new": new_panels["universe_count_summary"]["mean"],
        "rank_ic_old": result["model_oos"]["rank_ic"]["old"],
        "rank_ic_new": result["model_oos"]["rank_ic"]["new"],
        "portfolio_metrics_available": result["portfolio_oos"]["available"],
        "out": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
