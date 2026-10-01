"""Audit factor_v4 panel invariants and summarize monthly attrition."""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.panel_data import canonical_panel_hash  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit factor_v4 panels.")
    parser.add_argument("--panels", default="data/panels_v4")
    parser.add_argument("--compare", default="data/panels")
    parser.add_argument("--out", default="reports/factor_v4_audit.json")
    args = parser.parse_args(argv)
    panel_dir = Path(args.panels)
    files = sorted(panel_dir.glob("????-??.pkl"))
    if not files:
        raise FileNotFoundError(f"no panels in {panel_dir}")
    counts: list[int] = []
    coverages: list[float] = []
    excluded = Counter()
    feature_schemas: set[tuple[str, ...]] = set()
    failures: list[str] = []
    month_counts: dict[str, int] = {}
    for path in files:
        with path.open("rb") as handle:
            panel = pickle.load(handle)
        month = path.stem
        universe = [str(value) for value in panel["universe"]]
        frame_ids = panel["frame"]["stock_id"].astype(str).tolist()
        label_ids = panel["labels"].index.astype(str).tolist()
        columns = tuple(panel["feature_columns"])
        if panel.get("feature_version") != "factor_v4":
            failures.append(f"{month}: wrong feature_version")
        if universe != frame_ids or universe != label_ids:
            failures.append(f"{month}: universe/frame/labels mismatch")
        if any("missing" in column for column in panel["frame"].columns):
            failures.append(f"{month}: missingness column present")
        if panel.get("content_hash") != canonical_panel_hash(panel):
            failures.append(f"{month}: content hash mismatch")
        counts.append(len(universe))
        month_counts[month] = len(universe)
        feature_schemas.add(columns)
        coverages.extend(float(value) for value in panel["feature_coverage"].values())
        excluded.update(panel.get("eligibility", {}).get("excluded_counts", {}))

    comparison: dict[str, object] = {}
    compare_dir = Path(args.compare)
    common = []
    for month, v4_count in month_counts.items():
        old_path = compare_dir / f"{month}.pkl"
        if not old_path.is_file():
            continue
        with old_path.open("rb") as handle:
            old = pickle.load(handle)
        common.append((month, len(old.get("universe", [])), v4_count))
    if common:
        comparison = {
            "months": len(common),
            "mean_old_universe": statistics.fmean(row[1] for row in common),
            "mean_v4_universe": statistics.fmean(row[2] for row in common),
            "monthly": {
                month: {"old": old_count, "v4": v4_count, "delta": v4_count - old_count}
                for month, old_count, v4_count in common
            },
        }
    report = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "months": len(files),
        "feature_schema_count": len(feature_schemas),
        "feature_count": len(next(iter(feature_schemas))) if feature_schemas else 0,
        "eligible_count": {
            "min": min(counts),
            "max": max(counts),
            "mean": statistics.fmean(counts),
        },
        "minimum_raw_feature_coverage": min(coverages),
        "excluded_totals": dict(excluded.most_common()),
        "comparison": comparison,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "comparison"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
