"""Strict semantic comparison for legacy and optimized panel directories."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


def compare_panel(expected: dict, actual: dict, month: str) -> dict:
    for key in ("signal_date", "universe", "feature_columns", "feature_version"):
        if expected.get(key) != actual.get(key):
            raise AssertionError(f"{month}: {key} differs")
    expected_frame = expected["frame"].reset_index(drop=True)
    actual_frame = actual["frame"].reset_index(drop=True)
    assert_frame_equal(
        expected_frame,
        actual_frame,
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        check_like=False,
    )
    assert_series_equal(
        expected["labels"],
        actual["labels"],
        check_dtype=True,
        check_exact=True,
        check_names=True,
        check_like=False,
    )
    numeric = expected_frame.select_dtypes(include=[np.number]).columns
    diffs = []
    for column in numeric:
        left = expected_frame[column].to_numpy(dtype=float)
        right = actual_frame[column].to_numpy(dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            diffs.extend(np.abs(left[finite] - right[finite]).tolist())
    return {
        "month": month,
        "rows": len(expected_frame),
        "features": len(expected.get("feature_columns", [])),
        "max_abs_numeric_delta": max(diffs, default=0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args(argv)
    expected_dir, actual_dir = Path(args.expected), Path(args.actual)
    expected_files = {path.name for path in expected_dir.glob("????-??.pkl")}
    actual_files = {path.name for path in actual_dir.glob("????-??.pkl")}
    if expected_files != actual_files and not (args.allow_subset and actual_files <= expected_files):
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        print(f"panel file sets differ; missing={missing}, extra={extra}", file=sys.stderr)
        return 1
    summaries = []
    try:
        for filename in sorted(actual_files):
            with (expected_dir / filename).open("rb") as handle:
                expected = pickle.load(handle)
            with (actual_dir / filename).open("rb") as handle:
                actual = pickle.load(handle)
            summaries.append(compare_panel(expected, actual, filename[:7]))
    except Exception as exc:  # noqa: BLE001 - concise semantic diff failure.
        print(f"panel comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"months": len(summaries), "panels": summaries}, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"months": len(summaries), "panels": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
