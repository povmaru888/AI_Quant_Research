"""Refit on train+validation with frozen best params, then blind OOS test.

Stage 2 of 方案B: loads model_b1's best params and 13 feature columns,
trains one XGB on 2019-01..2023-12 (all 60 months, purge month included:
nothing is held out anymore), saves model_b2, then scores the untouched
2024 panels month by month (rank IC + top-decile spread, same as stage 1).

The eval split passed to train_xgb is the full set itself: with fixed
n_estimators there is no early stopping to be honest about, and every
row trains the deployed model.

Usage:
    python tools/refit_oos.py [--config config.yaml] [--panels data/panels]
        [--from models/model_b1] [--refit-start 2019-01] [--refit-end 2023-12]
        [--oos-start 2024-01] [--oos-end 2024-12] [--out models/model_b2]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import ModelArtifact  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.xgb_service import predict_xgb, rank_ic, train_xgb  # noqa: E402
from settings import load_settings  # noqa: E402


def _months(start: str, end: str) -> list[str]:
    months, cursor = [], start
    while cursor <= end:
        months.append(cursor)
        year, month = int(cursor[:4]), int(cursor[5:7])
        month += 1
        if month > 12:
            year, month = year + 1, 1
        cursor = f"{year}-{month:02d}"
    return months


def _load_panel(panels: Path, month: str) -> dict:
    with open(panels / f"{month}.pkl", "rb") as handle:
        return pickle.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refit frozen params, blind OOS.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panels", default="data/panels")
    parser.add_argument("--from", dest="src", default="models/model_b1")
    parser.add_argument("--refit-start", default="2019-01")
    parser.add_argument("--refit-end", default="2023-12")
    parser.add_argument("--oos-start", default="2024-01")
    parser.add_argument("--oos-end", default="2024-12")
    parser.add_argument("--out", default="models/model_b2")
    parser.add_argument("--model-version", default="xgb_b2")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    panels = Path(args.panels)
    with open(Path(args.src) / "report.json", encoding="utf-8") as handle:
        base = json.load(handle)
    if base.get("feature_version") != settings.features.feature_version:
        print(
            f"stale model report feature_version {base.get('feature_version')!r}; "
            f"expected {settings.features.feature_version!r}",
            file=sys.stderr,
        )
        return 1
    columns: list[str] = base["common_features"]
    params: dict = dict(base["best_params"])
    print(f"frozen params from {args.src}: {len(columns)} features")

    refit_months = _months(args.refit_start, args.refit_end)
    frames, labels = [], []
    for month in refit_months:
        panel = _load_panel(panels, month)
        if panel.get("feature_version") != settings.features.feature_version:
            print(f"[{month}] stale panel feature_version", file=sys.stderr)
            return 1
        missing = [c for c in columns if c not in panel["frame"].columns]
        if missing:
            print(f"[{month}] missing columns {missing}", file=sys.stderr)
            return 1
        frame = panel["frame"][["stock_id", *columns]].copy()
        aligned = panel["labels"].reindex(frame["stock_id"])
        keep = aligned.notna().to_numpy()
        frames.append(frame.loc[keep])
        labels.append(aligned.loc[keep].astype(int))
    big = pd.concat(frames, ignore_index=True)
    big_y = pd.concat(labels, ignore_index=True).astype(int)
    print(f"refit rows: {len(big)} over {len(refit_months)} months")
    train_x = big[columns]
    _, booster = train_xgb(
        train_x,
        big_y,
        train_x,
        big_y,
        params,
        args.model_version,
        settings.features.feature_version,
        f"manual-refit-{args.refit_start}-{args.refit_end}",
        f"refit-{args.refit_start}-{args.refit_end}",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "booster.ubj"))

    oos_months = _months(args.oos_start, args.oos_end)
    month_ics, month_spreads = [], []
    for month in oos_months:
        panel = _load_panel(panels, month)
        if panel.get("feature_version") != settings.features.feature_version:
            print(f"[{month}] stale panel feature_version", file=sys.stderr)
            return 1
        frame = panel["frame"][["stock_id", *columns]].copy()
        aligned = panel["labels"].reindex(frame["stock_id"])
        keep = aligned.notna().to_numpy()
        frame, truth_m = frame.loc[keep], aligned.loc[keep].astype(int)
        if frame.empty:
            print(f"[{month}] no labeled rows, skipped")
            continue
        scored = predict_xgb(
            ModelArtifact(
                run_id=f"refit-{args.refit_start}-{args.refit_end}",
                model_version=args.model_version,
                feature_version=settings.features.feature_version,
                parameter_version="manual-refit",
                feature_columns=tuple(columns),
                best_params=params,
                validation_rank_ic=float(base.get("validation_rank_ic") or 0.0),
            ),
            frame[["stock_id", *columns]],
            booster,
        )
        probs = scored.set_index("stock_id")["probability"]
        truth_m = truth_m.set_axis(frame["stock_id"].to_numpy())
        month_ics.append(rank_ic(probs, truth_m.loc[probs.index]))
        order = probs.sort_values(ascending=False)
        top_n = max(1, len(order) // 10)
        month_spreads.append(
            float(
                truth_m.loc[order.index[:top_n]].mean() - truth_m.loc[order.index[-top_n:]].mean()
            )
        )
    ics = [ic for ic in month_ics if np.isfinite(ic)]
    report = {
        "model_version": args.model_version,
        "feature_version": settings.features.feature_version,
        "refit_months": [refit_months[0], refit_months[-1], len(refit_months)],
        "refit_rows": len(big),
        "best_params": params,
        "common_features": columns,
        "oos_months": [oos_months[0], oos_months[-1], len(oos_months)],
        "oos_rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "oos_top_decile_spread": float(np.mean(month_spreads)) if month_spreads else float("nan"),
        "oos_monthly_ic": {
            month: (float(v) if np.isfinite(v) else None)
            for month, v in zip(oos_months, month_ics, strict=True)
        },
        "frozen_from": args.src,
    }
    print(
        f"OOS rank IC: {report['oos_rank_ic']:.4f}, spread: {report['oos_top_decile_spread']:.4f}"
    )
    with open(out / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
