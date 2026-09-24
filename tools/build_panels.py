"""Build monthly training panels (features + labels) with checkpoints.

One panel per month-end: universe -> PIT snapshot -> raw factors ->
preprocessed features -> 20d forward labels, mirroring _execute. Each
month is independent (PIT-safe); panels are pickled per month so reruns
resume and later stages (train/refit/OOS) never recompute.

Feature columns may differ per month (correlation dedup); the training
script intersects them. Nothing here trains a model.

Usage:
    python tools/build_panels.py [--config config.yaml]
        [--start 2019-01] [--end 2023-12] [--out data/panels]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.feature_preprocess_service import preprocess_features  # noqa: E402
from services.feature_service import calculate_raw_features  # noqa: E402
from services.label_service import build_labels  # noqa: E402
from services.pit_service import build_pit_snapshot  # noqa: E402
from services.universe_service import build_universe  # noqa: E402
from settings import load_settings  # noqa: E402


def _month_ends(store, start: str, end: str) -> list[str]:
    """Max trading day of each YYYY-MM in [start, end]."""
    days = store.trade_days(start + "-01", end + "-31")
    by_month: dict[str, str] = {}
    for day in days:
        if start[:7] <= day[:7] <= end[:7]:
            by_month[day[:7]] = day
    return [by_month[m] for m in sorted(by_month)]


def _labelable(prices, universe_ids, as_of, settings) -> list[str]:
    """Universe members with finite positive adj at t and t+20.

    build_labels fails a whole month on one bad stock (correct for live
    signals); panel building instead excludes those stocks so one gap
    cannot nuke a training month. No extra lookahead beyond the label.
    """
    as_of_str = as_of.isoformat()
    horizon = settings.label.horizon_trading_days
    sub = prices.loc[
        prices["stock_id"].isin(list(universe_ids)),
        ["stock_id", "trade_date", "close_adj"],
    ].sort_values(["stock_id", "trade_date"])
    eligible = []
    for stock_id, group in sub.groupby("stock_id", sort=False):
        dates = group["trade_date"].to_numpy()
        hit = np.flatnonzero(dates == as_of_str)
        if not len(hit):
            continue
        pos = int(hit[0])
        closes = pd.to_numeric(group["close_adj"], errors="coerce").to_numpy(dtype=float)
        if pos + horizon >= len(closes):
            continue
        base, ahead = closes[pos], closes[pos + horizon]
        if np.isfinite(base) and np.isfinite(ahead) and base > 0 and ahead > 0:
            eligible.append(str(stock_id))
    return eligible


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build monthly panels.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2019-01")
    parser.add_argument("--end", default="2023-12")
    parser.add_argument("--out", default="data/panels")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = build_store(settings)
    months = _month_ends(store, args.start, args.end)
    if not months:
        print("no month-ends in range", file=sys.stderr)
        return 1
    print(f"{len(months)} months: {months[0]}..{months[-1]}", flush=True)
    run_id = f"panels-{args.start}-{args.end}"
    prices = store.load_prices()
    print(f"prices: {len(prices)} bars", flush=True)
    done, failed = 0, []
    for signal_str in months:
        dest = out / f"{signal_str[:7]}.pkl"
        if dest.is_file():
            done += 1
            continue
        as_of = date.fromisoformat(signal_str)
        try:
            from runtime.db_store import DbStore, default_shares_path, get_engine, load_shares_cache

            monthly_store = DbStore(
                get_engine(settings),
                settings,
                as_of=as_of,
                shares_outstanding=load_shares_cache(
                    default_shares_path(settings.data.database_url)
                ),
            )
            universe = build_universe(prices, monthly_store.load_stocks(), as_of, settings, run_id)
            if not universe.included_ids:
                raise RuntimeError("empty universe")
            snapshot = build_pit_snapshot(
                universe,
                as_of,
                monthly_store.load_financials_snapshot(),
                monthly_store.load_institutional_snapshot(),
                prices,
            )
            raw = calculate_raw_features(
                snapshot,
                prices,
                as_of,
                monthly_store.load_financials_history(),
                monthly_store.load_institutional_history(),
            )
            features = preprocess_features(raw, settings, run_id, as_of)
            labels = build_labels(
                prices, _labelable(prices, universe.included_ids, as_of, settings), as_of, settings
            )
            if labels.empty:
                raise RuntimeError("no labels")
            with open(dest, "wb") as handle:
                pickle.dump(
                    {
                        "signal_date": signal_str,
                        "universe": list(universe.included_ids),
                        "feature_columns": list(features.feature_columns),
                        "feature_version": features.feature_version,
                        "frame": features.frame,
                        "labels": labels,
                    },
                    handle,
                )
        except Exception as exc:  # noqa: BLE001 - per-month isolation.
            failed.append((signal_str, f"{type(exc).__name__}: {exc}"))
            print(f"[{signal_str}] FAILED: {exc}", file=sys.stderr)
            continue
        done += 1
        print(f"[{done}/{len(months)}] {signal_str} n={len(universe.included_ids)}", flush=True)
        time.sleep(0.1)
    print(f"done {done}/{len(months)} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
