"""Run a frozen model through monthly signals with position carry-over.

Stage 3 of 方案B: for each month-end, build universe + PIT features with
the run's as_of, score with the frozen booster (no retraining), then
target holdings -> risk controls -> next-open orders, carrying positions
and compounding portfolio value month to month. Everything persists under
one run_id (predictions per month, signals, orders), so export_reports,
materialize_run and the dashboard work unchanged.

Usage:
    python tools/run_oos_portfolio.py [--config config.yaml]
        [--model models/model_b2] [--model-version xgb_b2]
        [--start 2024-01] [--end 2024-12] [--capital 30000000]
        [--run-id oos-2024-b2]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgboost import XGBClassifier  # noqa: E402

from contracts import ModelArtifact, PortfolioTarget  # noqa: E402
from runtime.db_store import (  # noqa: E402  # noqa: E402
    DbStore,
    default_shares_path,
    get_engine,
    load_shares_cache,
)
from runtime.dotenv import load_dotenv  # noqa: E402
from services.execution_service import create_orders  # noqa: E402
from services.feature_preprocess_service import preprocess_features  # noqa: E402
from services.feature_service import calculate_raw_features  # noqa: E402
from services.pit_service import build_pit_snapshot  # noqa: E402
from services.portfolio_service import build_target_holdings  # noqa: E402
from services.risk_service import apply_risk_controls  # noqa: E402
from services.universe_service import build_universe  # noqa: E402
from services.xgb_service import predict_xgb  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen-model OOS portfolio run.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="models/model_b2")
    parser.add_argument("--model-version", default="xgb_b2")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--capital", type=float, default=30_000_000.0)
    parser.add_argument("--run-id", default="oos-2024-b2")
    parser.add_argument("--panels", default="data/panels")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    with open(Path(args.model) / "report.json", encoding="utf-8") as handle:
        model_report = json.load(handle)
    if model_report.get("feature_version") != settings.features.feature_version:
        print(
            f"model feature_version {model_report.get('feature_version')!r} does not match "
            f"current {settings.features.feature_version!r}",
            file=sys.stderr,
        )
        return 1
    columns: list[str] = model_report.get("oos_columns") or model_report.get("common_features", [])
    if not columns:
        print("model report has no feature columns", file=sys.stderr)
        return 1
    booster = XGBClassifier()
    booster.load_model(str(Path(args.model) / "booster.ubj"))
    artifact = ModelArtifact(
        run_id=args.run_id,
        model_version=args.model_version,
        feature_version=settings.features.feature_version,
        parameter_version="manual-oos",
        feature_columns=tuple(columns),
        best_params=model_report.get("best_params", {}),
        validation_rank_ic=float(model_report.get("oos_rank_ic") or 0.0),
    )

    engine = get_engine(settings)
    shares = load_shares_cache(default_shares_path(settings.data.database_url))
    prices = DbStore(engine, settings, shares_outstanding=shares).load_prices()
    month_list = _months(args.start, args.end)
    trade_days = sorted(prices.loc[prices["stock_id"] != "TAIEX", "trade_date"].unique())
    signals: list[str] = []
    for month in month_list:
        candidates = [d for d in trade_days if d[:7] == month]
        if not candidates:
            print(f"[{month}] no trading days, skipped")
            continue
        signals.append(candidates[-1])
    if not signals:
        print("no signal dates", file=sys.stderr)
        return 1
    print(f"signals: {signals[0]}..{signals[-1]} ({len(signals)})")

    store = DbStore(engine, settings, shares_outstanding=shares)
    try:
        store.start_run(
            {
                "run_id": args.run_id,
                "data_end_date": signals[-1],
                "feature_version": settings.features.feature_version,
                "parameter_version": "manual-oos-b2",
                "model_version": args.model_version,
            }
        )
    except Exception:  # noqa: BLE001 - rerun reuses the run id.
        pass

    positions: dict[str, int] = {}
    adjusted_units: dict[str, float] = {}
    cash = float(args.capital)
    nav = float(args.capital)
    all_orders: list[pd.DataFrame] = []
    panels_dir = Path(args.panels) if args.panels else None
    for signal_str in signals:
        as_of = date.fromisoformat(signal_str)
        monthly = DbStore(engine, settings, as_of=as_of, shares_outstanding=shares)
        panel_file = (panels_dir / f"{signal_str[:7]}.pkl") if panels_dir else None
        if panel_file and panel_file.is_file():
            with open(panel_file, "rb") as handle:
                panel_data = pickle.load(handle)
            if panel_data.get("feature_version") != settings.features.feature_version:
                error = f"[{signal_str}] stale panel feature_version"
                print(error, file=sys.stderr)
                store.finish_run(args.run_id, "failed", error)
                return 1
            features_frame = panel_data["frame"]
        else:
            universe = build_universe(prices, monthly.load_stocks(), as_of, settings, args.run_id)
            if not universe.included_ids:
                print(f"[{signal_str}] empty universe, skipped")
                continue
            snapshot = build_pit_snapshot(
                universe,
                as_of,
                monthly.load_financials_snapshot(),
                monthly.load_institutional_snapshot(),
                prices,
            )
            raw = calculate_raw_features(
                snapshot,
                prices,
                as_of,
                monthly.load_financials_history(),
                monthly.load_institutional_history(),
            )
            features = preprocess_features(raw, settings, args.run_id, as_of)
            features_frame = features.frame
        available = [c for c in columns if c in features_frame.columns]
        if len(available) < len(columns):
            print(f"[{signal_str}] only {len(available)}/{len(columns)} features, skipped")
            continue
        scored = predict_xgb(artifact, features_frame[["stock_id", *columns]], booster)
        scored["prediction_date"] = signal_str
        store._run_id = args.run_id  # noqa: SLF001 - saves bind to this run.
        store._as_of = signal_str  # noqa: SLF001 - snapshot loads bind here.
        store.save_predictions(scored, args.model_version)
        prev_positions = pd.DataFrame({"stock_id": list(positions)})
        target: PortfolioTarget = build_target_holdings(
            scored, prev_positions, settings, args.run_id, as_of
        )
        active_stocks = [s for s, action in target.actions.items() if action in ("BUY", "HOLD")]
        returns_df = (
            monthly.load_returns(active_stocks)
            if active_stocks
            else pd.DataFrame(columns=["stock_id", "trade_date", "log_return"])
        )
        controlled = apply_risk_controls(
            target, returns_df, monthly.load_taiex(), as_of, settings
        )
        store.save_target_holdings(controlled, model_version=args.model_version)
        next_open = monthly.load_next_open(as_of)
        if next_open.empty:
            print(f"[{signal_str}] no next open, skipped")
            continue
        current = pd.DataFrame({"stock_id": list(positions), "shares": list(positions.values())})
        orders = create_orders(controlled, as_of, next_open, current, nav, settings, args.run_id)
        n_saved = store.save_orders(orders)
        if not orders.empty:
            all_orders.append(orders)
            from services.backtest_service import _apply_fill  # noqa: E402,SLF001

            opens = prices.set_index(["trade_date", "stock_id"])[["open", "open_adj"]]
            ledger = orders.sort_values(["execution_date", "order_id"])
            for record in ledger.to_dict("records"):
                key = (str(record["execution_date"]), str(record["stock_id"]))
                try:
                    raw_open = float(opens.loc[key, "open"])
                    adj_open = float(opens.loc[key, "open_adj"])
                except KeyError:
                    continue
                cash, _ = _apply_fill(
                    record, positions, adjusted_units, cash, 0.0, raw_open, adj_open
                )
        # Mark the portfolio on adjusted closes. If this month produced
        # fills, value it after execution on that session; otherwise use
        # the signal-date close. Raw share counts are for orders only.
        mark_date = (
            str(orders["execution_date"].max()) if not orders.empty else signal_str
        )
        day_closes = (
            prices.loc[prices["trade_date"] == mark_date]
            .set_index("stock_id")["close_adj"]
            .apply(pd.to_numeric, errors="coerce")
            .to_dict()
        )
        held_ids = [sid for sid, shares in positions.items() if shares > 0]
        missing_adj = [
            sid
            for sid in held_ids
            if sid not in day_closes
            or not pd.notna(day_closes[sid])
            or float(day_closes[sid]) <= 0
        ]
        if missing_adj:
            error = f"missing adjusted close for holdings {missing_adj} on {mark_date}"
            print(error, file=sys.stderr)
            store.finish_run(args.run_id, "failed", error)
            return 1
        nav = cash + sum(
            adjusted_units.get(sid, 0.0) * float(day_closes[sid]) for sid in held_ids
        )
        print(
            f"[{signal_str}] orders={n_saved} positions={len(positions)} nav={nav:,.0f}",
            flush=True,
        )
    try:
        store.finish_run(args.run_id, "succeeded")
    except ValueError:
        pass
    print(f"done months={len(all_orders)} final_positions={len(positions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
