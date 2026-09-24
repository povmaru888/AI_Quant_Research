"""Materialize post-hoc dashboard payloads for one rebalance run.

The research pipeline never persisted metrics, factor IC, explainability,
or sensitivity scenarios. This script computes them once per run into the
``run_artifacts`` table (migration 002); the dashboard reads them back so
no widget shows a missing value. Nothing here changes service code: it
replays the same tested functions on the run's own stored data.

Approximations are labeled in payload ``meta`` instead of hidden:

- ``model_explain`` comes from a PROXY model: same algorithm and seed,
  default hyper-parameters, trained on the signal month panel only
  (the tuned walk-forward booster was never persisted). It answers
  "what drove this month's scores", not "what the global model learned".
- ``predicted_volatility`` is the trailing-60d realized vol at the signal
  date (the forecast basis risk controls actually had).
- ``icir`` is mean/std of four weekly prediction ICs in the forward month.

Usage:
    python tools/materialize_run.py --run-id rebalance-2026-07-31b
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import session_scope  # noqa: E402
from models.research import Order, Prediction, Signal  # noqa: E402
from repositories import artifacts as artifacts_repo  # noqa: E402
from repositories import research as research_repo  # noqa: E402
from runtime.db_store import build_store, get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.backtest_service import run_backtest  # noqa: E402
from services.feature_preprocess_service import preprocess_features  # noqa: E402
from services.feature_service import calculate_raw_features  # noqa: E402
from services.label_service import build_labels  # noqa: E402
from services.metrics_service import (
    calculate_metrics,  # noqa: E402
    run_cost_sensitivity,  # noqa: E402
)
from services.pit_service import build_pit_snapshot  # noqa: E402
from services.universe_service import build_universe  # noqa: E402
from services.xgb_service import DEFAULT_PARAMS, rank_ic, train_xgb  # noqa: E402
from settings import load_settings  # noqa: E402

_FORWARD_DAYS = 20


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _close_panel(prices: pd.DataFrame, start: str) -> dict[str, list[float]]:
    """One-pass adjusted close history for everything on/after ``start``."""
    frame = prices.loc[prices["trade_date"] >= start, ["stock_id", "trade_date", "close_adj"]]
    frame = frame.copy()
    frame["close_adj"] = pd.to_numeric(frame["close_adj"], errors="coerce")
    # Keep missing adjusted bars in place: removing them would silently
    # shorten the forward trading-day horizon used by IC calculations.
    frame.loc[frame["close_adj"] <= 0, "close_adj"] = np.nan
    frame = frame.sort_values(["stock_id", "trade_date"])
    return (
        frame.groupby("stock_id")["close_adj"]
        .apply(lambda s: [float(x) for x in s.tolist()])
        .to_dict()
    )


def _forward_returns(panel: dict[str, list[float]], days: int) -> pd.Series:
    """Log return over the next ``days`` trading days per stock (indexed)."""
    rows = [
        {"stock_id": s, "fwd": float(np.log(c[days] / c[0]))}
        for s, c in panel.items()
        if len(c) > days
        and np.isfinite(c[0])
        and np.isfinite(c[days])
        and c[0] > 0
        and c[days] > 0
    ]
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index("stock_id")["fwd"]


def _weekly_ics(scores: pd.Series, panel: dict[str, list[float]]) -> list[float]:
    """Rank IC of one score vector against each of 4 weekly forward legs."""
    ics = []
    for week in range(1, 5):
        rows = [
            {
                "stock_id": s,
                "fwd": float(np.log(c[5 * week] / c[5 * (week - 1)])),
            }
            for s, c in panel.items()
            if len(c) > 5 * week
            and np.isfinite(c[5 * (week - 1)])
            and np.isfinite(c[5 * week])
            and c[5 * (week - 1)] > 0
            and c[5 * week] > 0
        ]
        if not rows:
            continue
        fwd = pd.DataFrame(rows).set_index("stock_id")["fwd"]
        ic = rank_ic(scores, fwd)
        if np.isfinite(ic):
            ics.append(float(ic))
    return ics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize run artifacts.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    store = build_store(settings)
    engine = get_engine(settings)
    if args.run_id:
        run_id = args.run_id
    else:
        from app import available_runs

        runs = available_runs(store, feature_version=settings.features.feature_version)
        if not runs:
            print("no succeeded rebalance runs", file=sys.stderr)
            return 1
        run_id = runs[-1]
    summary = store.load_run_summary(run_id)
    if summary["feature_version"] != settings.features.feature_version:
        print(
            f"cannot materialize {run_id}: feature_version "
            f"{summary['feature_version']!r} differs from current "
            f"{settings.features.feature_version!r}; rerun research with adjusted prices",
            file=sys.stderr,
        )
        return 1
    as_of = date.fromisoformat(summary["data_end_date"])
    as_of_str = summary["data_end_date"]
    print(f"materializing {run_id} as_of={as_of_str}", flush=True)
    from runtime.db_store import DbStore, default_shares_path, load_shares_cache

    store = DbStore(
        engine,
        settings,
        as_of=as_of,
        shares_outstanding=load_shares_cache(default_shares_path(settings.data.database_url)),
    )

    print("loading prices...", flush=True)
    prices = store.load_prices()
    print(f"prices: {len(prices)} bars", flush=True)
    print("replaying backtest...", flush=True)
    result = store.replay_backtest(run_id, prices)
    print("backtest done", flush=True)
    # Dashboard metrics describe the active window (first execution onward):
    # the pre-trade flat-cash stretch would corrupt annualization and rates.
    import dataclasses as _dc

    if not result.orders.empty:
        first_exec = str(result.orders["execution_date"].min())
        active = result.nav.loc[result.nav.index.astype(str) >= first_exec]
        with session_scope(engine) as session:
            qty_rows = session.execute(
                sqlalchemy.select(Order.stock_id, Order.side, Order.quantity)
                .where(Order.run_id == run_id)
            ).all()
        held = {}
        for sid, side, qty in qty_rows:
            held[str(sid)] = held.get(str(sid), 0) + (
                int(qty) if side == "BUY" else -int(qty)
            )
        if all(q == 0 for q in held.values()):
            last_exec = str(result.orders["execution_date"].max())
            active = active.loc[active.index.astype(str) <= last_exec]
            print(f"closed ledger: window ends {last_exec}", flush=True)
        result = _dc.replace(result, nav=active, start_date=str(active.index[0]))
        print(f"active window from {first_exec}: {len(active)} days", flush=True)

    # -- prediction IC -------------------------------------------------
    with session_scope(engine) as session:
        pred_rows = session.execute(
            sqlalchemy.select(Prediction.stock_id, Prediction.prediction_probability)
            .where(
                Prediction.run_id == run_id,
                Prediction.prediction_date == as_of_str,
            )
            .order_by(Prediction.stock_id)
        ).all()
    scores = pd.Series({s: float(p) for s, p in pred_rows}, dtype=float)
    panel = _close_panel(prices, as_of_str)
    month_ic = rank_ic(scores, _forward_returns(panel, _FORWARD_DAYS))
    weeklies = _weekly_ics(scores, panel)
    icir = float(np.mean(weeklies) / np.std(weeklies, ddof=1)) if len(weeklies) >= 2 else None

    # -- metrics ----------------------------------------------------------
    metrics_frame = calculate_metrics(result, rank_ic=month_ic, icir=icir)
    metrics = {k: _finite(v) for k, v in metrics_frame.iloc[0].to_dict().items()}
    nav = result.nav
    with session_scope(engine) as session:
        turnover_rows = session.execute(
            sqlalchemy.select(Order.quantity, Order.executed_price).where(Order.run_id == run_id)
        ).all()
    traded = sum(float(q) * float(x) for q, x in turnover_rows)
    avg_nav = float(nav.mean())
    turnover = traded / avg_nav if avg_nav > 0 else None
    metrics["turnover"] = _finite(turnover)
    nav_returns = nav.pct_change().dropna()
    if len(nav_returns) >= 2 and float(nav_returns.std(ddof=1)) > 0:
        metrics["realized_volatility"] = float(nav_returns.std(ddof=1) * np.sqrt(252))
    else:
        metrics["realized_volatility"] = None
    nav_days = nav.index.astype(str)
    months = sorted({str(d)[:7] for d in nav_days})
    monthly_returns = []
    for month in months:
        leg = nav.loc[nav_days.str[:7] == month]
        monthly_returns.append(
            {"month": month, "return": _finite(float(leg.iloc[-1] / leg.iloc[0] - 1))}
        )
    equity_curve = [
        {"date": str(d), "nav": _finite(float(v))}
        for d, v in zip(nav.index, nav.to_numpy(), strict=True)
    ]
    metrics_payload = {
        "metrics": metrics,
        "monthly_returns": monthly_returns,
        "oos_months": months,
        "equity_curve": equity_curve,
        "meta": {"icir_weeks": len(weeklies), "rank_ic_month": _finite(month_ic)},
    }

    # -- features (mirror _execute, then persist) --------------------------
    print("building universe...", flush=True)
    universe = build_universe(prices, store.load_stocks(), as_of, settings, run_id)
    print(f"universe: {len(universe.included_ids)} stocks", flush=True)
    snapshot = build_pit_snapshot(
        universe,
        as_of,
        store.load_financials_snapshot(),
        store.load_institutional_snapshot(),
        prices,
    )
    raw = calculate_raw_features(
        snapshot,
        prices,
        as_of,
        store.load_financials_history(),
        store.load_institutional_history(),
    )
    print("preprocessing features...", flush=True)
    features = preprocess_features(raw, settings, run_id, as_of)
    print(f"features: {features.frame.shape}, kept={len(features.feature_columns)}", flush=True)
    _save_features(engine, features, as_of_str, summary["feature_version"])

    # -- factor IC + monthly IC ---------------------------------------------
    fwd_series = _forward_returns(panel, _FORWARD_DAYS)
    wide = features.frame.set_index("stock_id")
    factor_rows = []
    for factor in features.feature_columns:
        ic = rank_ic(wide[factor].astype(float), fwd_series)
        if np.isfinite(ic):
            factor_rows.append({"factor": factor, "ic": float(ic)})
    factor_rows.sort(key=lambda r: abs(r["ic"]), reverse=True)
    monthly_payload = [{"month": as_of_str[:7], "ic": _finite(month_ic)}]

    # -- proxy model: importance + SHAP --------------------------------------
    print("training proxy model...", flush=True)
    labels = build_labels(prices, universe.included_ids, as_of, settings)
    explain_payload = _proxy_explain(features, labels, settings, summary, run_id, as_of_str)

    # -- sensitivity ----------------------------------------------------------
    print("running cost sensitivity...", flush=True)
    sensitivity_payload = _sensitivity(store, engine, run_id, prices, settings)

    with session_scope(engine) as session:
        artifacts_repo.save_artifact(session, run_id, "metrics", metrics_payload)
        artifacts_repo.save_artifact(session, run_id, "factor_ic", factor_rows)
        artifacts_repo.save_artifact(session, run_id, "monthly_ic", monthly_payload)
        artifacts_repo.save_artifact(session, run_id, "model_explain", explain_payload)
        artifacts_repo.save_artifact(session, run_id, "sensitivity", sensitivity_payload)
    print(
        f"saved: metrics={metrics} factors={len(factor_rows)} "
        f"shap={len(explain_payload['shap_top'])} "
        f"scenarios={len(sensitivity_payload)}"
    )
    return 0


def _save_features(engine, features, as_of_str: str, feature_version: str) -> int:
    frame = features.frame.copy()
    frame["rebalance_date"] = as_of_str
    frame["feature_version"] = feature_version
    with session_scope(engine) as session:
        return research_repo.save_features(session, frame)


def _proxy_explain(features, labels, settings, summary, run_id: str, as_of_str: str) -> dict:
    """Train a default-params proxy on the signal panel; gain + SHAP."""
    from sklearn.model_selection import train_test_split

    matrix = features.frame[list(features.feature_columns)]
    aligned = labels.reindex(features.frame["stock_id"]).astype(int)
    keep = aligned.notna().to_numpy()
    matrix, aligned = matrix.loc[keep], aligned.loc[keep]
    train_x, valid_x, train_y, valid_y = train_test_split(
        matrix,
        aligned,
        test_size=0.2,
        random_state=settings.project.random_state,
        stratify=aligned,
    )
    _, booster = train_xgb(
        train_x,
        train_y,
        valid_x,
        valid_y,
        {**DEFAULT_PARAMS, "random_state": settings.project.random_state},
        f"proxy_{as_of_str[:4]}{as_of_str[5:7]}",
        features.feature_version,
        summary["parameter_version"],
        run_id,
    )
    raw_gain = booster.feature_importances_
    importance = [
        {"feature": f, "gain": float(g)}
        for f, g in zip(features.feature_columns, raw_gain, strict=True)
    ]
    importance.sort(key=lambda r: r["gain"], reverse=True)
    import shap

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(matrix)
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    shap_top = [
        {"feature": f, "value": float(v)}
        for f, v in zip(features.feature_columns, mean_abs, strict=True)
    ]
    shap_top.sort(key=lambda r: abs(r["value"]), reverse=True)
    return {
        "shap_top": shap_top[:15],
        "feature_importance": importance,
        "method": "posthoc_xgb_default_params_seed42",
    }


def _sensitivity(store, engine, run_id: str, prices: pd.DataFrame, settings) -> list[dict]:
    """Cost-off baseline plus one row per slippage scenario (active window)."""
    with session_scope(engine) as session:
        order_rows = session.execute(
            sqlalchemy.select(
                Order.order_id,
                Order.run_id,
                Order.signal_date,
                Order.execution_date,
                Order.stock_id,
                Order.side,
                Order.quantity,
                Order.executed_price,
                Order.broker_fee,
                Order.transaction_tax,
                Order.slippage_cost,
                Order.total_cost,
            )
            .where(Order.run_id == run_id)
            .order_by(Order.execution_date, Order.order_id)
        ).all()
        weight_rows = session.execute(
            sqlalchemy.select(Signal.stock_id, Signal.target_weight).where(Signal.run_id == run_id)
        ).all()
    weights = {s: float(w) for s, w in weight_rows}
    orders = pd.DataFrame(
        [
            {
                "order_id": o,
                "run_id": r,
                "signal_date": s,
                "execution_date": e,
                "stock_id": sid,
                "side": side,
                "target_weight": weights.get(sid, 0.0),
                "target_shares": (0 if side == "SELL" else int(q)),
                "executed_price": float(x),
                "broker_fee": float(f),
                "transaction_tax": float(t),
                "slippage_cost": float(sl),
                "total_cost": float(tc),
            }
            for o, r, s, e, sid, side, q, x, f, t, sl, tc in order_rows
        ]
    )
    opens = prices.set_index(["trade_date", "stock_id"])["open"].apply(
        pd.to_numeric, errors="coerce"
    )
    first_exec = str(orders["execution_date"].min())
    prices = prices.loc[prices["trade_date"] >= first_exec]
    base = orders.copy()
    base["executed_price"] = [
        float(opens.get((e, sid), np.nan))
        for e, sid in zip(base["execution_date"], base["stock_id"], strict=True)
    ]
    for key in ("broker_fee", "transaction_tax", "slippage_cost", "total_cost"):
        base[key] = 0.0
    cash = store.load_portfolio_value()
    baseline = calculate_metrics(run_backtest(base, prices, cash, settings, run_id))
    scenarios = run_cost_sensitivity(orders, prices, cash, settings, run_id)
    rows = []
    for _, row in scenarios.iterrows():
        entry = {"scenario": f"slippage_{row['slippage_rate']:.4f}"}
        for stem in ("cagr", "sharpe", "max_drawdown"):
            entry[f"{stem}_before"] = _finite(baseline.iloc[0][stem])
            entry[f"{stem}_after"] = _finite(row[stem])
        entry["total_cost_before"] = 0.0
        entry["total_cost_after"] = _finite(row["total_cost"])
        rows.append(entry)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
