"""Train one model on month panels, validate once (方案B+refit step 1).

Train months are concatenated (purge month dropped); validation months are
scored once with rank IC, top-decile spread and distribution. Hyper-params
come from optimize_xgb on (train, valid). Nothing is refit here and OOS is
untouched: this stage ends with a validation report for sign-off.

Feature columns are intersected across months (per-month corr dedup may
differ); panel months missing labels for some stocks simply contribute
fewer rows (per-stock label tolerance lives in build_panels).

Usage:
    python tools/train_model.py [--config config.yaml] [--panels data/panels]
        [--train-start 2019-01] [--train-end 2022-11] [--purge-month 2022-12]
        [--valid-start 2023-01] [--valid-end 2023-12]
        [--trials 50] [--out models/model_b1]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from runtime.oos_data import load_symbol_prices  # noqa: E402
from services.feature_selection_service import select_stable_training_features  # noqa: E402
from services.feature_service import uses_training_coverage_selection  # noqa: E402
from services.lightgbm_regression_service import (  # noqa: E402
    optimize_lightgbm_regressor,
    predict_lightgbm,
    train_lightgbm_regressor,
)
from services.logistic_service import predict_logistic, train_logistic  # noqa: E402
from services.optimization_service import optimize_xgb  # noqa: E402
from services.ranking_service import optimize_xgb_ranker, train_xgb_ranker  # noqa: E402
from services.regression_service import (  # noqa: E402
    build_excess_return_labels_by_month,
    optimize_xgb_regressor,
    train_xgb_regressor,
)
from services.xgb_service import predict_xgb, rank_ic, train_xgb  # noqa: E402
from settings import load_settings  # noqa: E402


def _load_panels(panels: Path, months: list[str], feature_version: str) -> dict[str, dict]:
    out = {}
    for month in months:
        path = panels / f"{month}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing panel: {path}")
        with open(path, "rb") as handle:
            panel = pickle.load(handle)
        if panel.get("feature_version") != feature_version:
            raise ValueError(
                f"stale panel {path}: feature_version {panel.get('feature_version')!r}; "
                f"expected {feature_version!r}. Rebuild panels before training."
            )
        out[month] = panel
    return out


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


def _report_path(value: str) -> Path:
    path = Path(value)
    return path / "report.json" if path.is_dir() else path


def _load_parameter_report(value: str) -> tuple[Path, dict]:
    path = _report_path(value)
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"invalid model report: {path}")
    return path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train once, validate once.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panels", default="data/panels")
    parser.add_argument("--train-start", default="2019-01")
    parser.add_argument("--train-end", default="2022-11")
    parser.add_argument("--purge-month", default="2022-12")
    parser.add_argument("--valid-start", default="2023-01")
    parser.add_argument("--valid-end", default="2023-12")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--params-from",
        default=None,
        help="Reuse best_params from a prior report.json (skips Optuna).",
    )
    parser.add_argument(
        "--extra-train-panels",
        default=None,
        help="Add a second panel snapshot for each training month only.",
    )
    parser.add_argument("--out", default="models/model_b1")
    parser.add_argument("--model-version", default="xgb_b1")
    parser.add_argument(
        "--objective",
        choices=(
            "binary:logistic",
            "rank:pairwise",
            "reg:pseudohubererror",
            "lightgbm:huber",
            "logistic_regression",
        ),
        default="binary:logistic",
    )
    args = parser.parse_args(argv)
    if args.objective == "logistic_regression" and args.params_from:
        parser.error("--params-from is not supported for fixed Logistic Regression")
    if args.extra_train_panels and not args.params_from:
        parser.error("--extra-train-panels requires --params-from for a controlled experiment")
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    panels = Path(args.panels)
    train_months = _months(args.train_start, args.train_end)
    valid_months = _months(args.valid_start, args.valid_end)
    print(f"train months: {len(train_months)} ({train_months[0]}..{train_months[-1]})")
    print(f"purge month: {args.purge_month} (excluded)")
    print(f"valid months: {len(valid_months)} ({valid_months[0]}..{valid_months[-1]})")
    train_panels = _load_panels(panels, train_months, settings.features.feature_version)
    valid_panels = _load_panels(panels, valid_months, settings.features.feature_version)
    extra_train_panels = (
        _load_panels(
            Path(args.extra_train_panels), train_months, settings.features.feature_version
        )
        if args.extra_train_panels
        else {}
    )
    params_path: Path | None = None
    params_report: dict | None = None
    if args.params_from:
        params_path, params_report = _load_parameter_report(args.params_from)
        if params_report.get("feature_version") != settings.features.feature_version:
            raise ValueError(
                f"parameter report feature_version {params_report.get('feature_version')!r}; "
                f"expected {settings.features.feature_version!r}"
            )
        if params_report.get("training_objective") not in (None, args.objective):
            raise ValueError(
                f"parameter report objective {params_report.get('training_objective')!r}; "
                f"expected {args.objective!r}"
            )

    continuous_labels: dict[str, pd.Series] | None = None
    if args.objective in {"reg:pseudohubererror", "lightgbm:huber"}:
        all_panels = {**train_panels, **valid_panels}
        all_panels.update(
            {f"{month}@extra": panel for month, panel in extra_train_panels.items()}
        )
        stock_ids = {
            str(stock_id)
            for panel in all_panels.values()
            for stock_id in panel["frame"]["stock_id"].tolist()
        }
        price_frame = load_symbol_prices(
            get_engine(settings),
            stock_ids,
            start=min(str(panel["signal_date"]) for panel in all_panels.values()),
            columns=("stock_id", "trade_date", "close_adj"),
        )
        continuous_labels = build_excess_return_labels_by_month(
            all_panels, price_frame, settings.label.horizon_trading_days
        )

    feature_selection = None
    feature_schema_from = None
    if extra_train_panels:
        assert params_report is not None and params_path is not None
        columns = params_report.get("common_features")
        if not isinstance(columns, list) or not columns or not all(
            isinstance(column, str) and column for column in columns
        ):
            raise ValueError(f"no valid common_features in {params_path}")
        feature_schema_from = str(params_path)
        feature_selection = params_report.get("feature_selection")
        for snapshot, panel in [
            *((f"{month}@month_end", train_panels[month]) for month in train_months),
            *((f"{month}@extra", extra_train_panels[month]) for month in train_months),
            *((f"{month}@validation", valid_panels[month]) for month in valid_months),
        ]:
            missing_columns = [column for column in columns if column not in panel["frame"]]
            if missing_columns:
                raise ValueError(f"{snapshot} is missing fixed features {missing_columns}")
    elif uses_training_coverage_selection(settings.features.feature_version):
        columns, feature_selection = select_stable_training_features(train_panels, train_months)
    else:
        common = set(train_panels[train_months[0]]["feature_columns"])
        for months in (train_months, valid_months):
            for month in months:
                panels_dict = train_panels if month in train_panels else valid_panels
                common &= set(panels_dict[month]["feature_columns"])
        columns = sorted(common)
    print(f"common features: {len(columns)}")

    def assemble(snapshots: list[tuple[str, dict]]):
        frames, labels, groups = [], [], []
        for snapshot_key, panel in snapshots:
            frame = panel["frame"][["stock_id", *columns]].copy()
            frame["month"] = snapshot_key
            source_labels = (
                continuous_labels[snapshot_key]
                if continuous_labels is not None
                else panel["labels"]
            )
            aligned = source_labels.reindex(frame["stock_id"])
            keep = aligned.notna().to_numpy()
            frames.append(frame.loc[keep])
            labels.append(
                aligned.loc[keep].astype(float)
                if continuous_labels is not None
                else aligned.loc[keep].astype(int)
            )
            groups.append(int(keep.sum()))
        big_x = pd.concat(frames, ignore_index=True)
        big_y = pd.concat(labels, ignore_index=True)
        big_y = big_y.astype(float if continuous_labels is not None else int)
        return big_x, big_y, groups

    train_snapshots = [(month, train_panels[month]) for month in train_months]
    if extra_train_panels:
        train_snapshots.extend(
            (f"{month}@extra", extra_train_panels[month]) for month in train_months
        )
    valid_snapshots = [(month, valid_panels[month]) for month in valid_months]
    train_frame, train_y, train_groups = assemble(train_snapshots)
    valid_frame, valid_y, valid_groups = assemble(valid_snapshots)
    print(f"train rows: {len(train_frame)}, valid rows: {len(valid_frame)}")
    if continuous_labels is not None:
        print(
            f"train target std: {float(train_y.std()):.4f}, "
            f"valid target std: {float(valid_y.std()):.4f}"
        )
    else:
        print(f"train pos: {float(train_y.mean()):.3f}, valid pos: {float(valid_y.mean()):.3f}")
    train_x = train_frame[columns]
    valid_x = valid_frame[columns]

    run_id = f"train-{args.train_start}-{args.train_end}"
    training_started = time.perf_counter()
    if args.params_from:
        assert params_path is not None and params_report is not None
        params = params_report.get("best_params")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"no best_params in {params_path}")
        if args.objective == "lightgbm:huber":
            artifact, booster = train_lightgbm_regressor(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                params,
                args.model_version,
                settings.features.feature_version,
                f"fixed-{params_path.parent.name}",
                run_id,
            )
        elif args.objective == "reg:pseudohubererror":
            artifact, booster = train_xgb_regressor(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                params,
                args.model_version,
                settings.features.feature_version,
                f"fixed-{params_path.parent.name}",
                run_id,
            )
        elif args.objective == "rank:pairwise":
            artifact, booster = train_xgb_ranker(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                params,
                args.model_version,
                settings.features.feature_version,
                f"fixed-{params_path.parent.name}",
                run_id,
            )
        else:
            artifact, booster = train_xgb(
                train_x,
                train_y,
                valid_x,
                valid_y,
                params,
                args.model_version,
                settings.features.feature_version,
                f"fixed-{params_path.parent.name}",
                run_id,
            )
        trials = pd.DataFrame()
        print(f"reused fixed parameters from {params_path}")
    else:
        if args.objective == "logistic_regression":
            artifact, booster = train_logistic(
                train_x,
                train_y,
                valid_x,
                valid_y,
                args.model_version,
                settings.features.feature_version,
                f"fixed-{run_id}",
                run_id,
            )
            trials = pd.DataFrame()
        elif args.objective == "lightgbm:huber":
            artifact, booster, trials = optimize_lightgbm_regressor(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                settings,
                args.model_version,
                settings.features.feature_version,
                f"manual-{run_id}",
                run_id,
                n_trials=args.trials,
            )
        elif args.objective == "reg:pseudohubererror":
            artifact, booster, trials = optimize_xgb_regressor(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                settings,
                args.model_version,
                settings.features.feature_version,
                f"manual-{run_id}",
                run_id,
                n_trials=args.trials,
            )
        elif args.objective == "rank:pairwise":
            artifact, booster, trials = optimize_xgb_ranker(
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                valid_groups,
                settings,
                args.model_version,
                settings.features.feature_version,
                f"manual-{run_id}",
                run_id,
                n_trials=args.trials,
            )
        else:
            artifact, booster, trials = optimize_xgb(
                train_x,
                train_y,
                valid_x,
                valid_y,
                settings,
                args.model_version,
                settings.features.feature_version,
                f"manual-{run_id}",
                run_id,
                n_trials=args.trials,
            )
    print(f"best params: {artifact.best_params}")
    month_ics, month_spreads = [], []
    for month in valid_months:
        panel = valid_panels[month]
        frame = panel["frame"][["stock_id", *columns]].copy()
        source_labels = (
            continuous_labels[month] if continuous_labels is not None else panel["labels"]
        )
        aligned = source_labels.reindex(frame["stock_id"])
        keep = aligned.notna().to_numpy()
        frame = frame.loc[keep]
        truth_m = aligned.loc[keep].astype(float if continuous_labels is not None else int)
        if args.objective == "logistic_regression":
            scored = predict_logistic(artifact, frame[["stock_id", *columns]], booster)
        elif args.objective == "lightgbm:huber":
            scored = predict_lightgbm(artifact, frame[["stock_id", *columns]], booster)
        else:
            scored = predict_xgb(artifact, frame[["stock_id", *columns]], booster)
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
    ic = float(np.mean(ics)) if ics else float("nan")
    spread = float(np.mean(month_spreads)) if month_spreads else float("nan")
    report = {
        "model_version": args.model_version,
        "model_family": (
            "lightgbm"
            if args.objective == "lightgbm:huber"
            else "sklearn"
            if args.objective == "logistic_regression"
            else "xgboost"
        ),
        "library_version": (lightgbm.__version__ if args.objective == "lightgbm:huber" else None),
        "training_objective": args.objective,
        "target_definition": (
            "stock_simple_return_t_plus_20_minus_monthly_eligible_universe_mean"
            if continuous_labels is not None
            else "top_quantile_binary_relevance"
        ),
        "feature_version": settings.features.feature_version,
        "train_months": [train_months[0], train_months[-1], len(train_months)],
        "purge_month": args.purge_month,
        "valid_months": [valid_months[0], valid_months[-1], len(valid_months)],
        "common_features": columns,
        "feature_selection": feature_selection,
        "feature_schema_from": feature_schema_from,
        "train_rows": len(train_frame),
        "train_snapshot_count": len(train_snapshots),
        "month_end_train_rows": int(
            train_frame.loc[~train_frame["month"].str.endswith("@extra")].shape[0]
        ),
        "extra_train_rows": int(
            train_frame.loc[train_frame["month"].str.endswith("@extra")].shape[0]
        ),
        "extra_train_panels": args.extra_train_panels,
        "valid_rows": len(valid_frame),
        "best_params": artifact.best_params,
        "validation_rank_ic": float(ic),
        "validation_top_decile_spread": spread,
        "validation_monthly_ic": {
            month: (float(v) if np.isfinite(v) else None)
            for month, v in zip(valid_months, month_ics, strict=True)
        },
        "trials": len(trials),
        "parameters_from": args.params_from,
        "training_seconds": float(time.perf_counter() - training_started),
    }
    if args.objective == "logistic_regression":
        validation_probability = booster.predict_proba(valid_x)[:, 1]
        report.update(
            {
                "validation_roc_auc": float(roc_auc_score(valid_y, validation_probability)),
                "validation_log_loss": float(log_loss(valid_y, validation_probability)),
                "coefficients": {
                    column: float(value)
                    for column, value in zip(columns, booster.coef_[0], strict=True)
                },
                "intercept": float(booster.intercept_[0]),
            }
        )
    print(f"validation rank IC: {ic:.4f}, top-decile spread: {spread:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.objective == "logistic_regression":
        model_path = out / "model.joblib"
        joblib.dump(booster, model_path)
    elif args.objective == "lightgbm:huber":
        model_path = out / "booster.txt"
        booster.booster_.save_model(str(model_path))
    else:
        model_path = out / "booster.ubj"
        booster.save_model(str(model_path))
    report["model_file"] = model_path.name
    report["model_file_size_bytes"] = model_path.stat().st_size
    with open(out / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    with open(out / "trials.csv", "w", encoding="utf-8") as handle:
        trials.to_csv(handle, index=False, lineterminator="\n")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
