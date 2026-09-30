"""Recalculate model-selection diagnostics without changing portfolio results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import session_scope  # noqa: E402
from models.market import Price  # noqa: E402
from models.research import PipelineRun, Prediction  # noqa: E402
from repositories import artifacts as artifacts_repo  # noqa: E402
from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.selection_metrics_service import (  # noqa: E402
    build_selection_metrics,
    validate_selection_metrics,
)
from settings import load_settings  # noqa: E402


def _load_run_predictions(engine, run_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with session_scope(engine) as session:
        prediction_rows = session.execute(
            sqlalchemy.select(
                Prediction.run_id,
                Prediction.prediction_date,
                Prediction.stock_id,
                Prediction.prediction_probability,
                Prediction.rank,
            )
            .where(Prediction.run_id.in_(run_ids))
            .order_by(Prediction.run_id, Prediction.prediction_date, Prediction.stock_id)
        ).all()
        prediction_frame = pd.DataFrame(
            prediction_rows,
            columns=[
                "run_id",
                "prediction_date",
                "stock_id",
                "prediction_probability",
                "rank",
            ],
        )
        if prediction_frame.empty:
            return prediction_frame, pd.DataFrame(
                columns=["stock_id", "trade_date", "close_adj"]
            )
        stock_ids = prediction_frame["stock_id"].astype(str).unique().tolist()
        first_signal_date = str(prediction_frame["prediction_date"].min())
        price_rows = session.execute(
            sqlalchemy.select(Price.stock_id, Price.trade_date, Price.close_adj)
            .where(
                Price.stock_id.in_(stock_ids),
                Price.trade_date >= first_signal_date,
            )
            .order_by(Price.stock_id, Price.trade_date)
        ).all()
    prices = pd.DataFrame(price_rows, columns=["stock_id", "trade_date", "close_adj"])
    return prediction_frame, prices


def _selected_run_ids(engine, run_id: str | None, all_oos: bool) -> list[str]:
    with session_scope(engine) as session:
        statement = sqlalchemy.select(PipelineRun.run_id).where(
            PipelineRun.status == "succeeded"
        )
        if run_id is not None:
            statement = statement.where(PipelineRun.run_id == run_id)
        elif all_oos:
            statement = statement.where(PipelineRun.run_id.like("oos-%"))
        run_ids = list(session.execute(statement.order_by(PipelineRun.run_id)).scalars())
    if run_id is not None and not run_ids:
        raise ValueError(f"run is missing or not succeeded: {run_id}")
    if not run_ids:
        raise ValueError("no succeeded runs match the requested selection")
    return run_ids


def _preflight(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    run_ids: list[str],
) -> dict[str, dict]:
    if predictions.empty:
        raise ValueError("selected runs have no predictions")
    payloads = {}
    errors = []
    for run_id in run_ids:
        run_predictions = predictions.loc[predictions["run_id"] == run_id]
        if run_predictions.empty:
            print(f"skip {run_id}: no prediction rows", flush=True)
            continue
        payload = build_selection_metrics(run_predictions, prices)
        expected_months = run_predictions["prediction_date"].nunique()
        monthly = payload.get("monthly", [])
        if len(monthly) != expected_months:
            errors.append(
                f"{run_id}: expected {expected_months} monthly rows, got {len(monthly)}"
            )
        errors.extend(
            f"{run_id}: {reason}"
            for reason in validate_selection_metrics(payload, minimum_eligible=30)
        )
        payloads[run_id] = payload
    if errors:
        raise ValueError("preflight failed; no artifacts were changed:\n- " + "\n- ".join(errors))
    if not payloads:
        raise ValueError("selected runs have no prediction rows")
    return payloads


def _save_all(engine, payloads: dict[str, dict]) -> None:
    """Write all run artifacts in one transaction after successful preflight."""
    with session_scope(engine) as session:
        for run_id, payload in payloads.items():
            artifacts_repo.save_artifact(session, run_id, "monthly_ic", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recalculate model-selection diagnostics.")
    parser.add_argument("--config", default="config.yaml")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", help="recalculate one succeeded run")
    selection.add_argument(
        "--all-oos", action="store_true", help="recalculate every succeeded oos-* run"
    )
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
        engine = get_engine(settings)
        run_ids = _selected_run_ids(engine, args.run_id, args.all_oos)
        print(f"selected {len(run_ids)} runs", flush=True)
        predictions, prices = _load_run_predictions(engine, run_ids)
        print(
            f"loaded {len(predictions)} predictions and {len(prices)} adjusted-price rows",
            flush=True,
        )
        payloads = _preflight(predictions, prices, run_ids)
        _save_all(engine, payloads)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for run_id, payload in payloads.items():
        summary = payload["summary"]
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "months": len(payload["monthly"]),
                    "mean_continuous_rank_ic": summary["mean_continuous_rank_ic"],
                    "mean_top15_excess_return": summary["mean_top15_excess_return"],
                    "mean_top_bottom_spread": summary["mean_top_bottom_spread"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    print(f"saved selection metrics for {len(payloads)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
