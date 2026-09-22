"""P1-11: pipeline run repository (pipeline_runs table).

Same conventions as P1-06..P1-10: takes an active ``Session``, flushes but
never commits. Unlike the DataFrame-based repositories, this module works
with ORM objects directly. Run status transitions are enforced here: only
started -> succeeded and started -> failed are legal; terminal runs are
immutable (a rerun needs a new run_id).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.research import PipelineRun

STARTED = "started"
TERMINAL_STATUSES = ("succeeded", "failed")

_REQUIRED_KEYS = ("run_id", "data_end_date", "feature_version", "parameter_version")
_OPTIONAL_KEYS = (
    "run_time",
    "model_version",
    "universe_count",
    "position_count",
    "error_message",
)


def start_run(session: Session, metadata: dict) -> PipelineRun:
    """Create a ``started`` run from ``metadata``; return the persistent object."""
    unknown = [key for key in metadata if key not in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS)]
    if unknown:
        raise ValueError(f"unknown run metadata keys: {unknown}")
    missing = [key for key in _REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"missing run metadata keys: {missing}")
    run_id = metadata["run_id"]
    if not run_id or not str(run_id).strip():
        raise ValueError("start_run requires a non-empty 'run_id'")
    run = PipelineRun(
        run_id=run_id,
        run_time=metadata.get("run_time")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        data_end_date=metadata["data_end_date"],
        model_version=metadata.get("model_version"),
        feature_version=metadata["feature_version"],
        parameter_version=metadata["parameter_version"],
        universe_count=metadata.get("universe_count"),
        position_count=metadata.get("position_count"),
        status=STARTED,
        error_message=metadata.get("error_message"),
    )
    session.add(run)
    session.flush()
    return run


def finish_run(session: Session, run_id: str, status: str, error: str | None = None) -> PipelineRun:
    """Move a ``started`` run to ``succeeded`` or ``failed``."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal status: {status!r}")
    if status == "failed" and (error is None or not str(error).strip()):
        raise ValueError("finish_run to 'failed' requires a non-empty error")
    run = session.execute(
        select(PipelineRun).where(PipelineRun.run_id == run_id)
    ).scalar_one_or_none()
    if run is None:
        raise ValueError(f"unknown run_id: {run_id!r}")
    if run.status != STARTED:
        raise ValueError(f"run {run_id!r} is already {run.status!r}")
    run.status = status
    run.error_message = error if status == "failed" else None
    session.flush()
    return run
