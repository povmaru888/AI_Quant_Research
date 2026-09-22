"""P5-04: system health query (SDD section 16 operations).

Read-only snapshot for operators: latest succeeded/failed runs, data
end date, feature coverage, and price data lag. Safe on an empty
database (every field degrades to ``None``). Only whitelisted columns
are read, so tokens and sensitive settings can never appear here.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from models.market import Price
from models.research import Feature, PipelineRun


def get_system_health(session: Session, today: str | None = None) -> dict:
    """Return the health snapshot; ``today`` is an injectable ISO date."""
    anchor = _require_today(today)
    success = session.execute(
        select(PipelineRun.run_id, PipelineRun.data_end_date, PipelineRun.run_time)
        .where(PipelineRun.status == "succeeded")
        .order_by(desc(PipelineRun.run_time), desc(PipelineRun.run_id))
        .limit(1)
    ).first()
    failure = session.execute(
        select(PipelineRun.run_id, PipelineRun.run_time, PipelineRun.error_message)
        .where(PipelineRun.status == "failed")
        .order_by(desc(PipelineRun.run_time), desc(PipelineRun.run_id))
        .limit(1)
    ).first()
    latest_price_day = session.execute(select(func.max(Price.trade_date))).scalar()
    return {
        "latest_success": (
            {"run_id": success[0], "data_end_date": success[1], "run_time": success[2]}
            if success is not None
            else None
        ),
        "latest_failure": (
            {"run_id": failure[0], "run_time": failure[1], "error": failure[2]}
            if failure is not None
            else None
        ),
        "data_end_date": success[1] if success is not None else None,
        "feature_coverage": _feature_coverage(session),
        "data_lag_days": _lag_days(latest_price_day, anchor),
    }


def _require_today(today: str | None) -> str:
    if today is None:
        return datetime.now(timezone.utc).date().isoformat()
    if not isinstance(today, str):
        raise ValueError(f"invalid today: must be a string, got {today!r}")
    try:
        date.fromisoformat(today)
    except ValueError as exc:
        raise ValueError(f"invalid today: not a calendar date: {today!r}") from exc
    return today


def _feature_coverage(session: Session) -> float | None:
    latest = session.execute(select(func.max(Feature.rebalance_date))).scalar()
    if latest is None:
        return None
    factor_columns = [
        col.name
        for col in Feature.__table__.columns
        if col.name not in ("rebalance_date", "stock_id", "feature_version", "missing_flag")
    ]
    rows = session.execute(
        select(*[getattr(Feature, col) for col in factor_columns]).where(
            Feature.rebalance_date == latest
        )
    ).all()
    if not rows:
        return 0.0
    filled = sum(1 for row in rows for value in row if value is not None)
    return filled / (len(rows) * len(factor_columns))


def _lag_days(latest_price_day: str | None, anchor: str) -> int | None:
    if not latest_price_day:
        return None
    try:
        delta = date.fromisoformat(anchor) - date.fromisoformat(latest_price_day)
    except ValueError:
        return None
    return max(delta.days, 0)
