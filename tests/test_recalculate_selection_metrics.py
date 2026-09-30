"""Batch recalculation preflights all runs and commits artifacts atomically."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select

from database import session_scope
from models import Base
from models.market import Financial, Institutional, MarketValue, Price  # noqa: F401
from models.research import PipelineRun, RunArtifact
from models.security import Stock  # noqa: F401
from tools import recalculate_selection_metrics


def test_batch_artifact_write_rolls_back_every_run_on_failure(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        session.add_all(
            [
                PipelineRun(
                    run_id=run_id,
                    run_time="2020-01-01T00:00:00+00:00",
                    data_end_date="2020-01-31",
                    feature_version="v1",
                    parameter_version="p1",
                    status="succeeded",
                )
                for run_id in ("run-a", "run-b")
            ]
        )

    original_save = recalculate_selection_metrics.artifacts_repo.save_artifact

    def fail_after_second_write(session, run_id, kind, payload):
        original_save(session, run_id, kind, payload)
        if run_id == "run-b":
            raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        recalculate_selection_metrics.artifacts_repo,
        "save_artifact",
        fail_after_second_write,
    )
    payload = {"schema_version": 2, "monthly": [], "summary": {}}

    with pytest.raises(RuntimeError, match="injected write failure"):
        recalculate_selection_metrics._save_all(
            engine, {"run-a": payload, "run-b": payload}
        )

    with session_scope(engine) as session:
        assert session.execute(select(RunArtifact.run_id)).scalars().all() == []
    engine.dispose()
