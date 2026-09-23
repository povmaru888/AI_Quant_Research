"""Run artifacts repository (migration 002).

JSON dashboard payloads keyed by (run_id, kind); upsert semantics so
re-materialization is safe. Payloads must be JSON-serializable dicts/lists
(checked at write time, fail loud).
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.research import RunArtifact

_VALID_KINDS = frozenset({"metrics", "factor_ic", "monthly_ic", "model_explain", "sensitivity"})


def save_artifact(session: Session, run_id: str, kind: str, payload: dict | list) -> None:
    """Validate and upsert one artifact; flush, never commit."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    if kind not in _VALID_KINDS:
        raise ValueError(f"invalid kind: {kind!r}")
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact payload must be JSON-serializable: {exc}") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"artifact payload must be a dict or list, got {type(payload).__name__}")
    statement = sqlite_insert(RunArtifact).values(run_id=run_id, kind=kind, payload_json=text)
    statement = statement.on_conflict_do_update(
        index_elements=["run_id", "kind"],
        set_={"payload_json": statement.excluded.payload_json},
    )
    session.execute(statement)
    session.flush()


def load_artifact(session: Session, run_id: str, kind: str) -> dict | list | None:
    """Return the artifact payload, or None when absent."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"invalid kind: {kind!r}")
    text = session.execute(
        select(RunArtifact.payload_json).where(
            RunArtifact.run_id == run_id, RunArtifact.kind == kind
        )
    ).scalar()
    if text is None:
        return None
    return json.loads(text)
