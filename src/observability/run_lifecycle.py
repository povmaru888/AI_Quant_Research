"""P5-02: run lifecycle seam (SDD section 16 operations).

``managed_run`` wraps P1-11 ``start_run`` / ``finish_run`` so every job
(daily update, monthly rebalance, research run) opens and closes exactly
one pipeline run: success closes as ``succeeded``, any exception closes
as ``failed`` with a one-line error summary and re-raises untouched (job
exit-code logic stays with the caller). Structured logs bracket the run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from observability.logging import configure_logging
from repositories.runs import finish_run, start_run

ERROR_SUMMARY_LIMIT = 500


def summarize_error(exc: BaseException, limit: int = ERROR_SUMMARY_LIMIT) -> str:
    """Render a single-line error summary for ``error_message``."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    if not text:
        return "unknown error"
    if len(text) > limit:
        return text[:limit] + "\u2026[truncated]"
    return text


@contextmanager
def managed_run(
    session: Session,
    metadata: dict,
    logger: logging.Logger | None = None,
) -> Iterator[Any]:
    """Open a pipeline run; always close it, then return or re-raise."""
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid metadata: must be a dict, got {metadata!r}")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("invalid metadata: missing non-empty 'run_id'")
    active_logger = logger or configure_logging(run_id=run_id)
    run = start_run(session, metadata)
    active_logger.info("run started", extra={"event": "run_started", "run_id": run_id})
    try:
        yield run
    except Exception as exc:
        summary = summarize_error(exc) or "unknown error"
        finish_run(session, run_id, "failed", summary)
        active_logger.info(
            "run failed", extra={"event": "run_failed", "run_id": run_id, "error": summary}
        )
        raise
    finish_run(session, run_id, "succeeded")
    active_logger.info("run finished", extra={"event": "run_finished", "run_id": run_id})
