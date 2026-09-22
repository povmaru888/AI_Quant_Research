"""P5-01: structured logging (SDD section 16 operations).

Single-line JSON logs on stdlib ``logging``. Every record carries
``timestamp``, ``level``, ``event``, ``run_id``, and ``module``. Secrets
never reach the output: mapping arguments with sensitive keys are
masked, and any over-long field (e.g. a full external response body)
is truncated. Re-configuring is idempotent (no stacked handlers).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "taiwan_quant"
_HANDLER_MARKER = "_taiwan_quant_structured"
_MAX_FIELD_LENGTH = 2000

_SENSITIVE_SUBSTRINGS: tuple[str, ...] = ("token", "api_key", "apikey", "secret", "password")


def _is_sensitive_key(key: object) -> bool:
    if key == "authorization":
        return True
    return isinstance(key, str) and any(part in key.lower() for part in _SENSITIVE_SUBSTRINGS)


def _scrub(value):
    if isinstance(value, dict):
        return {
            key: ("***" if _is_sensitive_key(key) else _scrub(item)) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if isinstance(value, str) and len(value) > _MAX_FIELD_LENGTH:
        return value[:_MAX_FIELD_LENGTH] + "\u2026[truncated]"
    return value


class RunContextFilter(logging.Filter):
    """Inject run_id/event defaults and scrub sensitive mapping args."""

    def __init__(self, run_id: str | None) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id  # type: ignore[attr-defined]
        if not hasattr(record, "event"):
            record.event = "unspecified"  # type: ignore[attr-defined]
        if isinstance(record.args, dict):
            record.args = _scrub(dict(record.args))
        return True


class JsonFormatter(logging.Formatter):
    """Render the record as one JSON object with the required fields."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "event": _scrub(getattr(record, "event", "unspecified")),
            "run_id": getattr(record, "run_id", None),
            "module": record.module,
            "message": _scrub(message),
        }
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "event",
            "run_id",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = "***" if _is_sensitive_key(key) else _scrub(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(run_id: str | None = None, level: str | int = "INFO") -> logging.Logger:
    """Configure the shared logger; safe to call more than once."""
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError(f"invalid run_id: must be None or a non-empty string, got {run_id!r}")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            for context_filter in handler.filters:
                if isinstance(context_filter, RunContextFilter):
                    context_filter._run_id = run_id
            handler.setLevel(level)
            break
    else:
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        handler.addFilter(RunContextFilter(run_id))
        handler.setFormatter(JsonFormatter())
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
