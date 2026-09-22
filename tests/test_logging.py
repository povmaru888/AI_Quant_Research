"""P5-01 acceptance: structured logging fields and redaction."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from observability.logging import (
    LOGGER_NAME,
    JsonFormatter,
    RunContextFilter,
    configure_logging,
)


@pytest.fixture()
def captured() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    logger = configure_logging(run_id="run-001")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RunContextFilter("run-001"))
    logger.addHandler(handler)
    try:
        yield logger, stream
    finally:
        logger.removeHandler(handler)


def _entries(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_required_fields_present(captured) -> None:
    logger, stream = captured
    logger.info("sync done", extra={"event": "sync_completed", "rows": 12})
    (entry,) = _entries(stream)
    assert entry["level"] == "INFO"
    assert entry["event"] == "sync_completed"
    assert entry["run_id"] == "run-001"
    assert entry["module"] == "test_logging"
    assert entry["timestamp"].endswith("+00:00")
    assert entry["message"] == "sync done"
    assert entry["rows"] == 12


def test_event_defaults_and_run_id_none() -> None:
    logger = configure_logging(run_id=None)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RunContextFilter(None))
    logger.addHandler(handler)
    try:
        logger.warning("bare message")
    finally:
        logger.removeHandler(handler)
    (entry,) = _entries(stream)
    assert entry["event"] == "unspecified"
    assert entry["run_id"] is None


def test_secrets_masked(captured) -> None:
    logger, stream = captured
    logger.info(
        "calling %(host)s",
        {"host": "api.example.com", "finmind_token": "real-secret", "nested": {"api_key": "k"}},
        extra={"event": "fetch"},
    )
    logger.info(
        "key %(finmind_token)s used",
        {"finmind_token": "real-secret"},
        extra={"event": "fetch", "authorization": "Bearer abc", "nested": {"api_key": "k"}},
    )
    first, second = _entries(stream)
    assert "real-secret" not in stream.getvalue()
    assert first["message"] == "calling api.example.com"
    assert "finmind_token" not in first
    assert second["message"] == "key *** used"
    assert second["authorization"] == "***"
    assert second["nested"] == {"api_key": "***"}


def test_long_response_truncated(captured) -> None:
    logger, stream = captured
    body = "x" * 5000
    logger.info("response", extra={"event": "fetch", "response_body": body, "token_value": body})
    (entry,) = _entries(stream)
    assert entry["response_body"].endswith("[truncated]")
    assert len(entry["response_body"]) < len(body)
    assert entry["token_value"] == "***"


def test_reconfigure_is_idempotent() -> None:
    first = configure_logging(run_id="a")
    second = configure_logging(run_id="b")
    assert first is second
    markers = [
        handler
        for handler in second.handlers
        if getattr(handler, "_taiwan_quant_structured", False)
    ]
    assert len(markers) == 1
    assert logging.getLogger(LOGGER_NAME).propagate is False


def test_bad_run_id_rejected() -> None:
    with pytest.raises(ValueError, match="run_id"):
        configure_logging(run_id="  ")
