"""P5-06 acceptance: security workflow structure and rule effectiveness."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "security.yml"

# Mirrors of the in-workflow rules: fixtures must trip them here so we
# know the CI patterns actually catch offending content.
SECRET_PATTERN = re.compile(
    r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9-]+"
    r"|BEGIN [A-Z ]*PRIVATE KEY|sk-(live|test)-[A-Za-z0-9]{10,}"
)
SENSITIVE_FILE_PATTERN = re.compile(r"(^|/)\.env$|\.db$|\.sqlite3?$")


def _load_workflow() -> dict:
    assert WORKFLOW_PATH.is_file(), f"missing {WORKFLOW_PATH}"
    with WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def _triggers(data: dict):
    if "on" in data:
        return data["on"]
    return data[True]


def _steps(data: dict) -> list[dict]:
    steps = data["jobs"]["security"]["steps"]
    assert isinstance(steps, list) and steps
    return steps


def test_triggers_and_steps() -> None:
    data = _load_workflow()
    names = set(_triggers(data)) if isinstance(_triggers(data), (dict, list)) else set()
    assert "push" in names
    assert "pull_request" in names
    blobs = [f"{step.get('uses', '')}\n{step.get('run', '')}" for step in _steps(data)]
    assert any("pip-audit" in blob for blob in blobs), "missing dependency audit step"
    assert any("PRIVATE KEY" in blob or "ghp_" in blob for blob in blobs), (
        "missing secret scan step"
    )
    assert any(".env" in blob for blob in blobs), "missing sensitive file guard step"


def test_no_secret_echo() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in text
    for token in ("FINMIND_TOKEN", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"):
        assert token not in text


def test_secret_fixtures_trip_rule() -> None:
    # Fixtures are concatenated so this file's own text never matches.
    assert SECRET_PATTERN.search("key = " + "AKIA" + "IOSFODNN7EXAMPLE")
    assert SECRET_PATTERN.search("token = " + "ghp_" + "abcdefghijklmnopqrst")
    assert SECRET_PATTERN.search("-----BEGIN RSA " + "PRIVATE KEY-----")
    assert SECRET_PATTERN.search("sk-live-" + "abcdefghij1234")
    assert not SECRET_PATTERN.search("FINMIND_TOKEN=")
    assert not SECRET_PATTERN.search("see .env.example for variable names")


def test_sensitive_file_fixtures_trip_rule() -> None:
    assert SENSITIVE_FILE_PATTERN.search(".env")
    assert SENSITIVE_FILE_PATTERN.search("config/.env")
    assert SENSITIVE_FILE_PATTERN.search("database/quant.db")
    assert SENSITIVE_FILE_PATTERN.search("data/cache.sqlite3")
    assert not SENSITIVE_FILE_PATTERN.search(".env.example")
    assert not SENSITIVE_FILE_PATTERN.search("config.yaml")
