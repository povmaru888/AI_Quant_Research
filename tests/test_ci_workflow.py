"""P0-08 acceptance: CI workflow triggers, step order, and secret hygiene."""

from __future__ import annotations

from pathlib import Path

import yaml

CI_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

# YAML 1.1 parses the `on:` key as boolean True; accept both forms.
ORDERED_MARKERS = (
    "actions/checkout",
    "setup-python",
    "pip install -r requirements.txt",
    "pip check",
    "ruff check",
    "ruff format --check",
    "pytest",
)


def _load_workflow() -> dict:
    assert CI_PATH.is_file(), f"missing {CI_PATH}"
    with CI_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def _triggers(data: dict) -> dict | list:
    if "on" in data:
        return data["on"]
    return data[True]


def _steps(data: dict) -> list[dict]:
    steps = data["jobs"]["ci"]["steps"]
    assert isinstance(steps, list) and steps
    return steps


def test_triggers() -> None:
    triggers = _triggers(_load_workflow())
    names = set(triggers) if isinstance(triggers, (dict, list)) else set()
    assert "push" in names
    assert "pull_request" in names


def test_steps_order() -> None:
    blobs = [f"{step.get('uses', '')}\n{step.get('run', '')}" for step in _steps(_load_workflow())]
    positions: list[int] = []
    for marker in ORDERED_MARKERS:
        matches = [i for i, blob in enumerate(blobs) if marker in blob]
        assert matches, f"missing CI step: {marker}"
        positions.append(matches[0])
    assert positions == sorted(positions), f"CI steps out of order: {positions}"


def test_no_secret_echo() -> None:
    text = CI_PATH.read_text(encoding="utf-8")
    assert "secrets." not in text
    for token in ("FINMIND_TOKEN", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"):
        assert token not in text


def test_python_pinned() -> None:
    for step in _steps(_load_workflow()):
        if "setup-python" in str(step.get("uses", "")):
            assert str(step["with"]["python-version"]) == "3.12"
            return
    raise AssertionError("missing setup-python step")
