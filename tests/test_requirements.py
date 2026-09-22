"""P0-02 acceptance: requirements.txt pins all required dependencies."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"

# SDD 5.1 + TASK_ROADMAP P0-02 union, plus yfinance (SDD fallback_source)
# and shioaji (broker historical kbars backfill).
REQUIRED_PACKAGES = frozenset(
    {
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "optuna",
        "vectorbt",
        "sqlalchemy",
        "alembic",
        "requests",
        "pyyaml",
        "plotly",
        "streamlit",
        "shap",
        "pytest",
        "ruff",
        "yfinance",
        "shioaji",
    }
)

_PINNED_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+==[^;\s]+$")


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_requirements() -> dict[str, str]:
    assert REQUIREMENTS_PATH.is_file(), f"missing {REQUIREMENTS_PATH}"
    parsed: dict[str, str] = {}
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert _PINNED_PATTERN.match(line), f"unpinned requirement: {raw_line!r}"
        name, version = line.split("==", 1)
        key = _normalize(name)
        assert key not in parsed, f"duplicate requirement: {name}"
        parsed[key] = version.strip()
    return parsed


def test_required_packages_present() -> None:
    parsed = _parse_requirements()
    missing = sorted(REQUIRED_PACKAGES - set(parsed))
    assert not missing, f"missing packages: {missing}"


def test_versions_pinned_and_valid() -> None:
    from packaging.version import Version

    parsed = _parse_requirements()
    for version in parsed.values():
        Version(version)  # raises on invalid PEP 440


def test_installed_versions_match() -> None:
    parsed = _parse_requirements()
    mismatched: list[str] = []
    for name, pinned in parsed.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = metadata.version(name.replace("-", "_"))
        if installed != pinned:
            mismatched.append(f"{name}: pinned={pinned}, installed={installed}")
    assert not mismatched, f"version mismatch: {mismatched}"


def test_pip_check_clean() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
