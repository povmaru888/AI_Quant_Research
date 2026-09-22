"""P0-01 acceptance: package metadata in pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_pyproject() -> dict:
    assert PYPROJECT_PATH.is_file(), f"missing {PYPROJECT_PATH}"
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def test_package_name() -> None:
    data = _load_pyproject()
    assert data["project"]["name"] == "taiwan-quant-xgb"


def test_requires_python_minimum() -> None:
    data = _load_pyproject()
    spec: str = data["project"]["requires-python"]
    match = re.search(r"(\d+)\.(\d+)", spec)
    assert match is not None, f"unparsable requires-python: {spec}"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (3, 10), f"minimum Python must be >=3.10, got {spec}"


def test_src_is_package_source_root() -> None:
    data = _load_pyproject()
    where = data["tool"]["setuptools"]["packages"]["find"]["where"]
    assert "src" in where
    assert (PYPROJECT_PATH.parent / "src").is_dir()
    assert (PYPROJECT_PATH.parent / "src" / "__init__.py").is_file()


def test_pytest_discovers_tests_dir() -> None:
    data = _load_pyproject()
    testpaths = data["tool"]["pytest"]["ini_options"]["testpaths"]
    assert "tests" in testpaths
    assert (PYPROJECT_PATH.parent / "tests").is_dir()
