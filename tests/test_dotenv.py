"""Runtime dotenv loader acceptance (stdlib only, no file side effects)."""

from __future__ import annotations

import os
from pathlib import Path

from runtime.dotenv import load_dotenv


def test_load_dotenv_sets_missing_keys(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'FINMIND_TOKEN="abc123"\n# comment\nEMPTY=\nINVALID_LINE\n', encoding="utf-8"
    )
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    assert load_dotenv(env_file) == 2
    assert os.environ["FINMIND_TOKEN"] == "abc123"
    assert os.environ["EMPTY"] == ""


def test_load_dotenv_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FINMIND_TOKEN=file-value\n", encoding="utf-8")
    monkeypatch.setenv("FINMIND_TOKEN", "env-value")
    assert load_dotenv(env_file) == 0
    assert os.environ["FINMIND_TOKEN"] == "env-value"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == 0
