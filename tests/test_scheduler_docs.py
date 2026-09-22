"""P3-07 acceptance: scheduler doc contains procedures, no secrets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "windows-task-scheduler.md"
REPO = Path(__file__).resolve().parents[1]


def _text() -> str:
    assert DOC.is_file(), f"missing {DOC}"
    return DOC.read_text(encoding="utf-8")


def test_scheduler_doc_procedures() -> None:
    text = _text()
    assert "schtasks" in text
    assert "QuantDailyUpdate" in text and "QuantMonthlyRebalance" in text
    assert "09:00" in text
    assert "logs" in text
    assert "--signal-date" in text and "--as-of" in text
    assert "exit code" in text or "exit Code" in text or "exit" in text


def test_scheduler_doc_no_secrets() -> None:
    text = _text()
    assert "Bearer " not in text
    assert "sk-" not in text
    for line in text.splitlines():
        match = re.search(r"(?i)(token|api[_-]?key|secret)\s*=\s*(\S+)", line)
        assert match is None, line


def test_scheduler_doc_dry_run_commands_parse() -> None:
    text = _text()
    assert "daily_update --help" in text
    assert "monthly_rebalance --help" in text


def test_job_help_runs_from_repo_root() -> None:
    for module in ("src.jobs.daily_update", "src.jobs.monthly_rebalance"):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
