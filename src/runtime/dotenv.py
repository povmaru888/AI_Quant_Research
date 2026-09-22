"""Minimal .env loader (stdlib only; no new dependency).

Reads KEY=VALUE lines from the project-root ``.env`` into ``os.environ``
without overwriting variables that are already set. Missing file is a
no-op so unit tests and CI are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path | None = None) -> int:
    """Load .env into os.environ; return the number of keys set."""
    target = path or ENV_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return 0
    loaded = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'").strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded
