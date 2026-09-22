"""P0-05 acceptance: .env.example hygiene and name consistency."""

from __future__ import annotations

import re
from pathlib import Path

from settings import load_settings

ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / ".env.example"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

REQUIRED_NAMES = frozenset(
    {
        "FINMIND_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "SHIOAJI_API_KEY",
        "SHIOAJI_SECRET_KEY",
    }
)
# Explicit placeholders allowed besides blank; anything else is a leak.
ALLOWED_PLACEHOLDERS = frozenset({"", "placeholder", "changeme", "your-token-here"})
_SUSPICIOUS_PREFIXES = ("sk-", "xoxb-", "xoxp-", "AIza", "ya29.")


def _parse_env_example() -> dict[str, str]:
    assert ENV_EXAMPLE_PATH.is_file(), f"missing {ENV_EXAMPLE_PATH}"
    parsed: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"invalid line (missing '='): {raw_line!r}"
        key, _, value = line.partition("=")
        key = key.strip()
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key), f"invalid key: {key!r}"
        assert key not in parsed, f"duplicate key: {key}"
        parsed[key] = value.strip().strip('"').strip("'").strip()
    return parsed


def test_env_example_contains_required_names() -> None:
    parsed = _parse_env_example()
    assert set(parsed) == set(REQUIRED_NAMES), f"got {sorted(parsed)}"


def test_env_example_has_no_secret_values() -> None:
    parsed = _parse_env_example()
    for key, value in parsed.items():
        lowered = value.lower()
        assert lowered in ALLOWED_PLACEHOLDERS, f"{key} must be blank or placeholder"
        assert len(value) < 20, f"{key} value too long to be a placeholder"
        assert not value.lower().startswith(_SUSPICIOUS_PREFIXES), key


def test_names_consistent_with_settings() -> None:
    parsed = _parse_env_example()
    settings = load_settings(CONFIG_PATH, env={})
    assert settings.data.finmind_token_env in parsed
    assert settings.data.finmind_token_env == "FINMIND_TOKEN"
