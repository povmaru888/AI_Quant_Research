"""P5-05 acceptance: runtime secret validation (injected env only)."""

from __future__ import annotations

import pytest

from security.secrets import (
    SecretError,
    is_enabled,
    secret_env_name,
    validate_runtime_secrets,
)
from settings import Settings


def test_all_disabled_passes(settings: Settings) -> None:
    validate_runtime_secrets(settings, env={})


def test_enabled_without_secret_fails(settings: Settings) -> None:
    with pytest.raises(SecretError, match="FINMIND_TOKEN"):
        validate_runtime_secrets(settings, env={"FINMIND_ENABLED": "1"})
    with pytest.raises(SecretError, match="TELEGRAM_BOT_TOKEN"):
        validate_runtime_secrets(settings, env={"TELEGRAM_ENABLED": "true"})
    with pytest.raises(SecretError, match="GEMINI_API_KEY"):
        validate_runtime_secrets(settings, env={"GEMINI_ENABLED": "yes", "GEMINI_API_KEY": "  "})


def test_enabled_with_secret_passes(settings: Settings) -> None:
    validate_runtime_secrets(
        settings,
        env={
            "FINMIND_ENABLED": "1",
            "FINMIND_TOKEN": "finmind-secret-value",
            "TELEGRAM_ENABLED": "TRUE",
            "TELEGRAM_BOT_TOKEN": "telegram-secret-value",
            "GEMINI_ENABLED": "Yes",
            "GEMINI_API_KEY": "gemini-secret-value",
        },
    )


def test_errors_name_names_not_values(settings: Settings) -> None:
    with pytest.raises(SecretError, match="FINMIND_TOKEN") as exc_info:
        validate_runtime_secrets(settings, env={"FINMIND_ENABLED": "1"})
    assert "finmind-secret-value" not in str(exc_info.value)


def test_flag_parsing() -> None:
    assert is_enabled({"FINMIND_ENABLED": "1"}, "finmind_enabled") is True
    assert is_enabled({"FINMIND_ENABLED": "  True "}, "FINMIND_ENABLED") is True
    assert is_enabled({"FINMIND_ENABLED": "0"}, "finmind_enabled") is False
    assert is_enabled({}, "finmind_enabled") is False


def test_finmind_name_follows_settings(settings: Settings) -> None:
    assert secret_env_name("finmind_enabled", settings) == settings.data.finmind_token_env
    assert secret_env_name("telegram_enabled", settings) == "TELEGRAM_BOT_TOKEN"
    assert secret_env_name("gemini_enabled", settings) == "GEMINI_API_KEY"
    with pytest.raises(SecretError, match="Settings"):
        validate_runtime_secrets("nope", env={})  # type: ignore[arg-type]
