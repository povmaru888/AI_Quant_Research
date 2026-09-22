"""P5-05: runtime secret validation (SDD section 16 security).

Enabled integrations must present their secrets; disabled ones must
not be asked. Enablement lives in the environment (``*_ENABLED``), so
``Settings`` stays frozen as P0-04 defined it. Only variable *names*
appear in errors, never values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from settings import Settings

SECRET_ENV_NAMES: dict[str, str] = {
    "finmind_enabled": "FINMIND_TOKEN",
    "telegram_enabled": "TELEGRAM_BOT_TOKEN",
    "gemini_enabled": "GEMINI_API_KEY",
}

_TRUTHY = ("1", "true", "yes")


class SecretError(ValueError):
    """An enabled integration is missing its secret."""


def is_enabled(env: Mapping[str, str], flag: str) -> bool:
    """Return True when ``flag`` is set to a truthy value."""
    value = env.get(flag.upper(), env.get(flag))
    return isinstance(value, str) and value.strip().lower() in _TRUTHY


def secret_env_name(integration: str, settings: Settings) -> str:
    """Return the env var holding ``integration``'s secret.

    FinMind's name follows ``settings.data.finmind_token_env`` (P0-05
    convention); Telegram/Gemini names are fixed by ``.env.example``.
    """
    if integration == "finmind_enabled":
        return settings.data.finmind_token_env
    return SECRET_ENV_NAMES[integration]


def validate_runtime_secrets(settings: Settings, env: Mapping[str, str] | None = None) -> None:
    """Raise ``SecretError`` if an enabled integration lacks its secret."""
    if not isinstance(settings, Settings):
        raise SecretError(f"invalid settings: expected Settings, got {type(settings).__name__}")
    source: Mapping[str, str] = os.environ if env is None else env
    for flag in SECRET_ENV_NAMES:
        if not is_enabled(source, flag):
            continue
        name = secret_env_name(flag, settings)
        value = source.get(name)
        if not isinstance(value, str) or not value.strip():
            raise SecretError(f"missing secret for enabled integration: {name} is not set")
