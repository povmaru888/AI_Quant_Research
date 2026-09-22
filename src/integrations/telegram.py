"""P3-08: Telegram notification adapter (P1 priority, MVP-deferrable).

Sends a plain-text research summary via the Bot HTTP API. The token
travels in the URL path per Telegram's scheme and never appears in
exceptions. No real message is sent in unit tests (sender injection).
"""

from __future__ import annotations

from collections.abc import Callable

import requests

_API_BASE = "https://api.telegram.org"
_MAX_LENGTH = 4000


class TelegramError(Exception):
    """Telegram send failure; guaranteed token-free message."""


def _escape(text: str) -> str:
    return "".join(f"\\{c}" if c in "_*[]()~" else c for c in str(text))


def format_research_alert(summary: dict) -> str:
    """Render the research summary as plain text."""
    if not isinstance(summary, dict):
        raise ValueError("invalid summary: must be a dict")
    run_id = summary.get("run_id", "n/a")
    status = summary.get("status", "unknown")
    lines = [f"run {run_id}: {status}"]
    holdings = summary.get("top_holdings") or []
    lines.append("top holdings: " + (", ".join(_escape(h) for h in holdings[:15]) or "n/a"))
    if summary.get("max_drawdown") is not None:
        lines.append(f"max drawdown: {summary['max_drawdown']}")
    if summary.get("missing_notes"):
        lines.append(f"missing data: {summary['missing_notes']}")
    if status == "failed" and summary.get("error"):
        lines.append(f"error: {summary['error']}")
    text = "\n".join(lines)
    if len(text) > _MAX_LENGTH:
        text = text[:_MAX_LENGTH] + "\u2026(truncated)"
    return text


def send_research_alert(
    summary: dict,
    token: str,
    chat_id: str,
    sender: Callable[..., requests.Response] = requests.post,
    timeout: float = 30.0,
    enabled: bool = True,
) -> str:
    """Send the alert; return 'sent' or 'disabled'."""
    if not enabled:
        return "disabled"
    if not isinstance(token, str) or not token.strip():
        raise TelegramError("telegram send rejected: missing token")
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise TelegramError("telegram send rejected: missing chat_id")
    try:
        response = sender(
            f"{_API_BASE}/bot{token.strip()}/sendMessage",
            json={"chat_id": chat_id.strip(), "text": format_research_alert(summary)},
            timeout=timeout,
        )
    except Exception as exc:
        raise TelegramError(f"telegram transport failure: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise TelegramError(f"telegram HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception as exc:
        raise TelegramError("telegram undecodable body") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise TelegramError("telegram API error")
    return "sent"
