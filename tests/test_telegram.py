"""P3-08 acceptance: Telegram adapter (mocked sender, never real)."""

from __future__ import annotations

import pytest

from integrations.telegram import TelegramError, format_research_alert, send_research_alert

TOKEN = "bot-token-secret"


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = {"ok": True} if payload is None else payload

    def json(self):
        return self._payload


def _summary(**overrides) -> dict:
    base = {
        "run_id": "run-001",
        "status": "succeeded",
        "top_holdings": ["2330", "2317"],
        "max_drawdown": -0.12,
        "missing_notes": "dividend_yield uncovered",
    }
    base.update(overrides)
    return base


def test_format_research_alert_contents() -> None:
    text = format_research_alert(_summary())
    assert "run-001" in text and "succeeded" in text
    assert "2330" in text and "2317" in text
    assert "-0.12" in text and "dividend_yield" in text
    failed = format_research_alert(_summary(status="failed", error="covariance missing"))
    assert "covariance missing" in failed
    assert "covariance" not in format_research_alert(_summary())


def test_format_research_alert_escapes() -> None:
    text = format_research_alert(_summary(top_holdings=["A_B", "C*D"]))
    assert "A\\_B" in text and "C\\*D" in text
    with pytest.raises(ValueError, match="summary"):
        format_research_alert([])


def test_send_disabled_never_calls() -> None:
    calls: list = []

    def sender(*args, **kwargs):
        calls.append((args, kwargs))
        return _StubResponse()

    assert send_research_alert(_summary(), TOKEN, "123", sender=sender, enabled=False) == "disabled"
    assert calls == []


def test_send_success_posts_text() -> None:
    seen: dict = {}

    def sender(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        return _StubResponse()

    assert send_research_alert(_summary(), TOKEN, "123", sender=sender) == "sent"
    assert seen["url"].startswith("https://api.telegram.org/bot")
    assert seen["json"]["chat_id"] == "123"
    assert "run-001" in seen["json"]["text"]


def test_send_errors_hide_token() -> None:
    with pytest.raises(TelegramError, match="chat_id"):
        send_research_alert(_summary(), TOKEN, "  ")
    with pytest.raises(TelegramError, match="missing token"):
        send_research_alert(_summary(), " ", "123")

    def bad_status(*args, **kwargs):
        return _StubResponse(status_code=401, payload={"ok": False})

    with pytest.raises(TelegramError, match="HTTP 401") as exc_info:
        send_research_alert(_summary(), TOKEN, "123", sender=bad_status)
    assert TOKEN not in str(exc_info.value)

    def boom(*args, **kwargs):
        raise ConnectionError("down")

    with pytest.raises(TelegramError) as exc_info:
        send_research_alert(_summary(), TOKEN, "123", sender=boom)
    assert TOKEN not in str(exc_info.value)
