"""P3-09 acceptance: Gemini adapter (mocked poster, never real)."""

from __future__ import annotations

import inspect

import pytest

from integrations.gemini_report import GeminiError, build_prompt, generate_summary

KEY = "gemini-key-secret"


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _context(**overrides) -> dict:
    base = {
        "top_holdings": ["2330", "2317"],
        "shap_top": ["momentum_20d"],
        "market_regime": "above_ma60",
        "portfolio_risk": {"exposure": 1.0},
        "announcements": ["Q4 results approved for release"],
    }
    base.update(overrides)
    return base


def _ok_payload() -> dict:
    return {"candidates": [{"content": {"parts": [{"text": "穩健"}, {"text": "續抱"}]}}]}


def test_build_prompt_allow_list_and_constraint() -> None:
    prompt = build_prompt(_context())
    assert "Do not give investment advice" in prompt
    assert "2330" in prompt and "momentum_20d" in prompt
    assert "Q4 results" in prompt
    with pytest.raises(ValueError, match="disallowed keys"):
        build_prompt({**_context(), "account_balance": 999})
    with pytest.raises(ValueError, match="announcements"):
        build_prompt(_context(announcements="not-a-list"))


def test_generate_summary_success() -> None:
    seen: dict = {}

    def poster(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        return _StubResponse(200, _ok_payload())

    assert generate_summary(_context(), KEY, poster=poster) == "穩健續抱"
    assert "generateContent" in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == KEY
    assert "?" not in seen["url"]


def test_generate_summary_cannot_touch_research_state() -> None:
    params = list(inspect.signature(generate_summary).parameters)
    assert params == ["context", "api_key", "poster", "timeout", "model"]
    before = _context()
    snapshot = repr(sorted(before.items()))

    def poster(url, headers=None, json=None, timeout=None):
        return _StubResponse(200, _ok_payload())

    generate_summary(before, KEY, poster=poster)
    assert repr(sorted(before.items())) == snapshot


def test_generate_summary_errors_hide_key() -> None:
    def bad_status(*args, **kwargs):
        return _StubResponse(status_code=400, payload={"error": "bad key"})

    with pytest.raises(GeminiError, match="HTTP 400") as exc_info:
        generate_summary(_context(), KEY, poster=bad_status)
    assert KEY not in str(exc_info.value)

    def boom(*args, **kwargs):
        raise TimeoutError("slow")

    with pytest.raises(GeminiError, match="transport") as exc_info:
        generate_summary(_context(), KEY, poster=boom)
    assert KEY not in str(exc_info.value)

    with pytest.raises(GeminiError, match="missing api key"):
        generate_summary(_context(), " ", poster=boom)


def test_generate_summary_unreadable_body() -> None:
    def poster(*args, **kwargs):
        return _StubResponse(200, {"nope": []})

    with pytest.raises(GeminiError, match="unreadable"):
        generate_summary(_context(), KEY, poster=poster)
