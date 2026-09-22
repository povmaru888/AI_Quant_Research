"""P3-09: Gemini summary adapter (P1 priority, MVP-deferrable).

Renders a read-only Chinese/English digest via the Gemini REST API
without adding dependencies. Input is allow-listed (holdings, SHAP,
regime, risk, approved announcements); the output is a plain string
that can never mutate signals, weights, orders, or models. External
failures raise ``GeminiError`` so callers can absorb them and keep the
research run green (SDD section 16: LLM failure must not block).
"""

from __future__ import annotations

from collections.abc import Callable

import requests

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"

ALLOWED_CONTEXT_KEYS: tuple[str, ...] = (
    "top_holdings",
    "shap_top",
    "market_regime",
    "portfolio_risk",
    "announcements",
)

_CONSTRAINT = (
    "Summarize the provided portfolio research facts only. "
    "Do not give investment advice and do not invent numbers."
)


class GeminiError(Exception):
    """Gemini call failure; guaranteed key-free message."""


def build_prompt(context: dict) -> str:
    """Render the allow-listed context plus the system constraint."""
    if not isinstance(context, dict):
        raise ValueError("invalid context: must be a dict")
    unknown = [k for k in context if k not in ALLOWED_CONTEXT_KEYS]
    if unknown:
        raise ValueError(f"invalid context: disallowed keys {unknown}")
    announcements = context.get("announcements", [])
    if announcements and (
        not isinstance(announcements, list) or not all(isinstance(a, str) for a in announcements)
    ):
        raise ValueError("invalid context: announcements must be a list of strings")
    lines = [_CONSTRAINT]
    if context.get("top_holdings") is not None:
        lines.append(f"Top holdings: {list(context['top_holdings'])}")
    if context.get("shap_top") is not None:
        lines.append(f"SHAP top factors: {list(context['shap_top'])}")
    if context.get("market_regime") is not None:
        lines.append(f"Market regime: {context['market_regime']}")
    if context.get("portfolio_risk") is not None:
        lines.append(f"Portfolio risk: {context['portfolio_risk']}")
    if announcements:
        lines.append("Approved announcements:")
        lines.extend(f"- {item}" for item in announcements)
    return "\n".join(lines)


def generate_summary(
    context: dict,
    api_key: str,
    poster: Callable[..., requests.Response] = requests.post,
    timeout: float = 60.0,
    model: str = DEFAULT_MODEL,
) -> str:
    """Call Gemini and return the text digest."""
    prompt = build_prompt(context)
    if not isinstance(api_key, str) or not api_key.strip():
        raise GeminiError("gemini call rejected: missing api key")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"invalid model: {model!r}")
    try:
        response = poster(
            f"{_API_BASE}/{model.strip()}:generateContent",
            headers={"x-goog-api-key": api_key.strip(), "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
    except Exception as exc:
        raise GeminiError(f"gemini transport failure: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise GeminiError(f"gemini HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception as exc:
        raise GeminiError("gemini undecodable body") from exc
    try:
        candidates = payload["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise GeminiError("gemini unreadable response") from exc
    if not text:
        raise GeminiError("gemini empty response")
    return text
