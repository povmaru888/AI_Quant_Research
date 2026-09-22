"""Shared FinMind v4 plumbing (P3-01, P3-02).

The token travels in the Authorization header only and never appears
in exceptions, logs, or stored frames (SDD section 16).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(Exception):
    """FinMind fetch failure; guaranteed token-free message."""


def _require_range(start: str, end: str) -> None:
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, str):
            raise ValueError(f"invalid {name}: must be YYYY-MM-DD, got {value!r}")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid {name}: {value!r}") from exc
    if start > end:
        raise ValueError(f"invalid range: start {start!r} exceeds end {end!r}")


def _require_token(token: str) -> str:
    if not isinstance(token, str) or not token.strip():
        raise FinMindError("finmind request rejected: missing token")
    return token


def _get(
    dataset: str,
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> list[dict]:
    _require_range(start, end)
    credential = _require_token(token)
    try:
        response = requester(
            BASE_URL,
            params={"dataset": dataset, "start_date": start, "end_date": end},
            headers={"Authorization": f"Bearer {credential}"},
            timeout=timeout,
        )
    except Exception as exc:
        raise FinMindError(f"finmind {dataset} transport failure: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise FinMindError(f"finmind {dataset} HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception as exc:
        raise FinMindError(f"finmind {dataset} undecodable body") from exc
    if not isinstance(payload, dict) or payload.get("status") != 200:
        detail = payload.get("msg") if isinstance(payload, dict) else None
        raise FinMindError(f"finmind {dataset} API error: {detail}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise FinMindError(f"finmind {dataset} malformed data envelope")
    return data


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
