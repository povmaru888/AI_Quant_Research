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
    """FinMind fetch failure; guaranteed token-free message.

    ``retry_after`` is the server's suggested wait in seconds when the
    account/IP is throttled (``None`` otherwise); sync scripts sleep it
    off instead of treating the symbol as failed.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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


def _http_error(dataset: str, response: requests.Response) -> FinMindError:
    """Build an HTTP error carrying the server message (never the token).

    Only the generic ``msg`` and numeric ``retry_after`` are kept; raw
    bodies may echo credential tails and are never included.
    """
    detail: object = None
    wait: float | None = None
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - undecodable body stays generic.
        body = None
    if isinstance(body, dict):
        detail = body.get("msg")
        raw_wait = body.get("retry_after")
        if isinstance(raw_wait, (int, float)) and raw_wait > 0:
            wait = float(raw_wait)
    return FinMindError(
        f"finmind {dataset} HTTP {response.status_code}: {detail}", retry_after=wait
    )


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
    data_id: str | None = None,
) -> list[dict]:
    _require_range(start, end)
    credential = _require_token(token)
    params: dict[str, str] = {"dataset": dataset, "start_date": start, "end_date": end}
    if data_id is not None:
        # Per-stock query: required on free ("register") tokens, which reject
        # full-market requests. Omitted when None (paid-tier bulk path).
        params["data_id"] = data_id
    try:
        response = requester(
            BASE_URL,
            params=params,
            headers={"Authorization": f"Bearer {credential}"},
            timeout=timeout,
        )
    except Exception as exc:
        raise FinMindError(f"finmind {dataset} transport failure: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise _http_error(dataset, response)
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
