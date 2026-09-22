"""Sync-free helper predicates + FinMind throttle error shape (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sync_free import ban_wait_seconds, is_gated  # noqa: E402

from integrations.finmind import FinMindError, _http_error  # noqa: E402


class _StubResponse:
    def __init__(self, status_code: int, payload: object = None, broken: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._broken = broken

    def json(self):
        if self._broken:
            raise ValueError("nope")
        return self._payload


def test_http_error_carries_msg_and_retry_after() -> None:
    err = _http_error(
        "X", _StubResponse(403, {"msg": "ip banned", "retry_after": 1089, "token_tail": "...x"})
    )
    assert isinstance(err, FinMindError)
    assert "HTTP 403" in str(err)
    assert "ip banned" in str(err)
    assert err.retry_after == 1089.0
    assert "token_tail" not in str(err)


def test_http_error_undecodable_body() -> None:
    err = _http_error("X", _StubResponse(500, broken=True))
    assert "HTTP 500" in str(err)
    assert err.retry_after is None


def test_is_gated() -> None:
    assert is_gated(FinMindError("finmind X HTTP 402: ..."))
    assert is_gated(FinMindError("finmind X HTTP 403: ..."))
    assert not is_gated(FinMindError("finmind X HTTP 500: ..."))
    assert not is_gated(ValueError("boom"))


def test_ban_wait_seconds() -> None:
    assert ban_wait_seconds(FinMindError("ip banned", retry_after=1089)) == 1089.0
    assert ban_wait_seconds(FinMindError("HTTP 403", retry_after=10)) is None
    assert ban_wait_seconds(FinMindError("ip banned")) is None
    assert ban_wait_seconds(ValueError("ip banned")) is None


def test_sync_free_imports_without_side_effects() -> None:
    import sync_free

    assert callable(sync_free.main)
    with pytest.raises(SystemExit):
        sync_free.main(["--help"])
