"""Sync-free helper predicates + FinMind throttle error shape (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sync_free import ban_wait_seconds, is_gated, known_feed_skips  # noqa: E402

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


def test_known_402_skips_do_not_suppress_price_feeds(tmp_path) -> None:
    path = tmp_path / "skip_402.txt"
    path.write_text("2330\nfinancials:2317\ninstitutional:0050\n", encoding="utf-8")
    known = known_feed_skips(path)
    assert known["prices"] == known["price_adj"] == set()
    assert known["financials"] == {"2330", "2317"}
    assert known["institutional"] == {"2330", "0050"}


@pytest.mark.parametrize(
    ("coverage", "expected_status", "expected_code"),
    [((1, 1), "succeeded", 0), ((1, 0), "failed", 1)],
)
def test_default_sync_fetches_adjusted_despite_legacy_fundamental_skip(
    tmp_path, monkeypatch, coverage, expected_status, expected_code
) -> None:
    import pandas as pd
    import sync_free

    skip_path = tmp_path / "skip_402.txt"
    skip_path.write_text("2330\n", encoding="utf-8")
    calls: list[str] = []

    class Store:
        def start_run(self, metadata):
            pass

        def finish_run(self, run_id, status, error=None):
            calls.append(status)

        def upsert_prices(self, frame):
            calls.append("raw")
            return len(frame)

        def upsert_price_adj(self, frame):
            calls.append("adjusted")
            return len(frame)

        def load_adjusted_coverage(self, on):
            return coverage

    monkeypatch.setattr(sync_free, "load_dotenv", lambda: None)
    monkeypatch.setattr(sync_free, "load_settings", lambda path: object())
    monkeypatch.setattr(sync_free, "get_finmind_token", lambda settings: "tok")
    monkeypatch.setattr(sync_free, "build_store", lambda settings: Store())
    monkeypatch.setattr(
        sync_free, "fetch_prices", lambda *a, **k: pd.DataFrame({"stock_id": ["2330"]})
    )
    monkeypatch.setattr(
        sync_free, "fetch_price_adj", lambda *a, **k: pd.DataFrame({"stock_id": ["2330"]})
    )
    monkeypatch.setattr(
        sync_free,
        "fetch_financials",
        lambda *a, **k: pytest.fail("legacy gate should skip only financials"),
    )
    monkeypatch.setattr(
        sync_free,
        "fetch_institutional",
        lambda *a, **k: pytest.fail("legacy gate should skip only institutional"),
    )
    assert sync_free.main(
        [
            "--start", "2020-01-01", "--end", "2020-01-31", "--symbols", "2330",
            "--skip-file", str(skip_path), "--delay", "0",
        ]
    ) == expected_code
    assert calls == ["raw", "adjusted", expected_status]
