from __future__ import annotations

from pathlib import Path

from integrations.request_limiter import SqliteRequestLimiter


def test_sqlite_request_limiter_shares_rolling_budget(tmp_path: Path) -> None:
    now = [1000.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    path = tmp_path / "requests.sqlite3"
    first = SqliteRequestLimiter(
        path, limit=2, window_seconds=10, clock=lambda: now[0], sleeper=sleep
    )
    second = SqliteRequestLimiter(
        path, limit=2, window_seconds=10, clock=lambda: now[0], sleeper=sleep
    )
    first.acquire("prices")
    second.acquire("prices")
    third = first.acquire("prices")
    first.finish(third, 200)
    assert now[0] > 1010
    assert first.usage() == 1
