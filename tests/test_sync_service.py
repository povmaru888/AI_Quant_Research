"""P3-04 acceptance: retryable sync service (mocked feeds and store)."""

from __future__ import annotations

import pandas as pd

from integrations.finmind import FinMindError
from services.sync_service import SyncSummary, sync_market_data


class FakeStore:
    """Upsert-counting store; reruns overwrite the same keys."""

    def __init__(self) -> None:
        self.saved: dict[str, pd.DataFrame] = {}

    def _upsert(self, key: str, frame: pd.DataFrame) -> int:
        self.saved[key] = frame.copy()
        return len(frame)

    def upsert_prices(self, frame: pd.DataFrame) -> int:
        return self._upsert("prices", frame)

    def upsert_financials(self, frame: pd.DataFrame) -> int:
        return self._upsert("financials", frame)

    def upsert_institutional(self, frame: pd.DataFrame) -> int:
        return self._upsert("institutional", frame)


def _frame(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"a": range(rows)})


def test_sync_all_green(settings) -> None:
    store = FakeStore()
    summary = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "run-001",
        "tok",
        fetch_prices_fn=lambda s, e, t: _frame(2),
        fetch_financials_fn=lambda s, e, t: _frame(3),
        fetch_institutional_fn=lambda s, e, t: _frame(4),
    )
    assert isinstance(summary, SyncSummary)
    assert summary.ok
    assert (summary.prices.rows, summary.prices.attempts) == (2, 1)
    assert (summary.financials.rows, summary.institutional.rows) == (3, 4)
    assert not summary.fallback_used
    # Rerun is idempotent: same keys overwritten, no growth.
    again = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "run-002",
        "tok",
        fetch_prices_fn=lambda s, e, t: _frame(2),
        fetch_financials_fn=lambda s, e, t: _frame(3),
        fetch_institutional_fn=lambda s, e, t: _frame(4),
    )
    assert again.ok and len(store.saved["prices"]) == 2


def test_sync_prices_retry_then_success(settings) -> None:
    calls = {"n": 0}

    def flaky(start, end, token):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FinMindError("boom")
        return _frame(1)

    store = FakeStore()
    summary = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "r",
        "tok",
        fetch_prices_fn=flaky,
        fetch_financials_fn=lambda s, e, t: _frame(),
        fetch_institutional_fn=lambda s, e, t: _frame(),
        max_attempts=3,
    )
    assert summary.ok
    assert summary.prices.attempts == 3
    assert not summary.fallback_used


def test_sync_prices_fallback_only(settings) -> None:
    def dead(start, end, token):
        raise FinMindError("down")

    fallback = _frame(5)
    fallback.attrs["failed"] = []
    store = FakeStore()
    summary = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "r",
        "tok",
        fetch_prices_fn=dead,
        fetch_financials_fn=lambda s, e, t: _frame(),
        fetch_institutional_fn=lambda s, e, t: _frame(),
        fetch_fallback_fn=lambda syms, s, e: fallback,
        symbols=["2330"],
        max_attempts=2,
    )
    assert summary.fallback_used
    assert summary.prices.source == "yfinance"
    assert summary.prices.rows == 5
    assert summary.ok


def test_sync_feed_failure_has_no_fallback(settings) -> None:
    store = FakeStore()
    summary = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "r",
        "tok",
        fetch_prices_fn=lambda s, e, t: _frame(),
        fetch_financials_fn=lambda s, e, t: (_ for _ in ()).throw(FinMindError("gone")),
        fetch_institutional_fn=lambda s, e, t: _frame(),
        fetch_fallback_fn=lambda syms, s, e: _frame(9),
        symbols=["2330"],
    )
    assert not summary.ok
    assert summary.financials.error == "gone"
    assert summary.financials.rows == 0
    assert not summary.fallback_used


def test_sync_value_error_not_retried_and_missing_token(settings) -> None:
    calls = {"n": 0}

    def broken(start, end, token):
        calls["n"] += 1
        raise ValueError("bad columns")

    store = FakeStore()
    summary = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        store,
        "r",
        "tok",
        fetch_prices_fn=broken,
        fetch_financials_fn=lambda s, e, t: _frame(),
        fetch_institutional_fn=lambda s, e, t: _frame(),
    )
    assert calls["n"] == 1
    assert summary.prices.attempts == 1
    assert not summary.ok

    tokenless = sync_market_data(
        "2020-01-01",
        "2020-01-31",
        settings,
        FakeStore(),
        "r",
        None,
        fetch_prices_fn=lambda s, e, t: _frame(),
        fetch_financials_fn=lambda s, e, t: _frame(),
        fetch_institutional_fn=lambda s, e, t: _frame(),
    )
    assert tokenless.prices.attempts == 0
    assert tokenless.financials.error == "missing token"
    assert not tokenless.ok
