"""P3-04: retryable market data sync service (SDD 7.2).

Fetches the three feeds with retries, falls back to yfinance for prices
only, and persists via a ``SyncStore`` seam (upsert idempotency lives in
the store implementation, as in P1-07/P1-08). Only transient
``FinMindError`` is retried; programming errors (``ValueError``) fail
fast without burning attempts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from integrations.finmind import FinMindError
from integrations.finmind_fundamentals import fetch_financials, fetch_institutional
from integrations.finmind_prices import fetch_prices
from integrations.yfinance_prices import fetch_fallback_prices
from settings import Settings


@dataclass(frozen=True)
class FeedResult:
    """One feed's outcome; ``error`` is None on success."""

    name: str
    rows: int
    attempts: int
    source: str
    error: str | None = None


@dataclass(frozen=True)
class SyncSummary:
    """Per-run sync ledger."""

    run_id: str
    start: str
    end: str
    prices: FeedResult
    financials: FeedResult
    institutional: FeedResult
    fallback_used: bool = False

    @property
    def ok(self) -> bool:
        """True only when every feed landed without error."""
        feeds = (self.prices, self.financials, self.institutional)
        return all(feed.error is None for feed in feeds)


class SyncStore(Protocol):
    """Persistence seam; implementations must upsert (no duplicates on rerun)."""

    def upsert_prices(self, frame: pd.DataFrame) -> int: ...
    def upsert_financials(self, frame: pd.DataFrame) -> int: ...
    def upsert_institutional(self, frame: pd.DataFrame) -> int: ...


def sync_market_data(
    start: str,
    end: str,
    settings: Settings,
    store: SyncStore,
    run_id: str,
    token: str | None,
    fetch_prices_fn: Callable[..., pd.DataFrame] = fetch_prices,
    fetch_financials_fn: Callable[..., pd.DataFrame] = fetch_financials,
    fetch_institutional_fn: Callable[..., pd.DataFrame] = fetch_institutional,
    fetch_fallback_fn: Callable[..., pd.DataFrame] = fetch_fallback_prices,
    symbols: Sequence[str] = (),
    max_attempts: int = 3,
) -> SyncSummary:
    """Sync one date range; return the per-feed ledger."""
    _ = settings  # thresholds live downstream; kept for signature parity.
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: {run_id!r}")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError(f"invalid max_attempts: {max_attempts!r}")

    prices, fallback_used = _sync_prices(
        start, end, store, token, fetch_prices_fn, fetch_fallback_fn, symbols, max_attempts
    )
    financials = _sync_feed(
        "financials", start, end, store.upsert_financials, token, fetch_financials_fn, max_attempts
    )
    institutional = _sync_feed(
        "institutional",
        start,
        end,
        store.upsert_institutional,
        token,
        fetch_institutional_fn,
        max_attempts,
    )
    return SyncSummary(
        run_id=run_id,
        start=start,
        end=end,
        prices=prices,
        financials=financials,
        institutional=institutional,
        fallback_used=fallback_used,
    )


def _has_token(token: str | None) -> bool:
    return isinstance(token, str) and bool(token.strip())


def _attempt(
    fetcher: Callable[[], pd.DataFrame], max_attempts: int
) -> tuple[pd.DataFrame | None, int, str | None]:
    attempts = 0
    last_error = "no attempts made"
    while attempts < max_attempts:
        attempts += 1
        try:
            return fetcher(), attempts, None
        except ValueError:
            raise
        except FinMindError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - transient transport failure.
            last_error = f"{type(exc).__name__}: {exc}"
    return None, attempts, last_error


def _sync_feed(
    name: str,
    start: str,
    end: str,
    upsert: Callable[[pd.DataFrame], int],
    token: str | None,
    fetcher: Callable[..., pd.DataFrame],
    max_attempts: int,
) -> FeedResult:
    if not _has_token(token):
        return FeedResult(name=name, rows=0, attempts=0, source="finmind", error="missing token")
    try:
        frame, attempts, error = _attempt(lambda: fetcher(start, end, token), max_attempts)
    except ValueError as exc:
        return FeedResult(name=name, rows=0, attempts=1, source="finmind", error=str(exc))
    if error is not None or frame is None:
        return FeedResult(name=name, rows=0, attempts=attempts, source="finmind", error=error)
    stored = upsert(frame)
    return FeedResult(name=name, rows=stored, attempts=attempts, source="finmind")


def _sync_prices(
    start: str,
    end: str,
    store: SyncStore,
    token: str | None,
    fetch_prices_fn: Callable[..., pd.DataFrame],
    fetch_fallback_fn: Callable[..., pd.DataFrame],
    symbols: Sequence[str],
    max_attempts: int,
) -> tuple[FeedResult, bool]:
    if not _has_token(token):
        frame, attempts, error = None, 0, "missing token"
    else:
        try:
            frame, attempts, error = _attempt(
                lambda: fetch_prices_fn(start, end, token), max_attempts
            )
        except ValueError as exc:
            no_retry = FeedResult(
                name="prices", rows=0, attempts=1, source="finmind", error=str(exc)
            )
            return no_retry, False
    if error is None and frame is not None:
        return (
            FeedResult(
                name="prices",
                rows=store.upsert_prices(frame),
                attempts=attempts,
                source="finmind",
            ),
            False,
        )
    if not list(symbols):
        return (
            FeedResult(name="prices", rows=0, attempts=attempts, source="finmind", error=error),
            False,
        )
    try:
        fallback = fetch_fallback_fn(list(symbols), start, end)
    except Exception as exc:  # noqa: BLE001 - report both legs.
        combined = f"finmind: {error}; yfinance: {type(exc).__name__}: {exc}"
        failed_result = FeedResult(
            name="prices", rows=0, attempts=attempts, source="yfinance", error=combined
        )
        return failed_result, True
    stored = store.upsert_prices(fallback)
    failed = fallback.attrs.get("failed") if hasattr(fallback, "attrs") else None
    fallback_error = f"partial fallback failure: {failed}" if failed else None
    return (
        FeedResult(
            name="prices",
            rows=stored,
            attempts=attempts,
            source="yfinance",
            error=fallback_error,
        ),
        True,
    )
