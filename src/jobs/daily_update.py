"""P3-05: daily update CLI job (SDD 14.3).

Syncs the full configured window (idempotent upsert makes reruns safe)
and closes a PipelineRun. Partial syncs fail the job loudly: research
must never run on half a dataset (SDD section 16).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python -m src.jobs.*`

from services.sync_service import SyncStore, SyncSummary, sync_market_data
from settings import Settings, get_finmind_token, load_settings


class DailyStore(SyncStore, Protocol):
    """Sync persistence plus the job's read needs."""

    def load_latest_trade_date(self) -> str | None: ...
    def load_symbols(self) -> list[str]: ...
    def start_run(self, metadata: dict) -> None: ...
    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None: ...


def run_daily_update(
    as_of: date | None,
    settings: Settings,
    store: DailyStore,
    token: str | None,
    sync_fn=sync_market_data,
    run_id: str | None = None,
) -> dict:
    """Run one daily update; raise on partial sync."""
    resolved = as_of or _default_as_of(store)
    job_run_id = run_id or f"daily-{resolved.isoformat()}"
    store.start_run(
        {"run_id": job_run_id, "job": "daily_update", "data_end_date": resolved.isoformat()}
    )
    summary: SyncSummary = sync_fn(
        settings.data.price_start_date,
        resolved.isoformat(),
        settings,
        store,
        job_run_id,
        token,
        symbols=store.load_symbols(),
    )
    if not summary.ok:
        errors = "; ".join(
            f"{feed.name}: {feed.error}"
            for feed in (summary.prices, summary.financials, summary.institutional)
            if feed.error is not None
        )
        store.finish_run(job_run_id, "failed", errors or "unknown sync failure")
        raise RuntimeError(f"daily update {job_run_id} failed: {errors}")
    store.finish_run(job_run_id, "succeeded")
    return {
        "run_id": job_run_id,
        "as_of": resolved.isoformat(),
        "ok": True,
        "rows": {
            "prices": summary.prices.rows,
            "financials": summary.financials.rows,
            "institutional": summary.institutional.rows,
        },
        "fallback_used": summary.fallback_used,
    }


def _default_as_of(store: DailyStore) -> date:
    latest = store.load_latest_trade_date()
    if latest is None:
        return date.today()
    return date.fromisoformat(latest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily market data update job.")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to latest trade date.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser


def main(argv: list[str] | None = None, store_factory=None) -> int:
    """CLI entry; 0 on success, 1 on failure, 2 on bad arguments."""
    args = build_parser().parse_args(argv)
    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else None
    except ValueError:
        print(f"invalid --as-of: {args.as_of!r}, expected YYYY-MM-DD", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    try:
        from runtime.dotenv import load_dotenv

        load_dotenv()
        factory = store_factory or _build_store
        store = factory(settings)
    except NotImplementedError as exc:
        print(f"daily update unavailable: {exc}", file=sys.stderr)
        return 1
    token = get_finmind_token(settings)
    try:
        result = run_daily_update(as_of, settings, store, token)
    except Exception as exc:
        print(f"daily update failed: {exc}", file=sys.stderr)
        return 1
    print(f"run_id={result['run_id']} as_of={result['as_of']} rows={result['rows']}")
    return 0


def _build_store(settings: Settings) -> DailyStore:
    """Assemble the database-backed store (runtime wiring)."""
    from runtime.db_store import build_store

    return build_store(settings)


if __name__ == "__main__":
    raise SystemExit(main())
