"""Gap-driven historical price sync using official full-market daily files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.db_store import get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.price_gap_service import (  # noqa: E402
    execute_price_gap_plan,
    plan_price_gaps,
    write_profile,
)
from settings import get_finmind_token, load_settings  # noqa: E402


@contextmanager
def _build_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"price sync is already running; lock={path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _date_value(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _feeds(value: str) -> tuple[bool, bool]:
    feeds = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not feeds or not feeds <= {"raw", "adjusted"}:
        raise argparse.ArgumentTypeError("feeds must be raw, adjusted, or raw,adjusted")
    return "raw" in feeds, "adjusted" in feeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", required=True, type=_date_value)
    parser.add_argument("--end", required=True, type=_date_value)
    parser.add_argument("--feeds", default="raw,adjusted", type=_feeds)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-json", default="reports/price_sync_profile.json")
    parser.add_argument("--cache-dir", default="data/cache/official_prices")
    parser.add_argument("--lock-file", default="database/price_sync.lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        print("start date is after end date", file=sys.stderr)
        return 2
    load_dotenv()
    try:
        settings = load_settings(args.config)
        engine = get_engine(settings)
        include_raw, include_adjusted = args.feeds
        plan = plan_price_gaps(
            engine,
            args.start,
            args.end,
            include_raw=include_raw,
            include_adjusted=include_adjusted,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"cannot plan price sync: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    plan_report = {"mode": "dry-run", "plan": plan.to_dict()}
    if args.dry_run:
        write_profile(plan_report, args.profile_json)
        console_report = {"mode": "dry-run", "plan": plan.to_dict(include_symbols=False)}
        print(json.dumps(console_report, ensure_ascii=False, indent=2))
        return 0
    token = get_finmind_token(settings)
    try:
        with _build_lock(Path(args.lock_file)):
            result = execute_price_gap_plan(
                engine,
                plan,
                token=token,
                cache_dir=args.cache_dir,
            )
    except Exception as exc:
        print(f"price sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = {"mode": "execute", "plan": plan.to_dict(), "result": result}
    write_profile(report, args.profile_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["unresolved_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
