"""Parallel, resumable per-symbol FinMind financial-statement backfill."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.finmind_fundamentals import fetch_financials, fetch_institutional  # noqa: E402
from runtime.db_store import build_store  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill FinMind financials per stock.")
    parser.add_argument("--config", default="config.factor_v4.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols-file", required=True)
    parser.add_argument("--feed", choices=("financials", "institutional"), default="financials")
    parser.add_argument("--failures", default="logs/factor_v4_financial_failures.txt")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        raise RuntimeError("missing FINMIND_TOKEN")
    symbols = [
        value.strip()
        for value in Path(args.symbols_file).read_text(encoding="utf-8-sig").splitlines()
        if value.strip()
    ]
    store = build_store(settings)

    def fetch(stock_id: str):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                if args.feed == "financials":
                    frame = fetch_financials(
                        args.start, args.end, token, stock_id=stock_id, timeout=60.0
                    )
                else:
                    frame = fetch_institutional(
                        args.start,
                        args.end,
                        token,
                        stock_id=stock_id,
                        timeout=60.0,
                        include_floats=False,
                    )
                return stock_id, frame
            except Exception as exc:  # noqa: BLE001 - per-symbol retry boundary.
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"{stock_id}: {last_error}")

    failures: list[str] = []
    rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, stock_id): stock_id for stock_id in symbols}
        for completed, future in enumerate(as_completed(futures), 1):
            stock_id = futures[future]
            try:
                _, frame = future.result()
                rows += (
                    store.upsert_financials(frame)
                    if args.feed == "financials"
                    else store.upsert_institutional(frame)
                )
            except Exception as exc:  # noqa: BLE001 - retain checkpoint progress.
                failures.append(stock_id)
                print(f"[{completed}/{len(symbols)}] {stock_id} FAILED: {exc}", file=sys.stderr)
            if completed % 100 == 0:
                print(
                    f"[{completed}/{len(symbols)}] rows={rows} failures={len(failures)}", flush=True
                )
    Path(args.failures).write_text("\n".join(failures), encoding="utf-8")
    print(f"done symbols={len(symbols)} rows={rows} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
