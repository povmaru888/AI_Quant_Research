"""Backfill calibrated Yahoo adjusted prices into the local price table.

Yahoo and FinMind use different absolute adjusted-price bases. This command
requires a stable per-stock overlap ratio before it writes any adjusted bar.

Warm-up example:
    python tools/backfill_yfinance_adjusted.py --mode warmup \
        --start 2014-06-01 --end 2014-12-31 \
        --anchor-start 2015-01-01 --anchor-end 2015-02-28

Missing adjusted-date example:
    python tools/backfill_yfinance_adjusted.py --mode missing-adjusted \
        --date 2016-07-08 --date 2016-09-28 \
        --anchor-start 2016-06-01 --anchor-end 2016-10-31
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrations.yfinance_prices import fetch_bulk_prices_with_adjustments  # noqa: E402
from models.market import Price  # noqa: E402
from models.security import Stock  # noqa: E402
from runtime.db_store import build_store, get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from services.price_adjustment_service import calibrate_adjusted_prices  # noqa: E402
from settings import load_settings  # noqa: E402


def _as_date(name: str, value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be YYYY-MM-DD") from exc


def _target_rows(engine, args: argparse.Namespace) -> pd.DataFrame:
    with engine.connect() as connection:
        if args.mode == "warmup":
            rows = connection.execute(
                select(Stock.stock_id, Stock.market)
                .join(Price, Price.stock_id == Stock.stock_id)
                .where(Price.trade_date >= args.anchor_start)
                .where(Price.trade_date <= args.anchor_end)
                .where(Price.close_adj.is_not(None))
                .where(Stock.stock_id != "TAIEX")
                .distinct()
                .order_by(Stock.stock_id)
            ).all()
            return pd.DataFrame(rows, columns=["stock_id", "market"])

        rows = connection.execute(
            select(Price.stock_id, Stock.market, Price.trade_date)
            .join(Stock, Stock.stock_id == Price.stock_id)
            .where(Price.trade_date.in_(args.date))
            .where(Price.stock_id != "TAIEX")
            .where(Price.close_adj.is_(None))
            .order_by(Price.stock_id, Price.trade_date)
        ).all()
    return pd.DataFrame(rows, columns=["stock_id", "market", "trade_date"])


def _references(engine, start: str, end: str) -> pd.DataFrame:
    with engine.connect() as connection:
        rows = connection.execute(
            select(Price.stock_id, Price.trade_date, Price.close_adj)
            .where(Price.trade_date >= start)
            .where(Price.trade_date <= end)
            .where(Price.stock_id != "TAIEX")
            .where(Price.close_adj.is_not(None))
            .order_by(Price.stock_id, Price.trade_date)
        ).all()
    return pd.DataFrame(rows, columns=["stock_id", "trade_date", "close_adj"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.oos2020_stable.yaml")
    parser.add_argument("--mode", choices=("warmup", "missing-adjusted"), required=True)
    parser.add_argument("--start", default=None, help="Warm-up interval start.")
    parser.add_argument("--end", default=None, help="Warm-up interval end.")
    parser.add_argument("--date", action="append", default=[], help="Target missing-adjusted date; repeatable.")
    parser.add_argument("--anchor-start", required=True)
    parser.add_argument("--anchor-end", required=True)
    parser.add_argument("--batch", type=int, default=60)
    parser.add_argument("--max-relative-spread", type=float, default=0.001)
    parser.add_argument("--minimum-anchors", type=int, default=3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    try:
        args.anchor_start = _as_date("anchor-start", args.anchor_start)
        args.anchor_end = _as_date("anchor-end", args.anchor_end)
        if args.anchor_start > args.anchor_end or args.batch < 1:
            raise ValueError("invalid calibration range or batch size")
        if args.mode == "warmup":
            if args.start is None or args.end is None:
                raise ValueError("warmup mode requires --start and --end")
            args.start = _as_date("start", args.start)
            args.end = _as_date("end", args.end)
            if args.start > args.end:
                raise ValueError("warm-up start is after end")
        else:
            if not args.date:
                raise ValueError("missing-adjusted mode requires at least one --date")
            args.date = sorted({_as_date("date", value) for value in args.date})
            args.start, args.end = args.date[0], args.date[-1]
    except ValueError as exc:
        parser.error(str(exc))

    load_dotenv()
    settings = load_settings(args.config)
    store = build_store(settings)
    engine = get_engine(settings)
    targets = _target_rows(engine, args)
    if targets.empty:
        print("no target price rows")
        return 1
    symbols = targets["stock_id"].astype(str).drop_duplicates().tolist()
    market_map = dict(
        zip(targets["stock_id"].astype(str), targets["market"].fillna("TWSE"), strict=True)
    )
    fetch_start = min(args.start, args.anchor_start)
    fetch_end = max(args.end, args.anchor_end)
    print(
        f"mode={args.mode} target_symbols={len(symbols)} fetch={fetch_start}..{fetch_end}",
        flush=True,
    )
    raw, yahoo_adj = fetch_bulk_prices_with_adjustments(
        symbols, fetch_start, fetch_end, market_map, batch=args.batch
    )
    reference = _references(engine, args.anchor_start, args.anchor_end)
    adjusted, calibration = calibrate_adjusted_prices(
        yahoo_adj,
        reference,
        max_relative_spread=args.max_relative_spread,
        minimum_anchors=args.minimum_anchors,
    )
    if args.mode == "warmup":
        in_target = raw["trade_date"].between(args.start, args.end)
        target_raw = raw.loc[in_target].copy()
        target_adj = adjusted.loc[adjusted["trade_date"].between(args.start, args.end)].copy()
        valid_keys = target_adj[["stock_id", "trade_date"]].drop_duplicates()
        target_raw = target_raw.merge(valid_keys, on=["stock_id", "trade_date"], how="inner")
        raw_written = store.upsert_prices(target_raw)
        adjusted_written = store.upsert_price_adj(target_adj)
    else:
        missing_keys = set(
            map(tuple, targets[["stock_id", "trade_date"]].astype(str).to_numpy())
        )
        target_adj = adjusted.loc[adjusted["trade_date"].isin(args.date)].copy()
        target_adj["stock_id"] = target_adj["stock_id"].astype(str)
        target_adj["trade_date"] = target_adj["trade_date"].astype(str)
        target_adj = target_adj.loc[
            pd.MultiIndex.from_frame(target_adj[["stock_id", "trade_date"]]).isin(missing_keys)
        ]
        raw_written = 0
        adjusted_written = store.upsert_price_adj(target_adj)

    target_key_count = (
        len(set(map(tuple, targets[["stock_id", "trade_date"]].astype(str).to_numpy())))
        if args.mode == "missing-adjusted"
        else len(set(map(tuple, raw.loc[raw["trade_date"].between(args.start, args.end), ["stock_id", "trade_date"]].astype(str).to_numpy())))
    )
    calibrated_symbols = {stock_id for stock_id, item in calibration.items() if item["accepted"]}
    skipped = sorted(set(symbols) - calibrated_symbols)
    report = {
        "mode": args.mode,
        "target_start": args.start,
        "target_end": args.end,
        "anchor_start": args.anchor_start,
        "anchor_end": args.anchor_end,
        "yahoo_source": "Yahoo Finance Adj Close factor, OHLC scaled to Adj Close",
        "reference_source": "stored FinMind-adjusted close observations",
        "calibration_max_relative_spread": args.max_relative_spread,
        "minimum_anchors": args.minimum_anchors,
        "target_symbol_count": len(symbols),
        "target_row_count": target_key_count,
        "target_rows_written_raw": raw_written,
        "target_rows_written_adjusted": adjusted_written,
        "download_failed_symbols": sorted(set(raw.attrs.get("failed", []))),
        "uncalibrated_symbols": skipped,
        "calibration": calibration,
    }
    report_path = Path(args.report or f"reports/yfinance_{args.mode.replace('-', '_')}_{args.start}_{args.end}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"target_rows={target_key_count} calibrated={len(calibrated_symbols)}/{len(symbols)} "
        f"raw_written={raw_written} adjusted_written={adjusted_written} "
        f"failed={len(report['download_failed_symbols'])} uncalibrated={len(skipped)}",
        flush=True,
    )
    print(f"report={report_path}")
    return 0 if adjusted_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
