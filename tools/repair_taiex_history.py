"""Fill specific missing TAIEX daily bars from FinMind or TWSE history."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from models.market import Price  # noqa: E402
from runtime.db_store import build_store, get_engine  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402
import yfinance as yf  # noqa: E402

from repair_saturday_sessions import _fetch_taiex  # noqa: E402


def _fetch_taiex_yahoo(day: str) -> pd.DataFrame:
    raw = yf.download(
        "^TWII",
        start=day,
        end=(date.fromisoformat(day) + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo ^TWII returned no row for {day}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    columns = {str(column).strip().lower(): column for column in raw.columns}
    required = {"open", "high", "low", "close"}
    if not required.issubset(columns):
        raise ValueError(f"Yahoo ^TWII missing fields for {day}")
    result = pd.DataFrame(
        {
            "stock_id": "TAIEX",
            "trade_date": pd.to_datetime(raw.index).strftime("%Y-%m-%d"),
            "open": pd.to_numeric(raw[columns["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[columns["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[columns["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[columns["close"]], errors="coerce"),
            "volume": 0.0,
            "traded_value": 0.0,
            "source": "yfinance:^TWII",
        }
    ).dropna(subset=["open", "high", "low", "close"])
    if len(result) != 1 or result.iloc[0]["trade_date"] != day:
        raise ValueError(f"Yahoo ^TWII returned {len(result)} rows for {day}")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"Yahoo ^TWII has non-positive OHLC for {day}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.oos2020_stable.yaml")
    parser.add_argument("--date", action="append", required=True, help="Missing date; repeatable.")
    parser.add_argument("--report", default="reports/taiex_history_repair.json")
    args = parser.parse_args(argv)
    try:
        days = sorted({date.fromisoformat(value).isoformat() for value in args.date})
    except ValueError as exc:
        parser.error(f"invalid date: {exc}")
    load_dotenv()
    settings = load_settings(args.config)
    token = get_finmind_token(settings)
    if not token:
        print("missing FINMIND_TOKEN", file=sys.stderr)
        return 1
    store = build_store(settings)
    engine = get_engine(settings)
    rows = []
    for day in days:
        with engine.connect() as connection:
            exists = connection.execute(
                select(Price.trade_date).where(
                    Price.stock_id == "TAIEX", Price.trade_date == day
                )
            ).first()
        if exists:
            rows.append({"trade_date": day, "status": "already_present"})
            print(f"[{day}] already present")
            continue
        finmind_or_twse_error = None
        try:
            frame = _fetch_taiex(day, token)
        except Exception as exc:  # noqa: BLE001 - Yahoo is the final fallback.
            finmind_or_twse_error = f"{type(exc).__name__}: {exc}"
            frame = _fetch_taiex_yahoo(day)
        written = store.upsert_prices(frame)
        source = str(frame.iloc[0]["source"])
        rows.append(
            {
                "trade_date": day,
                "status": "inserted",
                "rows": written,
                "source": source,
                "finmind_or_twse_error_before_fallback": finmind_or_twse_error,
            }
        )
        print(f"[{day}] inserted={written} source={source}", flush=True)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
