"""Repair Taiwanese Saturday makeup trading sessions from FinMind.

The script defaults to a read-only API preflight. Pass ``--apply`` only after
reviewing the JSON report. It fetches all feeds for a date before changing the
database, writes a compressed preimage of affected rows, and commits each date
as a single SQLite transaction.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import create_engine_from_settings, session_scope  # noqa: E402
from integrations.finmind import _get  # noqa: E402
from integrations.finmind_fundamentals import fetch_institutional  # noqa: E402
from integrations.finmind_market_value import fetch_market_value_day  # noqa: E402
from integrations.finmind_prices import fetch_price_adj, fetch_prices  # noqa: E402
from models.security import Stock  # noqa: E402
from repositories import fundamentals as fundamentals_repo  # noqa: E402
from repositories import market_values as market_values_repo  # noqa: E402
from repositories import prices as prices_repo  # noqa: E402
from runtime.db_store import TAIEX_ID  # noqa: E402
from runtime.dotenv import load_dotenv  # noqa: E402
from runtime.panel_data import default_shares_path  # noqa: E402
from settings import get_finmind_token, load_settings  # noqa: E402

SATURDAY_SESSIONS = (
    "2016-01-30",
    "2016-06-04",
    "2016-09-10",
    "2017-02-18",
    "2017-06-03",
    "2017-09-30",
    "2018-03-31",
    "2018-12-22",
)
RETRY_COUNT = 5
REQUEST_DELAY_SECONDS = 0.6


def _retry(label, fetch):
    for attempt in range(RETRY_COUNT + 1):
        try:
            result = fetch()
            time.sleep(REQUEST_DELAY_SECONDS)
            return result
        except (PermissionError, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001 - retry network and provider 5xx/429.
            message = str(exc)
            if "returned no market values" in message:
                raise
            if "HTTP 401" in message or "HTTP 403" in message:
                raise RuntimeError(f"{label} access denied; stopping without retry") from exc
            if attempt == RETRY_COUNT:
                raise RuntimeError(
                    f"{label} failed after retries: {type(exc).__name__}: {exc}"
                ) from exc
            retry_after = getattr(exc, "retry_after", None)
            delay = min(60.0, float(retry_after) if retry_after else 1.5 * (2**attempt))
            print(
                f"{label}: temporary error; retry {attempt + 1}/{RETRY_COUNT} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("retry loop exhausted")


def _fetch_taiex(day: str, token: str) -> pd.DataFrame:
    source = "finmind"
    try:
        rows = _get("TaiwanVariousIndices", day, day, token)
        raw = pd.DataFrame(rows)
        required = {"stock_id", "date", "open", "high", "low", "close"}
        if raw.empty or not required.issubset(raw.columns):
            raise ValueError("empty or unexpected FinMind index schema")
        raw = raw.loc[raw["stock_id"].astype(str).eq("TAIEX")]
        if len(raw) != 1 or str(raw.iloc[0]["date"]) != day:
            raise ValueError(f"expected exactly one TAIEX row; got {len(raw)}")
        values = {
            key: pd.to_numeric(raw.iloc[0][key], errors="coerce")
            for key in ("open", "high", "low", "close")
        }
    except Exception:  # noqa: BLE001 - makeup-session index rows are absent from FinMind.
        response = requests.get(
            "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
            params={"date": day.replace("-", ""), "response": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        fields = payload.get("fields", [])
        rows = payload.get("data", [])
        if payload.get("stat") != "OK" or len(fields) < 5:
            raise ValueError(
                f"TWSE historical TAIEX endpoint returned invalid schema for {day}"
            ) from None
        roc_day = f"{int(day[:4]) - 1911:03d}/{day[5:7]}/{day[8:10]}"
        matches = [row for row in rows if len(row) >= 5 and str(row[0]).strip() == roc_day]
        if len(matches) != 1:
            raise ValueError(
                f"TWSE historical TAIEX endpoint has {len(matches)} rows for {day}"
            ) from None
        values = {
            key: pd.to_numeric(str(matches[0][index]).replace(",", ""), errors="coerce")
            for index, key in enumerate(("open", "high", "low", "close"), start=1)
        }
        source = "TWSE:TAIEX historical monthly OHLC"
    if (
        any(pd.isna(value) or value <= 0 for value in values.values())
        or values["high"] < values["low"]
    ):
        raise ValueError(f"invalid TAIEX OHLC values on {day}")
    return pd.DataFrame(
        [
            {
                "stock_id": TAIEX_ID,
                "trade_date": day,
                **values,
                "volume": 0.0,
                "traded_value": 0.0,
                "source": source,
            }
        ]
    )


def _fetch_tpex_market_values(day: str, known: set[str]) -> pd.DataFrame:
    """Fetch TPEx's date-specific official market-value table."""
    roc_day = f"{int(day[:4]) - 1911:03d}/{day[5:7]}/{day[8:10]}"
    response = _retry(
        f"{day} TPEx official market-value list",
        lambda: requests.get(
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyMarktVal",
            params={"date": roc_day, "response": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        ),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("date") != day.replace("-", "") or not payload.get("tables"):
        raise ValueError(f"TPEx official market-value response date mismatch for {day}")
    table = payload["tables"][0]
    fields = table.get("fields", [])
    if not {"股票代號", "市值(佰萬元)"}.issubset(set(fields)):
        raise ValueError(f"TPEx market-value schema changed for {day}")
    code_index, value_index = fields.index("股票代號"), fields.index("市值(佰萬元)")
    records = []
    for row in table.get("data", []):
        stock_id = str(row[code_index]).strip()
        market_value = pd.to_numeric(str(row[value_index]).replace(",", ""), errors="coerce")
        if stock_id in known and pd.notna(market_value) and market_value > 0:
            records.append(
                {
                    "trade_date": day,
                    "stock_id": stock_id,
                    "market_value": float(market_value) * 1_000_000,
                }
            )
    frame = pd.DataFrame(records, columns=["trade_date", "stock_id", "market_value"])
    if frame.empty or frame["stock_id"].duplicated().any():
        raise ValueError(f"TPEx returned an empty or duplicate market-value list for {day}")
    frame.attrs["market_value_source"] = "TPEx official daily market-value list"
    return frame


def _fetch_market_values(
    engine,
    day: str,
    token: str,
    known: set[str],
    eligible_ids: set[str],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer direct values, official TPEx values, then exact-day shares x close."""
    parts: list[pd.DataFrame] = []
    try:
        result = _retry(f"{day} TaiwanStockMarketValue", lambda: fetch_market_value_day(day, token))
        result = _only_known(result, known)
        if not result.empty:
            result.attrs["market_value_source"] = "FinMind:TaiwanStockMarketValue"
            parts.append(result)
    except Exception as direct_error:  # noqa: BLE001 - apply the approved same-day fallback.
        message = str(direct_error)
        if "returned no market values" not in message:
            raise

    # TPEx provides a historical daily list with exact values and issued shares.
    tpex = _fetch_tpex_market_values(day, known)
    parts.append(tpex)

    rows = _retry(
        f"{day} TaiwanStockShareholding issued shares",
        lambda: _get("TaiwanStockShareholding", day, day, token),
    )
    frame = pd.DataFrame(rows)
    required = {"date", "stock_id", "NumberOfSharesIssued"}
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise ValueError(
            f"same-day issued-share fallback unavailable for {day}; missing schema fields {missing}"
        )
    frame = frame.loc[:, ["date", "stock_id", "NumberOfSharesIssued"]].copy()
    frame = frame.rename(columns={"date": "trade_date", "NumberOfSharesIssued": "issued_shares"})
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["issued_shares"] = pd.to_numeric(frame["issued_shares"], errors="coerce")
    frame = frame.loc[
        frame["trade_date"].eq(day) & frame["stock_id"].isin(known) & frame["issued_shares"].gt(0)
    ]
    if frame["stock_id"].duplicated().any():
        raise ValueError(f"same-day issued-share fallback has duplicate stock IDs for {day}")
    # Include both the current FinMind price payload and pre-existing raw rows;
    # some days have legacy yfinance rows which are absent from the feed.
    with engine.connect() as connection:
        old_prices = pd.read_sql_query(
            text(
                "SELECT stock_id, close FROM prices WHERE trade_date = :day AND stock_id <> :index"
            ),
            connection,
            params={"day": day, "index": TAIEX_ID},
        )
    all_prices = pd.concat(
        [old_prices, prices.loc[:, ["stock_id", "close"]]], ignore_index=True
    ).drop_duplicates("stock_id", keep="last")
    all_prices["stock_id"] = all_prices["stock_id"].astype(str)
    all_prices["close"] = pd.to_numeric(all_prices["close"], errors="coerce")
    with engine.connect() as connection:
        formal_ids = set(
            connection.execute(text("SELECT stock_id FROM stocks WHERE market IN ('TWSE', 'TPEX')"))
            .scalars()
            .all()
        )
    expected = (
        set(all_prices.loc[all_prices["close"].gt(0), "stock_id"]) & formal_ids & eligible_ids
    )
    joined = frame.merge(all_prices, on="stock_id", how="inner", validate="one_to_one")
    derived = joined.assign(market_value=joined["issued_shares"] * joined["close"], trade_date=day)[
        ["trade_date", "stock_id", "market_value"]
    ]
    if not derived.empty:
        derived.attrs["market_value_source"] = (
            "FinMind:TaiwanStockShareholding NumberOfSharesIssued x same-day nominal close"
        )
        parts.append(derived)
    combined = pd.concat(parts, ignore_index=True).drop_duplicates("stock_id", keep="first")
    missing_caps = expected - set(combined["stock_id"].astype(str))
    previous_fallback_count = 0
    if missing_caps:
        today_prices = pd.concat(
            [old_prices, prices.loc[:, ["stock_id", "close"]]], ignore_index=True
        ).drop_duplicates("stock_id", keep="last")
        today_prices["stock_id"] = today_prices["stock_id"].astype(str)
        today_prices["close"] = pd.to_numeric(today_prices["close"], errors="coerce")
        # Local historical market values have gaps too, so obtain the broad
        # market's previous-session snapshot before applying the approved
        # previous MV / previous nominal close x current nominal close formula.
        with engine.connect() as connection:
            previous_day = connection.execute(
                text(
                    "SELECT MAX(trade_date) FROM prices "
                    "WHERE stock_id <> :index AND trade_date < :day"
                ),
                {"index": TAIEX_ID, "day": day},
            ).scalar_one()
            prior_closes = (
                pd.read_sql_query(
                    text(
                        "SELECT stock_id, close AS previous_close FROM prices "
                        "WHERE trade_date = :previous_day AND stock_id <> :index"
                    ),
                    connection,
                    params={"previous_day": previous_day, "index": TAIEX_ID},
                )
                if previous_day
                else pd.DataFrame(columns=["stock_id", "previous_close"])
            )
            prior_db_caps = (
                pd.read_sql_query(
                    text(
                        "SELECT stock_id, market_value AS previous_market_value "
                        "FROM market_values WHERE trade_date = :previous_day"
                    ),
                    connection,
                    params={"previous_day": previous_day},
                )
                if previous_day
                else pd.DataFrame(columns=["stock_id", "previous_market_value"])
            )
        previous_caps = prior_closes.merge(prior_db_caps, on="stock_id", how="left")
        if previous_day:
            prior_feed = _retry(
                f"{previous_day} TaiwanStockMarketValue fallback",
                lambda: fetch_market_value_day(previous_day, token),
            )
            if not prior_feed.empty:
                prior_feed = _only_known(prior_feed, known).rename(
                    columns={"market_value": "feed_market_value"}
                )
                previous_caps = previous_caps.merge(
                    prior_feed.loc[:, ["stock_id", "feed_market_value"]],
                    on="stock_id",
                    how="left",
                    validate="one_to_one",
                )
                previous_caps["previous_market_value"] = previous_caps[
                    "previous_market_value"
                ].fillna(previous_caps["feed_market_value"])
                previous_caps = previous_caps.drop(columns="feed_market_value")
            prior_tpex = _fetch_tpex_market_values(previous_day, known).rename(
                columns={"market_value": "tpex_market_value"}
            )
            previous_caps = previous_caps.merge(
                prior_tpex.loc[:, ["stock_id", "tpex_market_value"]],
                on="stock_id",
                how="left",
                validate="one_to_one",
            )
            previous_caps["previous_market_value"] = previous_caps[
                "previous_market_value"
            ].fillna(previous_caps["tpex_market_value"])
        previous_caps = previous_caps.loc[
            previous_caps["stock_id"].astype(str).isin(missing_caps)
        ].copy()
        if not previous_caps.empty:
            previous_caps["stock_id"] = previous_caps["stock_id"].astype(str)
            previous_caps["previous_close"] = pd.to_numeric(
                previous_caps["previous_close"], errors="coerce"
            )
            previous_caps["previous_market_value"] = pd.to_numeric(
                previous_caps["previous_market_value"], errors="coerce"
            )
            prices_for_fallback = today_prices.loc[
                today_prices["stock_id"].isin(missing_caps) & today_prices["close"].gt(0)
            ].rename(columns={"close": "current_close"})
            inferred = previous_caps.merge(
                prices_for_fallback, on="stock_id", how="inner", validate="one_to_one"
            )
            inferred = inferred.loc[
                inferred["previous_close"].gt(0) & inferred["previous_market_value"].gt(0)
            ].assign(
                market_value=lambda rows: (
                    (rows["previous_market_value"] / rows["previous_close"]) * rows["current_close"]
                ),
                trade_date=day,
            )[["trade_date", "stock_id", "market_value"]]
            if not inferred.empty:
                previous_fallback_count = len(inferred)
                inferred.attrs["market_value_source"] = (
                    f"previous-day market value / nominal close x same-day close ({previous_day})"
                )
                parts.append(inferred)
                combined = pd.concat(parts, ignore_index=True).drop_duplicates(
                    "stock_id", keep="first"
                )
    else:
        previous_day = None
    covered = set(combined["stock_id"].astype(str))
    coverage = len(covered & expected) / len(expected) if expected else 0.0
    missing_ids = sorted(expected - covered)
    with engine.connect() as connection:
        missing_markets = pd.read_sql_query(
            select(Stock.stock_id, Stock.market).where(Stock.stock_id.in_(missing_ids)),
            connection,
        )
    missing_by_market = missing_markets["market"].value_counts().to_dict()
    result = combined.loc[combined["market_value"].gt(0)].reset_index(drop=True)
    sources = sorted(
        {part.attrs.get("market_value_source", "FinMind market-value data") for part in parts}
    )
    result.attrs["source_content_hash"] = market_values_repo.canonical_day_hash(result)
    result.attrs["source_row_count"] = len(result)
    result.attrs["market_value_source"] = "; ".join(sources)
    result.attrs["reconstruction_coverage"] = coverage
    result.attrs["previous_market_value_fallback_count"] = previous_fallback_count
    result.attrs["previous_market_value_fallback_day"] = previous_day
    result.attrs["missing_market_value_count"] = len(missing_ids)
    result.attrs["missing_market_value_by_market"] = missing_by_market
    result.attrs["missing_market_value_examples"] = missing_ids[:50]
    return result


def _only_known(frame: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["stock_id"] = result["stock_id"].astype(str)
    return result.loc[result["stock_id"].isin(known)].reset_index(drop=True)


def _validate_feed(name: str, frame: pd.DataFrame, day: str, *, allow_empty: bool = False) -> None:
    if frame.empty and not allow_empty:
        raise ValueError(f"{name} returned no rows for known securities on trading day {day}")
    if not frame.empty:
        if frame["trade_date"].astype(str).ne(day).any():
            raise ValueError(f"{name} returned unexpected trade dates for {day}")
        if frame["stock_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} returned duplicate stock IDs for {day}")


def _fetch_day(
    engine, day: str, token: str, known: set[str], eligible_ids: set[str]
) -> dict[str, pd.DataFrame]:
    prices = _only_known(
        _retry(f"{day} TaiwanStockPrice", lambda: fetch_prices(day, day, token)), known
    )
    feeds = {
        "prices": prices,
        "price_adj": _only_known(
            _retry(f"{day} TaiwanStockPriceAdj", lambda: fetch_price_adj(day, day, token)), known
        ),
        "institutional": _only_known(
            _retry(
                f"{day} institutional and margin",
                lambda: fetch_institutional(day, day, token, include_floats=False),
            ),
            known,
        ),
        "market_value": _fetch_market_values(engine, day, token, known, eligible_ids, prices),
        "taiex": _retry(f"{day} TaiwanVariousIndices", lambda: _fetch_taiex(day, token)),
    }
    for name in ("prices", "price_adj", "institutional", "market_value", "taiex"):
        _validate_feed(name, feeds[name], day)
    raw_keys = set(map(tuple, feeds["prices"][["stock_id", "trade_date"]].astype(str).to_numpy()))
    adj_keys = set(
        map(tuple, feeds["price_adj"][["stock_id", "trade_date"]].astype(str).to_numpy())
    )
    if adj_keys - raw_keys:
        feeds["price_adj"] = (
            feeds["price_adj"]
            .loc[
                feeds["price_adj"].apply(
                    lambda row: (str(row.stock_id), str(row.trade_date)) in raw_keys, axis=1
                )
            ]
            .reset_index(drop=True)
        )
    return feeds


def _existing_ids(engine) -> set[str]:
    with engine.connect() as connection:
        return set(connection.execute(text("SELECT stock_id FROM stocks")).scalars().all())


def _existing_rows(engine, table: str, day: str) -> list[dict]:
    with engine.connect() as connection:
        result = connection.execute(
            text(f"SELECT * FROM {table} WHERE trade_date = :day"), {"day": day}
        )
        return [dict(row._mapping) for row in result]


def _preimage(engine, day: str) -> dict:
    return {
        "trade_date": day,
        "prices": _existing_rows(engine, "prices", day),
        "institutional": _existing_rows(engine, "institutional", day),
        "market_values": _existing_rows(engine, "market_values", day),
        "market_value_sync_days": _existing_rows(engine, "market_value_sync_days", day),
    }


def _changed_rows(
    engine, table: str, frame: pd.DataFrame, key_columns: tuple[str, ...]
) -> pd.DataFrame:
    if frame.empty:
        return frame
    with engine.connect() as connection:
        old = pd.read_sql_query(
            text(f"SELECT * FROM {table} WHERE trade_date = :day"),
            connection,
            params={"day": str(frame.iloc[0]["trade_date"])},
        )
    if old.empty:
        return frame
    old = old.set_index(list(key_columns))
    changed = []
    for row in frame.to_dict(orient="records"):
        key = tuple(row[column] for column in key_columns)
        if key not in old.index:
            changed.append(row)
            continue
        prior = old.loc[key]
        if isinstance(prior, pd.DataFrame):
            prior = prior.iloc[0]
        if any(
            not _same_value(row.get(col), prior.get(col))
            for col in frame.columns
            if col not in key_columns
        ):
            changed.append(row)
    return pd.DataFrame(changed, columns=frame.columns)


def _same_value(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _apply_day(engine, day: str, feeds: dict[str, pd.DataFrame]) -> dict[str, int | str]:
    day_date = date.fromisoformat(day)
    prices = feeds["prices"]
    adj = feeds["price_adj"]
    institutional = feeds["institutional"]
    changed_prices = _changed_rows(engine, "prices", prices, ("trade_date", "stock_id"))
    changed_adj = _changed_rows(engine, "prices", adj, ("trade_date", "stock_id"))
    changed_inst = _changed_rows(engine, "institutional", institutional, ("trade_date", "stock_id"))
    with session_scope(engine) as session:
        price_count = (
            prices_repo.upsert_prices(session, changed_prices) if not changed_prices.empty else 0
        )
        # Raw and adjusted upserts share the same per-day transaction.
        adj_count = (
            prices_repo.upsert_price_adj(session, changed_adj) if not changed_adj.empty else 0
        )
        inst_count = (
            fundamentals_repo.upsert_institutional(session, changed_inst)
            if not changed_inst.empty
            else 0
        )
        mv = feeds["market_value"]
        source_hash = mv.attrs.get("source_content_hash")
        if mv.attrs.get("reconstruction_coverage", 0) >= 0.99:
            mv_count = market_values_repo.upsert_market_value_day(
                session,
                day_date,
                mv,
                source_content_hash=source_hash,
                source=mv.attrs.get("market_value_source", "FinMind:TaiwanStockMarketValue"),
            )
            mv_status = "succeeded"
        else:
            mv_count = market_values_repo.upsert_partial_market_value_day(
                session,
                day_date,
                mv,
                source=mv.attrs.get("market_value_source", "partial verified market values"),
            )
            mv_status = "failed (partial rows retained; not eligible for PIT panels)"
    return {
        "prices_changed": price_count,
        "price_adj_changed": adj_count,
        "institutional_changed": inst_count,
        "market_value_rows": mv_count,
        "market_value_sync_status": mv_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default=SATURDAY_SESSIONS[0])
    parser.add_argument("--end", default=SATURDAY_SESSIONS[-1])
    parser.add_argument(
        "--apply", action="store_true", help="write validated feeds to the database"
    )
    parser.add_argument("--report", default="reports/saturday-session-repair.json")
    args = parser.parse_args(argv)
    try:
        start, end = (
            date.fromisoformat(args.start).isoformat(),
            date.fromisoformat(args.end).isoformat(),
        )
        if start > end:
            raise ValueError("start must be on or before end")
        days = [day for day in SATURDAY_SESSIONS if start <= day <= end]
        if not days:
            raise ValueError("requested range contains no known Saturday trading sessions")
        load_dotenv()
        settings = load_settings(args.config)
        shares_path = default_shares_path(settings.data.database_url)
        if shares_path is None or not shares_path.is_file():
            raise ValueError("database/shares.json is required to identify equity securities")
        shares_payload = json.loads(shares_path.read_text(encoding="utf-8"))
        if not isinstance(shares_payload, dict):
            raise ValueError("database/shares.json must contain a stock-ID object")
        eligible_ids = {str(stock_id) for stock_id in shares_payload}
    except Exception as exc:  # noqa: BLE001
        print(f"configuration error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    token = get_finmind_token(settings)
    if not token:
        print("missing FinMind token", file=sys.stderr)
        return 1
    engine = create_engine_from_settings(settings)
    report = {
        "mode": "apply" if args.apply else "preflight",
        "dates": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        known = _existing_ids(engine)
        if TAIEX_ID not in known:
            raise RuntimeError("TAIEX security row is missing from stocks table")
        prepared: list[tuple[str, dict[str, pd.DataFrame], dict]] = []
        for day in days:
            print(f"Fetching and validating feeds for {day}", flush=True)
            feeds = _fetch_day(engine, day, token, known, eligible_ids)
            date_report = {
                "trade_date": day,
                "counts": {key: len(frame) for key, frame in feeds.items()},
                "market_value_source": feeds["market_value"].attrs.get("market_value_source"),
                "market_value_coverage": feeds["market_value"].attrs.get("reconstruction_coverage"),
                "previous_market_value_fallback_count": feeds["market_value"].attrs.get(
                    "previous_market_value_fallback_count", 0
                ),
                "previous_market_value_fallback_day": feeds["market_value"].attrs.get(
                    "previous_market_value_fallback_day"
                ),
                "missing_market_value_count": feeds["market_value"].attrs.get(
                    "missing_market_value_count", 0
                ),
                "missing_market_value_by_market": feeds["market_value"].attrs.get(
                    "missing_market_value_by_market", {}
                ),
                "missing_market_value_examples": feeds["market_value"].attrs.get(
                    "missing_market_value_examples", []
                ),
            }
            date_report["existing_counts"] = {
                table: len(_existing_rows(engine, table, day))
                for table in ("prices", "institutional", "market_values", "market_value_sync_days")
            }
            date_report["new_or_changed_price_ids"] = sorted(
                set(feeds["prices"].stock_id.astype(str))
                - {str(row["stock_id"]) for row in _existing_rows(engine, "prices", day)}
            )
            report["dates"].append(date_report)
            prepared.append((day, feeds, date_report))
        if args.apply:
            backup_dir = (
                Path("reports")
                / "saturday_session_preimage"
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            backup_dir.mkdir(parents=True, exist_ok=True)
            for day, _feeds, date_report in prepared:
                backup_path = backup_dir / f"{day}.json.gz"
                with gzip.open(backup_path, "wt", encoding="utf-8") as backup:
                    json.dump(_preimage(engine, day), backup, ensure_ascii=False, default=str)
                date_report["preimage"] = str(backup_path)
            for day, feeds, date_report in prepared:
                date_report["written"] = _apply_day(engine, day, feeds)
                print(f"[{day}] committed {date_report['written']}", flush=True)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report saved: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"repair stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
