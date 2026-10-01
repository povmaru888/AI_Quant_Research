"""Bounded, panel-only bulk data access for monthly panel construction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd

from contracts import UniverseSnapshot
from runtime.db_store import default_shares_path
from services.feature_service import (
    allows_partial_pit_market_values,
    is_factor_v4_feature_version,
    uses_pit_market_values,
)
from services.universe_service import build_universe
from settings import Settings

PANEL_FORMAT_VERSION = 2
PANEL_BUILDER_VERSION = "batch-index-v4-eligibility-r3"
_TAIEX_ID = "TAIEX"
_PRICE_COLUMNS = (
    "stock_id",
    "trade_date",
    "close",
    "high_adj",
    "close_adj",
    "volume",
    "traded_value",
)
_INSTITUTIONAL_COLUMNS = (
    "stock_id",
    "trade_date",
    "foreign_net_buy",
    "trust_net_buy",
    "margin_balance",
    "short_balance",
    "float_shares",
)
_FINANCIAL_COLUMNS = (
    "stock_id",
    "report_period",
    "announcement_date",
    "available_date",
    "revenue",
    "net_income",
    "equity",
    "assets",
    "operating_income",
    "operating_cash_flow",
)


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_fingerprint(settings: Settings) -> str:
    """Fingerprint all panel semantics and the dedicated builder revision."""
    payload = {
        "panel_format_version": PANEL_FORMAT_VERSION,
        "panel_builder_version": PANEL_BUILDER_VERSION,
        "universe": asdict(settings.universe),
        "features": asdict(settings.features),
        "label": asdict(settings.label),
    }
    return canonical_json_hash(payload)


def database_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("panel builder requires a file-backed SQLite database")
    raw = unquote(database_url[len(prefix) :])
    if not raw or raw == ":memory:":
        raise ValueError("panel builder requires a file-backed SQLite database")
    return Path(raw).resolve()


def _stat(path: Path) -> dict[str, int] | None:
    try:
        item = path.stat()
    except OSError:
        return None
    return {"size": item.st_size, "mtime_ns": item.st_mtime_ns}


def source_fingerprint(settings: Settings) -> str:
    """Return a cheap, exact source watermark without hashing the large DB."""
    path = database_path(settings.data.database_url)
    shares_path = default_shares_path(settings.data.database_url)
    if shares_path and shares_path.is_file():
        shares_hash = hashlib.sha256(shares_path.read_bytes()).hexdigest()
    else:
        shares_hash = hashlib.sha256(b"missing-shares-file").hexdigest()
    db_uri = "file:" + quote(path.as_posix(), safe="/:\\") + "?mode=ro"
    with sqlite3.connect(db_uri, uri=True, timeout=30.0) as conn:
        try:
            revision = int(
                conn.execute("SELECT revision FROM source_revision WHERE singleton=1").fetchone()[0]
            )
        except sqlite3.Error as exc:
            raise RuntimeError("database migration 004 is required; run tools/init_db.py") from exc
    return canonical_json_hash(
        {
            "database": _stat(path),
            "wal": _stat(Path(str(path) + "-wal")),
            "revision": revision,
            "shares_sha256": shares_hash,
        }
    )


def canonical_panel_hash(panel: dict) -> str:
    """Hash semantic panel content independently of pickle byte layout."""
    digest = hashlib.sha256()
    for key in ("signal_date", "universe", "feature_columns", "feature_version"):
        digest.update(
            json.dumps(panel.get(key), ensure_ascii=False, sort_keys=True, default=str).encode()
        )
        digest.update(b"\0")
    if "feature_coverage" in panel:
        digest.update(
            json.dumps(panel["feature_coverage"], ensure_ascii=False, sort_keys=True).encode()
        )
        digest.update(b"\0")
    if "eligibility" in panel:
        digest.update(json.dumps(panel["eligibility"], ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\0")
    for name in ("frame", "labels"):
        frame = panel[name]
        digest.update(name.encode())
        if isinstance(frame, pd.Series):
            metadata = {
                "name": frame.name,
                "dtype": str(frame.dtype),
                "index_name": frame.index.name,
                "index_dtype": str(frame.index.dtype),
            }
        else:
            metadata = {
                "columns": list(frame.columns),
                "dtypes": [str(value) for value in frame.dtypes],
                "index_name": frame.index.name,
                "index_dtype": str(frame.index.dtype),
            }
        digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode())
        digest.update(
            pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy().tobytes()
        )
    return digest.hexdigest()


class PreparedPanelData:
    """One-snapshot, bounded price arrays plus month-specific PIT slices."""

    def __init__(self, settings: Settings, start: str, end: str, run_id: str) -> None:
        self.settings = settings
        self.start = start
        self.end = end
        self.run_id = run_id
        self.path = database_path(settings.data.database_url)
        self.conn = self._connect()
        self.conn.execute("BEGIN")
        self.source_revision = self._read_revision()
        self.rows_loaded: dict[str, int] = {}
        self.price_by_stock: dict[str, dict[str, np.ndarray]] = {}
        self.stocks = self._load_stocks()
        self.shares = self._load_shares()
        self.financials = pd.DataFrame(columns=list(_FINANCIAL_COLUMNS))
        self.market_values = pd.DataFrame(columns=["trade_date", "stock_id", "market_value"])
        self.market_values_by_day: dict[str, dict[str, float]] = {}
        self.universe_by_month: dict[str, UniverseSnapshot] = {}
        self.universe_stocks: dict[str, pd.DataFrame] = {}
        self.inst_by_month: dict[str, pd.DataFrame] = {}

    def _connect(self) -> sqlite3.Connection:
        uri = "file:" + quote(self.path.as_posix(), safe="/:\\") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
        conn.row_factory = None
        return conn

    def _read_revision(self) -> int:
        try:
            row = self.conn.execute(
                "SELECT revision FROM source_revision WHERE singleton=1"
            ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("database migration 004 is required; run tools/init_db.py") from exc
        if row is None:
            raise RuntimeError("database source_revision row is missing; run tools/init_db.py")
        return int(row[0])

    def close(self) -> None:
        self.conn.close()

    def month_ends(self) -> list[str]:
        start_date = self.start + "-01"
        end_month = date.fromisoformat(self.end + "-01")
        if end_month.month == 12:
            next_month = date(end_month.year + 1, 1, 1)
        else:
            next_month = date(end_month.year, end_month.month + 1, 1)
        end_date = (next_month.fromordinal(next_month.toordinal() - 1)).isoformat()
        rows = self.conn.execute(
            "SELECT DISTINCT trade_date FROM prices "
            "WHERE stock_id != ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            (_TAIEX_ID, start_date, end_date),
        ).fetchall()
        by_month: dict[str, str] = {}
        for (trade_day,) in rows:
            month = trade_day[:7]
            if self.start <= month <= self.end:
                by_month[month] = trade_day
        return [by_month[month] for month in sorted(by_month)]

    def _load_stocks(self) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT stock_id, stock_name, market, listed_date, delisted_date, industry "
            "FROM stocks WHERE stock_id != ? ORDER BY stock_id",
            (_TAIEX_ID,),
        ).fetchall()
        self.rows_loaded["stocks"] = len(rows)
        frame = pd.DataFrame(
            rows,
            columns=[
                "stock_id",
                "stock_name",
                "market",
                "listed_date",
                "delisted_date",
                "industry",
            ],
        )
        frame["flags"] = ""
        return frame

    def _load_shares(self) -> dict[str, dict]:
        path = default_shares_path(self.settings.data.database_url)
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_prices(self, months: list[str]) -> None:
        if not months:
            return
        first_day, last_day = months[0], months[-1]
        horizon = self.settings.label.horizon_trading_days
        history_rows = 121
        calendar_end = last_day
        if is_factor_v4_feature_version(self.settings.features.feature_version):
            history_rows = max(
                history_rows,
                self.settings.features.required_adjusted_price_rows + 10,
            )
            # Future TAIEX dates are used only to validate that t+20 equity
            # rows are real sessions; feature slicing still stops at signal day.
            calendar_end = "9999-12-31"
        sql = f"""
        WITH before_ranked AS (
            SELECT stock_id, trade_date,
                   ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY trade_date DESC) AS rn
            FROM prices
            WHERE stock_id != ? AND trade_date < ?
        ),
        after_ranked AS (
            SELECT stock_id, trade_date,
                   ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY trade_date ASC) AS rn
            FROM prices
            WHERE stock_id != ? AND trade_date > ?
        ),
        selected AS (
            SELECT {", ".join(_PRICE_COLUMNS)} FROM prices
             WHERE stock_id != ? AND trade_date >= ? AND trade_date <= ?
            UNION ALL
            SELECT p.stock_id, p.trade_date, p.close, p.high_adj, p.close_adj,
                   p.volume, p.traded_value
              FROM prices AS p JOIN before_ranked AS r USING (stock_id, trade_date)
             WHERE r.rn <= ?
            UNION ALL
            SELECT p.stock_id, p.trade_date, p.close, p.high_adj, p.close_adj,
                   p.volume, p.traded_value
              FROM prices AS p JOIN after_ranked AS r USING (stock_id, trade_date)
             WHERE r.rn <= ?
            UNION ALL
            SELECT stock_id, trade_date, close, high_adj, close_adj, volume, traded_value
              FROM prices WHERE stock_id = ? AND trade_date <= ?
        )
        SELECT {", ".join(_PRICE_COLUMNS)} FROM selected ORDER BY stock_id, trade_date
        """
        params = (
            _TAIEX_ID,
            first_day,
            _TAIEX_ID,
            last_day,
            _TAIEX_ID,
            first_day,
            last_day,
            history_rows,
            horizon,
            _TAIEX_ID,
            calendar_end,
        )
        cursor = self.conn.execute(sql, params)
        current_id: str | None = None
        columns: list[list] = [[] for _ in _PRICE_COLUMNS]
        total = 0

        def flush_group() -> None:
            nonlocal columns, current_id
            if current_id is None:
                return
            dates = np.asarray(columns[1], dtype="U10")
            self.price_by_stock[current_id] = {
                "trade_date": dates,
                "close": pd.to_numeric(pd.Series(columns[2]), errors="coerce").to_numpy(
                    dtype=float
                ),
                "high_adj": pd.to_numeric(pd.Series(columns[3]), errors="coerce").to_numpy(
                    dtype=float
                ),
                "close_adj": pd.to_numeric(pd.Series(columns[4]), errors="coerce").to_numpy(
                    dtype=float
                ),
                "volume": pd.to_numeric(pd.Series(columns[5]), errors="coerce").to_numpy(
                    dtype=float
                ),
                "traded_value": pd.to_numeric(pd.Series(columns[6]), errors="coerce").to_numpy(
                    dtype=float
                ),
            }
            columns = [[] for _ in _PRICE_COLUMNS]

        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            for row in batch:
                stock_id = str(row[0])
                if current_id is not None and stock_id != current_id:
                    flush_group()
                current_id = stock_id
                for index, value in enumerate(row):
                    columns[index].append(value)
                total += 1
        flush_group()
        if is_factor_v4_feature_version(self.settings.features.feature_version):
            market = self.price_by_stock.get(_TAIEX_ID)
            if market is None or len(market["trade_date"]) == 0:
                raise RuntimeError("factor_v4 requires a non-empty TAIEX trading calendar")
            active_dates: set[str] = set()
            for stock_id, history in self.price_by_stock.items():
                if stock_id == _TAIEX_ID:
                    continue
                active_dates.update(
                    str(day)
                    for day, value in zip(
                        history["trade_date"], history["traded_value"], strict=True
                    )
                    if np.isfinite(value) and value > 0
                )
            calendar = market["trade_date"][
                np.isin(market["trade_date"], np.asarray(sorted(active_dates), dtype="U10"))
            ]
            for stock_id, history in self.price_by_stock.items():
                if stock_id == _TAIEX_ID:
                    continue
                keep = np.isin(history["trade_date"], calendar)
                self.price_by_stock[stock_id] = {
                    column: values[keep] for column, values in history.items()
                }
        self.rows_loaded["prices"] = total

    def _load_market_values(self, months: list[str]) -> None:
        if not uses_pit_market_values(self.settings.features.feature_version) or not months:
            return
        signals = [month_end for month_end in months]
        placeholders = ",".join("?" for _ in signals)
        usable_statuses = (
            "'succeeded', 'partial'"
            if allows_partial_pit_market_values(self.settings.features.feature_version)
            else "'succeeded'"
        )
        rows = self.conn.execute(
            "SELECT mv.trade_date, mv.stock_id, mv.market_value FROM market_values AS mv "
            "JOIN market_value_sync_days AS sync ON sync.trade_date = mv.trade_date "
            f"WHERE sync.status IN ({usable_statuses}) AND mv.trade_date IN ({placeholders}) "
            "ORDER BY mv.trade_date, mv.stock_id",
            signals,
        ).fetchall()
        self.market_values = pd.DataFrame(rows, columns=["trade_date", "stock_id", "market_value"])
        self.market_values_by_day = {
            day: group.set_index("stock_id")["market_value"].astype(float).to_dict()
            for day, group in self.market_values.groupby("trade_date", sort=False)
        }
        self.rows_loaded["market_values"] = len(rows)

    def _market_cap(self, stock_id: str, as_of: str) -> float | None:
        if uses_pit_market_values(self.settings.features.feature_version):
            value = self.market_values_by_day.get(as_of, {}).get(stock_id)
            if value is None:
                return None
            return value if np.isfinite(value) and value > 0 else None
        entry = self.shares.get(stock_id)
        if not isinstance(entry, dict):
            return None
        history = self.price_by_stock.get(stock_id)
        close = None
        if history is not None:
            pos = int(np.searchsorted(history["trade_date"], as_of, side="right")) - 1
            if pos >= 0:
                close = float(history["close"][pos])
        shares = entry.get("shares")
        if close is not None and isinstance(shares, (int, float)) and shares > 0:
            return close * float(shares)
        cached_cap = entry.get("market_cap")
        if isinstance(cached_cap, (int, float)) and cached_cap > 0:
            return float(cached_cap)
        return None

    def _stocks_as_of(self, signal_day: str) -> pd.DataFrame:
        active = (
            self.stocks.loc[
                self.stocks["listed_date"].isna() | self.stocks["listed_date"].le(signal_day)
            ]
            .loc[self.stocks["delisted_date"].isna() | self.stocks["delisted_date"].gt(signal_day)]
            .copy()
        )
        active["market_cap"] = [
            self._market_cap(str(stock_id), signal_day) for stock_id in active["stock_id"]
        ]
        return active.reset_index(drop=True)

    def _price_frame(self, stock_ids: list[str], signal_day: str, window: int) -> pd.DataFrame:
        frames = []
        for stock_id in stock_ids:
            history = self.price_by_stock.get(str(stock_id))
            if history is None:
                continue
            stop = int(np.searchsorted(history["trade_date"], signal_day, side="right"))
            if stop == 0:
                continue
            start = max(0, stop - window)
            indices = np.arange(start, stop)
            frames.append(self._frame_from_indices(str(stock_id), history, indices))
        if not frames:
            return pd.DataFrame(columns=list(_PRICE_COLUMNS))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _frame_from_indices(
        stock_id: str, history: dict[str, np.ndarray], indices: np.ndarray
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock_id": stock_id,
                "trade_date": history["trade_date"][indices],
                "close": history["close"][indices],
                "high_adj": history["high_adj"][indices],
                "close_adj": history["close_adj"][indices],
                "volume": history["volume"][indices],
                "traded_value": history["traded_value"][indices],
            }
        )

    def _universe_prices(self, signal_day: str) -> pd.DataFrame:
        rows = []
        for stock_id, history in self.price_by_stock.items():
            if stock_id == _TAIEX_ID:
                continue
            stop = int(np.searchsorted(history["trade_date"], signal_day, side="right"))
            if stop == 0 or history["trade_date"][stop - 1] != signal_day:
                continue
            start = max(0, stop - 20)
            rows.append(self._frame_from_indices(stock_id, history, np.arange(start, stop)))
        if not rows:
            return pd.DataFrame(columns=list(_PRICE_COLUMNS))
        return pd.concat(rows, ignore_index=True)

    def build_universes(self, months: list[str]) -> None:
        for signal_day in months:
            stocks = self._stocks_as_of(signal_day)
            universe = build_universe(
                self._universe_prices(signal_day),
                stocks,
                date.fromisoformat(signal_day),
                self.settings,
                self.run_id,
            )
            self.universe_by_month[signal_day] = universe
            self.universe_stocks[signal_day] = stocks

    def load_financials(self) -> None:
        rows = self.conn.execute(
            "SELECT stock_id, report_period, announcement_date, available_date, "
            "revenue, net_income, "
            "equity, assets, operating_income, operating_cash_flow FROM financials "
            "ORDER BY stock_id, available_date"
        ).fetchall()
        self.financials = pd.DataFrame(rows, columns=list(_FINANCIAL_COLUMNS))
        self.rows_loaded["financials"] = len(rows)

    def load_institutional(self, months: list[str]) -> None:
        stock_signals: dict[str, list[str]] = defaultdict(list)
        for signal_day, universe in self.universe_by_month.items():
            for stock_id in universe.included_ids:
                stock_signals[str(stock_id)].append(signal_day)
        stock_ids = sorted(stock_signals)
        rows_by_month: dict[str, list[tuple]] = {day: [] for day in months}
        total_rows = 0
        chunk_size = 350
        for offset in range(0, len(stock_ids), chunk_size):
            ids = stock_ids[offset : offset + chunk_size]
            placeholders = ",".join("?" for _ in ids)
            sql = (
                "SELECT stock_id, trade_date, foreign_net_buy, trust_net_buy, margin_balance, "
                "short_balance, float_shares FROM institutional "
                f"WHERE stock_id IN ({placeholders}) AND trade_date <= ? "
                "ORDER BY stock_id, trade_date"
            )
            cursor = self.conn.execute(sql, (*ids, months[-1]))
            current_id: str | None = None
            group: list[tuple] = []

            def flush_group() -> None:  # noqa: B023
                nonlocal group, current_id
                if current_id is None or not group:  # noqa: B023
                    group = []
                    return
                days = np.asarray([row[1] for row in group], dtype="U10")
                for signal_day in stock_signals.get(current_id, ()):  # noqa: B023
                    stop = int(np.searchsorted(days, signal_day, side="right"))
                    if stop:
                        rows_by_month[signal_day].extend(group[max(0, stop - 21) : stop])
                group = []

            while True:
                batch = cursor.fetchmany(20_000)
                if not batch:
                    break
                total_rows += len(batch)
                for row in batch:
                    stock_id = str(row[0])
                    if current_id is not None and stock_id != current_id:
                        flush_group()
                    current_id = stock_id
                    group.append(row)
            flush_group()
        self.inst_by_month = {
            signal_day: pd.DataFrame(rows, columns=list(_INSTITUTIONAL_COLUMNS))
            for signal_day, rows in rows_by_month.items()
        }
        self.rows_loaded["institutional_scanned"] = total_rows
        self.rows_loaded["institutional_retained"] = sum(
            len(rows) for rows in rows_by_month.values()
        )

    def prepare(self, months: list[str]) -> dict[str, float]:
        """Load bounded source frames and construct month-level lookups."""
        import time

        timings: dict[str, float] = {}
        started = time.perf_counter()
        self._load_prices(months)
        timings["prices"] = time.perf_counter() - started
        started = time.perf_counter()
        self._load_market_values(months)
        self.build_universes(months)
        timings["market_values_and_universes"] = time.perf_counter() - started
        started = time.perf_counter()
        self.load_financials()
        timings["financials"] = time.perf_counter() - started
        started = time.perf_counter()
        self.load_institutional(months)
        timings["institutional"] = time.perf_counter() - started
        return timings

    def month_inputs(self, signal_day: str) -> dict[str, pd.DataFrame | UniverseSnapshot]:
        universe = self.universe_by_month[signal_day]
        included = list(universe.included_ids)
        feature_prices = self._price_frame(included, signal_day, 121)
        taiex = self.price_by_stock.get(_TAIEX_ID)
        if taiex is not None:
            stop = int(np.searchsorted(taiex["trade_date"], signal_day, side="right"))
            if stop:
                indices = np.arange(stop)
                index_frame = self._frame_from_indices(_TAIEX_ID, taiex, indices)
                feature_prices = pd.concat([feature_prices, index_frame], ignore_index=True)
        day_prices = self._price_frame(included, signal_day, 1)
        label_frames = []
        horizon = self.settings.label.horizon_trading_days
        for stock_id in included:
            history = self.price_by_stock.get(str(stock_id))
            if history is None:
                continue
            start = int(np.searchsorted(history["trade_date"], signal_day, side="left"))
            stop = min(len(history["trade_date"]), start + horizon + 1)
            if start < stop and history["trade_date"][start] == signal_day:
                label_frames.append(
                    pd.DataFrame(
                        {
                            "stock_id": str(stock_id),
                            "trade_date": history["trade_date"][start:stop],
                            "close_adj": history["close_adj"][start:stop],
                        }
                    )
                )
        labels_prices = (
            pd.concat(label_frames, ignore_index=True)
            if label_frames
            else pd.DataFrame(columns=["stock_id", "trade_date", "close_adj"])
        )
        inst_history = self.inst_by_month.get(
            signal_day, pd.DataFrame(columns=list(_INSTITUTIONAL_COLUMNS))
        )
        market_values = self.market_values.loc[
            self.market_values["trade_date"].eq(signal_day)
        ].copy()
        return {
            "universe": universe,
            "stocks": self.universe_stocks[signal_day],
            "universe_prices": self._universe_prices(signal_day),
            "prices": pd.concat([day_prices, feature_prices], ignore_index=True).drop_duplicates(
                subset=["stock_id", "trade_date"], keep="last"
            ),
            "feature_prices": feature_prices,
            "labels_prices": labels_prices,
            "financials": self.financials,
            "institutional": inst_history,
            "market_values": market_values,
        }

    def labelable_ids(self, signal_day: str, stock_ids: list[str]) -> list[str]:
        horizon = self.settings.label.horizon_trading_days
        out = []
        for stock_id in stock_ids:
            history = self.price_by_stock.get(str(stock_id))
            if history is None:
                continue
            start = int(np.searchsorted(history["trade_date"], signal_day, side="left"))
            ahead = start + horizon
            if (
                start < len(history["trade_date"])
                and history["trade_date"][start] == signal_day
                and ahead < len(history["trade_date"])
            ):
                base = history["close_adj"][start]
                future = history["close_adj"][ahead]
                if np.isfinite(base) and np.isfinite(future) and base > 0 and future > 0:
                    out.append(str(stock_id))
        return out
