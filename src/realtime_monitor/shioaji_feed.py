"""Shioaji snapshots and streaming adapter with a delayed-price fallback."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import urllib.request
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from realtime_monitor.engine import LiveQuote
from realtime_monitor.oos_bridge import FundSeed

TAIPEI = ZoneInfo("Asia/Taipei")
TWSE_INDEX_URL = (
    "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?"
    "ex_ch=tse_t00.tw&json=1&delay=0"
)


class DatabaseQuoteFeed:
    """Development fallback: publish the latest stored nominal closes once."""

    def __init__(self, database_path: str | Path, callback: Callable[[LiveQuote], None]) -> None:
        self.database_path = Path(database_path)
        self.callback = callback

    def start(self, symbols: list[str]) -> str:
        with sqlite3.connect(str(self.database_path)) as connection:
            for symbol in [*symbols, "TAIEX"]:
                row = connection.execute(
                    "SELECT trade_date, close FROM prices WHERE stock_id = ? AND close > 0 "
                    "ORDER BY trade_date DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
                if row:
                    amount = (
                        self._stored_market_amount(connection, str(row[0]))
                        if symbol == "TAIEX"
                        else None
                    )
                    self.callback(
                        LiveQuote(
                            code=symbol,
                            price=float(row[1]),
                            timestamp=f"{row[0]}T13:30:00+08:00",
                            source="database-delayed",
                            amount=amount,
                        )
                    )
        return "Shioaji credentials unavailable; using latest stored closes"

    @staticmethod
    def _stored_market_amount(connection: sqlite3.Connection, trade_date: str) -> float | None:
        price_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(prices)").fetchall()
        }
        if "traded_value" not in price_columns:
            return None
        stock_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(stocks)").fetchall()
        }
        if "market" in stock_columns:
            row = connection.execute(
                "SELECT SUM(p.traded_value) FROM prices p "
                "JOIN stocks s ON s.stock_id = p.stock_id "
                "WHERE p.trade_date = ? AND s.market = 'TWSE'",
                (trade_date,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT SUM(traded_value) FROM prices "
                "WHERE trade_date = ? AND stock_id != 'TAIEX'",
                (trade_date,),
            ).fetchone()
        value = float(row[0]) if row and row[0] is not None else None
        return value if value is not None and value > 0 else None

    def reconfigure(self, symbols: list[str]) -> str:
        return self.start(symbols)

    def stop(self) -> None:
        return None


class TwseIndexQuoteFeed:
    """Official TWSE MIS fallback for the cash-market weighted index."""

    def __init__(self, callback: Callable[[LiveQuote], None]) -> None:
        self.callback = callback

    @staticmethod
    def _number(value):
        try:
            return float(str(value).replace(",", "")) if value not in {None, "-"} else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def quote_from_payload(cls, payload: dict) -> LiveQuote:
        rows = payload.get("msgArray") or ()
        if not rows:
            raise ValueError("TWSE MIS returned no TAIEX row")
        row = rows[0]
        day = str(row.get("d") or row.get("^") or "")
        clock = str(row.get("t") or row.get("%") or "")
        if len(day) != 8 or not clock:
            raise ValueError("TWSE MIS returned an invalid market timestamp")
        moment = datetime.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S").replace(
            tzinfo=TAIPEI
        )
        price = cls._number(row.get("z"))
        if price is None or price <= 0:
            raise ValueError("TWSE MIS returned an invalid TAIEX price")
        return LiveQuote(
            code="TAIEX",
            price=price,
            timestamp=moment.isoformat(),
            source="twse-mis",
            reference=cls._number(row.get("y")),
            open=cls._number(row.get("o")),
            high=cls._number(row.get("h")),
            low=cls._number(row.get("l")),
            volume=cls._number(row.get("m")),
        )

    def refresh(self) -> LiveQuote:
        request = urllib.request.Request(
            TWSE_INDEX_URL,
            headers={"User-Agent": "Mozilla/5.0 AI-Quant-Research/1.0"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        quote = self.quote_from_payload(payload)
        self.callback(quote)
        return quote


class ShioajiQuoteFeed:
    def __init__(self, callback: Callable[[LiveQuote], None], *, simulation: bool = True) -> None:
        self.callback = callback
        self.simulation = simulation
        self.api = None
        self._symbols: set[str] = set()
        self._stock_contracts: dict[str, object] = {}
        self._index_contract = None
        self._index_reference: float | None = None
        self._api_lock = threading.RLock()

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY"))

    @staticmethod
    def _number(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _emit(self, code: str, quote, source: str = "shioaji-stream") -> None:
        price = self._number(getattr(quote, "close", None))
        if not price or price <= 0:
            return
        timestamp = getattr(quote, "datetime", None) or datetime.now(timezone.utc)
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        bid_prices = getattr(quote, "bid_price", None) or ()
        ask_prices = getattr(quote, "ask_price", None) or ()
        reference = self._number(getattr(quote, "reference", None))
        if reference is None:
            change_price = self._number(getattr(quote, "change_price", None))
            if change_price is not None and price - change_price > 0:
                reference = price - change_price
        normalized_code = "TAIEX" if code == "IX0001" else str(code)
        if normalized_code == "TAIEX" and reference is not None:
            self._index_reference = reference
        self.callback(
            LiveQuote(
                code=normalized_code,
                price=price,
                timestamp=str(timestamp),
                source=source,
                reference=reference,
                open=self._number(getattr(quote, "open", None)),
                high=self._number(getattr(quote, "high", None)),
                low=self._number(getattr(quote, "low", None)),
                volume=self._number(
                    getattr(quote, "total_volume", None) or getattr(quote, "vol_sum", None)
                ),
                amount=self._number(
                    getattr(quote, "total_amount", None) or getattr(quote, "amount_sum", None)
                ),
                bid=self._number(bid_prices[0]) if bid_prices else None,
                ask=self._number(ask_prices[0]) if ask_prices else None,
            )
        )

    def start(self, symbols: list[str]) -> str:
        import shioaji as sj

        self.api = sj.Shioaji(simulation=self.simulation)
        self.api.login(os.environ["SHIOAJI_API_KEY"], os.environ["SHIOAJI_SECRET_KEY"])
        contracts = [self.api.Contracts.Stocks[symbol] for symbol in symbols]
        index_contract = self.api.Contracts.Indexs.TSE["IX0001"]
        self._stock_contracts = {str(contract.code): contract for contract in contracts}
        self._symbols = set(self._stock_contracts)
        self._index_contract = index_contract
        for snapshot in self.api.snapshots([*contracts, index_contract]):
            self._emit(str(snapshot.code), snapshot, "shioaji-snapshot")

        @self.api.on_quote_stk_v1()
        def on_stock(exchange, quote):  # noqa: ARG001
            self._emit(str(quote.code), quote)

        @self.api.on_quote_idx_v1()
        def on_index(exchange, quote):  # noqa: ARG001
            self._emit(str(quote.code), quote)

        for contract in contracts:
            self.api.quote.subscribe(contract, quote_type=sj.constant.QuoteType.Quote, version="v1")
        self.api.quote.subscribe(
            index_contract, quote_type=sj.constant.QuoteType.Quote, version="v1"
        )
        return f"Shioaji connected; subscribed to {len(contracts)} holdings and TAIEX"

    def reconfigure(self, symbols: list[str]) -> str:
        """Update subscriptions when a newly materialized monthly book appears."""
        if self.api is None:
            raise RuntimeError("Shioaji is not connected")
        import shioaji as sj

        wanted = set(symbols)
        for symbol in sorted(self._symbols - wanted):
            contract = self._stock_contracts.pop(symbol, None)
            if contract is not None:
                self.api.quote.unsubscribe(
                    contract, quote_type=sj.constant.QuoteType.Quote, version="v1"
                )
        new_contracts = []
        for symbol in sorted(wanted - self._symbols):
            contract = self.api.Contracts.Stocks[symbol]
            self._stock_contracts[symbol] = contract
            new_contracts.append(contract)
            self.api.quote.subscribe(contract, quote_type=sj.constant.QuoteType.Quote, version="v1")
        if new_contracts:
            for snapshot in self.api.snapshots(new_contracts):
                self._emit(str(snapshot.code), snapshot, "shioaji-snapshot")
        self._symbols = wanted
        return f"Shioaji connected; subscribed to {len(wanted)} holdings and TAIEX"

    def refresh_index_quote(self, session_date: str) -> LiveQuote | None:
        """Refresh TAIEX, using the current K bar when simulation snapshots are stale."""
        if self.api is None or self._index_contract is None:
            raise RuntimeError("Shioaji is not connected")
        bars = self._bars(self._index_contract, session_date)
        rows = self._minute_rows(bars, session_date)
        if not rows:
            return None
        moment, price = rows[-1]
        volumes = [self._number(value) or 0.0 for value in bars.get("Volume") or ()]
        amounts = [self._number(value) or 0.0 for value in bars.get("Amount") or ()]
        quote = LiveQuote(
            code="TAIEX",
            price=price,
            timestamp=moment.isoformat(),
            source="shioaji-index-kbar",
            reference=self._index_reference,
            open=self._number((bars.get("Open") or [None])[0]),
            high=max(
                (self._number(value) or 0.0 for value in bars.get("High") or ()),
                default=None,
            ),
            low=min(
                (
                    value
                    for value in (self._number(item) for item in bars.get("Low") or ())
                    if value
                ),
                default=None,
            ),
            volume=sum(volumes) or None,
            amount=sum(amounts) or None,
        )
        self.callback(quote)
        return quote

    @staticmethod
    def _bar_datetime(value: object) -> datetime:
        """Decode Shioaji's nanosecond timestamp as a Taiwan wall-clock value."""
        stamp = int(value) / 1_000_000_000
        return datetime.fromtimestamp(stamp, tz=timezone.utc).replace(tzinfo=TAIPEI)

    def _bars(self, contract: object, session_date: str) -> dict:
        if self.api is None:
            raise RuntimeError("Shioaji is not connected")
        with self._api_lock:
            result = self.api.kbars(contract, start=session_date, end=session_date)
        return result.dict() if hasattr(result, "dict") else dict(result)

    @classmethod
    def _minute_rows(cls, bars: dict, session_date: str) -> list[tuple[datetime, float]]:
        rows: list[tuple[datetime, float]] = []
        for stamp, close in zip(bars.get("ts") or (), bars.get("Close") or (), strict=False):
            moment = cls._bar_datetime(stamp)
            value = cls._number(close)
            if (
                moment.date().isoformat() == session_date
                and time(9) <= moment.time() <= time(13, 30)
                and value is not None
                and value > 0
            ):
                rows.append((moment.replace(second=0, microsecond=0), value))
        return sorted(rows)

    def load_intraday_history(
        self,
        seed: FundSeed,
        fallback_prices: dict[str, float],
        *,
        target_date: str | None = None,
        allow_previous: bool = True,
        lookback_days: int = 14,
    ) -> dict:
        """Rebuild one requested session, or the latest session when allowed."""
        if self.api is None or self._index_contract is None:
            raise RuntimeError("Shioaji is not connected")

        session_date = None
        index_bars: dict = {}
        index_rows: list[tuple[datetime, float]] = []
        requested = (
            datetime.fromisoformat(target_date).date()
            if target_date
            else datetime.now(TAIPEI).date()
        )
        attempts = max(1, lookback_days) if allow_previous else 1
        for offset in range(attempts):
            candidate = (requested - timedelta(days=offset)).isoformat()
            bars = self._bars(self._index_contract, candidate)
            rows = self._minute_rows(bars, candidate)
            if rows:
                session_date = candidate
                index_bars = bars
                index_rows = rows
                break
        if session_date is None:
            return {
                "session_date": None,
                "market_open": "09:00",
                "market_close": "13:30",
                "requested_date": target_date,
                "points": [],
            }

        session_open = datetime.combine(
            datetime.fromisoformat(session_date).date(), time(9), tzinfo=TAIPEI
        )
        index_open = self._number((index_bars.get("Open") or [None])[0])
        if index_open and index_rows[0][0] > session_open:
            index_rows.insert(0, (session_open, index_open))

        stock_rows: dict[str, list[tuple[datetime, float]]] = {}
        stock_opens: dict[str, float | None] = {}
        for holding in seed.holdings:
            contract = self._stock_contracts.get(holding.stock_id)
            bars = self._bars(contract, session_date) if contract is not None else {}
            stock_rows[holding.stock_id] = self._minute_rows(bars, session_date)
            stock_opens[holding.stock_id] = self._number((bars.get("Open") or [None])[0])

        positions: dict[str, int] = {holding.stock_id: 0 for holding in seed.holdings}
        last_prices: dict[str, float] = {
            holding.stock_id: float(
                stock_opens[holding.stock_id]
                or (
                    fallback_prices.get(holding.stock_id, holding.anchor_price)
                    if not stock_rows[holding.stock_id]
                    else stock_rows[holding.stock_id][0][1]
                )
            )
            for holding in seed.holdings
        }
        points: list[dict] = []
        for moment, taiex in index_rows:
            for holding in seed.holdings:
                rows = stock_rows[holding.stock_id]
                cursor = positions[holding.stock_id]
                while cursor < len(rows) and rows[cursor][0] <= moment:
                    last_prices[holding.stock_id] = rows[cursor][1]
                    cursor += 1
                positions[holding.stock_id] = cursor
            nav = seed.anchor_cash + sum(
                holding.shares * last_prices[holding.stock_id] for holding in seed.holdings
            )
            points.append(
                {
                    "session_date": session_date,
                    "time": moment.isoformat(),
                    "nav": nav,
                    "taiex": taiex,
                    "quote_revision": moment.isoformat(),
                }
            )
        return {
            "session_date": session_date,
            "market_open": "09:00",
            "market_close": "13:30",
            "requested_date": target_date,
            "points": points,
        }

    def stop(self) -> None:
        if self.api is not None:
            try:
                self.api.logout()
            finally:
                self.api = None
                self._symbols.clear()
                self._stock_contracts.clear()
