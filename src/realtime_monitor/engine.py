"""Thread-safe fund valuation and quote state."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from realtime_monitor.oos_bridge import FundSeed


@dataclass(frozen=True)
class LiveQuote:
    code: str
    price: float
    timestamp: str
    source: str
    reference: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    amount: float | None = None
    bid: float | None = None
    ask: float | None = None


class FundEngine:
    def __init__(self, seed: FundSeed) -> None:
        self.seed = seed
        self._quotes: dict[str, LiveQuote] = {}
        self._connected = False
        self._connection_message = "waiting for quote service"
        self._lock = threading.RLock()

    def set_connection(self, connected: bool, message: str) -> None:
        with self._lock:
            self._connected = connected
            self._connection_message = message

    def update_quote(self, quote: LiveQuote) -> None:
        if not math.isfinite(quote.price) or quote.price <= 0:
            return
        with self._lock:
            current = self._quotes.get(quote.code)
            if current is None or quote.timestamp >= current.timestamp:
                self._quotes[quote.code] = quote

    def replace_seed(self, seed: FundSeed) -> None:
        """Atomically roll the live book forward after a new materialized signal."""
        with self._lock:
            self.seed = seed
            keep = {holding.stock_id for holding in seed.holdings} | {"TAIEX", "IX0001"}
            self._quotes = {code: quote for code, quote in self._quotes.items() if code in keep}

    def _holding_rows(self) -> tuple[list[dict], float, int]:
        rows: list[dict] = []
        total_equity = 0.0
        live_count = 0
        for holding in self.seed.holdings:
            quote = self._quotes.get(holding.stock_id)
            price = quote.price if quote else holding.anchor_price
            market_value = holding.shares * price
            pnl = holding.shares * (price - holding.anchor_price)
            if quote:
                live_count += 1
            total_equity += market_value
            rows.append(
                {
                    **asdict(holding),
                    "last_price": price,
                    "quote_time": quote.timestamp if quote else holding.anchor_price_date,
                    "quote_source": quote.source if quote else "oos-anchor",
                    "previous_close": quote.reference if quote else None,
                    "bid": quote.bid if quote else None,
                    "ask": quote.ask if quote else None,
                    "day_change": (
                        price / quote.reference - 1
                        if quote and quote.reference and quote.reference > 0
                        else None
                    ),
                    "market_value": market_value,
                    "pnl_since_anchor": pnl,
                }
            )
        nav = self.seed.anchor_cash + total_equity
        for row in rows:
            row["live_weight"] = row["market_value"] / nav if nav > 0 else None
            row["nav_contribution"] = row["pnl_since_anchor"] / self.seed.anchor_nav
        return rows, nav, live_count

    def snapshot(self) -> dict:
        with self._lock:
            holdings, nav, live_count = self._holding_rows()
            now = datetime.now(timezone.utc).isoformat()
            taiex = self._quotes.get("TAIEX") or self._quotes.get("IX0001")
            quote_revision = max((quote.timestamp for quote in self._quotes.values()), default=None)
            taiex_above_ma60 = (
                taiex.price >= self.seed.taiex_ma60
                if taiex and self.seed.taiex_ma60 is not None
                else None
            )
            pnl = nav - self.seed.anchor_nav
            quality = (
                "live"
                if live_count == len(holdings) and self._connected
                else "partial"
                if live_count
                else "anchor"
            )
            return {
                "as_of": now,
                "run_id": self.seed.run_id,
                "signal_date": self.seed.signal_date,
                "execution_date": self.seed.execution_date,
                "anchor_date": self.seed.anchor_date,
                "initial_capital": self.seed.initial_capital,
                "anchor_nav": self.seed.anchor_nav,
                "nav": nav,
                "unit_nav": nav / (self.seed.initial_capital / 10.0),
                "cash": self.seed.anchor_cash,
                "equity_value": nav - self.seed.anchor_cash,
                "equity_exposure": (nav - self.seed.anchor_cash) / nav if nav > 0 else 0.0,
                "pnl_since_anchor": pnl,
                "return_since_anchor": pnl / self.seed.anchor_nav,
                "return_since_inception": nav / self.seed.initial_capital - 1,
                "quote_coverage": live_count / len(holdings) if holdings else 0.0,
                "valuation_quality": quality,
                "quote_connected": self._connected,
                "quote_message": self._connection_message,
                "quote_revision": quote_revision,
                "market": asdict(taiex) if taiex else None,
                "taiex_ma60": self.seed.taiex_ma60,
                "taiex_above_ma60": taiex_above_ma60,
                "holdings": holdings,
            }
