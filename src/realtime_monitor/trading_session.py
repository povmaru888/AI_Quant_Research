"""TWSE trading-day and cash-session state for the live monitor."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class SessionState:
    now: datetime
    is_trading_day: bool | None
    phase: str
    target_date: str | None
    data_status: str


class TradingCalendar:
    """Official TWSE holidays plus explicit emergency/makeup overrides."""

    def __init__(self, store) -> None:
        self.store = store
        self._holidays: dict[int, set[str]] = {}
        self._loaded: set[int] = set()
        self._closed_overrides, self._open_overrides = self._parse_overrides()

    @staticmethod
    def _parse_overrides() -> tuple[set[str], set[str]]:
        raw = os.getenv("TWSE_TRADING_DAY_OVERRIDES", "").strip()
        if not raw:
            return set(), set()
        payload = json.loads(raw)
        return set(payload.get("closed", ())), set(payload.get("open", ()))

    def refresh_year(self, year: int) -> bool:
        roc_year = year - 1911
        url = (
            "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
            f"?response=json&queryYear={roc_year}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                payload = json.load(response)
            if payload.get("stat") != "ok":
                raise ValueError(f"TWSE calendar returned {payload.get('stat')!r}")
            dates = {str(row[0]) for row in payload.get("data", ()) if row}
            self.store.save_calendar_year(year, sorted(dates), "twse")
            self._holidays[year] = dates
            self._loaded.add(year)
            return True
        except Exception:  # noqa: BLE001 - cached calendar remains authoritative.
            cached = self.store.load_calendar_year(year)
            if cached is None:
                return False
            self._holidays[year] = set(cached)
            self._loaded.add(year)
            return True

    def is_trading_day(self, day: date) -> bool | None:
        value = day.isoformat()
        if value in self._open_overrides:
            return True
        if value in self._closed_overrides:
            return False
        if day.year not in self._loaded and not self.refresh_year(day.year):
            return None
        if day.weekday() >= 5:
            return False
        return value not in self._holidays[day.year]

    def previous_trading_day(self, day: date) -> date | None:
        candidate = day - timedelta(days=1)
        for _ in range(20):
            status = self.is_trading_day(candidate)
            if status is True:
                return candidate
            if status is None:
                return None
            candidate -= timedelta(days=1)
        return None

    def state(self, now: datetime | None = None) -> SessionState:
        current = (now or datetime.now(TAIPEI)).astimezone(TAIPEI)
        trading = self.is_trading_day(current.date())
        if trading is None:
            return SessionState(current, None, "unknown", None, "calendar_unavailable")
        if not trading:
            previous = self.previous_trading_day(current.date())
            return SessionState(
                current,
                False,
                "non_trading",
                previous.isoformat() if previous else None,
                "historical",
            )
        clock = current.timetz().replace(tzinfo=None)
        if clock < time(8, 45):
            previous = self.previous_trading_day(current.date())
            return SessionState(
                current,
                True,
                "previous_close",
                previous.isoformat() if previous else None,
                "historical",
            )
        if clock < time(9):
            return SessionState(current, True, "clearing", current.date().isoformat(), "waiting")
        if clock < time(13, 30):
            return SessionState(current, True, "live", current.date().isoformat(), "backfilling")
        return SessionState(current, True, "closed", current.date().isoformat(), "closing")
