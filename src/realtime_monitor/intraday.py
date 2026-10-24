"""Taiwan cash-session coordinates for live fund valuations."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


def _previous_weekday(value: date) -> date:
    value -= timedelta(days=1)
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def intraday_point(snapshot: dict, *, session_date_override: str | None = None) -> dict:
    """Map a valuation to the Taiwan 09:00-13:30 cash-session axis."""
    raw_time = snapshot.get("quote_revision") or snapshot["as_of"]
    observed = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=TAIPEI)
    observed = observed.astimezone(TAIPEI)
    session_date = observed.date()
    session_clock = observed.timetz().replace(tzinfo=None)
    if session_date_override is not None:
        authoritative_date = date.fromisoformat(session_date_override)
        if authoritative_date != session_date:
            session_clock = time(13, 30)
        session_date = authoritative_date
    elif session_date.weekday() >= 5:
        while session_date.weekday() >= 5:
            session_date -= timedelta(days=1)
        session_clock = time(13, 30)
    elif session_clock < time(9):
        session_date = _previous_weekday(session_date)
        session_clock = time(13, 30)
    elif session_clock > time(13, 30):
        session_clock = time(13, 30)
    session_time = datetime.combine(session_date, session_clock, tzinfo=TAIPEI)
    market = snapshot.get("market") or {}
    return {
        "session_date": session_date.isoformat(),
        "time": session_time.isoformat(),
        "nav": snapshot["nav"],
        "taiex": market.get("price"),
        "quote_revision": snapshot.get("quote_revision"),
    }


def intraday_payload(snapshots: list[dict]) -> dict:
    """Keep only the latest session and collapse closed-market heartbeats."""
    points = [intraday_point(snapshot) for snapshot in snapshots]
    latest_date = max((point["session_date"] for point in points), default=None)
    selected = [point for point in points if point["session_date"] == latest_date]
    deduplicated = {point["time"]: point for point in selected}
    return {
        "session_date": latest_date,
        "market_open": "09:00",
        "market_close": "13:30",
        "points": [deduplicated[key] for key in sorted(deduplicated)],
    }


def merge_intraday_payload(backfill: dict, snapshot: dict) -> dict:
    """Merge the authoritative Shioaji session with one newer live valuation."""
    session_date = backfill.get("session_date")
    if not session_date:
        return intraday_payload([snapshot])
    current = intraday_point(snapshot, session_date_override=session_date)
    points = {
        str(point["time"]): point
        for point in backfill.get("points", ())
        if point.get("session_date") == session_date
    }
    points[current["time"]] = current
    return {
        "session_date": session_date,
        "market_open": "09:00",
        "market_close": "13:30",
        "previous_close_nav": backfill.get("previous_close_nav"),
        "previous_close_taiex": backfill.get("previous_close_taiex"),
        "points": [points[key] for key in sorted(points)],
    }
