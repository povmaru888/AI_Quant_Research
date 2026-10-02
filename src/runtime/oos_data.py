"""Bounded market-data reads for OOS scoring and materialization.

The general :class:`DbStore` intentionally exposes full-table loaders.  OOS
runs already have small monthly panels, so loading every equity bar is wasted
I/O.  This module keeps the same price semantics while limiting SQL reads to
the symbols and dates that a run can actually touch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select

from models.market import Price

TAIEX_ID = "TAIEX"
_CHUNK_SIZE = 500
_QUOTE_COLUMNS = ("stock_id", "trade_date", "open", "open_adj", "close_adj")


def _chunks(values: Sequence[str], size: int = _CHUNK_SIZE) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _frame(rows: list[tuple], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns)).reset_index(drop=True)


def month_end_signal_dates(engine: Engine, months: Sequence[str]) -> list[str]:
    """Return the final non-index trading date for each requested month."""
    if not months:
        return []
    bounds = []
    for month in months:
        period = pd.Period(month, freq="M")
        bounds.append(
            (
                period.start_time.strftime("%Y-%m-%d"),
                (period + 1).start_time.strftime("%Y-%m-%d"),
            )
        )
    with engine.connect() as connection:
        values = [
            connection.execute(
                select(func.max(Price.trade_date)).where(
                    Price.stock_id != TAIEX_ID,
                    Price.trade_date >= start,
                    Price.trade_date < stop,
                )
            ).scalar_one_or_none()
            for start, stop in bounds
        ]
    return [str(value) for value in values if value is not None]


def load_symbol_prices(
    engine: Engine,
    stock_ids: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    columns: Sequence[str] = _QUOTE_COLUMNS,
) -> pd.DataFrame:
    """Load selected price columns for a bounded symbol set in safe chunks."""
    ids = sorted({str(stock_id) for stock_id in stock_ids if str(stock_id)})
    if not ids:
        return pd.DataFrame(columns=list(columns))
    model_columns = [getattr(Price, column) for column in columns]
    rows: list[tuple] = []
    with engine.connect() as connection:
        for batch in _chunks(ids):
            statement = select(*model_columns).where(Price.stock_id.in_(batch))
            if start is not None:
                statement = statement.where(Price.trade_date >= start)
            if end is not None:
                statement = statement.where(Price.trade_date <= end)
            rows.extend(
                connection.execute(statement.order_by(Price.stock_id, Price.trade_date)).all()
            )
    return _frame(rows, columns)


def load_calendar(
    engine: Engine, *, start: str | None = None, end: str | None = None
) -> list[str]:
    statement = select(Price.trade_date).where(Price.stock_id == TAIEX_ID)
    if start is not None:
        statement = statement.where(Price.trade_date >= start)
    if end is not None:
        statement = statement.where(Price.trade_date <= end)
    with engine.connect() as connection:
        rows = connection.execute(statement.order_by(Price.trade_date)).all()
    return [str(row[0]) for row in rows]


def next_equity_dates(engine: Engine, signal_dates: Sequence[str]) -> dict[str, str]:
    """Resolve each signal to the next official TAIEX trading session."""
    with engine.connect() as connection:
        pairs = [
            (
                signal_date,
                connection.execute(
                    select(func.min(Price.trade_date)).where(
                        Price.stock_id == TAIEX_ID, Price.trade_date > signal_date
                    )
                ).scalar_one_or_none(),
            )
            for signal_date in signal_dates
        ]
    return {signal: str(day) for signal, day in pairs if day is not None}


def load_taiex(engine: Engine, *, end: str | None = None) -> pd.DataFrame:
    statement = select(Price.trade_date, Price.close).where(Price.stock_id == TAIEX_ID)
    if end is not None:
        statement = statement.where(Price.trade_date <= end)
    with engine.connect() as connection:
        rows = connection.execute(statement.order_by(Price.trade_date)).all()
    return _frame(rows, ("trade_date", "close"))


def compact_backtest_prices(
    engine: Engine, stock_ids: Iterable[str], *, start: str, end: str
) -> pd.DataFrame:
    """Return traded quotes plus index-only rows that preserve the full calendar."""
    quotes = load_symbol_prices(engine, stock_ids, start=start, end=end)
    calendar = load_symbol_prices(
        engine, [TAIEX_ID], start=start, end=end, columns=_QUOTE_COLUMNS
    )
    return pd.concat([calendar, quotes], ignore_index=True).sort_values(
        ["trade_date", "stock_id"], kind="mergesort", ignore_index=True
    )


@dataclass
class PreparedOOSData:
    """Small in-memory view used by every month of a panel-backed OOS run."""

    returns: pd.DataFrame
    taiex: pd.DataFrame
    quotes: pd.DataFrame
    next_dates: dict[str, str]
    rows_loaded: int

    @classmethod
    def load(
        cls, engine: Engine, candidate_ids: Iterable[str], signal_dates: Sequence[str]
    ) -> PreparedOOSData:
        candidates = sorted({str(stock_id) for stock_id in candidate_ids})
        final_signal = max(signal_dates)
        history = load_symbol_prices(
            engine,
            candidates,
            end=final_signal,
            columns=("stock_id", "trade_date", "close_adj"),
        )
        history["close_adj"] = pd.to_numeric(history["close_adj"], errors="coerce")
        history.loc[history["close_adj"] <= 0, "close_adj"] = np.nan
        history = history.sort_values(["stock_id", "trade_date"], kind="mergesort")
        taiex = load_taiex(engine, end=final_signal)
        returns = _calendar_aligned_returns(history, taiex["trade_date"].astype(str).tolist())

        next_dates = next_equity_dates(engine, signal_dates)
        needed_dates = set(signal_dates) | set(next_dates.values())
        quote_rows = load_symbol_prices(
            engine,
            candidates,
            start=min(needed_dates),
            end=max(needed_dates),
            columns=_QUOTE_COLUMNS,
        )
        quotes = quote_rows.loc[quote_rows["trade_date"].isin(needed_dates)].reset_index(drop=True)
        return cls(
            returns=returns,
            taiex=taiex,
            quotes=quotes,
            next_dates=next_dates,
            rows_loaded=len(history) + len(taiex) + len(quotes),
        )

    def returns_for(self, stock_ids: Iterable[str]) -> pd.DataFrame:
        wanted = {str(stock_id) for stock_id in stock_ids}
        if not wanted:
            return self.returns.iloc[0:0].copy()
        return self.returns.loc[self.returns["stock_id"].isin(wanted)].reset_index(drop=True)

    def next_open(self, signal_date: str) -> pd.DataFrame:
        execution_date = self.next_dates.get(signal_date)
        if execution_date is None:
            return pd.DataFrame(columns=["stock_id", "trade_date", "open"])
        return self.quotes.loc[
            self.quotes["trade_date"] == execution_date, ["stock_id", "trade_date", "open"]
        ].reset_index(drop=True)

    def quote_lookup(self) -> pd.DataFrame:
        return self.quotes.set_index(["trade_date", "stock_id"])


def _calendar_aligned_returns(history: pd.DataFrame, calendar: Sequence[str]) -> pd.DataFrame:
    """Treat absent bars on official sessions as zero-return suspensions.

    A stored row whose adjusted close is missing remains invalid and is not
    bridged. Only a completely absent stock row is forward-filled, which
    matches portfolio valuation while a listed stock is suspended.
    """
    calendar_index = pd.Index([str(day) for day in calendar], name="trade_date")
    parts: list[pd.DataFrame] = []
    for stock_id, group in history.groupby("stock_id", sort=False):
        ordered = group.sort_values("trade_date", kind="mergesort")
        original_dates = pd.Index(ordered["trade_date"].astype(str))
        if original_dates.empty:
            continue
        days = calendar_index[
            (calendar_index >= str(original_dates.min()))
            & (calendar_index <= str(original_dates.max()))
        ]
        closes = ordered.set_index(ordered["trade_date"].astype(str))["close_adj"]
        aligned = closes.reindex(days)
        absent = ~days.isin(original_dates)
        carried = aligned.ffill()
        aligned.loc[absent] = carried.loc[absent]
        log_return = np.log(aligned / aligned.shift(1))
        part = pd.DataFrame(
            {
                "stock_id": str(stock_id),
                "trade_date": days,
                "log_return": log_return.to_numpy(dtype=float),
            }
        ).dropna(subset=["log_return"])
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["stock_id", "trade_date", "log_return"])
    return pd.concat(parts, ignore_index=True)
