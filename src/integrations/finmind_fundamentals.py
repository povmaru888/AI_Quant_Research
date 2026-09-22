"""P3-02: FinMind fundamentals and institutional clients (SDD 7.1).

Live-API notes (verified 2026-09-22 against api.finmindtrade.com):

- ``TaiwanStockFinancialStatements`` returns LONG rows
  ``(date, stock_id, type, value)`` where ``date`` is the period-END date
  and ``type`` is an account code (``Revenue``, ``OperatingIncome``,
  ``IncomeAfterTaxes``, ``EquityAttributableToOwnersOfParent``, ...).
  It carries neither announcement dates nor balance-sheet/cash-flow
  accounts, so ``assets``/``operating_cash_flow`` stay NULL and
  ``available_date`` falls back to statutory filing deadlines + 1 day
  (conservative: never earlier than the real publication, so no lookahead).
- ``TaiwanStockInstitutionalInvestorsBuySell`` returns LONG rows
  ``(date, stock_id, name, buy, sell)``; nets are pivoted per investor
  group. ``TaiwanStockMarginPurchaseShortSale`` is wide with
  ``MarginPurchaseTodayBalance``/``ShortSaleTodayBalance``.
- Free ("register") tokens reject full-market requests (HTTP 400) but
  accept per-stock ones, so every fetcher takes an optional ``stock_id``
  forwarded as ``data_id``. ``TaiwanStockTradingDailyReport`` is gated even
  per-stock, hence the ``include_floats`` switch (free tier runs without
  float shares; ``float_shares`` stays NULL).

Vendor field names never reach downstream modules.
"""

from __future__ import annotations

from calendar import monthrange
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

from integrations.finmind import _get, _numeric

SOURCE = "finmind"
FINANCIALS_DATASET = "TaiwanStockFinancialStatements"
INSTITUTIONAL_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"
MARGIN_DATASET = "TaiwanStockMarginPurchaseShortSale"
FLOAT_DATASET = "TaiwanStockTradingDailyReport"

FINANCIAL_COLUMNS: tuple[str, ...] = (
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
    "source",
)

INSTITUTIONAL_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "trade_date",
    "foreign_net_buy",
    "trust_net_buy",
    "margin_balance",
    "short_balance",
    "float_shares",
    "source",
)

# Long-format account code -> output column. Missing accounts stay NULL
# (assets / operating_cash_flow are not published by this dataset).
_FIN_ACCOUNT_MAP = {
    "Revenue": "revenue",
    "OperatingIncome": "operating_income",
    "EquityAttributableToOwnersOfParent": "equity",
}
_FIN_NET_CANDIDATES = (
    "IncomeAfterTaxes",
    "NetIncome",
    "IncomeFromContinuingOperations",
)

# Investor `name` values -> net-flow groups (nets = buy - sell).
_FOREIGN_NAMES = frozenset({"Foreign_Investor", "Foreign_Dealer_Self"})
_TRUST_NAMES = frozenset({"Investment_Trust"})

_MARGIN_MAP = {
    "MarginPurchaseTodayBalance": "margin_balance",
    "ShortSaleTodayBalance": "short_balance",
}

# Quarter-end month -> quarter; statutory filing deadline (month, day).
# Q4 (annual) deadline falls in the following year.
_QUARTER_OF_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}
_FILING_DEADLINE = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}


def _statutory_dates(period_end: str) -> tuple[str, str, str] | None:
    """Return (report_period, announcement_date, available_date)."""
    try:
        day = date.fromisoformat(str(period_end))
    except (TypeError, ValueError):
        return None
    quarter = _QUARTER_OF_MONTH.get(day.month)
    if quarter is None:
        return None
    deadline_month, deadline_day = _FILING_DEADLINE[quarter]
    deadline_year = day.year + (1 if quarter == 4 else 0)
    last_day = monthrange(deadline_year, deadline_month)[1]
    announcement = date(deadline_year, deadline_month, min(deadline_day, last_day))
    available = announcement + timedelta(days=1)
    return (f"{day.year}Q{quarter}", announcement.isoformat(), available.isoformat())


def fetch_financials(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
    stock_id: str | None = None,
) -> pd.DataFrame:
    """Fetch long-format statements, pivot accounts, attach PIT dates."""
    rows = _get(FINANCIALS_DATASET, start, end, token, requester, timeout, data_id=stock_id)
    if not rows:
        return pd.DataFrame(columns=[*FINANCIAL_COLUMNS])
    by_period: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("stock_id")
        period_end = row.get("date")
        if not sid or not period_end:
            continue
        key = (str(sid), str(period_end))
        accounts = by_period.setdefault(key, {})
        code = str(row.get("type", ""))
        column = _FIN_ACCOUNT_MAP.get(code)
        if column is not None:
            try:
                accounts[column] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
        elif code in _FIN_NET_CANDIDATES and "net_income" not in accounts:
            try:
                accounts["net_income"] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    records = []
    for (sid, period_end), accounts in sorted(by_period.items()):
        if not accounts:
            continue
        dates = _statutory_dates(period_end)
        if dates is None:
            continue
        report_period, announcement, available = dates
        records.append(
            {
                "stock_id": sid,
                "report_period": report_period,
                "announcement_date": announcement,
                "available_date": available,
                "revenue": accounts.get("revenue"),
                "net_income": accounts.get("net_income"),
                "equity": accounts.get("equity"),
                "assets": None,
                "operating_income": accounts.get("operating_income"),
                "operating_cash_flow": None,
                "source": SOURCE,
            }
        )
    frame = pd.DataFrame(records, columns=[*FINANCIAL_COLUMNS])
    return frame.reset_index(drop=True)


def _pivot_investors(rows: list[dict]) -> pd.DataFrame:
    """Pivot long (date, stock_id, name, buy, sell) rows to net flows."""
    if not rows:
        return pd.DataFrame(columns=["stock_id", "trade_date", "foreign_net_buy", "trust_net_buy"])
    frame = pd.DataFrame(rows)
    if not {"stock_id", "date", "name"}.issubset(frame.columns):
        raise ValueError(f"finmind {INSTITUTIONAL_DATASET} missing fields")
    frame["buy"] = pd.to_numeric(frame.get("buy"), errors="coerce").fillna(0.0)
    frame["sell"] = pd.to_numeric(frame.get("sell"), errors="coerce").fillna(0.0)
    frame["net"] = frame["buy"] - frame["sell"]
    frame["group"] = frame["name"].map(
        lambda n: (
            "foreign_net_buy"
            if n in _FOREIGN_NAMES
            else ("trust_net_buy" if n in _TRUST_NAMES else None)
        )
    )
    frame = frame.loc[frame["group"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=["stock_id", "trade_date", "foreign_net_buy", "trust_net_buy"])
    pivoted = (
        frame.groupby(["stock_id", "date", "group"])["net"]
        .sum()
        .unstack("group")
        .reset_index()
        .rename(columns={"date": "trade_date"})
    )
    for column in ("foreign_net_buy", "trust_net_buy"):
        if column not in pivoted.columns:
            pivoted[column] = 0.0
    return pivoted[["stock_id", "trade_date", "foreign_net_buy", "trust_net_buy"]]


def _frame_margin(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or not {"stock_id", "date"}.issubset(frame.columns):
        return pd.DataFrame(columns=["stock_id", "trade_date", *_MARGIN_MAP.values()])
    keep = ["stock_id", "date", *[c for c in _MARGIN_MAP if c in frame.columns]]
    frame = frame[[c for c in keep if c in frame.columns]].rename(
        columns={"date": "trade_date", **_MARGIN_MAP}
    )
    return frame


def fetch_institutional(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
    stock_id: str | None = None,
    include_floats: bool = True,
) -> pd.DataFrame:
    """Fetch and merge institutional flows, margin balances, float shares."""
    investors = _get(INSTITUTIONAL_DATASET, start, end, token, requester, timeout, data_id=stock_id)
    margin = _get(MARGIN_DATASET, start, end, token, requester, timeout, data_id=stock_id)
    floats: list[dict] = []
    if include_floats:
        floats = _get(FLOAT_DATASET, start, end, token, requester, timeout, data_id=stock_id)

    parts: list[pd.DataFrame] = []
    nets = _pivot_investors(investors)
    if not nets.empty:
        parts.append(nets)
    balances = _frame_margin(margin)
    if not balances.empty:
        parts.append(balances)
    frame = pd.DataFrame(floats)
    if not frame.empty and {"stock_id", "date"}.issubset(frame.columns):
        candidates = ["Float_Shares", "Outstanding_Shares", "float_shares"]
        source_col = next((c for c in candidates if c in frame.columns), None)
        if source_col is not None:
            frame = frame[["stock_id", "date", source_col]].rename(
                columns={"date": "trade_date", source_col: "float_shares"}
            )
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=[*INSTITUTIONAL_COLUMNS])

    merged = parts[0]
    for extra in parts[1:]:
        merged = merged.merge(extra, on=["stock_id", "trade_date"], how="outer")
    merged = _numeric(
        merged,
        ["foreign_net_buy", "trust_net_buy", "margin_balance", "short_balance", "float_shares"],
    )
    merged = merged.dropna(subset=["stock_id", "trade_date"])
    for column in ("foreign_net_buy", "trust_net_buy"):
        # No flow record on a covered date means no flow.
        if column not in merged.columns:
            merged[column] = 0.0
        else:
            merged[column] = merged[column].fillna(0.0)
    for column in ("margin_balance", "short_balance", "float_shares"):
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["source"] = SOURCE
    return merged[[*INSTITUTIONAL_COLUMNS]].reset_index(drop=True)


__all__ = [
    "FINANCIAL_COLUMNS",
    "INSTITUTIONAL_COLUMNS",
    "fetch_financials",
    "fetch_institutional",
]
