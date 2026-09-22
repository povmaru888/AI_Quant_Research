"""P3-02: FinMind fundamentals and institutional clients (SDD 7.1).

Financials come from ``TaiwanStockFinancialStatements``; chips merge
three datasets (institutional buy/sell, margin balances, daily float
shares) on (stock_id, trade_date). Vendor field names never reach
downstream modules. ``available_date`` is announcement date plus one
calendar day (conservative ETL assumption, consistent with the P2-03
PIT key and the P1-08 CHECK).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

from integrations.finmind import _get, _numeric

SOURCE = "finmind"
FINANCIALS_DATASET = "TaiwanStockFinancialStatements"
INSTITUTIONAL_DATASET = "InstitutionalInvestorsBuySell"
MARGIN_DATASET = "MarginPurchaseShortSale"
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

_FINANCIAL_MAP = {
    "stock_id": "stock_id",
    "date": "announcement_date",
    "revenue": "revenue",
    "net_income": "net_income",
    "equity": "equity",
    "assets": "assets",
    "operating_income": "operating_income",
    "operating_cash_flow": "operating_cash_flow",
}


def _available_day(announcement: str) -> str | None:
    try:
        day = date.fromisoformat(announcement)
    except (TypeError, ValueError):
        return None
    return (day + timedelta(days=1)).isoformat()


def fetch_financials(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and standardize financial reports."""
    rows = _get(FINANCIALS_DATASET, start, end, token, requester, timeout)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*FINANCIAL_COLUMNS])
    missing = [c for c in (*_FINANCIAL_MAP, "report_period") if c not in frame.columns]
    if missing:
        raise ValueError(f"finmind {FINANCIALS_DATASET} missing fields {missing}")
    frame = frame.rename(columns=_FINANCIAL_MAP)
    frame["report_period"] = (
        frame["report_period"].astype(str).str.replace("/", "", regex=False).str.strip()
    )
    frame["available_date"] = frame["announcement_date"].apply(
        lambda v: _available_day(str(v)) if pd.notna(v) else None
    )
    frame = _numeric(
        frame,
        ["revenue", "net_income", "equity", "assets", "operating_income", "operating_cash_flow"],
    )
    required = ["stock_id", "announcement_date", "available_date", "report_period"]
    frame = frame.dropna(subset=required)
    frame = frame.loc[frame["report_period"] != ""]
    frame = frame.loc[frame["available_date"] >= frame["announcement_date"]]
    frame["source"] = SOURCE
    return frame[[*FINANCIAL_COLUMNS]].reset_index(drop=True)


def fetch_institutional(
    start: str,
    end: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and merge institutional, margin, and float-share feeds."""
    investors = _get(INSTITUTIONAL_DATASET, start, end, token, requester, timeout)
    margin = _get(MARGIN_DATASET, start, end, token, requester, timeout)
    floats = _get(FLOAT_DATASET, start, end, token, requester, timeout)

    parts: list[pd.DataFrame] = []
    frame = pd.DataFrame(investors)
    if not frame.empty:
        rename = {"Foreign_Investor_BuySell": "foreign_net_buy", "Trust_BuySell": "trust_net_buy"}
        keep = ["stock_id", "date", *[c for c in rename if c in frame.columns]]
        frame = frame[[c for c in keep if c in frame.columns]].rename(
            columns={"date": "trade_date", **rename}
        )
        parts.append(frame)
    frame = pd.DataFrame(margin)
    if not frame.empty and {"stock_id", "date"}.issubset(frame.columns):
        rename = {"Margin_Balance": "margin_balance", "Short_Balance": "short_balance"}
        keep = ["stock_id", "date", *[c for c in rename if c in frame.columns]]
        frame = frame[[c for c in keep if c in frame.columns]].rename(
            columns={"date": "trade_date", **rename}
        )
        parts.append(frame)
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
    for column in (
        "foreign_net_buy",
        "trust_net_buy",
        "margin_balance",
        "short_balance",
        "float_shares",
    ):
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["source"] = SOURCE
    return merged[[*INSTITUTIONAL_COLUMNS]].reset_index(drop=True)
