"""FinMind TaiwanStockMonthRevenue client with conservative PIT dates."""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from integrations.finmind import _get
from repositories.monthly_revenues import MONTHLY_REVENUE_COLUMNS

DATASET = "TaiwanStockMonthRevenue"
SOURCE = "finmind"


def _historical_available_date(source_date: date) -> str:
    """Use the 11th of the publication month when no observation time exists."""
    return source_date.replace(day=11).isoformat()


def fetch_monthly_revenues(
    start: str,
    end: str,
    token: str,
    requester=requests.get,
    timeout: float = 90.0,
    stock_id: str | None = None,
) -> pd.DataFrame:
    rows = _get(DATASET, start, end, token, requester, timeout, data_id=stock_id)
    if not rows:
        return pd.DataFrame(columns=list(MONTHLY_REVENUE_COLUMNS))
    raw = pd.DataFrame(rows)
    required = {"date", "stock_id", "revenue", "revenue_month", "revenue_year"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"finmind {DATASET} missing fields {missing}")
    raw["stock_id"] = raw["stock_id"].astype(str).str.strip()
    raw["source_date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw["revenue"] = pd.to_numeric(raw["revenue"], errors="coerce")
    raw["revenue_year"] = pd.to_numeric(raw["revenue_year"], errors="coerce")
    raw["revenue_month"] = pd.to_numeric(raw["revenue_month"], errors="coerce")
    create = (
        raw.get("create_time", pd.Series("", index=raw.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    parsed_create = pd.to_datetime(create.where(create.ne("")), errors="coerce")
    available = []
    for source_text, created in zip(raw["source_date"], parsed_create, strict=True):
        if pd.isna(source_text):
            available.append(None)
        elif pd.notna(created):
            available.append(max(date.fromisoformat(source_text), created.date()).isoformat())
        else:
            available.append(_historical_available_date(date.fromisoformat(source_text)))
    frame = pd.DataFrame(
        {
            "stock_id": raw["stock_id"],
            "revenue_year": raw["revenue_year"],
            "revenue_month": raw["revenue_month"],
            "revenue": raw["revenue"],
            "source_date": raw["source_date"],
            "create_time": parsed_create.dt.strftime("%Y-%m-%d"),
            "available_date": available,
            "country": raw.get("country"),
            "source": SOURCE,
        }
    )
    valid = (
        frame["stock_id"].ne("")
        & frame["source_date"].notna()
        & frame["available_date"].notna()
        & frame["revenue"].ge(0)
        & frame["revenue_year"].between(1900, 9999)
        & frame["revenue_month"].between(1, 12)
    )
    frame = frame.loc[valid].copy()
    frame["revenue_year"] = frame["revenue_year"].astype(int)
    frame["revenue_month"] = frame["revenue_month"].astype(int)
    frame = frame.sort_values(["stock_id", "revenue_year", "revenue_month", "available_date"])
    frame = frame.drop_duplicates(["stock_id", "revenue_year", "revenue_month"], keep="last")
    return frame.loc[:, list(MONTHLY_REVENUE_COLUMNS)].reset_index(drop=True)
