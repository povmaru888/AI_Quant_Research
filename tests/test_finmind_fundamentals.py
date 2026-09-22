"""P3-02 acceptance: FinMind fundamentals and institutional clients.

Live-schema tests (verified 2026-09-22): statements and investor flows
arrive in LONG format; margin balances use TodayBalance keys.
"""

from __future__ import annotations

import pandas as pd
import pytest

from integrations.finmind import FinMindError
from integrations.finmind_fundamentals import (
    FINANCIAL_COLUMNS,
    INSTITUTIONAL_COLUMNS,
    INSTITUTIONAL_DATASET,
    MARGIN_DATASET,
    fetch_financials,
    fetch_institutional,
)

TOKEN = "secret-token-xyz"


class _StubResponse:
    def __init__(self, payload: object) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _ok(data: list[dict]) -> _StubResponse:
    return _StubResponse({"status": 200, "msg": "success", "data": data})


def _account(stock_id: str, period_end: str, code: str, value: object) -> dict:
    return {"stock_id": stock_id, "date": period_end, "type": code, "value": value}


def _by_dataset(mapping: dict[str, list[dict]]):
    def requester(url, params=None, headers=None, timeout=None, **kwargs):
        return _ok(mapping[params["dataset"]])

    return requester


def test_fetch_financials_pivots_long_rows() -> None:
    rows = [
        _account("2330", "2020-03-31", "Revenue", "592644201000"),
        _account("2330", "2020-03-31", "OperatingIncome", "249018306000"),
        _account("2330", "2020-03-31", "IncomeAfterTaxes", "225221263000"),
        _account("2330", "2020-03-31", "EquityAttributableToOwnersOfParent", "225484877000"),
    ]
    frame = fetch_financials(
        "2020-01-01",
        "2020-12-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": rows}),
    )
    assert list(frame.columns) == list(FINANCIAL_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["stock_id"] == "2330"
    assert row["report_period"] == "2020Q1"
    # Statutory Q1 deadline 05-15, available one day later (no lookahead).
    assert row["announcement_date"] == "2020-05-15"
    assert row["available_date"] == "2020-05-16"
    assert row["revenue"] == pytest.approx(592644201000.0)
    assert row["net_income"] == pytest.approx(225221263000.0)
    assert row["equity"] == pytest.approx(225484877000.0)
    assert row["operating_income"] == pytest.approx(249018306000.0)
    assert row["assets"] is None or bool(pd.isna(row["assets"]))
    assert row["source"] == "finmind"


def test_fetch_financials_q4_deadline_next_year() -> None:
    rows = [_account("2330", "2019-12-31", "Revenue", "100000")]
    frame = fetch_financials(
        "2019-01-01",
        "2020-12-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": rows}),
    )
    assert frame.iloc[0]["report_period"] == "2019Q4"
    assert frame.iloc[0]["announcement_date"] == "2020-03-31"
    assert frame.iloc[0]["available_date"] == "2020-04-01"


def test_fetch_financials_net_income_fallbacks() -> None:
    rows = [_account("2330", "2020-06-30", "NetIncome", "-5")]
    frame = fetch_financials(
        "2020-01-01",
        "2020-12-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": rows}),
    )
    assert frame.iloc[0]["net_income"] == pytest.approx(-5.0)


def test_fetch_financials_drops_empty_periods() -> None:
    rows = [
        _account("2330", "2020-03-31", "Revenue", "100000"),
        _account("2330", "2020-06-30", "EPS", "7.5"),  # no mapped account.
        _account("bad-date", "not-a-date", "Revenue", "1"),
        _account("2330", "2021-13-99", "Revenue", "1"),
    ]
    frame = fetch_financials(
        "2020-01-01",
        "2021-12-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": rows}),
    )
    assert frame["stock_id"].tolist() == ["2330"]
    assert frame.iloc[0]["report_period"] == "2020Q1"


def test_fetch_financials_forwards_data_id() -> None:
    seen: dict = {}

    def requester(url, params=None, headers=None, timeout=None, **kwargs):
        seen.update(params)
        return _ok([])

    fetch_financials("2020-01-01", "2020-01-31", TOKEN, requester=requester, stock_id="2330")
    assert seen["data_id"] == "2330"
    seen.clear()
    fetch_financials("2020-01-01", "2020-01-31", TOKEN, requester=requester)
    assert "data_id" not in seen


def _flow(stock_id: str, day: str, name: str, buy: object, sell: object) -> dict:
    return {"stock_id": stock_id, "date": day, "name": name, "buy": buy, "sell": sell}


def test_fetch_institutional_pivots_long_flows() -> None:
    mapping = {
        INSTITUTIONAL_DATASET: [
            _flow("2330", "2019-12-31", "Foreign_Investor", "300", "200"),
            _flow("2330", "2019-12-31", "Foreign_Dealer_Self", "50", "20"),
            _flow("2330", "2019-12-31", "Investment_Trust", "80", "30"),
            _flow("2330", "2019-12-31", "Dealer_self", "999", "1"),  # ignored.
        ],
        MARGIN_DATASET: [
            {
                "stock_id": "2330",
                "date": "2019-12-31",
                "MarginPurchaseTodayBalance": "1000",
                "ShortSaleTodayBalance": "100",
            }
        ],
        "TaiwanStockTradingDailyReport": [],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert list(frame.columns) == list(INSTITUTIONAL_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["foreign_net_buy"] == pytest.approx(130.0)
    assert row["trust_net_buy"] == pytest.approx(50.0)
    assert row["margin_balance"] == pytest.approx(1000.0)
    assert row["short_balance"] == pytest.approx(100.0)
    assert bool(pd.isna(row["float_shares"]))


def test_fetch_institutional_tolerates_missing_feed() -> None:
    mapping = {
        INSTITUTIONAL_DATASET: [_flow("2330", "2019-12-31", "Foreign_Investor", "100", "0")],
        MARGIN_DATASET: [],
        "TaiwanStockTradingDailyReport": [],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert len(frame) == 1
    assert frame.iloc[0]["foreign_net_buy"] == pytest.approx(100.0)
    assert bool(frame.iloc[0].isna()["margin_balance"])


def test_fetch_institutional_all_empty() -> None:
    mapping = {
        INSTITUTIONAL_DATASET: [],
        MARGIN_DATASET: [],
        "TaiwanStockTradingDailyReport": [],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert list(frame.columns) == list(INSTITUTIONAL_COLUMNS)
    assert frame.empty


def test_fetch_institutional_include_floats() -> None:
    mapping = {
        INSTITUTIONAL_DATASET: [],
        MARGIN_DATASET: [],
        "TaiwanStockTradingDailyReport": [
            {"stock_id": "2330", "date": "2019-12-31", "Float_Shares": "1000000"}
        ],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert frame.iloc[0]["float_shares"] == pytest.approx(1000000.0)


def test_fetch_errors_hide_token() -> None:
    with pytest.raises(FinMindError) as exc_info:
        fetch_financials("2019-01-01", "2020-01-31", " ", requester=lambda *a, **k: None)
    assert TOKEN not in str(exc_info.value)
