"""P3-02 acceptance: FinMind fundamentals and institutional clients."""

from __future__ import annotations

import pytest

from integrations.finmind import FinMindError
from integrations.finmind_fundamentals import (
    FINANCIAL_COLUMNS,
    INSTITUTIONAL_COLUMNS,
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


def _financial_row(**overrides) -> dict:
    row = {
        "stock_id": "2330",
        "date": "2020-01-15",
        "report_period": "2019/Q4",
        "revenue": "100000",
        "net_income": "20000",
        "equity": "500000",
        "assets": "800000",
        "operating_income": "15000",
        "operating_cash_flow": "12000",
    }
    row.update(overrides)
    return row


def _by_dataset(mapping: dict[str, list[dict]]):
    def requester(url, params=None, headers=None, timeout=None):
        return _ok(mapping[params["dataset"]])

    return requester


def test_fetch_financials_mapping_and_available_date() -> None:
    frame = fetch_financials(
        "2019-01-01",
        "2020-01-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": [_financial_row()]}),
    )
    assert list(frame.columns) == list(FINANCIAL_COLUMNS)
    row = frame.iloc[0].to_dict()
    assert row["report_period"] == "2019Q4"
    assert row["announcement_date"] == "2020-01-15"
    assert row["available_date"] == "2020-01-16"
    assert row["revenue"] == pytest.approx(100000.0)
    assert row["operating_cash_flow"] == pytest.approx(12000.0)
    assert row["source"] == "finmind"


def test_fetch_financials_drops_bad_rows() -> None:
    rows = [
        _financial_row(),
        _financial_row(stock_id="no-period", report_period=""),
        _financial_row(stock_id="bad-date", date="not-a-date"),
        _financial_row(stock_id="zero-rev", revenue="0", net_income="-5"),
    ]
    frame = fetch_financials(
        "2019-01-01",
        "2020-01-31",
        TOKEN,
        requester=_by_dataset({"TaiwanStockFinancialStatements": rows}),
    )
    assert frame["stock_id"].tolist() == ["2330", "zero-rev"]
    assert frame.loc[frame["stock_id"] == "zero-rev", "net_income"].iloc[0] == pytest.approx(-5.0)


def test_fetch_institutional_merges_three_feeds() -> None:
    mapping = {
        "InstitutionalInvestorsBuySell": [
            {
                "stock_id": "2330",
                "date": "2019-12-31",
                "Foreign_Investor_BuySell": "100",
                "Trust_BuySell": "50",
            }
        ],
        "MarginPurchaseShortSale": [
            {
                "stock_id": "2330",
                "date": "2019-12-31",
                "Margin_Balance": "1000",
                "Short_Balance": "100",
            }
        ],
        "TaiwanStockTradingDailyReport": [
            {"stock_id": "2330", "date": "2019-12-31", "Float_Shares": "1000000"}
        ],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert list(frame.columns) == list(INSTITUTIONAL_COLUMNS)
    row = frame.iloc[0].to_dict()
    assert row["foreign_net_buy"] == pytest.approx(100.0)
    assert row["trust_net_buy"] == pytest.approx(50.0)
    assert row["margin_balance"] == pytest.approx(1000.0)
    assert row["short_balance"] == pytest.approx(100.0)
    assert row["float_shares"] == pytest.approx(1000000.0)


def test_fetch_institutional_tolerates_missing_feed() -> None:
    mapping = {
        "InstitutionalInvestorsBuySell": [
            {"stock_id": "2330", "date": "2019-12-31", "Foreign_Investor_BuySell": "100"}
        ],
        "MarginPurchaseShortSale": [],
        "TaiwanStockTradingDailyReport": [],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert len(frame) == 1
    assert frame.iloc[0]["foreign_net_buy"] == pytest.approx(100.0)
    assert bool(frame.iloc[0].isna()["margin_balance"])


def test_fetch_institutional_all_empty() -> None:
    mapping = {
        "InstitutionalInvestorsBuySell": [],
        "MarginPurchaseShortSale": [],
        "TaiwanStockTradingDailyReport": [],
    }
    frame = fetch_institutional("2019-12-01", "2019-12-31", TOKEN, requester=_by_dataset(mapping))
    assert list(frame.columns) == list(INSTITUTIONAL_COLUMNS)
    assert frame.empty


def test_fetch_errors_hide_token() -> None:
    with pytest.raises(FinMindError) as exc_info:
        fetch_financials("2019-01-01", "2020-01-31", " ", requester=lambda *a, **k: None)
    assert TOKEN not in str(exc_info.value)
