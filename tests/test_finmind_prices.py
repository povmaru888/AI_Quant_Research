"""P3-01 acceptance: FinMind price client (HTTP mocked, no network)."""

from __future__ import annotations

import pytest

from integrations.finmind import FinMindError
from integrations.finmind_prices import PRICE_COLUMNS, fetch_prices

TOKEN = "secret-token-abc"


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _ok(rows: list[dict]) -> _StubResponse:
    return _StubResponse(200, {"status": 200, "msg": "success", "data": rows})


def _row(**overrides) -> dict:
    row = {
        "stock_id": "2330",
        "date": "2019-12-31",
        "open": "300.0",
        "max": "305.0",
        "min": "298.0",
        "close": "303.0",
        "Trading_Volume": "10000",
        "Trading_money": "3000000",
    }
    row.update(overrides)
    return row


def test_fetch_prices_mapping() -> None:
    seen: dict = {}

    def requester(url, params=None, headers=None, timeout=None):
        seen["auth"] = headers.get("Authorization")
        seen["params"] = params
        return _ok([_row()])

    frame = fetch_prices("2019-12-01", "2019-12-31", TOKEN, requester=requester)
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert frame.iloc[0].to_dict() == {
        "stock_id": "2330",
        "trade_date": "2019-12-31",
        "open": 300.0,
        "high": 305.0,
        "low": 298.0,
        "close": 303.0,
        "volume": 10000.0,
        "traded_value": 3000000.0,
        "source": "finmind",
    }
    assert seen["auth"] == f"Bearer {TOKEN}"
    assert seen["params"]["dataset"] == "TaiwanStockPrice"


def test_fetch_prices_missing_token() -> None:
    with pytest.raises(FinMindError, match="missing token") as exc_info:
        fetch_prices("2019-12-01", "2019-12-31", "  ", requester=lambda *a, **k: None)
    assert TOKEN not in str(exc_info.value)


def test_fetch_prices_http_and_api_errors_hide_token() -> None:
    def boom(*args, **kwargs):
        raise TimeoutError("nope")

    with pytest.raises(FinMindError) as exc_info:
        fetch_prices("2019-12-01", "2019-12-31", TOKEN, requester=boom)
    assert TOKEN not in str(exc_info.value)

    bad_status = lambda *a, **k: _StubResponse(500, {})  # noqa: E731
    with pytest.raises(FinMindError, match="HTTP 500") as exc_info:
        fetch_prices("2019-12-01", "2019-12-31", TOKEN, requester=bad_status)
    assert TOKEN not in str(exc_info.value)

    api_error = lambda *a, **k: _StubResponse(  # noqa: E731
        200, {"status": 400, "msg": "invalid token", "data": []}
    )
    with pytest.raises(FinMindError, match="API error") as exc_info:
        fetch_prices("2019-12-01", "2019-12-31", TOKEN, requester=api_error)
    assert TOKEN not in str(exc_info.value)


def test_fetch_prices_drops_bad_bars() -> None:
    rows = [
        _row(),
        _row(stock_id="bad-zero", open="0"),
        _row(stock_id="bad-range", max="290.0"),
        _row(stock_id="bad-nan", close="n/a"),
    ]
    frame = fetch_prices("2019-12-01", "2019-12-31", TOKEN, requester=lambda *a, **k: _ok(rows))
    assert frame["stock_id"].tolist() == ["2330"]


def test_fetch_prices_rejects_bad_range() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        fetch_prices("2019-12-31", "2019-12-01", TOKEN, requester=lambda *a, **k: None)
    with pytest.raises(ValueError, match="invalid start"):
        fetch_prices("2019-13-01", "2019-12-31", TOKEN, requester=lambda *a, **k: None)
