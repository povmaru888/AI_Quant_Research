from __future__ import annotations

from integrations.official_listing_dates import _parse_date, fetch_official_listing_dates


class _Response:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return self.rows


def test_fetches_and_normalizes_twse_tpex_listing_dates() -> None:
    def requester(url, **_kwargs):
        if "twse" in url:
            return _Response([{"公司代號": "2330", "上市日期": "19940905"}])
        return _Response([{"公司代號": "6488", "上櫃日期": "103/03/27"}])

    frame = fetch_official_listing_dates(requester=requester)

    assert frame.set_index("stock_id")["listed_date"].to_dict() == {
        "2330": "1994-09-05",
        "6488": "2014-03-27",
    }
    assert _parse_date("1130102") == "2024-01-02"
