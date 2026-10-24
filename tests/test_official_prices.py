from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.official_prices import fetch_official_price_day


class Response:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _payload() -> dict:
    return {
        "stat": "OK",
        "tables": [
            {
                "fields": [
                    "證券代號",
                    "證券名稱",
                    "成交股數",
                    "成交金額",
                    "開盤價",
                    "最高價",
                    "最低價",
                    "收盤價",
                ],
                "data": [
                    ["2330", "台積電", "1,000", "600,000", "590", "605", "588", "600"],
                    ["9999", "停牌", "0", "0", "--", "--", "--", "--"],
                ],
            }
        ],
    }


def test_fetch_official_price_day_maps_units_and_caches(tmp_path: Path) -> None:
    calls = 0

    def requester(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response(_payload())

    first = fetch_official_price_day("TWSE", "2026-10-23", cache_dir=tmp_path, requester=requester)
    second = fetch_official_price_day(
        "TWSE",
        "2026-10-23",
        cache_dir=tmp_path,
        requester=lambda *args, **kwargs: pytest.fail("cache should avoid HTTP"),
    )
    assert calls == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert first.content_hash == second.content_hash
    row = first.frame.iloc[0]
    assert row["stock_id"] == "2330"
    assert row["volume"] == 1000
    assert row["traded_value"] == 600000
    assert row["source"] == "TWSE:official-daily-close"


def test_tpex_alias_schema_is_supported(tmp_path: Path) -> None:
    payload = {
        "tables": [
            {
                "fields": [
                    "代號",
                    "成交股數",
                    "成交金額(元)",
                    "開盤",
                    "最高",
                    "最低",
                    "收盤",
                ],
                "data": [["6488", "2,000", "200,000", "98", "103", "97", "100"]],
            }
        ]
    }
    result = fetch_official_price_day(
        "TPEX",
        "2026-10-23",
        cache_dir=tmp_path,
        requester=lambda *args, **kwargs: Response(payload),
    )
    assert result.frame.loc[0, "stock_id"] == "6488"
    assert json.loads(__import__("gzip").decompress(next(tmp_path.rglob("*.gz")).read_bytes()))


def test_mismatched_response_date_fails(tmp_path: Path) -> None:
    payload = _payload()
    payload["date"] = "20261022"
    with pytest.raises(ValueError, match="date mismatch"):
        fetch_official_price_day(
            "TWSE",
            "2026-10-23",
            cache_dir=tmp_path,
            requester=lambda *args, **kwargs: Response(payload),
        )


def test_corrupt_cache_is_refetched(tmp_path: Path) -> None:
    cached = tmp_path / "twse" / "2026-10-23.json.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not gzip")
    result = fetch_official_price_day(
        "TWSE",
        "2026-10-23",
        cache_dir=tmp_path,
        requester=lambda *args, **kwargs: Response(_payload()),
    )
    assert result.cache_hit is False
    assert len(result.frame) == 1
