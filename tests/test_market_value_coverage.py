"""Exchange-verified exclusions used by the PIT market-value coverage audit."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tools.check_market_value_coverage import _load_nonformal_exclusions


ROOT = Path(__file__).resolve().parents[1]


def test_nonformal_market_value_exclusions_have_official_evidence() -> None:
    exclusions = _load_nonformal_exclusions(
        ROOT / "data" / "pit_v3_nonformal_mainboard_exclusions.csv"
    )

    assert len(exclusions) == 92
    assert sum(row["market"] == "TPEX" for row in exclusions.values()) == 70
    assert sum(row["market"] == "TWSE" for row in exclusions.values()) == 22
    assert all(row["official_source_url"].startswith("https://") for row in exclusions.values())
    for (signal_date, _), row in exclusions.items():
        if row["formal_listing_date"]:
            assert date.fromisoformat(signal_date) < date.fromisoformat(row["formal_listing_date"])


def test_nonformal_exclusions_reject_duplicate_stock_date_rows(tmp_path: Path) -> None:
    path = tmp_path / "exclusions.csv"
    row = (
        "signal_date,stock_id,market,reason,formal_listing_date,official_source_url,evidence\n"
        "2020-01-31,1234,TWSE,before_listing,2020-02-01,https://example.com,official\n"
        "2020-01-31,1234,TWSE,before_listing,2020-02-01,https://example.com,official\n"
    )
    path.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate nonformal exclusion"):
        _load_nonformal_exclusions(path)
