from __future__ import annotations

import sqlite3

import pytest

from runtime.panel_data import PreparedPanelData


def test_midmonth_signal_dates_use_last_trading_day_on_or_before_fifteenth() -> None:
    data = object.__new__(PreparedPanelData)
    data.start = "2024-01"
    data.end = "2024-03"
    data.conn = sqlite3.connect(":memory:")
    data.conn.execute("CREATE TABLE prices (stock_id TEXT, trade_date TEXT)")
    data.conn.executemany(
        "INSERT INTO prices VALUES (?, ?)",
        [
            ("2330", "2024-01-15"),
            ("2330", "2024-01-16"),
            ("2330", "2024-02-14"),
            ("2330", "2024-02-16"),
            ("2330", "2024-03-14"),
            ("2330", "2024-03-15"),
            ("TAIEX", "2024-03-15"),
        ],
    )
    try:
        assert data.midmonth_signal_dates() == [
            "2024-01-15",
            "2024-02-14",
            "2024-03-15",
        ]
        with pytest.raises(ValueError, match="1 through 28"):
            data.midmonth_signal_dates(29)
    finally:
        data.conn.close()
