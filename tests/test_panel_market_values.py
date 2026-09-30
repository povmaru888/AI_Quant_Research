from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from runtime.panel_data import PreparedPanelData


def test_stable_pit_loader_uses_verified_partial_rows_but_regular_v3_does_not(
    settings, tmp_path: Path
) -> None:
    db_path = tmp_path / "partial-pit.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE source_revision (singleton INTEGER PRIMARY KEY, revision INTEGER NOT NULL);
            INSERT INTO source_revision(singleton, revision) VALUES (1, 0);
            CREATE TABLE stocks (
                stock_id TEXT PRIMARY KEY, stock_name TEXT, market TEXT NOT NULL,
                listed_date TEXT, delisted_date TEXT, industry TEXT
            );
            CREATE TABLE market_values (
                trade_date TEXT NOT NULL, stock_id TEXT NOT NULL,
                market_value REAL NOT NULL, source TEXT NOT NULL
            );
            CREATE TABLE market_value_sync_days (
                trade_date TEXT PRIMARY KEY, status TEXT NOT NULL,
                row_count INTEGER NOT NULL, content_hash TEXT, synced_at TEXT
            );
            INSERT INTO market_values VALUES ('2016-01-30', '2330', 100.0, 'FinMind');
            INSERT INTO market_value_sync_days VALUES ('2016-01-30', 'partial', 1, 'hash', 'now');
            INSERT INTO market_values VALUES ('2016-02-26', '2330', 110.0, 'FinMind');
            INSERT INTO market_value_sync_days VALUES ('2016-02-26', 'succeeded', 1, 'hash', 'now');
            """
        )

    stable_settings = replace(
        settings,
        data=replace(settings.data, database_url=f"sqlite:///{db_path.as_posix()}"),
        features=replace(settings.features, feature_version="factor_adj_pit_v3_stable"),
    )
    legacy_pit_settings = replace(
        stable_settings,
        features=replace(stable_settings.features, feature_version="factor_adj_pit_v3"),
    )
    stable = PreparedPanelData(stable_settings, "2016-01", "2016-02", "stable-test")
    legacy = PreparedPanelData(legacy_pit_settings, "2016-01", "2016-02", "legacy-test")
    try:
        stable._load_market_values(["2016-01-30", "2016-02-26"])
        legacy._load_market_values(["2016-01-30", "2016-02-26"])
        assert set(stable.market_values["trade_date"]) == {"2016-01-30", "2016-02-26"}
        assert set(legacy.market_values["trade_date"]) == {"2016-02-26"}
    finally:
        stable.close()
        legacy.close()
