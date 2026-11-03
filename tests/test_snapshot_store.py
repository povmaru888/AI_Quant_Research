from __future__ import annotations

import json
import sqlite3

import pandas as pd

from runtime.snapshot_store import SnapshotStore


def test_snapshot_store_serves_dashboard_contract(tmp_path):
    path = tmp_path / "snapshot.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (run_id TEXT PRIMARY KEY, status TEXT,
              parameter_version TEXT, feature_version TEXT, sort_order INTEGER);
            CREATE TABLE payloads (run_id TEXT, kind TEXT, as_of TEXT,
              payload_json TEXT, PRIMARY KEY(run_id, kind, as_of));
            """
        )
        connection.execute(
            "INSERT INTO runs VALUES ('run-1','succeeded','params','factor_v4',0)"
        )
        payloads = {
            ("overview", ""): {"run_id": "run-1"},
            ("holding_dates", ""): ["2025-01-31"],
            ("portfolio", "2025-01"): {
                "holdings": [{"stock_id": "2330"}],
                "risk_state": {"signal_date": "2025-01-31"},
            },
            ("model", ""): {"shap_top": []},
            ("risk", ""): {"turnover": 1.0},
            ("comparison", ""): [{"scenario": "with_cost"}],
        }
        connection.executemany(
            "INSERT INTO payloads VALUES ('run-1',?,?,?)",
            [(kind, as_of, json.dumps(payload)) for (kind, as_of), payload in payloads.items()],
        )

    store = SnapshotStore(path)
    assert store.list_runs("succeeded")[0]["feature_version"] == "factor_v4"
    assert store.get_run_status("run-1") == "succeeded"
    assert store.load_run_summary("run-1") == {"run_id": "run-1"}
    assert store.list_holding_dates("run-1") == ["2025-01-31"]
    assert isinstance(store.load_holdings("run-1", "2025-01"), pd.DataFrame)
    assert store.load_holdings("run-1", "2025-01").iloc[0]["stock_id"] == "2330"
    assert store.load_portfolio_risk("run-1", "2025-01")["signal_date"] == "2025-01-31"
    assert store.load_model_data("run-1")["shap_top"] == []
    assert store.load_risk("run-1")["turnover"] == 1.0
    assert store.load_comparison("run-1").iloc[0]["scenario"] == "with_cost"
