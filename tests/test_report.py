"""P4-08 acceptance: report exporter (fake store, tmp output dir)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from report import ReportPaths, export_reports


class FakeStore:
    def __init__(self, status: str = "succeeded") -> None:
        self._status = status

    def get_run_status(self, run_id: str) -> str:
        if run_id == "missing":
            raise KeyError(run_id)
        return self._status

    def load_run_summary(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "data_end_date": "2020-02-29",
            "feature_version": "v1",
            "model_version": "xgb_202002",
            "parameter_version": "params_abc",
            "metrics": {"cagr": 0.12, "sharpe": 1.5},
            "oos_months": ["2020-02"],
        }

    def load_performance(self, run_id: str) -> pd.DataFrame:
        assert run_id
        return pd.DataFrame(
            {"date": ["2020-02-03", "2020-02-04"], "nav": [1.0, 1.02], "extra": [0, 0]},
            index=[7, 8],
        )

    def load_factor_ic(self, run_id: str) -> pd.DataFrame:
        assert run_id
        return pd.DataFrame({"factor": ["momentum_20d"], "ic": [0.08]})


def test_export_creates_four_files(tmp_path: Path) -> None:
    paths = export_reports("run-001", tmp_path / "reports", FakeStore())
    assert isinstance(paths, ReportPaths)
    for path in (
        paths.performance_csv,
        paths.factor_ic_csv,
        paths.run_summary_json,
        paths.report_pdf,
    ):
        assert path.is_file()

    performance = pd.read_csv(paths.performance_csv)
    assert list(performance.columns) == ["date", "nav", "extra"]
    assert performance["nav"].tolist() == [1.0, 1.02]

    factor_ic = pd.read_csv(paths.factor_ic_csv)
    assert list(factor_ic.columns) == ["factor", "ic"]
    assert factor_ic["factor"].tolist() == ["momentum_20d"]

    summary = json.loads(paths.run_summary_json.read_text(encoding="utf-8"))
    assert summary["run_id"] == "run-001"
    assert summary["data_end_date"] == "2020-02-29"
    assert summary["model_version"] == "xgb_202002"
    assert summary["parameter_version"] == "params_abc"
    assert summary["metrics"]["cagr"] == 0.12
    assert summary["oos_months"] == ["2020-02"]

    raw = paths.report_pdf.read_bytes()
    assert raw.startswith(b"%PDF")
    assert b"run-001" in raw


def test_export_rejects_bad_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not completed"):
        export_reports("run-001", tmp_path, FakeStore(status="failed"))
    with pytest.raises(ValueError, match="unknown run"):
        export_reports("missing", tmp_path, FakeStore())
    with pytest.raises(ValueError, match="run_id"):
        export_reports("  ", tmp_path, FakeStore())

    class BadSummary(FakeStore):
        def load_run_summary(self, run_id: str) -> dict:
            return {"run_id": run_id}

    with pytest.raises(ValueError, match="missing"):
        export_reports("run-001", tmp_path, BadSummary())

    class BadPerformance(FakeStore):
        def load_performance(self, run_id: str) -> pd.DataFrame:
            return pd.DataFrame({"date": ["2020-02-03"]})

    with pytest.raises(ValueError, match="missing columns"):
        export_reports("run-001", tmp_path, BadPerformance())
