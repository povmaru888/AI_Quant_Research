"""Small cache and atomic-write failure fixtures for the batch panel builder."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from tools import build_panels
from runtime.panel_data import PANEL_FORMAT_VERSION, canonical_panel_hash


def _panel() -> dict:
    panel = {
        "signal_date": "2024-12-31",
        "universe": ["2330"],
        "feature_columns": ["momentum_20d"],
        "feature_version": "factor_adj_v2",
        "frame": pd.DataFrame(
            {"stock_id": ["2330"], "missing_flag": [0], "momentum_20d": [0.25]}
        ),
        "labels": pd.Series(
            [1], index=pd.Index(["2330"], name="stock_id"), name="2024-12-31", dtype="int64"
        ),
        "panel_format_version": PANEL_FORMAT_VERSION,
        "build_fingerprint": "build-fingerprint",
        "source_fingerprint": "source-fingerprint",
    }
    panel["content_hash"] = canonical_panel_hash(panel)
    return panel


def test_valid_panel_cache_requires_schema_fingerprints_and_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "2024-12.pkl"
    panel = _panel()
    build_panels._write_panel(path, panel)
    assert build_panels._read_valid_panel(
        path, "factor_adj_v2", "build-fingerprint", "source-fingerprint"
    )
    assert not build_panels._read_valid_panel(
        path, "factor_adj_v2", "wrong-build", "source-fingerprint"
    )

    panel["frame"].loc[0, "momentum_20d"] = 0.5
    with path.open("wb") as handle:
        pickle.dump(panel, handle)
    assert not build_panels._read_valid_panel(
        path, "factor_adj_v2", "build-fingerprint", "source-fingerprint"
    )
    path.write_bytes(b"truncated-pickle")
    assert not build_panels._read_valid_panel(
        path, "factor_adj_v2", "build-fingerprint", "source-fingerprint"
    )


def test_atomic_pickle_failure_keeps_previous_panel_intact(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "2024-12.pkl"
    build_panels._write_panel(path, _panel())
    before = path.read_bytes()

    def fail_dump(*_args, **_kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(build_panels.pickle, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated interrupted write"):
        build_panels._write_panel(path, _panel())
    assert path.read_bytes() == before


def test_build_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    output = tmp_path / "panels"
    output.mkdir()
    lock = build_panels._lock(output)
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            build_panels._lock(output)
    finally:
        lock.unlink(missing_ok=True)
