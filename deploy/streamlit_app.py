"""Streamlit Community Cloud entrypoint for the read-only research dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app as dashboard_app  # noqa: E402
from runtime.snapshot_store import SnapshotStore  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent / "dashboard_snapshot.sqlite"


dashboard_app.main(store_factory=lambda: SnapshotStore(SNAPSHOT))
