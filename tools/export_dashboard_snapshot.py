"""Export a compact, read-only snapshot for Streamlit Community Cloud."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime.db_store import build_store  # noqa: E402
from services import dashboard_service  # noqa: E402
from settings import load_settings  # noqa: E402


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _encode(value) -> str:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict("records")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def export_snapshot(config: Path, output: Path) -> dict[str, int]:
    settings = load_settings(config, env=os.environ)
    store = build_store(settings)
    rows = [
        row
        for row in store.list_runs(status="succeeded")
        if not str(row.get("parameter_version", "")).startswith("job:")
    ]
    if not rows:
        raise RuntimeError("no completed research runs found")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    payload_count = 0
    with closing(sqlite3.connect(temporary)) as destination:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                parameter_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                sort_order INTEGER NOT NULL
            );
            CREATE TABLE payloads (
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                as_of TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, kind, as_of)
            );
            """
        )
        destination.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("created_at", datetime.now().astimezone().isoformat()),
        )
        for order, row in enumerate(rows):
            run_id = str(row["run_id"])
            destination.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    row["status"],
                    row.get("parameter_version") or "",
                    row.get("feature_version") or "",
                    order,
                ),
            )
            overview = dashboard_service.get_overview(run_id, store)
            model = dashboard_service.get_model_data(run_id, store)
            risk = dashboard_service.get_risk(run_id, store)
            comparison = dashboard_service.get_comparison(run_id, store)
            dates = store.list_holding_dates(run_id)
            fixed_payloads = {
                "overview": overview,
                "model": model,
                "risk": risk,
                "comparison": comparison,
                "holding_dates": dates,
            }
            for kind, payload in fixed_payloads.items():
                destination.execute(
                    "INSERT INTO payloads VALUES (?, ?, '', ?)",
                    (run_id, kind, _encode(payload)),
                )
                payload_count += 1
            for month in sorted({value[:7] for value in dates}):
                portfolio = dashboard_service.get_portfolio(run_id, month, store)
                serializable = {
                    "holdings": portfolio["holdings"].to_dict("records"),
                    "risk_state": portfolio["risk_state"],
                }
                destination.execute(
                    "INSERT INTO payloads VALUES (?, 'portfolio', ?, ?)",
                    (run_id, month, _encode(serializable)),
                )
                payload_count += 1
        destination.execute("CREATE INDEX payload_lookup ON payloads(run_id, kind, as_of)")
        destination.commit()
        destination.execute("VACUUM")
    os.replace(temporary, output)
    return {"runs": len(rows), "payloads": payload_count, "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deploy" / "dashboard_snapshot.sqlite"
    )
    args = parser.parse_args()
    result = export_snapshot(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
