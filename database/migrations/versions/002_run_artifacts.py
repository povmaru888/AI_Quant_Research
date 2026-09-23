"""002: run artifacts table (post-hoc materialized dashboard payloads).

The research pipeline never persisted metrics, factor IC, explainability,
or sensitivity scenarios. Rather than rewriting frozen services, a
materialization script computes them once per run into this table and the
dashboard reads them back. Payloads are JSON documents keyed by
(run_id, kind); see tools/materialize_run.py for producers.

Uses only stdlib ``sqlite3`` like 001; the caller owns the connection.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "002"

ARTIFACTS_SQL = """
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f+00:00', 'NOW')),
    PRIMARY KEY (run_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_run ON run_artifacts(run_id);
"""

# Documented kinds written by tools/materialize_run.py.
KINDS: tuple[str, ...] = (
    "metrics",  # nine SDD metrics + monthly_returns + oos_months + nav stats.
    "factor_ic",  # [{factor, ic}] mean rank IC over the signal forward month.
    "monthly_ic",  # [{month, ic}] prediction rank IC (+ weekly rows for ICIR).
    "model_explain",  # {shap_top, feature_importance, method} from proxy model.
    "sensitivity",  # [{scenario, *_before, *_after}] cost on/off comparison.
)


def upgrade(conn: sqlite3.Connection) -> None:
    """Apply 002 (idempotent)."""
    conn.executescript(ARTIFACTS_SQL)
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Remove 002 objects, leaving 001 intact."""
    conn.execute("DROP INDEX IF EXISTS idx_run_artifacts_run")
    conn.execute("DROP TABLE IF EXISTS run_artifacts")
    conn.commit()
