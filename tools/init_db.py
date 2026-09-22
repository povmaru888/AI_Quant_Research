"""Create the SQLite database file and apply the initial schema migration.

Usage:
    python tools/init_db.py [--config config.yaml]

Idempotent: the migration uses CREATE TABLE IF NOT EXISTS.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from settings import load_settings  # noqa: E402

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "migrations"
    / "versions"
    / "001_initial_schema.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("initial_schema", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sqlite_file(database_url: str) -> Path:
    scheme = urlparse(database_url).scheme
    if scheme != "sqlite":
        raise ValueError(f"only sqlite database_url is supported: {database_url!r}")
    raw = unquote(database_url[len("sqlite:///") :])
    if not raw or raw == ":memory:":
        raise ValueError("init_db needs a file database_url, got in-memory")
    return Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create DB and apply migrations.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        db_path = _sqlite_file(settings.data.database_url)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migration = _load_migration()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        migration.upgrade(conn)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()
    print(f"database ready: {db_path} tables={len(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
