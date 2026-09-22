"""P1-02: SQLAlchemy engine and session transaction boundary.

MVP supports SQLite only (per SDD). Schema creation belongs to P1-01;
this module only connects. Every pooled connection enables foreign keys
via a ``connect`` event listener.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from settings import Settings

_SQLITE_PREFIX = "sqlite:///"


def _sqlite_file_path(database_url: str) -> Path | None:
    """Return the filesystem path for a file SQLite URL, else None."""
    if not database_url.startswith(_SQLITE_PREFIX):
        return None
    raw = unquote(database_url[len(_SQLITE_PREFIX) :])
    if not raw or raw == ":memory:":
        return None
    return Path(raw)


def _enable_foreign_keys(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_engine_from_settings(settings: Settings) -> Engine:
    """Build a SQLite Engine from ``settings.data.database_url``."""
    database_url = settings.data.database_url
    if urlparse(database_url).scheme != "sqlite":
        raise ValueError(f"only sqlite database_url is supported: {database_url!r}")
    file_path = _sqlite_file_path(database_url)
    if file_path is not None and file_path.parent != Path():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Yield a Session: commit on success, rollback and re-raise on error."""
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
