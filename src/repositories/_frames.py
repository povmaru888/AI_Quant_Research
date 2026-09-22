"""Shared DataFrame helpers for P1-06..P1-11 repositories."""

from __future__ import annotations

import pandas as pd


def to_records(
    rows: pd.DataFrame,
    allowed: frozenset[str],
    key: str | tuple[str, ...],
    entity: str,
) -> list[dict]:
    """Validate columns and convert a frame to insert-ready record dicts.

    Raises ``ValueError`` on missing key columns or unknown columns;
    converts NaN to None. An empty frame yields ``[]``.
    """
    keys = (key,) if isinstance(key, str) else tuple(key)
    missing = [col for col in keys if col not in rows.columns]
    if missing:
        raise ValueError(f"{entity} requires columns: {missing}")
    unknown = [col for col in rows.columns if col not in allowed]
    if unknown:
        raise ValueError(f"unknown {entity} columns: {unknown}")
    return [
        {name: (None if pd.isna(value) else value) for name, value in record.items()}
        for record in rows.to_dict(orient="records")
    ]
