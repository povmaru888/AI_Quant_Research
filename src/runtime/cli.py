"""Shared CLI helpers for tools/ scripts."""

from __future__ import annotations


def parse_symbols(value: str) -> list[str]:
    """Split a comma list of stock codes, restoring PowerShell-stripped zeros.

    PowerShell coerces ``--symbols 0051,0052`` to the numeric array @(51, 52),
    arriving as ``"51,52"``. TW codes are zero-padded, so all-digit parts
    shorter than 4 chars are left-padded (``51`` -> ``0051``); longer codes
    (``00878``) and lettered codes (``00400A``) pass through untouched.
    """
    codes: list[str] = []
    for part in (value or "").split(","):
        code = part.strip()
        if code.isdigit() and len(code) < 4:
            code = code.zfill(4)
        if code:
            codes.append(code)
    return codes
