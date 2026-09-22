"""Runtime CLI helpers acceptance (no network)."""

from __future__ import annotations

from runtime.cli import parse_symbols


def test_parse_symbols_restores_powershell_stripped_zeros() -> None:
    assert parse_symbols("51,52") == ["0051", "0052"]
    assert parse_symbols("2330, 2317") == ["2330", "2317"]
    assert parse_symbols("00878") == ["00878"]
    assert parse_symbols("00400A") == ["00400A"]
    assert parse_symbols("") == []
    assert parse_symbols(" 2330 ,, ") == ["2330"]
