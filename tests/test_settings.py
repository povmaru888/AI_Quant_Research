"""P0-03 acceptance (baseline): config.yaml structure and value ranges.

P0-04: typed loader `load_settings` tests below the baseline block.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from settings import SettingsError, get_finmind_token, load_settings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_raw_config() -> dict:
    assert CONFIG_PATH.is_file(), f"missing {CONFIG_PATH}"
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_required_sections_present() -> None:
    data = _load_raw_config()
    for section in (
        "project",
        "data",
        "universe",
        "features",
        "label",
        "validation",
        "portfolio",
        "execution",
    ):
        assert section in data, f"missing section: {section}"
    assert data["project"]["name"] == "taiwan-quant-xgb"
    assert data["data"]["finmind_token_env"] == "FINMIND_TOKEN"
    assert data["data"]["fallback_source"] == "yfinance"
    assert "min_market_cap_twd" in data["universe"]
    assert "top_n" in data["portfolio"]
    assert "broker_fee_rate" in data["execution"]


def test_purge_equals_label_horizon_20() -> None:
    data = _load_raw_config()
    assert data["label"]["horizon_trading_days"] == 20
    assert data["validation"]["purge_trading_days"] == 20


def test_value_ranges() -> None:
    data = _load_raw_config()
    assert data["features"]["winsor_lower_quantile"] == 0.01
    assert data["features"]["winsor_upper_quantile"] == 0.99
    assert data["features"]["correlation_threshold"] == 0.85
    assert 0 < data["label"]["top_quantile"] <= 1
    assert data["portfolio"]["top_n"] > 0
    assert data["portfolio"]["hold_rank_threshold"] >= data["portfolio"]["top_n"]
    assert 0 < data["portfolio"]["max_individual_weight"] <= 1
    assert 0 < data["portfolio"]["max_equity_exposure"] <= 1
    for key in (
        "broker_fee_rate",
        "sell_tax_rate",
        "slippage_rate_per_side",
    ):
        assert data["execution"][key] >= 0, key


def test_no_secrets() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "FINMIND_TOKEN" in text  # env var *name* only
    for token in ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"):
        assert token not in text


# --- P0-04: load_settings -------------------------------------------------


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return path


def test_load_settings_success() -> None:
    settings = load_settings(CONFIG_PATH, env={})
    assert settings.project.name == "taiwan-quant-xgb"
    assert settings.data.finmind_token_env == "FINMIND_TOKEN"
    assert settings.portfolio.top_n == 15
    assert settings.label.horizon_trading_days == 20
    assert settings.validation.purge_trading_days == 20
    with pytest.raises(FrozenInstanceError):
        settings.portfolio.top_n = 99  # type: ignore[misc]


def test_load_settings_missing_key(tmp_path: Path) -> None:
    data = copy.deepcopy(_load_raw_config())
    del data["portfolio"]["top_n"]
    path = _write_config(tmp_path, data)
    with pytest.raises(SettingsError, match="portfolio.top_n"):
        load_settings(path, env={})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("execution", "broker_fee_rate", -0.001),
        ("execution", "sell_tax_rate", -1.0),
        ("portfolio", "top_n", 0),
        ("portfolio", "top_n", -5),
    ],
)
def test_load_settings_invalid_values(
    tmp_path: Path, section: str, key: str, value: float | int
) -> None:
    data = copy.deepcopy(_load_raw_config())
    data[section][key] = value
    path = _write_config(tmp_path, data)
    with pytest.raises(SettingsError, match=key):
        load_settings(path, env={})


def test_load_settings_hold_rank_below_top_n(tmp_path: Path) -> None:
    data = copy.deepcopy(_load_raw_config())
    data["portfolio"]["top_n"] = 15
    data["portfolio"]["hold_rank_threshold"] = 5
    path = _write_config(tmp_path, data)
    with pytest.raises(SettingsError, match="hold_rank_threshold"):
        load_settings(path, env={})


def test_token_from_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMIND_TOKEN", "fake-secret-value")
    settings = load_settings(CONFIG_PATH, env={"FINMIND_TOKEN": "fake-secret-value"})
    assert settings.data.finmind_token_env == "FINMIND_TOKEN"
    assert "fake-secret-value" not in repr(settings)
    assert get_finmind_token(settings, {"FINMIND_TOKEN": "fake-secret-value"}) == (
        "fake-secret-value"
    )
