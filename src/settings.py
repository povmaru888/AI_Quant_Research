"""P0-04: typed, immutable settings loader.

Tokens are never stored in ``Settings``; only the environment variable
*name* (e.g. ``FINMIND_TOKEN``) is kept. Resolve values lazily via
:func:`get_finmind_token`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SettingsError(ValueError):
    """Raised when settings are missing or invalid."""


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    timezone: str
    random_state: int


@dataclass(frozen=True)
class DataSettings:
    database_url: str
    finmind_token_env: str
    price_start_date: str
    fallback_source: str


@dataclass(frozen=True)
class UniverseSettings:
    min_market_cap_twd: float
    min_avg_traded_value_20d_twd: float
    min_price_twd: float
    excluded_flags: tuple[str, ...] = field(default_factory=tuple)
    min_listing_age_trading_days: int = 0


@dataclass(frozen=True)
class FeaturesSettings:
    winsor_lower_quantile: float
    winsor_upper_quantile: float
    correlation_threshold: float
    volatility_window: int
    feature_version: str
    required_adjusted_price_rows: int = 0
    required_financial_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LabelSettings:
    horizon_trading_days: int
    top_quantile: float


@dataclass(frozen=True)
class ValidationSettings:
    train_years: int
    validation_years: int
    test_years: int
    purge_trading_days: int
    optuna_trials: int


@dataclass(frozen=True)
class PortfolioSettings:
    top_n: int
    hold_rank_threshold: int
    max_individual_weight: float
    target_annual_volatility: float
    max_equity_exposure: float
    taiex_ma_window: int
    reduced_exposure_below_ma: float


@dataclass(frozen=True)
class ExecutionSettings:
    signal_time: str
    execution_time: str
    broker_fee_rate: float
    sell_tax_rate: float
    slippage_rate_per_side: float


@dataclass(frozen=True)
class Settings:
    project: ProjectSettings
    data: DataSettings
    universe: UniverseSettings
    features: FeaturesSettings
    label: LabelSettings
    validation: ValidationSettings
    portfolio: PortfolioSettings
    execution: ExecutionSettings


def _section(data: Mapping, name: str) -> Mapping:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise SettingsError(f"missing required section: {name}")
    return section


def _require(section: Mapping, section_name: str, key: str):
    if key not in section:
        raise SettingsError(f"missing required setting: {section_name}.{key}")
    return section[key]


def _require_str(section: Mapping, section_name: str, key: str) -> str:
    value = _require(section, section_name, key)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"invalid {section_name}.{key}: must be a non-empty string")
    return value


def _require_positive_int(section: Mapping, section_name: str, key: str) -> int:
    value = _require(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SettingsError(f"invalid {section_name}.{key}: must be a positive int")
    return value


def _require_non_negative_number(section: Mapping, section_name: str, key: str) -> float:
    value = _require(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SettingsError(f"invalid {section_name}.{key}: must be >= 0")
    return float(value)


def _require_positive_number(section: Mapping, section_name: str, key: str) -> float:
    value = _require(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SettingsError(f"invalid {section_name}.{key}: must be > 0")
    return float(value)


def _require_unit_interval(section: Mapping, section_name: str, key: str) -> float:
    value = _require(section, section_name, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"invalid {section_name}.{key}: must be a number in (0, 1]")
    if not 0 < float(value) <= 1:
        raise SettingsError(f"invalid {section_name}.{key}: must be in (0, 1]")
    return float(value)


def load_settings(path: str | Path, env: Mapping[str, str] | None = None) -> Settings:
    """Load YAML config into immutable typed ``Settings``.

    ``env`` is accepted for API compatibility but token *values* are never
    read into ``Settings``; only the variable name from YAML is kept.
    """
    _ = env  # tokens resolve lazily via get_finmind_token
    config_path = Path(path)
    if not config_path.is_file():
        raise SettingsError(f"settings file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SettingsError("invalid settings: top level must be a mapping")

    project = _section(data, "project")
    data_sec = _section(data, "data")
    universe = _section(data, "universe")
    features = _section(data, "features")
    label = _section(data, "label")
    validation = _section(data, "validation")
    portfolio = _section(data, "portfolio")
    execution = _section(data, "execution")

    random_state_raw = _require(project, "project", "random_state")
    if (
        isinstance(random_state_raw, bool)
        or not isinstance(random_state_raw, int)
        or random_state_raw < 0
    ):
        raise SettingsError("invalid project.random_state: must be an int >= 0")
    project_settings = ProjectSettings(
        name=_require_str(project, "project", "name"),
        timezone=_require_str(project, "project", "timezone"),
        random_state=random_state_raw,
    )

    finmind_token_env = _require_str(data_sec, "data", "finmind_token_env")
    if not _ENV_NAME_PATTERN.match(finmind_token_env):
        raise SettingsError("invalid data.finmind_token_env: must be an env var name")
    price_start_date = _require_str(data_sec, "data", "price_start_date")
    if not _DATE_PATTERN.match(price_start_date):
        raise SettingsError("invalid data.price_start_date: must be YYYY-MM-DD")
    try:
        date.fromisoformat(price_start_date)
    except ValueError as exc:
        raise SettingsError("invalid data.price_start_date: not a calendar date") from exc
    data_settings = DataSettings(
        database_url=_require_str(data_sec, "data", "database_url"),
        finmind_token_env=finmind_token_env,
        price_start_date=price_start_date,
        fallback_source=_require_str(data_sec, "data", "fallback_source"),
    )

    excluded = _require(universe, "universe", "excluded_flags")
    if not isinstance(excluded, (list, tuple)) or not all(
        isinstance(v, str) and v.strip() for v in excluded
    ):
        raise SettingsError("invalid universe.excluded_flags: must be a list of strings")
    universe_settings = UniverseSettings(
        min_market_cap_twd=_require_positive_number(universe, "universe", "min_market_cap_twd"),
        min_avg_traded_value_20d_twd=_require_positive_number(
            universe, "universe", "min_avg_traded_value_20d_twd"
        ),
        min_price_twd=_require_positive_number(universe, "universe", "min_price_twd"),
        excluded_flags=tuple(excluded),
        min_listing_age_trading_days=int(universe.get("min_listing_age_trading_days", 0)),
    )

    winsor_lower = _require(features, "features", "winsor_lower_quantile")
    winsor_upper = _require(features, "features", "winsor_upper_quantile")
    if (
        isinstance(winsor_lower, bool)
        or isinstance(winsor_upper, bool)
        or not isinstance(winsor_lower, (int, float))
        or not isinstance(winsor_upper, (int, float))
        or not 0 <= float(winsor_lower) < float(winsor_upper) <= 1
    ):
        raise SettingsError("invalid features winsor quantiles: require 0 <= lower < upper <= 1")
    features_settings = FeaturesSettings(
        winsor_lower_quantile=float(winsor_lower),
        winsor_upper_quantile=float(winsor_upper),
        correlation_threshold=_require_unit_interval(features, "features", "correlation_threshold"),
        volatility_window=_require_positive_int(features, "features", "volatility_window"),
        feature_version=_require_str(features, "features", "feature_version"),
        required_adjusted_price_rows=int(features.get("required_adjusted_price_rows", 0)),
        required_financial_fields=tuple(features.get("required_financial_fields", ())),
    )
    if universe_settings.min_listing_age_trading_days < 0:
        raise SettingsError("invalid universe.min_listing_age_trading_days: must be >= 0")
    if features_settings.required_adjusted_price_rows < 0:
        raise SettingsError("invalid features.required_adjusted_price_rows: must be >= 0")
    if not all(
        isinstance(value, str) and value.strip()
        for value in features_settings.required_financial_fields
    ):
        raise SettingsError("invalid features.required_financial_fields: must be strings")

    label_settings = LabelSettings(
        horizon_trading_days=_require_positive_int(label, "label", "horizon_trading_days"),
        top_quantile=_require_unit_interval(label, "label", "top_quantile"),
    )

    validation_settings = ValidationSettings(
        train_years=_require_positive_int(validation, "validation", "train_years"),
        validation_years=_require_positive_int(validation, "validation", "validation_years"),
        test_years=_require_positive_int(validation, "validation", "test_years"),
        purge_trading_days=_require_positive_int(validation, "validation", "purge_trading_days"),
        optuna_trials=_require_positive_int(validation, "validation", "optuna_trials"),
    )

    top_n = _require_positive_int(portfolio, "portfolio", "top_n")
    hold_rank = _require_positive_int(portfolio, "portfolio", "hold_rank_threshold")
    if hold_rank < top_n:
        raise SettingsError("invalid portfolio.hold_rank_threshold: must be >= top_n")
    reduced_exposure = _require(portfolio, "portfolio", "reduced_exposure_below_ma")
    if (
        isinstance(reduced_exposure, bool)
        or not isinstance(reduced_exposure, (int, float))
        or not 0 <= float(reduced_exposure) <= 1
    ):
        raise SettingsError("invalid portfolio.reduced_exposure_below_ma: must be in [0, 1]")
    portfolio_settings = PortfolioSettings(
        top_n=top_n,
        hold_rank_threshold=hold_rank,
        max_individual_weight=_require_unit_interval(
            portfolio, "portfolio", "max_individual_weight"
        ),
        target_annual_volatility=_require_positive_number(
            portfolio, "portfolio", "target_annual_volatility"
        ),
        max_equity_exposure=_require_unit_interval(portfolio, "portfolio", "max_equity_exposure"),
        taiex_ma_window=_require_positive_int(portfolio, "portfolio", "taiex_ma_window"),
        reduced_exposure_below_ma=float(reduced_exposure),
    )

    execution_settings = ExecutionSettings(
        signal_time=_require_str(execution, "execution", "signal_time"),
        execution_time=_require_str(execution, "execution", "execution_time"),
        broker_fee_rate=_require_non_negative_number(execution, "execution", "broker_fee_rate"),
        sell_tax_rate=_require_non_negative_number(execution, "execution", "sell_tax_rate"),
        slippage_rate_per_side=_require_non_negative_number(
            execution, "execution", "slippage_rate_per_side"
        ),
    )

    return Settings(
        project=project_settings,
        data=data_settings,
        universe=universe_settings,
        features=features_settings,
        label=label_settings,
        validation=validation_settings,
        portfolio=portfolio_settings,
        execution=execution_settings,
    )


def get_finmind_token(settings: Settings, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve the FinMind token from the environment only."""
    source = env if env is not None else os.environ
    return source.get(settings.data.finmind_token_env)
