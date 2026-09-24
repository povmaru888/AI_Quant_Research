"""P2-09 acceptance: Optuna optimization service."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd

from services.optimization_service import SEARCH_SPACE, optimize_xgb

rng = np.random.default_rng(11)


def _dataset(n: int) -> tuple[pd.DataFrame, pd.Series]:
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    return pd.DataFrame({"f1": f1, "f2": f2}), pd.Series(((f1 + f2) > 0).astype(int))


def test_optimize_xgb_returns_best_and_log(settings, monkeypatch) -> None:
    import services.optimization_service as opt_module

    real_suggest = opt_module._suggest

    def tiny_suggest(trial) -> dict:
        params = real_suggest(trial)
        params["n_estimators"] = 10
        return params

    monkeypatch.setattr(opt_module, "_suggest", tiny_suggest)
    real_prepare = opt_module._prepare_training
    prepared_calls = 0

    def counted_prepare(*args):
        nonlocal prepared_calls
        prepared_calls += 1
        return real_prepare(*args)

    monkeypatch.setattr(opt_module, "_prepare_training", counted_prepare)
    train_x, train_y = _dataset(60)
    valid_x, valid_y = _dataset(30)
    artifact, booster, trial_log = optimize_xgb(
        train_x, train_y, valid_x, valid_y, settings, "v", "f", "p", "r", n_trials=2
    )
    assert booster is not None
    assert list(trial_log.columns) == ["trial", *SEARCH_SPACE, "random_state", "rank_ic"]
    assert len(trial_log) == 2
    assert artifact.validation_rank_ic == trial_log["rank_ic"].max()
    assert artifact.best_params["random_state"] == settings.project.random_state
    assert prepared_calls == 1


def test_optimize_xgb_trial_count_defaults_to_settings(settings, monkeypatch) -> None:
    import services.optimization_service as opt_module

    real_suggest = opt_module._suggest

    def tiny_suggest(trial) -> dict:
        params = real_suggest(trial)
        params["n_estimators"] = 5
        return params

    monkeypatch.setattr(opt_module, "_suggest", tiny_suggest)
    single = dataclasses.replace(
        settings, validation=dataclasses.replace(settings.validation, optuna_trials=1)
    )
    train_x, train_y = _dataset(40)
    valid_x, valid_y = _dataset(20)
    _, _, trial_log = optimize_xgb(train_x, train_y, valid_x, valid_y, single, "v", "f", "p", "r")
    assert len(trial_log) == 1


def test_optimize_xgb_ties_keep_first_trial(settings, monkeypatch) -> None:
    import services.optimization_service as opt_module

    real_suggest = opt_module._suggest

    def tiny_suggest(trial) -> dict:
        params = real_suggest(trial)
        params["n_estimators"] = 5
        return params

    monkeypatch.setattr(opt_module, "_suggest", tiny_suggest)
    train_x, train_y = _dataset(40)
    valid_x, _ = _dataset(20)
    artifact, _, trial_log = optimize_xgb(
        train_x, train_y, valid_x, pd.Series([0] * 20), settings, "v", "f", "p", "r", n_trials=2
    )
    assert trial_log["rank_ic"].tolist() == [-1.0, -1.0]
    assert all(artifact.best_params[name] == trial_log.loc[0, name] for name in SEARCH_SPACE)


def test_optimize_xgb_test_data_cannot_enter() -> None:
    params = list(inspect.signature(optimize_xgb).parameters)
    assert not any("test" in name for name in params), params
