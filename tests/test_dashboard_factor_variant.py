from __future__ import annotations

from app import _factor_version, _filter_run_rows, _run_label


def test_beta60_ablation_uses_research_feature_variant() -> None:
    row = {
        "run_id": "oos-2024-factor-v4-ablation-no-beta60d-pseudohuber-optuna-NOrefit",
        "model_version": "xgb_oos2024_factor_v4_ablation_no_beta60d_pseudohuber",
        "feature_version": "factor_v4",
    }

    assert _factor_version(row) == "factor_v4_ablation_no_beta60d"
    assert _run_label(row).endswith("[factor_v4_ablation_no_beta60d]")
    assert _filter_run_rows([row], factor_version="factor_v4_ablation_no_beta60d") == [row]


def test_regular_factor_v4_run_keeps_persisted_version() -> None:
    row = {
        "run_id": "oos-2024-factor-v4-pseudohuber-NOrefit",
        "model_version": "xgb_oos2024_factor_v4_pseudohuber",
        "feature_version": "factor_v4",
    }

    assert _factor_version(row) == "factor_v4"
