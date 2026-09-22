"""P4-06: risk page (SDD 14.1 風險).

Renders equity exposure, predicted/realized volatility, MDD, turnover,
and the TAIEX regime from a ``get_risk`` dict, plus a verifiable MA60
exposure check: below the MA60 the exposure must not exceed the cap.
Missing values degrade to ``N/A`` / "insufficient data", never an
exception.
"""

from __future__ import annotations

import math

BELOW_MA60 = "below_ma60"


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def check_ma60_exposure(
    market_regime: object, equity_exposure: object, exposure_cap: object
) -> str:
    """Return the MA60 exposure verdict: 符合 / 超限 / 不適用 / 資料不足."""
    if market_regime != BELOW_MA60:
        if market_regime is None:
            return "資料不足"
        return "不適用"
    exposure, cap = _number(equity_exposure), _number(exposure_cap)
    if exposure is None or cap is None:
        return "資料不足"
    return "符合" if exposure <= cap else "超限"


def render_risk(risk_data: dict, st=None) -> None:
    """Render the risk section for one risk payload."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    if not isinstance(risk_data, dict):
        raise ValueError("invalid risk_data: must be a dict")
    st.header("風險")

    def _line(label: str, value: object) -> None:
        number = _number(value)
        st.write(f"{label}：{number:.4f}" if number is not None else f"{label}：N/A")

    _line("股票曝險", risk_data.get("equity_exposure"))
    _line("預測波動", risk_data.get("predicted_volatility"))
    _line("實際波動", risk_data.get("realized_volatility"))
    _line("MDD", risk_data.get("max_drawdown"))
    _line("Turnover", risk_data.get("turnover"))
    regime = risk_data.get("market_regime")
    st.write(f"TAIEX regime：{regime if isinstance(regime, str) and regime else 'N/A'}")
    verdict = check_ma60_exposure(
        regime, risk_data.get("equity_exposure"), risk_data.get("exposure_cap")
    )
    st.write(f"MA60 曝險限制：{verdict}")
