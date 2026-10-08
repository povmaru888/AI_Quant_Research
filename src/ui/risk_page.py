"""P4-06: risk page (SDD 14.1 風險).

Renders equity exposure, predicted/realized volatility, MDD, turnover,
and the TAIEX regime from a ``get_risk`` dict, plus a verifiable MA60
exposure check: below the MA60 the exposure must not exceed the cap.
Missing values degrade to ``N/A`` / "insufficient data", never an
exception.
"""

from __future__ import annotations

import math

from ui.theme import comparison_kpi_html, format_value, kpi_html, value_class

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
    if risk_data.get("comparison_mode") is True:
        primary = risk_data.get("primary")
        comparison = risk_data.get("comparison")
        if not isinstance(primary, dict) or not isinstance(comparison, dict):
            raise ValueError("invalid risk comparison payload")
        st.header("風險比較")
        caption = getattr(st, "caption", None)
        if callable(caption):
            caption(
                f"目前：{risk_data.get('primary_run_id', '')}｜"
                f"比較：{risk_data.get('comparison_run_id', '')}"
            )
        cards = (
            ("股票曝險", "equity_exposure", "percent"),
            ("預測波動", "predicted_volatility", "percent"),
            ("實際波動", "realized_volatility", "percent"),
            ("最大回撤", "max_drawdown", "percent"),
            ("Turnover", "turnover", "multiple"),
        )
        columns_fn = getattr(st, "columns", None)
        if callable(columns_fn):
            for col, (label, key, kind) in zip(columns_fn(5), cards, strict=False):
                first, second = _number(primary.get(key)), _number(comparison.get(key))
                delta = first - second if first is not None and second is not None else None
                col.markdown(
                    comparison_kpi_html(
                        label,
                        format_value(first, kind),
                        format_value(second, kind),
                        format_value(delta, kind) if delta is not None else None,
                        value_class(delta),
                        "比較 OOS",
                    ),
                    unsafe_allow_html=True,
                )
        else:
            for label, key, kind in cards:
                st.write(
                    f"{label}：目前 {format_value(primary.get(key), kind)}｜"
                    f"比較 {format_value(comparison.get(key), kind)}"
                )
        for label, payload in (("目前 OOS", primary), ("比較 OOS", comparison)):
            verdict = check_ma60_exposure(
                payload.get("market_regime"),
                payload.get("equity_exposure"),
                payload.get("exposure_cap"),
            )
            st.write(
                f"{label}：市場狀態 {payload.get('market_regime') or 'N/A'}｜"
                f"MA60 曝險限制 {verdict}"
            )
        return
    st.header("風險")

    def _line(label: str, value: object) -> None:
        number = _number(value)
        st.write(f"{label}：{number:.4f}" if number is not None else f"{label}：N/A")

    columns_fn = getattr(st, "columns", None)
    markdown = getattr(st, "markdown", None)
    cards = (
        ("股票曝險", risk_data.get("equity_exposure"), "percent"),
        ("預測波動", risk_data.get("predicted_volatility"), "percent"),
        ("實際波動", risk_data.get("realized_volatility"), "percent"),
        ("最大回撤", risk_data.get("max_drawdown"), "percent"),
        ("Turnover", risk_data.get("turnover"), "multiple"),
    )
    if callable(columns_fn) and callable(markdown):
        cols = columns_fn(5)
        for col, (label, value, kind) in zip(cols, cards, strict=False):
            col.markdown(
                kpi_html(label, format_value(value, kind), tone=value_class(value)),
                unsafe_allow_html=True,
            )
    else:
        _line("股票曝險", risk_data.get("equity_exposure"))
        _line("預測波動", risk_data.get("predicted_volatility"))
        _line("實際波動", risk_data.get("realized_volatility"))
        _line("MDD", risk_data.get("max_drawdown"))
        _line("Turnover", risk_data.get("turnover"))
    regime = risk_data.get("market_regime")
    regime_text = regime if isinstance(regime, str) and regime else "N/A"
    verdict = check_ma60_exposure(
        regime, risk_data.get("equity_exposure"), risk_data.get("exposure_cap")
    )
    if callable(markdown):
        regime_tone = "negative" if regime == BELOW_MA60 else "positive" if regime else "neutral"
        verdict_tone = (
            "positive" if verdict == "符合" else "negative" if verdict == "超限" else "neutral"
        )
        markdown(
            f'<span class="status {regime_tone}">TAIEX：{regime_text}</span>'
            f'<span class="status {verdict_tone}">MA60 曝險限制：{verdict}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(f"TAIEX regime：{regime_text}")
        st.write(f"MA60 曝險限制：{verdict}")
