"""Human-readable definitions for the factor_v4 model inputs."""

from __future__ import annotations

FACTOR_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "價格與動能",
        (
            ("momentum_20d", "最近 20 個交易日的還原權值報酬。"),
            ("momentum_60d", "最近 60 個交易日的還原權值報酬。"),
            ("momentum_120d", "最近 120 個交易日的還原權值報酬。"),
            (
                "momentum_20d_ex_5d",
                "跳過最近 5 日，再計算之前 20 日動能，用來減少短期反轉干擾。",
            ),
            ("ma20_ma60_gap", "20 日均價 ÷ 60 日均價 − 1，觀察中短期趨勢。"),
            ("rsi14", "14 日 RSI，衡量近期上漲與下跌力量。"),
            ("price_ma20_gap", "現價 ÷ 20 日均價 − 1，衡量股價偏離短期均線的程度。"),
            ("close_60d_high", "現價 ÷ 最近 60 日最高還原價；越接近 1，越接近波段高點。"),
        ),
    ),
    (
        "成交量與流動性",
        (
            ("volume_ma20_ma60", "20 日平均成交量 ÷ 60 日平均成交量 − 1，判斷近期是否放量。"),
            ("turnover_60d", "最近 60 日平均成交股數 ÷ 當時發行股數。"),
            (
                "amihud_illiquidity",
                "平均「價格變動幅度 ÷ 成交金額」；越高代表流動性越差，至少要求 50 個有效交易日。",
            ),
        ),
    ),
    (
        "估值與公司規模",
        (
            ("earnings_yield", "最新可得淨利 ÷ PIT 市值，可視為本益比的倒數；負值視為缺值。"),
            ("book_to_market", "股東權益 ÷ PIT 市值，類似股價淨值比的倒數。"),
            ("sales_yield", "最新可得營收 ÷ PIT 市值，類似股價營收比的倒數。"),
            ("log_market_cap", "PIT 市值取自然對數，代表公司規模。"),
        ),
    ),
    (
        "獲利與成長",
        (
            ("roe", "最新可得淨利 ÷ 股東權益，衡量股東資金的獲利效率。"),
            ("revenue_yoy", "最新季營收 ÷ 四季前營收 − 1，代表季度營收年增率。"),
            (
                "revenue_mom",
                "目前實際為最新季營收 ÷ 前一季營收 − 1，也就是 QoQ，不是真正的月增率。",
            ),
            ("operating_income_qoq", "最新季營業利益 ÷ 前一季營業利益 − 1。"),
            ("operating_margin", "最新可得營業利益 ÷ 營收，代表本業獲利率。"),
        ),
    ),
    (
        "風險",
        (
            ("volatility_60d", "最近 60 日對數報酬的年化標準差；越高表示價格波動越大。"),
            (
                "beta_60d",
                "最近 60 日股票報酬相對 TAIEX 報酬的敏感度；Beta 大於 1 通常會放大大盤漲跌。",
            ),
            ("max_drawdown_120d", "最近 120 日從高點到低點的最大跌幅；越負代表跌幅越深。"),
        ),
    ),
    (
        "法人與籌碼",
        (
            ("foreign_net_buy_to_issued_shares", "最近 20 日外資買賣超合計 ÷ 發行股數。"),
            ("trust_net_buy_to_issued_shares", "最近 20 日投信買賣超合計 ÷ 發行股數。"),
            ("margin_balance_change", "最新融資餘額 ÷ 20 個交易日前融資餘額 − 1。"),
            ("short_margin_ratio", "融券餘額 ÷ 融資餘額，衡量多空籌碼比例。"),
        ),
    ),
)


def _table_markdown(rows: tuple[tuple[str, str], ...]) -> str:
    body = "\n".join(f"| `{feature}` | {description} |" for feature, description in rows)
    return f"| 特徵 | 簡單定義 |\n| --- | --- |\n{body}"


def render_factor_guide(st) -> None:
    """Render the standalone factor guide without loading research data."""
    if st.button("← 返回研究儀表板", type="secondary"):
        st.query_params.clear()
        st.rerun()
        return
    st.title("Factor v4 特徵因子解說")
    st.caption("共 27 個固定特徵。技術因子使用還原權值價格；估值與公司規模使用訊號日 PIT 市值。")
    for heading, rows in FACTOR_GROUPS:
        st.subheader(heading)
        st.markdown(_table_markdown(rows))
    st.info(
        "已知語意注意事項：factor_v4 的 revenue_mom 實際計算季度 QoQ。"
        "真正的月營收 MoM／YoY 將於後續 feature version 定案時接入。"
    )
