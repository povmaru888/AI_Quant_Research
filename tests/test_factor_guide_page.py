"""Factor guide page content and navigation tests."""

from __future__ import annotations

from ui.factor_guide_page import FACTOR_GROUPS, render_factor_guide


class FakeSt:
    def __init__(self, *, back_pressed: bool = False) -> None:
        self.calls: list[tuple] = []
        self.query_params = {"view": "factors"}
        self.back_pressed = back_pressed

    def button(self, label, **kwargs) -> bool:
        self.calls.append(("button", label, kwargs))
        return self.back_pressed

    def rerun(self) -> None:
        self.calls.append(("rerun",))

    def markdown(self, text, **kwargs) -> None:
        self.calls.append(("markdown", text, kwargs))

    def title(self, text) -> None:
        self.calls.append(("title", text))

    def caption(self, text) -> None:
        self.calls.append(("caption", text))

    def subheader(self, text) -> None:
        self.calls.append(("subheader", text))

    def info(self, text) -> None:
        self.calls.append(("info", text))


def test_factor_guide_has_all_factor_v4_definitions() -> None:
    features = [feature for _, rows in FACTOR_GROUPS for feature, _ in rows]
    assert len(features) == 27
    assert len(set(features)) == 27
    assert "revenue_mom" in features
    assert "foreign_net_buy_to_issued_shares" in features


def test_factor_guide_renders_sections_tables_and_back_button() -> None:
    st = FakeSt()
    render_factor_guide(st)

    assert ("button", "← 返回研究儀表板", {"type": "secondary"}) in st.calls
    assert ("title", "Factor v4 特徵因子解說") in st.calls
    assert [call[1] for call in st.calls if call[0] == "subheader"] == [
        heading for heading, _ in FACTOR_GROUPS
    ]
    markdown = [call[1] for call in st.calls if call[0] == "markdown"]
    assert any("`momentum_20d`" in item for item in markdown)
    assert any("`short_margin_ratio`" in item for item in markdown)
    assert "QoQ" in next(call[1] for call in st.calls if call[0] == "info")


def test_back_button_clears_query_params_and_reruns() -> None:
    st = FakeSt(back_pressed=True)

    render_factor_guide(st)

    assert st.query_params == {}
    assert ("rerun",) in st.calls
    assert not any(call[0] == "title" for call in st.calls)
