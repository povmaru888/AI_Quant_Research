"""P4-01 acceptance: app shell without launching Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from app import ALL_FILTER, PAGES, _filter_run_rows, _oos_year, available_runs, main


class FakeSidebar:
    def __init__(self, st: FakeSt) -> None:
        self._st = st

    def selectbox(self, label, options, **kwargs):
        self._st.calls.append(("selectbox", label, list(options), kwargs))
        values = list(options)
        selected = getattr(self._st, "selected_selectboxes", {}).get(label)
        if selected in values:
            return selected
        return values[kwargs.get("index", 0)]

    def text_input(self, label, value="", **kwargs):
        self._st.calls.append(("text_input", label, value))
        return value

    def radio(self, label, options, **kwargs):
        self._st.calls.append(("radio", label, list(options)))
        selected = getattr(self._st, "selected_page", None)
        return selected if selected in options else list(options)[0]


class FakeSt:
    def __init__(self) -> None:
        self.calls: list = []
        self.sidebar = FakeSidebar(self)
        self.selected_selectboxes: dict[str, str] = {}

    def title(self, text) -> None:
        self.calls.append(("title", text))

    def info(self, text) -> None:
        self.calls.append(("info", text))

    def error(self, text) -> None:
        self.calls.append(("error", text))


class FakeStore:
    def __init__(self, rows: list[dict], honor_status_arg: bool = True) -> None:
        self._rows = rows
        self.honor_status_arg = honor_status_arg
        self.seen: list = []

    def list_runs(self, status=None):
        self.seen.append(status)
        if isinstance(self._rows, list) and self.honor_status_arg and status is not None:
            return [r for r in self._rows if r.get("status") == status]
        return self._rows


class FakeHoldingsStore(FakeStore):
    def get_run_status(self, run_id):
        return "succeeded"

    def list_holding_dates(self, run_id):
        return ["2024-01-31", "2024-02-29"]


def _run_ids(*statuses: str) -> list[dict]:
    return [{"run_id": f"run-{i:02d}", "status": s} for i, s in enumerate(statuses)]


def test_available_runs_keeps_only_succeeded() -> None:
    store = FakeStore(_run_ids("succeeded", "failed", "started", "succeeded"))
    assert available_runs(store) == ["run-00", "run-03"]
    assert store.seen == ["succeeded"]


def test_available_runs_filters_even_if_store_ignores_arg() -> None:
    store = FakeStore(_run_ids("succeeded", "failed"), honor_status_arg=False)
    assert available_runs(store) == ["run-00"]


def test_available_runs_skips_daily_job_runs() -> None:
    rows = [
        {
            "run_id": "daily-2026-09-22",
            "status": "succeeded",
            "parameter_version": "job:daily_update",
        },
        {
            "run_id": "rebalance-2026-07-31b",
            "status": "succeeded",
            "parameter_version": "params_abc",
        },
        {"run_id": "legacy", "status": "succeeded"},
    ]
    assert available_runs(FakeStore(rows)) == ["rebalance-2026-07-31b", "legacy"]


def test_available_runs_keeps_mixed_feature_versions() -> None:
    rows = [
        {"run_id": "old", "status": "succeeded", "feature_version": "factor_v1"},
        {"run_id": "adjusted", "status": "succeeded", "feature_version": "factor_adj_v2"},
    ]
    assert available_runs(FakeStore(rows)) == ["old", "adjusted"]


def test_available_runs_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="run_id"):
        available_runs(FakeStore([{"status": "succeeded"}]))
    with pytest.raises(ValueError, match="must return a list"):
        available_runs(FakeStore("nope"))  # type: ignore[arg-type]


def test_run_filters_support_all_factor_version_and_oos_year() -> None:
    rows = [
        {"run_id": "oos-2024-a", "feature_version": "factor_v4"},
        {"run_id": "oos-2023-b", "feature_version": "factor_v4"},
        {"run_id": "oos-2024-c", "feature_version": "factor_adj_v2"},
        {"run_id": "rebalance-live", "feature_version": "factor_v4"},
    ]
    assert _filter_run_rows(rows) == rows
    assert [row["run_id"] for row in _filter_run_rows(rows, "factor_v4", "2024")] == [
        "oos-2024-a"
    ]
    assert [row["run_id"] for row in _filter_run_rows(rows, ALL_FILTER, "2024")] == [
        "oos-2024-a",
        "oos-2024-c",
    ]
    assert _oos_year(rows[-1]) is None


def test_main_happy_path_dispatches_first_page() -> None:
    st = FakeSt()
    store = FakeStore(_run_ids("succeeded", "failed", "succeeded"))
    seen: list = []

    def _loader(name):
        def load(run_id, as_of):
            seen.append(("load", name, run_id, as_of))
            return name

        return load

    def _page(name):
        def render(s, payload):
            seen.append(("render", name, payload))

        return render

    loaders = {name: _loader(name) for name in PAGES}
    pages = {name: _page(name) for name in PAGES}

    main(st=st, store=store, load_settings_fn=lambda: object(), loaders=loaders, pages=pages)

    selectbox = next(
        c for c in st.calls if c[0] == "selectbox" and c[1] == "研究 run（僅顯示已完成）"
    )
    assert selectbox[2] == ["run-00", "run-02"]
    assert ("load", "總覽", "run-00", "") in seen
    assert ("render", "總覽", "總覽") in seen
    assert not [c for c in st.calls if c[0] in ("info", "error")]


def test_main_labels_runs_with_their_feature_versions() -> None:
    st = FakeSt()
    rows = [
        {
            "run_id": "oos-2020-b2",
            "status": "succeeded",
            "feature_version": "factor_adj_v2",
        },
        {
            "run_id": "oos-2020-b2-stable-schema",
            "status": "succeeded",
            "feature_version": "factor_adj_pit_v3_stable",
        },
    ]
    seen: list[str] = []
    main(
        st=st,
        store=FakeStore(rows),
        load_settings_fn=lambda: object(),
        loaders={"總覽": lambda run_id, _as_of: seen.append(run_id) or "payload"},
        pages={"總覽": lambda _st, _payload: None},
    )

    selectbox = next(
        call
        for call in st.calls
        if call[0] == "selectbox" and call[1] == "研究 run（僅顯示已完成）"
    )
    format_func = selectbox[3]["format_func"]
    assert format_func("oos-2020-b2") == "oos-2020-b2 [factor_adj_v2]"
    assert format_func("oos-2020-b2-stable-schema") == (
        "oos-2020-b2-stable-schema [factor_adj_pit_v3_stable]"
    )
    assert seen == ["oos-2020-b2"]


def test_main_filters_run_selector_by_factor_and_oos_year() -> None:
    st = FakeSt()
    st.selected_selectboxes = {
        "依 Factor Version 篩選": "factor_v4",
        "依 OOS 年份篩選": "2023",
    }
    rows = [
        {"run_id": "oos-2024-v4", "status": "succeeded", "feature_version": "factor_v4"},
        {"run_id": "oos-2023-v4", "status": "succeeded", "feature_version": "factor_v4"},
        {"run_id": "oos-2023-v2", "status": "succeeded", "feature_version": "factor_adj_v2"},
        {"run_id": "rebalance-live", "status": "succeeded", "feature_version": "factor_v4"},
    ]
    seen: list[str] = []
    main(
        st=st,
        store=FakeStore(rows),
        load_settings_fn=lambda: object(),
        loaders={"總覽": lambda run_id, _as_of: seen.append(run_id) or "payload"},
        pages={"總覽": lambda _st, _payload: None},
    )
    run_select = next(
        call
        for call in st.calls
        if call[0] == "selectbox" and call[1] == "研究 run（僅顯示已完成）"
    )
    assert run_select[2] == ["oos-2023-v4"]
    assert seen == ["oos-2023-v4"]


def test_main_offers_month_selector_for_portfolio() -> None:
    st = FakeSt()
    st.selected_page = "投組"
    store = FakeHoldingsStore(_run_ids("succeeded"))
    seen: list = []
    main(
        st=st,
        store=store,
        load_settings_fn=lambda: object(),
        loaders={"投組": lambda run_id, as_of: seen.append((run_id, as_of)) or "holdings"},
        pages={"投組": lambda _st, payload: seen.append(payload)},
    )
    month_select = next(
        call
        for call in st.calls
        if call[0] == "selectbox" and call[1] == "持股年月（只顯示權重大於 0 的持股）"
    )
    assert month_select[2] == ["2024-01", "2024-02"]
    assert month_select[3]["index"] == 1
    assert seen == [("run-00", "2024-02"), "holdings"]


def test_main_empty_states_never_raise() -> None:
    def no_source() -> object:
        raise RuntimeError("no db")

    assert (
        main(st=FakeSt(), store=None, load_settings_fn=lambda: object(), store_factory=no_source)
        is None
    )
    st = FakeSt()
    main(st=st, store=FakeStore([]), load_settings_fn=lambda: object())
    assert any(c[0] == "info" for c in st.calls)

    st = FakeSt()

    def boom() -> object:
        raise RuntimeError("yaml broken")

    main(st=st, store=FakeStore(_run_ids("succeeded")), load_settings_fn=boom)
    assert any(c[0] == "error" and "設定" in c[1] for c in st.calls)


def test_main_auto_connects_when_store_missing() -> None:
    st = FakeSt()
    store = FakeStore(_run_ids("succeeded"))
    seen: list = []
    loaders = {"總覽": lambda run_id, as_of: seen.append(run_id) or "payload"}
    pages = {"總覽": lambda s, p: seen.append(p)}
    main(
        st=st,
        store=None,
        load_settings_fn=lambda: object(),
        loaders=loaders,
        pages=pages,
        store_factory=lambda: store,
    )
    assert seen == ["run-00", "payload"]
    assert not [c for c in st.calls if c[0] in ("info", "error")]


def test_main_loader_failure_becomes_error_state() -> None:
    st = FakeSt()

    def boom(run_id, as_of):
        raise RuntimeError("db down")

    rendered: list = []
    main(
        st=st,
        store=FakeStore(_run_ids("succeeded")),
        load_settings_fn=lambda: object(),
        loaders={"總覽": boom},
        pages={"總覽": lambda s, p: rendered.append(p)},
    )
    assert rendered == []
    assert any(c[0] == "error" for c in st.calls)


def test_pages_cover_five_sections() -> None:
    assert PAGES == ("總覽", "投組", "模型", "風險", "研究比較")


def test_app_module_importable_standalone() -> None:
    assert app.CONFIG_PATH.name == "config.yaml"
    assert app.SUCCEEDED == "succeeded"
