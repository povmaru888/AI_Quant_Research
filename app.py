"""P4-01: Streamlit application shell (SDD 14.1).

Thin shell only: settings load, succeeded-run selector, date filter,
five-page navigation, and actionable empty states (never a stack trace).
Page payloads come from injected ``loaders`` and rendering from injected
``pages`` so tests run without launching Streamlit. Defaults lazy-import
``services.dashboard_service`` and ``ui.*`` (land in P4-02..P4-07); while
those modules are absent the shell reports "pages not ready" instead of
crashing, and with no store it tries the database first, reporting
"no data source connected" only if that fails.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

PAGES: tuple[str, ...] = ("總覽", "投組", "模型", "風險", "研究比較")
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SUCCEEDED = "succeeded"


class RunStore(Protocol):
    """Minimal run-listing seam; the Phase 5 DB store satisfies this."""

    def list_runs(self, status: str | None = None) -> list[dict]: ...


def _available_run_rows(store: RunStore) -> list[dict]:
    """Return succeeded RESEARCH run rows, preserving store order.

    Daily sync jobs share the runs table; they are filtered by their
    ``job:`` parameter_version so the selector only offers rebalance runs.
    Stores that do not report a parameter_version keep every succeeded run.
    """
    rows = store.list_runs(status=SUCCEEDED)
    if not isinstance(rows, list):
        raise ValueError("invalid runs: list_runs must return a list")
    run_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid run row: must be a dict, got {row!r}")
        if row.get("status", SUCCEEDED) != SUCCEEDED:
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"invalid run row: bad run_id {run_id!r}")
        marker = row.get("parameter_version", "")
        if isinstance(marker, str) and marker.startswith("job:"):
            continue
        run_rows.append(row)
    return run_rows


def available_runs(store: RunStore) -> list[str]:
    """Return all succeeded RESEARCH run ids, regardless of feature version."""
    return [row["run_id"] for row in _available_run_rows(store)]


def _run_label(row: dict) -> str:
    run_id = row["run_id"]
    feature_version = row.get("feature_version")
    if isinstance(feature_version, str) and feature_version.strip():
        return f"{run_id} [{feature_version}]"
    return run_id


def _default_load_settings():
    from settings import load_settings

    return load_settings(CONFIG_PATH, env=os.environ)


def _default_loaders(store=None) -> dict[str, Callable[[str, str], object]]:
    from services import dashboard_service as dashboard

    dashboard_store = store if store is not None else _dashboard_store()
    return {
        "總覽": lambda run_id, as_of: dashboard.get_overview(run_id, dashboard_store),
        "投組": lambda run_id, as_of: dashboard.get_holdings(run_id, as_of, dashboard_store),
        "模型": lambda run_id, as_of: dashboard.get_model_data(run_id, dashboard_store),
        "風險": lambda run_id, as_of: dashboard.get_risk(run_id, dashboard_store),
        "研究比較": lambda run_id, as_of: dashboard.get_comparison(run_id, dashboard_store),
    }


def _default_pages() -> dict[str, Callable]:
    from ui import comparison_page, model_page, overview_page, portfolio_page, risk_page

    # Page convention is (payload, st); the shell dispatches (st, payload).
    return {
        "總覽": lambda st, payload: overview_page.render_overview(payload, st),
        "投組": lambda st, payload: portfolio_page.render_portfolio(payload, st),
        "模型": lambda st, payload: model_page.render_model(payload, st),
        "風險": lambda st, payload: risk_page.render_risk(payload, st),
        "研究比較": lambda st, payload: comparison_page.render_comparison(payload, st),
    }


def _dashboard_store():
    """Return the database-backed store (runtime wiring)."""
    from runtime.db_store import build_store

    return build_store(_default_load_settings())


def main(
    st=None,
    store: RunStore | None = None,
    load_settings_fn: Callable[[], object] | None = None,
    loaders: dict[str, Callable[[str, str], object]] | None = None,
    pages: dict[str, Callable] | None = None,
    store_factory: Callable[[], RunStore] | None = None,
) -> None:
    """Render the shell; all failures surface as page states, never raises."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    st.title("台股多因子量化交易")
    try:
        settings = (load_settings_fn or _default_load_settings)()
    except Exception as exc:
        st.error(f"設定載入失敗：{exc}")
        return
    if store is None:
        try:
            store = (store_factory or _dashboard_store)()
        except Exception:
            st.info("尚未連接資料來源：請先完成資料同步後再回來查看。")
            return
    try:
        run_rows = _available_run_rows(store)
        run_ids = [row["run_id"] for row in run_rows]
        run_labels = {row["run_id"]: _run_label(row) for row in run_rows}
    except Exception as exc:
        st.error(f"讀取研究紀錄失敗：{exc}")
        return
    if not run_ids:
        st.info("尚無已完成的研究 run：請先執行月度訊號 job。")
        return
    run_id = st.sidebar.selectbox(
        "研究 run（僅顯示已完成）",
        run_ids,
        format_func=lambda value: run_labels[value],
    )
    page = st.sidebar.radio("頁面", list(PAGES))
    try:
        if page == "投組" and callable(getattr(store, "list_holding_dates", None)):
            from services.dashboard_service import get_holding_months

            months = get_holding_months(run_id, store)
            if not months:
                st.info("此研究 run 尚無可查詢的持股月份。")
                return
            as_of = st.sidebar.selectbox(
                "持股年月（只顯示權重大於 0 的持股）",
                months,
                index=len(months) - 1,
            )
        else:
            as_of = st.sidebar.text_input("資料截止日（空白為最新）", "")
        active_loaders = loaders if loaders is not None else _default_loaders(store)
        active_pages = pages if pages is not None else _default_pages()
        payload = active_loaders[page](run_id, as_of)
        active_pages[page](st, payload)
    except Exception as exc:
        st.error(f"頁面載入失敗：{exc}")


if __name__ == "__main__":
    main()
