"""P4-01: Streamlit application shell (SDD 14.1).

Thin shell only: settings load, succeeded-run selector, date filter,
five-page navigation, and actionable empty states (never a stack trace).
Page payloads come from injected ``loaders`` and rendering from injected
``pages`` so tests run without launching Streamlit. Defaults lazy-import
``services.dashboard_service`` and ``ui.*`` (land in P4-02..P4-07); while
those modules are absent the shell reports "pages not ready" instead of
crashing, and with no store it reports "no data source connected".
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


def available_runs(store: RunStore) -> list[str]:
    """Return succeeded run ids, preserving store order."""
    rows = store.list_runs(status=SUCCEEDED)
    if not isinstance(rows, list):
        raise ValueError("invalid runs: list_runs must return a list")
    run_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid run row: must be a dict, got {row!r}")
        if row.get("status", SUCCEEDED) != SUCCEEDED:
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"invalid run row: bad run_id {run_id!r}")
        run_ids.append(run_id)
    return run_ids


def _default_load_settings():
    from settings import load_settings

    return load_settings(CONFIG_PATH, env=os.environ)


def _default_loaders() -> dict[str, Callable[[str, str], object]]:
    from services import dashboard_service as dashboard

    return {
        "總覽": lambda run_id, as_of: dashboard.get_overview(run_id, _dashboard_store()),
        "投組": lambda run_id, as_of: dashboard.get_holdings(run_id, as_of, _dashboard_store()),
        "模型": lambda run_id, as_of: dashboard.get_model_data(run_id, _dashboard_store()),
        "風險": lambda run_id, as_of: dashboard.get_risk(run_id, _dashboard_store()),
        "研究比較": lambda run_id, as_of: dashboard.get_comparison(run_id, _dashboard_store()),
    }


def _default_pages() -> dict[str, Callable]:
    from ui import comparison_page, model_page, overview_page, portfolio_page, risk_page

    return {
        "總覽": overview_page.render_overview,
        "投組": portfolio_page.render_portfolio,
        "模型": model_page.render_model,
        "風險": risk_page.render_risk,
        "研究比較": comparison_page.render_comparison,
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
) -> None:
    """Render the shell; all failures surface as page states, never raises."""
    if st is None:
        import streamlit as streamlit

        st = streamlit
    st.title("台股多因子量化交易")
    try:
        (load_settings_fn or _default_load_settings)()
    except Exception as exc:
        st.error(f"設定載入失敗：{exc}")
        return
    if store is None:
        st.info("尚未連接資料來源：請先完成資料同步後再回來查看。")
        return
    try:
        run_ids = available_runs(store)
    except Exception as exc:
        st.error(f"讀取研究紀錄失敗：{exc}")
        return
    if not run_ids:
        st.info("尚無已完成的研究 run：請先執行月度訊號 job。")
        return
    run_id = st.sidebar.selectbox("研究 run（僅顯示已完成）", run_ids)
    as_of = st.sidebar.text_input("資料截止日（空白為最新）", "")
    page = st.sidebar.radio("頁面", list(PAGES))
    try:
        active_loaders = loaders if loaders is not None else _default_loaders()
        active_pages = pages if pages is not None else _default_pages()
        payload = active_loaders[page](run_id, as_of)
        active_pages[page](st, payload)
    except Exception as exc:
        st.error(f"頁面載入失敗：{exc}")


if __name__ == "__main__":
    main()
