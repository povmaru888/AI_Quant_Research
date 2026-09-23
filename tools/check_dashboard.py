"""Live dashboard check: all five pages render against the real database.

Uses the real loaders/pages with a recording stand-in for Streamlit
(the same call path `streamlit run app.py` takes). Any exception means
the dashboard is broken for that page. Not a pytest file: it needs the
production database with at least one succeeded rebalance run.

Usage:
    python tools/check_dashboard.py [--run-id rebalance-2026-07-31b]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from runtime.db_store import build_store  # noqa: E402
from settings import load_settings  # noqa: E402


class FakeSidebar:
    def __init__(self, st: FakeSt, run_id: str) -> None:
        self._st = st
        self._run_id = run_id

    def selectbox(self, label, options, **kwargs):
        assert self._run_id in list(options), f"run {self._run_id} not offered"
        return self._run_id

    def text_input(self, label, value="", **kwargs):
        return value

    def radio(self, label, options, **kwargs):
        return self._page

    def bind(self, page: str) -> None:
        self._page = page


class FakeSt:
    def __init__(self, run_id: str) -> None:
        self.calls: list = []
        self.sidebar = FakeSidebar(self, run_id)

    def _record(self, name: str, *args) -> None:
        text = " ".join(str(a)[:80] for a in args)
        self.calls.append((name, text))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self._record(name, *args)

        return method


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render all pages headlessly.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    settings = load_settings("config.yaml")
    store = build_store(settings)
    run_ids = app.available_runs(store)
    print(f"selectable runs: {run_ids}")
    run_id = args.run_id or (run_ids[0] if run_ids else None)
    if run_id is None:
        print("no succeeded rebalance runs", file=sys.stderr)
        return 1
    loaders = app._default_loaders()
    pages = app._default_pages()
    failures = 0
    for page in app.PAGES:
        st = FakeSt(run_id)
        st.sidebar.bind(page)
        try:
            payload = loaders[page](run_id, "")
            pages[page](st, payload)
        except Exception as exc:  # noqa: BLE001 - report, don't stop.
            print(f"[{page}] FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        kinds = sorted({c[0] for c in st.calls})
        print(f"[{page}] ok calls={len(st.calls)} widgets={kinds}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
