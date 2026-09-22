"""P3-05 acceptance: daily update CLI job (mocked store and sync)."""

from __future__ import annotations

import pandas as pd

from jobs.daily_update import main, run_daily_update
from services.sync_service import FeedResult, SyncSummary


class FakeStore:
    def __init__(self, latest: str | None = "2020-01-31") -> None:
        self.latest = latest
        self.started: dict | None = None
        self.finished: tuple[str, str | None] | None = None

    def upsert_prices(self, frame: pd.DataFrame) -> int:
        return len(frame)

    def upsert_financials(self, frame: pd.DataFrame) -> int:
        return len(frame)

    def upsert_institutional(self, frame: pd.DataFrame) -> int:
        return len(frame)

    def load_latest_trade_date(self) -> str | None:
        return self.latest

    def load_symbols(self) -> list[str]:
        return ["2330"]

    def start_run(self, metadata: dict) -> None:
        self.started = dict(metadata)

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        self.finished = (status, error)


def _feed(name: str, rows: int) -> FeedResult:
    return FeedResult(name=name, rows=rows, attempts=1, source="finmind")


def _ok_summary(run_id: str = "daily-2020-01-31") -> SyncSummary:
    return SyncSummary(
        run_id=run_id,
        start="2015-01-01",
        end="2020-01-31",
        prices=_feed("prices", 10),
        financials=_feed("financials", 5),
        institutional=_feed("institutional", 7),
    )


def _bad_summary() -> SyncSummary:
    broken = FeedResult(name="financials", rows=0, attempts=3, source="finmind", error="gone")
    return SyncSummary(
        run_id="r",
        start="s",
        end="e",
        prices=_feed("prices", 1),
        financials=broken,
        institutional=_feed("institutional", 1),
    )


def test_run_daily_update_success(settings, capsys) -> None:
    store = FakeStore()
    result = run_daily_update(
        None, settings, store, "tok", sync_fn=lambda *a, **k: _ok_summary(), run_id="run-001"
    )
    assert result["run_id"] == "run-001"
    assert result["as_of"] == "2020-01-31"
    assert result["rows"] == {"prices": 10, "financials": 5, "institutional": 7}
    assert store.finished == ("succeeded", None)


def test_run_daily_update_failure_closes_failed(settings) -> None:
    store = FakeStore()
    try:
        run_daily_update(
            None, settings, store, "tok", sync_fn=lambda *a, **k: _bad_summary(), run_id="r"
        )
    except RuntimeError as exc:
        assert "financials: gone" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert store.finished is not None and store.finished[0] == "failed"


def test_main_exit_codes(tmp_path, capsys, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "project: {name: x, timezone: Asia/Taipei, random_state: 1}\n", encoding="utf-8"
    )
    full = (
        "data: {database_url: sqlite:///x.db, finmind_token_env: FINMIND_TOKEN,"
        " price_start_date: '2015-01-01', fallback_source: yfinance}\n"
        "universe: {min_market_cap_twd: 1, min_avg_traded_value_20d_twd: 1, min_price_twd: 1,"
        " excluded_flags: []}\n"
        "features: {winsor_lower_quantile: 0.01, winsor_upper_quantile: 0.99,"
        " correlation_threshold: 0.85, volatility_window: 60, feature_version: v}\n"
        "label: {horizon_trading_days: 20, top_quantile: 0.2}\n"
        "validation: {train_years: 4, validation_years: 1, test_years: 1,"
        " purge_trading_days: 20, optuna_trials: 1}\n"
        "portfolio: {top_n: 15, hold_rank_threshold: 30, max_individual_weight: 0.1,"
        " target_annual_volatility: 0.15, max_equity_exposure: 1.0, taiex_ma_window: 60,"
        " reduced_exposure_below_ma: 0.5}\n"
        "execution: {signal_time: month_end_close, execution_time: next_trading_day_open,"
        " broker_fee_rate: 0.001425, sell_tax_rate: 0.003, slippage_rate_per_side: 0.001}\n"
    )
    config.write_text(config.read_text(encoding="utf-8") + full, encoding="utf-8")
    store = FakeStore()
    import jobs.daily_update as job

    monkeypatch.setattr(
        job,
        "run_daily_update",
        lambda *a, **k: {
            "run_id": "daily-2020-01-31",
            "as_of": "2020-01-31",
            "rows": {},
            "fallback_used": False,
        },
    )
    code = main(["--config", str(config)], store_factory=lambda settings: store)
    assert code == 0
    out = capsys.readouterr().out
    assert "run_id=daily-2020-01-31" in out
    assert "tok" not in out

    assert main(["--as-of", "not-a-date"]) == 2
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
