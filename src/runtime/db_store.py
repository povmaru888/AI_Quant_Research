"""Database-backed store for jobs, dashboard, and reports.

Phase 5 left three wiring points unimplemented (``_build_store`` in both
jobs, ``_dashboard_store`` in ``app.py``). This module fills them with one
``DbStore`` class that satisfies every store Protocol in the codebase:

- ``SyncStore`` + ``DailyStore`` (daily_update)
- ``ResearchStore`` + ``RebalanceStore`` (monthly_rebalance)
- ``RunStore`` + ``DashboardStore`` + ``ReportStore`` (app.py, report.py)

Design notes (all consequences of frozen service contracts):

- Services call snapshot loaders with **no date argument**, so the store is
  bound to one ``as_of`` date. The binding comes from ``start_run``
  metadata (``data_end_date``), which both jobs always provide; it can also
  be passed to the constructor.
- The current ``run_id`` is captured at ``start_run`` so ``save_*`` calls
  (whose signatures carry no run id) can fill the ``run_id`` columns the
  schema requires.
- Daily-job metadata (``{run_id, job, data_end_date}``) lacks the research
  run keys, so ``start_run`` defaults ``feature_version`` from settings and
  ``parameter_version`` to ``job:<name>``; the ``job`` key itself is stripped
  before hitting the repository (unknown keys raise).
- ``prices.stock_id`` has a foreign key to ``stocks``. Full-market FinMind
  frames contain stocks the seed may not know yet, so upserts create missing
  parents first (``market="UNKNOWN"``; the universe gates never read
  ``market``, and the seed overwrites them with real values later).
- Execution-shape orders (``target_shares``) and order-shape rows
  (``quantity/open_price/notional``) differ. The store derives the latter
  from executed costs: ``notional = total_cost / rate_sum``,
  ``quantity = round(notional / executed_price)``,
  ``open_price`` = executed price with slippage removed. Exact whenever the
  configured cost rates are positive.
- ``save_target_holdings`` maps ``PortfolioTarget`` actions onto the
  ``signals`` table; ``rank`` is looked up from the predictions the same run
  saved moments earlier (research always saves predictions first).
- ``load_stocks`` derives ``market_cap`` as latest close × cached shares
  (``database/shares.json`` next to the DB file, built by
  ``tools/cache_shares.py`` from yfinance; ETFs which publish no share
  count fall back to their cached market cap). Stocks without either are
  excluded downstream with reason ``market_cap:missing``. ``flags`` is
  empty for all stocks: no KY/disposal/full-cash-settlement source is
  wired yet, so that universe gate currently passes everything
  (known gap, documented).
- First-run simplifications (documented, fail-safe): current holdings and
  previous positions are empty, portfolio value is the seed capital, model
  explainability (SHAP/importance/IC) is ``None``, and the cost comparison
  is an empty scenario frame. Performance NAV is replayed from the run's
  own orders via ``run_backtest``.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, case, delete, func, select
from sqlalchemy.exc import OperationalError

from database import create_engine_from_settings, session_scope
from models.market import Financial, Institutional, MarketValue, MarketValueSyncDay, Price
from models.research import Feature, Order, PipelineRun, PortfolioDaily, Prediction, Signal
from models.security import Stock
from repositories import artifacts as artifacts_repo
from repositories import fundamentals as fundamentals_repo
from repositories import market_values as market_values_repo
from repositories import prices as prices_repo
from repositories import research as research_repo
from repositories import runs as runs_repo
from repositories import stocks as stocks_repo
from repositories import trading as trading_repo
from services.backtest_service import run_backtest
from services.benchmark_service import build_benchmark_payload
from services.dashboard_service import HOLDING_COLUMNS
from services.feature_service import (
    allows_partial_pit_market_values,
    uses_pit_market_values,
)
from settings import Settings

TAIEX_ID = "TAIEX"
UNKNOWN_MARKET = "UNKNOWN"
INITIAL_CAPITAL = 30_000_000.0
VOLATILITY_LOOKBACK_DAYS = 60

_PRICE_LOAD_COLUMNS = (
    "stock_id",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
    "volume",
    "traded_value",
)

_FIN_HISTORY_COLUMNS = (
    "stock_id",
    "report_period",
    "available_date",
    "revenue",
    "operating_income",
)


def _annualized_volatility_60d(closes: Collection[float]) -> float:
    """Annualized standard deviation of 60 adjusted daily log returns."""
    values = np.asarray(list(closes), dtype=float)
    required = VOLATILITY_LOOKBACK_DAYS + 1
    if len(values) != required or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return float("nan")
    returns = np.diff(np.log(values))
    return float(np.std(returns, ddof=1) * np.sqrt(252))

_engine_cache: dict[str, Engine] = {}


def get_engine(settings: Settings) -> Engine:
    """Return a cached Engine per database_url (safe for Streamlit reruns)."""
    key = settings.data.database_url
    engine = _engine_cache.get(key)
    if engine is None:
        engine = create_engine_from_settings(settings)
        _engine_cache[key] = engine
    return engine


def build_store(settings: Settings) -> DbStore:
    """Assemble the database-backed store for jobs and the dashboard."""
    return DbStore(
        get_engine(settings),
        settings,
        shares_outstanding=load_shares_cache(default_shares_path(settings.data.database_url)),
    )


def default_shares_path(database_url: str) -> Path | None:
    """Locate database/shares.json next to a file SQLite database."""
    if not database_url.startswith("sqlite:///"):
        return None
    raw = database_url[len("sqlite:///") :]
    if not raw or raw == ":memory:":
        return None
    return Path(raw).parent / "shares.json"


def load_shares_cache(path: Path | None) -> dict[str, dict[str, float | str]]:
    """Load {stock_id: {shares, market_cap, as_of}}; missing file -> {}."""
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _model_version_for(signal_date: str, feature_version: str) -> str:
    """Key models by signal month and feature semantics."""
    return f"xgb_{signal_date[:4]}{signal_date[5:7]}_{feature_version}"


def _frame(rows: list[dict], columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(columns))
    return frame.reset_index(drop=True)


class DbStore:
    """One store for every Protocol; see the module docstring for semantics."""

    def __init__(
        self,
        engine: Engine,
        settings: Settings,
        as_of: date | str | None = None,
        initial_capital: float = INITIAL_CAPITAL,
        shares_outstanding: dict[str, dict[str, float | str]] | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        if isinstance(as_of, date):
            as_of = as_of.isoformat()
        self._as_of = as_of
        self._run_id: str | None = None
        if not initial_capital > 0:
            raise ValueError(f"invalid initial_capital: {initial_capital!r}")
        self._initial_capital = float(initial_capital)
        self._shares = shares_outstanding or {}

    def bind(self, run_id: str, as_of: date | str | None = None) -> None:
        """Attach to an existing run without creating a run row."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"invalid run_id: {run_id!r}")
        self.get_run_status(run_id)
        self._run_id = run_id
        if as_of is not None:
            self._as_of = as_of.isoformat() if isinstance(as_of, date) else as_of

    # -- session helper ----------------------------------------------------

    def _scope(self):
        return session_scope(self._engine)

    def _require_run(self) -> str:
        if self._run_id is None:
            raise ValueError("store has no run: call start_run first")
        return self._run_id

    def _require_as_of(self) -> str:
        if self._as_of is None:
            raise ValueError("store has no as_of: bind via start_run data_end_date")
        return self._as_of

    # -- run lifecycle (jobs + research share pipeline_runs) ---------------

    def start_run(self, metadata: dict) -> None:
        record = dict(metadata)
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"invalid run metadata: bad run_id {run_id!r}")
        job = record.pop("job", "adhoc")
        record.setdefault("feature_version", self._settings.features.feature_version)
        record.setdefault("parameter_version", f"job:{job}")
        data_end = record.get("data_end_date")
        if isinstance(data_end, str):
            self._as_of = data_end
        with self._scope() as session:
            runs_repo.start_run(session, record)
        self._run_id = run_id

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._scope() as session:
            runs_repo.finish_run(session, run_id, status, error)

    def list_runs(self, status: str | None = None) -> list[dict]:
        with self._scope() as session:
            statement = select(
                PipelineRun.run_id,
                PipelineRun.status,
                PipelineRun.parameter_version,
                PipelineRun.feature_version,
            ).order_by(PipelineRun.run_time, PipelineRun.run_id)
            if status is not None:
                statement = statement.where(PipelineRun.status == status)
            rows = session.execute(statement).all()
        return [
            {
                "run_id": run_id,
                "status": row_status,
                "parameter_version": param,
                "feature_version": feature_version,
            }
            for run_id, row_status, param, feature_version in rows
        ]

    def get_run_status(self, run_id: str) -> str:
        with self._scope() as session:
            status = session.execute(
                select(PipelineRun.status).where(PipelineRun.run_id == run_id)
            ).scalar()
        if status is None:
            raise ValueError(f"unknown run: {run_id!r}")
        return status

    # -- sync (idempotent upserts, parents first for the stocks FK) --------

    def _ensure_stocks(self, session, stock_ids: list[str]) -> None:
        if not stock_ids:
            return
        existing = set(
            session.execute(select(Stock.stock_id).where(Stock.stock_id.in_(stock_ids)))
            .scalars()
            .all()
        )
        missing = [s for s in stock_ids if s not in existing]
        if missing:
            stocks_repo.upsert_stocks(
                session, pd.DataFrame({"stock_id": missing, "market": UNKNOWN_MARKET})
            )

    @staticmethod
    def _frame_ids(frame: pd.DataFrame) -> list[str]:
        if frame.empty:
            return []
        return pd.Series(frame["stock_id"]).dropna().astype(str).unique().tolist()

    def upsert_prices(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._scope() as session:
            self._ensure_stocks(session, self._frame_ids(frame))
            return prices_repo.upsert_prices(session, frame)

    def upsert_price_adj(self, frame: pd.DataFrame) -> int:
        """Fill adj columns on existing bars; dates without raw bars skip."""
        if frame.empty:
            return 0
        with self._scope() as session:
            return prices_repo.upsert_price_adj(session, frame)

    def upsert_financials(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._scope() as session:
            self._ensure_stocks(session, self._frame_ids(frame))
            return fundamentals_repo.upsert_financials(session, frame)

    def upsert_institutional(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._scope() as session:
            self._ensure_stocks(session, self._frame_ids(frame))
            return fundamentals_repo.upsert_institutional(session, frame)

    def upsert_market_value_day(
        self, trade_day: date, frame: pd.DataFrame, *, source_content_hash: str | None = None
    ) -> int:
        """Persist one validated FinMind market-value day and its checkpoint."""
        with self._scope() as session:
            self._ensure_stocks(session, self._frame_ids(frame))
            return market_values_repo.upsert_market_value_day(
                session, trade_day, frame, source_content_hash=source_content_hash
            )

    def mark_market_value_sync_day(self, trade_day: str, status: str) -> None:
        """Persist a failed/running sync checkpoint without market values."""
        with self._scope() as session:
            market_values_repo.mark_sync_day(session, trade_day, status)

    def load_completed_market_value_days(self, start: str, end: str) -> set[str]:
        with self._scope() as session:
            return market_values_repo.load_completed_days(session, start, end)

    # -- daily extras --------------------------------------------------------

    def load_latest_trade_date(self) -> str | None:
        with self._scope() as session:
            return session.execute(
                select(func.max(Price.trade_date)).where(Price.stock_id != TAIEX_ID)
            ).scalar()

    def trade_days(self, start: str | None = None, end: str | None = None) -> list[str]:
        """Sorted distinct stock trading days in [start, end] (TAIEX excluded)."""
        with self._scope() as session:
            statement = select(Price.trade_date).where(Price.stock_id != TAIEX_ID).distinct()
            if start is not None:
                statement = statement.where(Price.trade_date >= start)
            if end is not None:
                statement = statement.where(Price.trade_date <= end)
            return sorted(row[0] for row in session.execute(statement).all())

    def load_adjusted_coverage(self, on: date | str) -> tuple[int, int]:
        """Return raw and fully adjusted stock-bar counts for one date."""
        day = on.isoformat() if isinstance(on, date) else date.fromisoformat(on).isoformat()
        has_adjusted_ohlc = (
            (Price.open_adj > 0)
            & (Price.high_adj > 0)
            & (Price.low_adj > 0)
            & (Price.close_adj > 0)
        )
        with self._scope() as session:
            raw, adjusted = session.execute(
                select(
                    func.count(),
                    func.coalesce(func.sum(case((has_adjusted_ohlc, 1), else_=0)), 0),
                ).where(Price.trade_date == day, Price.stock_id != TAIEX_ID)
            ).one()
        return int(raw), int(adjusted)

    def load_symbols(self) -> list[str]:
        with self._scope() as session:
            return list(
                session.execute(
                    select(Stock.stock_id)
                    .where(
                        Stock.stock_id != TAIEX_ID,
                        Stock.delisted_date.is_(None),
                    )
                    .order_by(Stock.stock_id)
                )
                .scalars()
                .all()
            )

    # -- research loads ------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        """Raw and adjusted history; raw TAIEX is the market index.

        Stock research must use ``*_adj``. Raw OHLC remains available only
        for actual order fills and conversion to adjusted backtest units.
        """
        with self._scope() as session:
            rows = session.execute(
                select(
                    Price.stock_id,
                    Price.trade_date,
                    Price.open,
                    Price.high,
                    Price.low,
                    Price.close,
                    Price.open_adj,
                    Price.high_adj,
                    Price.low_adj,
                    Price.close_adj,
                    Price.volume,
                    Price.traded_value,
                ).order_by(Price.stock_id, Price.trade_date)
            ).all()
        return _frame(rows, _PRICE_LOAD_COLUMNS)

    def load_stocks(self) -> pd.DataFrame:
        """Active stocks plus derived market_cap; TAIEX pseudo-row excluded."""
        as_of = self._require_as_of()
        as_of_date = date.fromisoformat(as_of)
        with self._scope() as session:
            actives = stocks_repo.get_active_stocks(session, as_of_date)
            caps = (
                self._pit_market_caps(session, as_of)
                if uses_pit_market_values(self._settings.features.feature_version)
                else self._market_caps(session, as_of)
            )
        frame = actives.loc[actives["stock_id"] != TAIEX_ID].copy()
        frame["flags"] = ""
        frame["market_cap"] = frame["stock_id"].map(caps)
        return frame.reset_index(drop=True)

    def load_market_value_snapshot(self) -> pd.DataFrame:
        """Load direct PIT market values for this store's signal date."""
        as_of = date.fromisoformat(self._require_as_of())
        with self._scope() as session:
            return market_values_repo.load_market_value_snapshot(
                session,
                as_of,
                include_partial=allows_partial_pit_market_values(
                    self._settings.features.feature_version
                ),
            )

    def _pit_market_caps(self, session, as_of: str) -> dict[str, float]:
        statuses = (
            ("succeeded", "partial")
            if allows_partial_pit_market_values(self._settings.features.feature_version)
            else ("succeeded",)
        )
        rows = session.execute(
            select(MarketValue.stock_id, MarketValue.market_value)
            .join(MarketValueSyncDay, MarketValueSyncDay.trade_date == MarketValue.trade_date)
            .where(MarketValue.trade_date == as_of)
            .where(MarketValueSyncDay.status.in_(statuses))
        ).all()
        return {stock_id: float(market_value) for stock_id, market_value in rows}

    def _market_caps(self, session, as_of: str) -> dict[str, float]:
        """Latest nominal close times cached shares (ETF cached-cap fallback).

        Market-cap and minimum-price gates use tradeable nominal units;
        adjusted history is used for returns, labels and factor signals.
        Stocks with neither get no cap and are excluded downstream with
        reason ``market_cap:missing``.
        """
        if not self._shares:
            return {}
        latest_close = (
            select(Price.close)
            .where(
                Price.stock_id == Stock.stock_id,
                Price.trade_date <= as_of,
            )
            .order_by(Price.trade_date.desc())
            .limit(1)
            .correlate(Stock)
            .scalar_subquery()
        )
        rows = session.execute(
            select(Stock.stock_id, latest_close).where(Stock.stock_id != TAIEX_ID)
        ).all()
        caps: dict[str, float] = {}
        for stock_id, close in rows:
            entry = self._shares.get(stock_id)
            if not isinstance(entry, dict):
                continue
            shares = entry.get("shares")
            if close is not None and isinstance(shares, (int, float)) and shares > 0:
                caps[stock_id] = float(close) * float(shares)
                continue
            cached_cap = entry.get("market_cap")
            if isinstance(cached_cap, (int, float)) and cached_cap > 0:
                caps[stock_id] = float(cached_cap)
        return caps

    def load_financials_snapshot(self) -> pd.DataFrame:
        as_of = date.fromisoformat(self._require_as_of())
        with self._scope() as session:
            return fundamentals_repo.load_pit_financials(session, as_of)

    def load_financials_history(self) -> pd.DataFrame:
        as_of = self._require_as_of()
        with self._scope() as session:
            rows = session.execute(
                select(
                    Financial.stock_id,
                    Financial.report_period,
                    Financial.available_date,
                    Financial.revenue,
                    Financial.operating_income,
                )
                .where(Financial.available_date <= as_of)
                .order_by(Financial.stock_id, Financial.available_date)
            ).all()
        return _frame(rows, _FIN_HISTORY_COLUMNS)

    def load_institutional_snapshot(self) -> pd.DataFrame:
        as_of = self._require_as_of()
        columns = (
            "stock_id",
            "trade_date",
            "foreign_net_buy",
            "trust_net_buy",
            "margin_balance",
            "short_balance",
            "float_shares",
        )
        with self._scope() as session:
            latest = (
                select(
                    Institutional.stock_id,
                    func.max(Institutional.trade_date).label("day"),
                )
                .where(Institutional.trade_date <= as_of)
                .group_by(Institutional.stock_id)
                .subquery()
            )
            rows = session.execute(
                select(
                    Institutional.stock_id,
                    Institutional.trade_date,
                    Institutional.foreign_net_buy,
                    Institutional.trust_net_buy,
                    Institutional.margin_balance,
                    Institutional.short_balance,
                    Institutional.float_shares,
                )
                .join(
                    latest,
                    (Institutional.stock_id == latest.c.stock_id)
                    & (Institutional.trade_date == latest.c.day),
                )
                .order_by(Institutional.stock_id)
            ).all()
        return _frame(rows, columns)

    def load_institutional_history(self) -> pd.DataFrame:
        as_of = self._require_as_of()
        columns = (
            "stock_id",
            "trade_date",
            "foreign_net_buy",
            "trust_net_buy",
            "margin_balance",
        )
        with self._scope() as session:
            rows = session.execute(
                select(
                    Institutional.stock_id,
                    Institutional.trade_date,
                    Institutional.foreign_net_buy,
                    Institutional.trust_net_buy,
                    Institutional.margin_balance,
                )
                .where(Institutional.trade_date <= as_of)
                .order_by(Institutional.stock_id, Institutional.trade_date)
            ).all()
        return _frame(rows, columns)

    def load_returns(self, stock_ids: Collection[str] | None = None) -> pd.DataFrame:
        """Adjusted log returns for requested stocks, or all non-TAIEX stocks."""
        columns = ("stock_id", "trade_date", "close_adj")
        statement = select(Price.stock_id, Price.trade_date, Price.close_adj).where(
            Price.stock_id != TAIEX_ID
        )
        if stock_ids is not None:
            ids = list(stock_ids)
            if not ids:
                return _frame([], ("stock_id", "trade_date", "log_return"))
            statement = statement.where(Price.stock_id.in_(ids))
        with self._scope() as session:
            rows = session.execute(statement.order_by(Price.stock_id, Price.trade_date)).all()
        frame = _frame(rows, columns)
        if frame.empty:
            return _frame([], ("stock_id", "trade_date", "log_return"))
        frame["close_adj"] = pd.to_numeric(frame["close_adj"], errors="coerce")
        # Keep missing adjusted bars in the shift so a gap cannot be
        # misreported as a one-session return on the following date.
        frame.loc[frame["close_adj"] <= 0, "close_adj"] = np.nan
        previous_close = frame.groupby("stock_id")["close_adj"].shift(1)
        frame["log_return"] = np.log(frame["close_adj"] / previous_close)
        frame = frame.dropna(subset=["log_return"])
        return frame.loc[:, ["stock_id", "trade_date", "log_return"]].reset_index(drop=True)

    def load_taiex(self) -> pd.DataFrame:
        """Load the unadjusted market index; TAIEX has no adjusted OHLC."""
        with self._scope() as session:
            rows = session.execute(
                select(Price.trade_date, Price.close)
                .where(Price.stock_id == TAIEX_ID)
                .order_by(Price.trade_date)
            ).all()
        return _frame(rows, ("trade_date", "close"))

    def load_next_open(self, as_of: date) -> pd.DataFrame:
        """Load raw next-session opens for executable broker orders."""
        if not isinstance(as_of, date):
            raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
        with self._scope() as session:
            day = session.execute(
                select(func.min(Price.trade_date)).where(
                    Price.trade_date > as_of.isoformat(),
                    Price.stock_id == TAIEX_ID,
                )
            ).scalar()
            if day is None:
                rows: list = []
            else:
                rows = session.execute(
                    select(Price.stock_id, Price.trade_date, Price.open)
                    .where(Price.trade_date == day, Price.stock_id != TAIEX_ID)
                    .order_by(Price.stock_id)
                ).all()
        return _frame(rows, ("stock_id", "trade_date", "open"))

    def load_current_holdings(self) -> pd.DataFrame:
        """Empty until position carry-over is wired (first-run correct)."""
        return _frame([], ("stock_id", "shares"))

    def load_previous_positions(self) -> pd.DataFrame:
        with self._scope() as session:
            day = session.execute(select(func.max(Signal.signal_date))).scalar()
            if day is None:
                ids: list[str] = []
            else:
                ids = list(
                    session.execute(
                        select(Signal.stock_id)
                        .where(Signal.signal_date == day)
                        .where(Signal.signal.in_(("BUY", "HOLD")))
                        .order_by(Signal.stock_id)
                    )
                    .scalars()
                    .all()
                )
        return _frame([{"stock_id": s} for s in ids], ("stock_id",))

    def load_portfolio_value(self) -> float:
        return self._initial_capital

    def _artifact(self, run_id: str, kind: str):
        """Load one materialized payload; None when absent or unmigrated."""
        try:
            with self._scope() as session:
                return artifacts_repo.load_artifact(session, run_id, kind)
        except OperationalError as exc:
            if "no such table" in str(exc):
                return None
            raise

    def load_run_summary(self, run_id: str) -> dict:
        with self._scope() as session:
            row = session.execute(
                select(
                    PipelineRun.run_id,
                    PipelineRun.data_end_date,
                    PipelineRun.feature_version,
                    PipelineRun.model_version,
                    PipelineRun.parameter_version,
                ).where(PipelineRun.run_id == run_id)
            ).one_or_none()
        if row is None:
            raise ValueError(f"unknown run: {run_id!r}")
        run_id_v, data_end, feature_v, model_v, param_v = row
        payload = self._artifact(run_id, "metrics") or {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        oos_months = payload.get("oos_months", []) if isinstance(payload, dict) else []
        summary = {
            "run_id": run_id_v,
            "data_end_date": data_end,
            "feature_version": feature_v,
            "model_version": model_v or _model_version_for(data_end, feature_v),
            "parameter_version": param_v,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "oos_months": oos_months if isinstance(oos_months, list) else [],
        }
        if isinstance(payload, dict):
            for key in ("equity_curve", "monthly_returns"):
                if payload.get(key) is not None:
                    summary[key] = payload[key]
        equity_curve = summary.get("equity_curve")
        if isinstance(equity_curve, list) and equity_curve:
            days = [
                row.get("date")
                for row in equity_curve
                if isinstance(row, dict) and isinstance(row.get("date"), str)
            ]
            if days:
                with self._scope() as session:
                    taiex_rows = session.execute(
                        select(Price.trade_date, Price.close)
                        .where(
                            Price.stock_id == TAIEX_ID,
                            Price.trade_date >= min(days),
                            Price.trade_date <= max(days),
                        )
                        .order_by(Price.trade_date)
                    ).all()
                taiex = pd.DataFrame(taiex_rows, columns=["trade_date", "close"])
                summary.update(build_benchmark_payload(equity_curve, taiex))
        return summary

    # -- research saves ------------------------------------------------------

    def save_predictions(self, predictions: pd.DataFrame, model_version: str) -> int:
        run_id = self._require_run()
        frame = predictions.copy()
        frame["run_id"] = run_id
        frame["model_version"] = model_version
        if "probability" in frame.columns and "prediction_probability" not in frame.columns:
            # Service contract (xgb_service) vs schema (P1-09) naming.
            frame = frame.rename(columns={"probability": "prediction_probability"})
        with self._scope() as session:
            return research_repo.save_predictions(session, frame)

    def save_target_holdings(self, target, model_version: str | None = None) -> None:
        """Persist target actions onto signals; rank comes from predictions."""
        run_id = self._require_run()
        as_of = self._require_as_of()
        version = model_version or _model_version_for(
            as_of, self._settings.features.feature_version
        )
        with self._scope() as session:
            rank_rows = session.execute(
                select(Prediction.stock_id, Prediction.rank).where(
                    Prediction.run_id == run_id,
                    Prediction.prediction_date == as_of,
                    Prediction.model_version == version,
                )
            ).all()
            ranks = {stock_id: int(rank) for stock_id, rank in rank_rows}
            fallback_rank = max(ranks.values()) + 1 if ranks else 10**6
            rows = []
            for stock_id, action in target.actions.items():
                rows.append(
                    {
                        "signal_date": as_of,
                        "stock_id": stock_id,
                        "run_id": run_id,
                        "signal": action,
                        "rank": ranks.get(stock_id, fallback_rank),
                        "target_weight": float(target.weights.get(stock_id, 0.0)),
                    }
                )
            trading_repo.save_signals(session, pd.DataFrame(rows))

    def _to_order_rows(self, orders: pd.DataFrame, run_id: str) -> pd.DataFrame:
        """Map execution-shape orders onto the order-shape orders table."""
        cfg = self._settings.execution
        fee_rate = float(cfg.broker_fee_rate)
        tax_rate = float(cfg.sell_tax_rate)
        slip = float(cfg.slippage_rate_per_side)
        records = []
        for record in orders.to_dict("records"):
            side = record["side"]
            rate_sum = fee_rate + (tax_rate if side == "SELL" else 0.0) + slip
            if rate_sum <= 0:
                raise ValueError("cannot derive order quantities: cost rates are zero")
            total_cost = float(record["total_cost"])
            executed = float(record["executed_price"])
            if not executed > 0:
                raise ValueError(f"invalid executed_price: {executed!r}")
            quantity = int(round((total_cost / rate_sum) / executed))
            if quantity <= 0:
                raise ValueError(f"invalid derived quantity {quantity} for {record['order_id']!r}")
            if side == "BUY":
                open_price = executed / (1 + slip)
            elif side == "SELL":
                open_price = executed / (1 - slip)
            else:
                raise ValueError(f"invalid side: {side!r}")
            records.append(
                {
                    "order_id": record["order_id"],
                    "run_id": run_id,
                    "signal_date": record["signal_date"],
                    "execution_date": record["execution_date"],
                    "stock_id": record["stock_id"],
                    "side": side,
                    "quantity": quantity,
                    "open_price": open_price,
                    "executed_price": executed,
                    "notional": quantity * executed,
                    "broker_fee": float(record["broker_fee"]),
                    "transaction_tax": float(record["transaction_tax"]),
                    "slippage_cost": float(record["slippage_cost"]),
                    "total_cost": total_cost,
                }
            )
        return pd.DataFrame(records)

    def save_orders(self, orders: pd.DataFrame) -> int:
        if orders.empty:
            return 0
        run_id = self._require_run()
        with self._scope() as session:
            return trading_repo.save_orders(session, self._to_order_rows(orders, run_id))

    def save_oos_month(
        self,
        predictions: pd.DataFrame,
        target,
        orders: pd.DataFrame,
        model_version: str,
    ) -> int:
        """Atomically replace one OOS month's predictions, signals and orders."""
        run_id = self._require_run()
        as_of = self._require_as_of()
        pred = predictions.copy()
        pred["run_id"] = run_id
        pred["model_version"] = model_version
        if "probability" in pred.columns and "prediction_probability" not in pred.columns:
            pred = pred.rename(columns={"probability": "prediction_probability"})
        ranks = {
            str(row.stock_id): int(row.rank)
            for row in pred[["stock_id", "rank"]].itertuples(index=False)
        }
        fallback_rank = max(ranks.values()) + 1 if ranks else 10**6
        signal_rows = pd.DataFrame(
            [
                {
                    "signal_date": as_of,
                    "stock_id": stock_id,
                    "run_id": run_id,
                    "signal": action,
                    "rank": ranks.get(stock_id, fallback_rank),
                    "target_weight": float(target.weights.get(stock_id, 0.0)),
                }
                for stock_id, action in target.actions.items()
            ]
        )
        order_rows = self._to_order_rows(orders, run_id) if not orders.empty else pd.DataFrame()
        with self._scope() as session:
            session.execute(
                delete(Prediction).where(
                    Prediction.run_id == run_id,
                    Prediction.prediction_date == as_of,
                    Prediction.model_version == model_version,
                )
            )
            session.execute(
                delete(Signal).where(Signal.run_id == run_id, Signal.signal_date == as_of)
            )
            session.execute(delete(Order).where(Order.run_id == run_id, Order.signal_date == as_of))
            prediction_count = research_repo.save_predictions(session, pred)
            if not signal_rows.empty:
                trading_repo.save_signals(session, signal_rows)
            if not order_rows.empty:
                trading_repo.save_orders(session, order_rows)
        return prediction_count + len(signal_rows) + len(order_rows)

    # -- rebalance extras ----------------------------------------------------

    def verify_month_end(self, signal_date: date) -> bool:
        if not isinstance(signal_date, date):
            raise ValueError(f"invalid signal_date: {signal_date!r}")
        prefix = signal_date.isoformat()[:7]
        with self._scope() as session:
            latest = session.execute(
                select(func.max(Price.trade_date)).where(
                    Price.trade_date.like(f"{prefix}%"),
                    Price.stock_id != TAIEX_ID,
                )
            ).scalar()
        return latest == signal_date.isoformat()

    def replace_orders(self, run_id: str, signal_date: str, orders: pd.DataFrame) -> int:
        if orders.empty:
            return 0
        with self._scope() as session:
            return trading_repo.save_orders(session, self._to_order_rows(orders, run_id))

    # -- dashboard / report --------------------------------------------------

    def list_holding_dates(self, run_id: str) -> list[str]:
        """List signal dates available for a run, in chronological order."""
        with self._scope() as session:
            rows = session.execute(
                select(Signal.signal_date)
                .where(Signal.run_id == run_id)
                .distinct()
                .order_by(Signal.signal_date)
            ).all()
        return [str(row[0]) for row in rows]

    def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
        summary = self.load_run_summary(run_id)
        signal_date = None
        with self._scope() as session:
            if not as_of:
                signal_date = session.execute(
                    select(func.max(Signal.signal_date)).where(Signal.run_id == run_id)
                ).scalar_one_or_none()
            elif len(as_of) == 7:
                month_start = date.fromisoformat(f"{as_of}-01")
                if month_start.strftime("%Y-%m") != as_of:
                    raise ValueError(f"invalid as_of month: {as_of!r}")
                if month_start.month == 12:
                    month_end = date(month_start.year + 1, 1, 1)
                else:
                    month_end = date(month_start.year, month_start.month + 1, 1)
                signal_date = session.execute(
                    select(func.max(Signal.signal_date)).where(
                        Signal.run_id == run_id,
                        Signal.signal_date >= month_start.isoformat(),
                        Signal.signal_date < month_end.isoformat(),
                    )
                ).scalar_one_or_none()
            else:
                requested_date = date.fromisoformat(as_of)
                if requested_date.isoformat() != as_of:
                    raise ValueError(f"invalid as_of date: {as_of!r}")
                signal_date = session.execute(
                    select(func.max(Signal.signal_date)).where(
                        Signal.run_id == run_id,
                        Signal.signal_date == as_of,
                    )
                ).scalar_one_or_none()

            if signal_date is None:
                rows = []
                raw_volatility: dict[str, float] = {}
            else:
                statement = (
                    select(
                        Signal.stock_id,
                        Stock.stock_name,
                        Signal.rank,
                        Prediction.prediction_probability,
                        Signal.target_weight,
                        Feature.volatility_60d,
                        Feature.beta_60d,
                    )
                    .select_from(Signal)
                    .outerjoin(Stock, Stock.stock_id == Signal.stock_id)
                    .outerjoin(
                        Prediction,
                        and_(
                            Prediction.stock_id == Signal.stock_id,
                            Prediction.prediction_date == Signal.signal_date,
                            Prediction.run_id == Signal.run_id,
                            Prediction.model_version == summary["model_version"],
                        ),
                    )
                    .outerjoin(
                        Feature,
                        and_(
                            Feature.stock_id == Signal.stock_id,
                            Feature.rebalance_date == Signal.signal_date,
                            Feature.feature_version == summary["feature_version"],
                        ),
                    )
                    .where(
                        Signal.run_id == run_id,
                        Signal.signal_date == signal_date,
                        Signal.target_weight > 0,
                    )
                    .order_by(Signal.rank, Signal.stock_id)
                )
                rows = session.execute(statement).all()
                stock_ids = [str(row[0]) for row in rows]
                raw_volatility = {}
                if stock_ids:
                    price_rows = session.execute(
                        select(Price.stock_id, Price.trade_date, Price.close_adj)
                        .where(
                            Price.stock_id.in_(stock_ids),
                            Price.trade_date <= signal_date,
                        )
                        .order_by(Price.stock_id, Price.trade_date.desc())
                    ).all()
                    histories: dict[str, list[tuple[str, float | None]]] = {}
                    for stock_id, trade_date, close_adj in price_rows:
                        history = histories.setdefault(str(stock_id), [])
                        if len(history) < VOLATILITY_LOOKBACK_DAYS + 1:
                            history.append((str(trade_date), close_adj))
                    for stock_id, history in histories.items():
                        ordered_closes = [close for _, close in reversed(history)]
                        raw_volatility[stock_id] = _annualized_volatility_60d(ordered_closes)

        frame = pd.DataFrame(rows, columns=list(HOLDING_COLUMNS))
        if not frame.empty:
            frame["stock_id"] = frame["stock_id"].astype(str)
            frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
            frame["weight"] = pd.to_numeric(frame["weight"], errors="raise").astype(float)
            frame["volatility_60d"] = frame["stock_id"].map(raw_volatility)
        return frame.loc[:, list(HOLDING_COLUMNS)].reset_index(drop=True)

    def load_model_data(self, run_id: str) -> dict:
        self.load_run_summary(run_id)  # unknown run raises here.
        with self._scope() as session:
            probs = list(
                session.execute(
                    select(Prediction.prediction_probability)
                    .where(Prediction.run_id == run_id)
                    .order_by(Prediction.prediction_probability.desc())
                )
                .scalars()
                .all()
            )
        explain = self._artifact(run_id, "model_explain") or {}
        monthly = self._artifact(run_id, "monthly_ic")
        shap_top = explain.get("shap_top") if isinstance(explain, dict) else None
        importance = explain.get("feature_importance") if isinstance(explain, dict) else None
        return {
            "shap_top": shap_top,
            "feature_importance": importance,
            "monthly_ic": monthly if isinstance(monthly, (list, dict)) else None,
            "prediction_dist": [float(p) for p in probs],
        }

    def load_risk(self, run_id: str) -> dict:
        self.load_run_summary(run_id)  # unknown run raises here.
        with self._scope() as session:
            exposure = session.execute(
                select(PortfolioDaily.equity_exposure)
                .where(PortfolioDaily.run_id == run_id)
                .order_by(PortfolioDaily.trade_date.desc())
                .limit(1)
            ).scalar_one_or_none()
            if exposure is None:
                latest_signal_date = session.execute(
                    select(func.max(Signal.signal_date)).where(Signal.run_id == run_id)
                ).scalar_one_or_none()
                exposure = (
                    session.execute(
                        select(func.sum(Signal.target_weight)).where(
                            Signal.run_id == run_id,
                            Signal.signal_date == latest_signal_date,
                        )
                    ).scalar_one_or_none()
                    if latest_signal_date is not None
                    else None
                )
            taiex = session.execute(
                select(Price.trade_date, Price.close)
                .where(Price.stock_id == TAIEX_ID)
                .order_by(Price.trade_date)
            ).all()
        exposure = float(exposure) if exposure is not None else 0.0
        regime = "unknown"
        closes = [float(c) for _, c in taiex if c is not None]
        window = self._settings.portfolio.taiex_ma_window
        if len(closes) >= window and window > 0:
            regime = "above_ma60" if closes[-1] >= sum(closes[-window:]) / window else "below_ma60"
        stats = self._artifact(run_id, "metrics") or {}
        metrics = stats.get("metrics", {}) if isinstance(stats, dict) else {}
        predicted = None
        if len(closes) >= 60:
            trailing = pd.Series(closes[-60:]).pct_change().dropna()
            if len(trailing) >= 2 and float(trailing.std(ddof=1)) > 0:
                predicted = float(trailing.std(ddof=1) * np.sqrt(252))
        return {
            "equity_exposure": exposure,
            "predicted_volatility": predicted,
            "realized_volatility": metrics.get("realized_volatility"),
            "max_drawdown": metrics.get("max_drawdown"),
            "turnover": metrics.get("turnover"),
            "market_regime": regime,
            "exposure_cap": float(self._settings.portfolio.max_equity_exposure),
        }

    def load_comparison(self, run_id: str) -> pd.DataFrame:
        """Cost sensitivity scenarios; empty but valid until materialized."""
        self.load_run_summary(run_id)  # unknown run raises here.
        payload = self._artifact(run_id, "sensitivity")
        if isinstance(payload, list) and payload:
            return pd.DataFrame(payload).reset_index(drop=True)
        return pd.DataFrame({"scenario": pd.Series(dtype=str)})

    def load_performance(self, run_id: str) -> pd.DataFrame:
        """Load the bounded materialized curve, or replay as a fallback."""
        payload = self._artifact(run_id, "metrics")
        if isinstance(payload, dict) and isinstance(payload.get("equity_curve"), list):
            curve = pd.DataFrame(payload["equity_curve"])
            if not curve.empty and {"date", "nav"}.issubset(curve.columns):
                return curve.loc[:, ["date", "nav"]].reset_index(drop=True)
        nav = self.replay_backtest(run_id).nav
        return pd.DataFrame({"date": list(nav.index.astype(str)), "nav": list(nav.to_numpy())})

    def replay_backtest(
        self,
        run_id: str,
        prices: pd.DataFrame | None = None,
        *,
        signal_end_date: str | None = None,
    ):
        """Replay the run's orders into a BacktestResult (shared by readers)."""
        self.load_run_summary(run_id)  # unknown run raises here.
        if signal_end_date is not None:
            parsed = date.fromisoformat(signal_end_date)
            if parsed.isoformat() != signal_end_date:
                raise ValueError(f"invalid signal_end_date: {signal_end_date!r}")
        with self._scope() as session:
            order_query = select(
                Order.order_id,
                Order.run_id,
                Order.signal_date,
                Order.execution_date,
                Order.stock_id,
                Order.side,
                Order.quantity,
                Order.executed_price,
                Order.broker_fee,
                Order.transaction_tax,
                Order.slippage_cost,
                Order.total_cost,
            ).where(Order.run_id == run_id)
            if signal_end_date is not None:
                order_query = order_query.where(Order.signal_date <= signal_end_date)
            order_rows = session.execute(
                order_query.order_by(Order.execution_date, Order.order_id)
            ).all()
            weight_query = select(Signal.stock_id, Signal.target_weight).where(
                Signal.run_id == run_id
            )
            if signal_end_date is not None:
                weight_query = weight_query.where(Signal.signal_date <= signal_end_date)
            weight_rows = session.execute(weight_query).all()
        weights = {s: float(w) for s, w in weight_rows}
        orders = pd.DataFrame(
            [
                {
                    "order_id": o,
                    "run_id": r,
                    "signal_date": s,
                    "execution_date": e,
                    "stock_id": sid,
                    "side": side,
                    "target_weight": weights.get(sid, 0.0),
                    "target_shares": (0 if side == "SELL" else int(q)),
                    "executed_price": float(x),
                    "broker_fee": float(f),
                    "transaction_tax": float(t),
                    "slippage_cost": float(sl),
                    "total_cost": float(tc),
                }
                for o, r, s, e, sid, side, q, x, f, t, sl, tc in order_rows
            ]
        )
        if prices is None:
            prices = self.load_prices()
        prices = prices.loc[:, ["stock_id", "trade_date", "open", "open_adj", "close_adj"]]
        return run_backtest(orders, prices, self._initial_capital, self._settings, run_id)

    def load_factor_ic(self, run_id: str) -> pd.DataFrame:
        """Materialized per-factor IC; empty but valid until materialized."""
        self.load_run_summary(run_id)  # unknown run raises here.
        payload = self._artifact(run_id, "factor_ic")
        if isinstance(payload, list) and payload:
            frame = pd.DataFrame(payload)
            if set(frame.columns) >= {"factor", "ic"}:
                return frame.loc[:, ["factor", "ic"]].reset_index(drop=True)
        return pd.DataFrame(
            {
                "factor": pd.Series(dtype=str),
                "ic": pd.Series(dtype=float),
            }
        )
