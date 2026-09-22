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
- ``load_stocks`` derives ``market_cap`` as latest close × latest
  ``float_shares`` (both point-in-time); stocks without either are excluded
  downstream with reason ``market_cap:missing``. ``flags`` is empty for all
  stocks: no KY/disposal/full-cash-settlement source is wired yet, so that
  universe gate currently passes everything (known gap, documented).
- First-run simplifications (documented, fail-safe): current holdings and
  previous positions are empty, portfolio value is the seed capital, model
  explainability (SHAP/importance/IC) is ``None``, and the cost comparison
  is an empty scenario frame. Performance NAV is replayed from the run's
  own orders via ``run_backtest``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select

from database import create_engine_from_settings, session_scope
from models.market import Financial, Institutional, Price
from models.research import Feature, Order, PipelineRun, Prediction, Signal
from models.security import Stock
from repositories import fundamentals as fundamentals_repo
from repositories import prices as prices_repo
from repositories import research as research_repo
from repositories import runs as runs_repo
from repositories import stocks as stocks_repo
from repositories import trading as trading_repo
from services.backtest_service import run_backtest
from services.dashboard_service import HOLDING_COLUMNS
from settings import Settings

TAIEX_ID = "TAIEX"
UNKNOWN_MARKET = "UNKNOWN"
INITIAL_CAPITAL = 30_000_000.0

_PRICE_LOAD_COLUMNS = (
    "stock_id",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
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
    return DbStore(get_engine(settings), settings)


def _model_version_for(signal_date: str) -> str:
    """Mirror the research model_version rule: xgb_<as_of YYYYMM>."""
    return f"xgb_{signal_date[:4]}{signal_date[5:7]}"


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
            statement = select(PipelineRun.run_id, PipelineRun.status).order_by(
                PipelineRun.run_time, PipelineRun.run_id
            )
            if status is not None:
                statement = statement.where(PipelineRun.status == status)
            rows = session.execute(statement).all()
        return [{"run_id": run_id, "status": row_status} for run_id, row_status in rows]

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

    # -- daily extras --------------------------------------------------------

    def load_latest_trade_date(self) -> str | None:
        with self._scope() as session:
            return session.execute(
                select(func.max(Price.trade_date)).where(Price.stock_id != TAIEX_ID)
            ).scalar()

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
        """Full price history including the TAIEX pseudo-row (market beta)."""
        with self._scope() as session:
            rows = session.execute(
                select(
                    Price.stock_id,
                    Price.trade_date,
                    Price.open,
                    Price.high,
                    Price.low,
                    Price.close,
                    Price.volume,
                    Price.traded_value,
                ).order_by(Price.stock_id, Price.trade_date)
            ).all()
        return _frame(
            [dict(zip(_PRICE_LOAD_COLUMNS, row, strict=True)) for row in rows],
            _PRICE_LOAD_COLUMNS,
        )

    def load_stocks(self) -> pd.DataFrame:
        """Active stocks plus derived market_cap; TAIEX pseudo-row excluded."""
        as_of = self._require_as_of()
        as_of_date = date.fromisoformat(as_of)
        with self._scope() as session:
            actives = stocks_repo.get_active_stocks(session, as_of_date)
            caps = self._market_caps(session, as_of)
        frame = actives.loc[actives["stock_id"] != TAIEX_ID].copy()
        frame["flags"] = ""
        frame["market_cap"] = frame["stock_id"].map(caps)
        return frame.reset_index(drop=True)

    def _market_caps(self, session, as_of: str) -> dict[str, float]:
        """Latest close × latest float_shares per stock (point-in-time)."""
        closes = session.execute(
            select(Price.stock_id, func.max(Price.trade_date))
            .where(Price.trade_date <= as_of, Price.stock_id != TAIEX_ID)
            .group_by(Price.stock_id)
        ).all()
        if not closes:
            return {}
        # Per-stock latest lookup (stock count is small vs bar count).
        latest_close: dict[str, float] = {}
        for stock_id, _ in closes:
            day = session.execute(
                select(func.max(Price.trade_date)).where(
                    Price.trade_date <= as_of, Price.stock_id == stock_id
                )
            ).scalar()
            if day is None:
                continue
            close = session.execute(
                select(Price.close).where(Price.stock_id == stock_id, Price.trade_date == day)
            ).scalar()
            if close is not None:
                latest_close[stock_id] = float(close)
        floats: dict[str, float] = {}
        for stock_id in latest_close:
            day = session.execute(
                select(func.max(Institutional.trade_date)).where(
                    Institutional.trade_date <= as_of,
                    Institutional.stock_id == stock_id,
                )
            ).scalar()
            if day is None:
                continue
            shares = session.execute(
                select(Institutional.float_shares).where(
                    Institutional.stock_id == stock_id,
                    Institutional.trade_date == day,
                )
            ).scalar()
            if shares is not None:
                floats[stock_id] = float(shares)
        return {
            stock_id: latest_close[stock_id] * floats[stock_id]
            for stock_id in latest_close
            if stock_id in floats
        }

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
        return _frame(
            [dict(zip(_FIN_HISTORY_COLUMNS, row, strict=True)) for row in rows],
            _FIN_HISTORY_COLUMNS,
        )

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
        return _frame([dict(zip(columns, row, strict=True)) for row in rows], columns)

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
        return _frame([dict(zip(columns, row, strict=True)) for row in rows], columns)

    def load_returns(self) -> pd.DataFrame:
        """Log returns per stock over full history (TAIEX excluded)."""
        frame = self.load_prices()
        frame = frame.loc[frame["stock_id"] != TAIEX_ID].sort_values(["stock_id", "trade_date"])
        if frame.empty:
            return _frame([], ("stock_id", "trade_date", "log_return"))
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.loc[frame["close"] > 0]
        frame["log_return"] = frame.groupby("stock_id")["close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        frame = frame.dropna(subset=["log_return"])
        return frame.loc[:, ["stock_id", "trade_date", "log_return"]].reset_index(drop=True)

    def load_taiex(self) -> pd.DataFrame:
        with self._scope() as session:
            rows = session.execute(
                select(Price.trade_date, Price.close)
                .where(Price.stock_id == TAIEX_ID)
                .order_by(Price.trade_date)
            ).all()
        return _frame(
            [{"trade_date": day, "close": close} for day, close in rows],
            ("trade_date", "close"),
        )

    def load_next_open(self, as_of: date) -> pd.DataFrame:
        if not isinstance(as_of, date):
            raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
        with self._scope() as session:
            day = session.execute(
                select(func.min(Price.trade_date)).where(
                    Price.trade_date > as_of.isoformat(),
                    Price.stock_id != TAIEX_ID,
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
        return _frame(
            [{"stock_id": s, "trade_date": d, "open": o} for s, d, o in rows],
            ("stock_id", "trade_date", "open"),
        )

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
        return {
            "run_id": run_id_v,
            "data_end_date": data_end,
            "feature_version": feature_v,
            "model_version": model_v or _model_version_for(data_end),
            "parameter_version": param_v,
            "metrics": {},
            "oos_months": [],
        }

    # -- research saves ------------------------------------------------------

    def save_predictions(self, predictions: pd.DataFrame, model_version: str) -> int:
        run_id = self._require_run()
        frame = predictions.copy()
        frame["run_id"] = run_id
        frame["model_version"] = model_version
        with self._scope() as session:
            return research_repo.save_predictions(session, frame)

    def save_target_holdings(self, target) -> None:
        """Persist target actions onto signals; rank comes from predictions."""
        run_id = self._require_run()
        as_of = self._require_as_of()
        model_version = _model_version_for(as_of)
        with self._scope() as session:
            rank_rows = session.execute(
                select(Prediction.stock_id, Prediction.rank).where(
                    Prediction.prediction_date == as_of,
                    Prediction.model_version == model_version,
                )
            ).all()
            ranks = {stock_id: int(rank) for stock_id, rank in rank_rows}
            rows = []
            for stock_id, action in target.actions.items():
                if stock_id not in ranks:
                    raise ValueError(f"no prediction rank for {stock_id!r} on {as_of}")
                rows.append(
                    {
                        "signal_date": as_of,
                        "stock_id": stock_id,
                        "run_id": run_id,
                        "signal": action,
                        "rank": ranks[stock_id],
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

    def load_holdings(self, run_id: str, as_of: str) -> pd.DataFrame:
        summary = self.load_run_summary(run_id)
        signal_date = summary["data_end_date"]
        with self._scope() as session:
            signals = session.execute(
                select(Signal.stock_id, Signal.target_weight).where(Signal.run_id == run_id)
            ).all()
            preds = session.execute(
                select(
                    Prediction.stock_id,
                    Prediction.rank,
                    Prediction.prediction_probability,
                ).where(Prediction.run_id == run_id)
            ).all()
            feats = session.execute(
                select(
                    Feature.stock_id,
                    Feature.volatility_60d,
                    Feature.beta_60d,
                ).where(Feature.rebalance_date == signal_date)
            ).all()
        _ = as_of  # holdings are fixed per run; as_of only selects the run view.
        frame = pd.DataFrame([{"stock_id": s, "weight": float(w)} for s, w in signals])
        pred_frame = pd.DataFrame(
            [
                {"stock_id": s, "rank": int(r), "prediction_probability": float(p)}
                for s, r, p in preds
            ]
        )
        feat_frame = pd.DataFrame(
            [{"stock_id": s, "volatility_60d": v, "beta_60d": b} for s, v, b in feats]
        )
        if not frame.empty:
            frame = frame.merge(pred_frame, on="stock_id", how="left")
            frame = frame.merge(feat_frame, on="stock_id", how="left")
        for column in HOLDING_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
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
        return {
            "shap_top": None,
            "feature_importance": None,
            "monthly_ic": None,
            "prediction_dist": [float(p) for p in probs],
        }

    def load_risk(self, run_id: str) -> dict:
        self.load_run_summary(run_id)  # unknown run raises here.
        with self._scope() as session:
            weights = list(
                session.execute(select(Signal.target_weight).where(Signal.run_id == run_id))
                .scalars()
                .all()
            )
            taiex = session.execute(
                select(Price.trade_date, Price.close)
                .where(Price.stock_id == TAIEX_ID)
                .order_by(Price.trade_date)
            ).all()
        exposure = float(sum(float(w) for w in weights)) if weights else 0.0
        regime = "unknown"
        closes = [float(c) for _, c in taiex if c is not None]
        window = self._settings.portfolio.taiex_ma_window
        if len(closes) >= window and window > 0:
            regime = "above_ma60" if closes[-1] >= sum(closes[-window:]) / window else "below_ma60"
        return {
            "equity_exposure": exposure,
            "predicted_volatility": None,
            "realized_volatility": None,
            "max_drawdown": None,
            "turnover": None,
            "market_regime": regime,
            "exposure_cap": float(self._settings.portfolio.max_equity_exposure),
        }

    def load_comparison(self, run_id: str) -> pd.DataFrame:
        """No sensitivity scenarios are persisted yet; empty but valid."""
        self.load_run_summary(run_id)  # unknown run raises here.
        return pd.DataFrame({"scenario": pd.Series(dtype=str)})

    def load_performance(self, run_id: str) -> pd.DataFrame:
        """Replay the run's own orders into a (date, nav) frame."""
        self.load_run_summary(run_id)  # unknown run raises here.
        with self._scope() as session:
            order_rows = session.execute(
                select(
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
                )
                .where(Order.run_id == run_id)
                .order_by(Order.execution_date, Order.order_id)
            ).all()
            weight_rows = session.execute(
                select(Signal.stock_id, Signal.target_weight).where(Signal.run_id == run_id)
            ).all()
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
                    "target_shares": int(q),
                    "executed_price": float(x),
                    "broker_fee": float(f),
                    "transaction_tax": float(t),
                    "slippage_cost": float(sl),
                    "total_cost": float(tc),
                }
                for o, r, s, e, sid, side, q, x, f, t, sl, tc in order_rows
            ]
        )
        prices = self.load_prices().loc[:, ["stock_id", "trade_date", "close"]]
        result = run_backtest(orders, prices, self._initial_capital, self._settings, run_id)
        nav = result.nav
        return pd.DataFrame({"date": list(nav.index.astype(str)), "nav": list(nav.to_numpy())})

    def load_factor_ic(self, run_id: str) -> pd.DataFrame:
        """Factor IC needs persisted features; empty but valid for now."""
        self.load_run_summary(run_id)  # unknown run raises here.
        return pd.DataFrame(
            {
                "factor": pd.Series(dtype=str),
                "ic": pd.Series(dtype=float),
            }
        )
