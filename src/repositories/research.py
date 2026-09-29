"""P1-09: features and predictions repository.

Same conventions as P1-06..P1-08: takes an active ``Session``, flushes but
never commits; writes are idempotent via SQLite ON CONFLICT DO UPDATE.
``rank`` is written by the caller (P2-08); this module only persists it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models.research import Feature, Prediction
from repositories._frames import to_records

FACTOR_COLUMNS = (
    "momentum_20d",
    "momentum_60d",
    "momentum_120d",
    "momentum_20d_ex_5d",
    "ma20_ma60_gap",
    "rsi14",
    "price_ma20_gap",
    "volume_ma20_ma60",
    "earnings_yield",
    "book_to_market",
    "sales_yield",
    "dividend_yield",
    "roe",
    "roa",
    "revenue_yoy",
    "operating_income_qoq",
    "accrual_assets",
    "volatility_60d",
    "beta_60d",
    "max_drawdown_120d",
    "turnover_60d",
    "foreign_net_buy_float",
    "trust_net_buy_float",
    "margin_balance_change",
    "close_60d_high",
    "log_market_cap",
    "amihud_illiquidity",
    "short_margin_ratio",
    "operating_margin",
    "revenue_mom",
)

FEATURE_COLUMNS = (
    "rebalance_date",
    "stock_id",
    "feature_version",
    *FACTOR_COLUMNS,
    "foreign_net_buy_to_issued_shares",
    "trust_net_buy_to_issued_shares",
    "missing_flag",
)

PREDICTION_COLUMNS = (
    "prediction_date",
    "stock_id",
    "run_id",
    "model_version",
    "prediction_probability",
    "rank",
)

_FEATURE_COLUMNS = frozenset(col.name for col in Feature.__table__.columns)
_FEATURE_KEY = ("rebalance_date", "stock_id", "feature_version")
_PREDICTION_COLUMNS = frozenset(col.name for col in Prediction.__table__.columns)
_PREDICTION_KEY = ("prediction_date", "stock_id", "model_version")


def _save(session: Session, entity, key: tuple[str, ...], records: list[dict]) -> int:
    statement = sqlite_insert(entity)
    update_columns = [col for col in records[0] if col not in key]
    if update_columns:
        statement = statement.on_conflict_do_update(
            index_elements=list(key),
            set_={col: statement.excluded[col] for col in update_columns},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=list(key))
    session.execute(statement, records)
    session.flush()
    return len(records)


def save_features(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update feature rows; return the number of rows written."""
    records = to_records(rows, _FEATURE_COLUMNS, _FEATURE_KEY, "Feature")
    if not records:
        return 0
    return _save(session, Feature, _FEATURE_KEY, records)


def save_predictions(session: Session, rows: pd.DataFrame) -> int:
    """Insert or update prediction rows; return the number of rows written."""
    records = to_records(rows, _PREDICTION_COLUMNS, _PREDICTION_KEY, "Prediction")
    if not records:
        return 0
    return _save(session, Prediction, _PREDICTION_KEY, records)


def load_predictions(session: Session, on: date, model_version: str) -> pd.DataFrame:
    """Load one prediction date + model version, ordered by probability DESC."""
    if not model_version or not model_version.strip():
        raise ValueError("load_predictions requires a non-empty model_version")
    statement = (
        select(
            Prediction.prediction_date,
            Prediction.stock_id,
            Prediction.run_id,
            Prediction.model_version,
            Prediction.prediction_probability,
            Prediction.rank,
        )
        .where(
            Prediction.prediction_date == on.isoformat(),
            Prediction.model_version == model_version,
        )
        .order_by(Prediction.prediction_probability.desc(), Prediction.stock_id)
    )
    result = session.execute(statement).all()
    return pd.DataFrame(result, columns=list(PREDICTION_COLUMNS))
