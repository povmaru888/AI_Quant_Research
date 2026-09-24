"""P2-04: raw factor service (SDD section 9.2).

Computes the 30 SDD factor columns from a PIT snapshot plus price
history. Only data available at ``as_of`` is used; zero denominators
never yield infinities (they become NaN and raise ``missing_flag``).

MVP approximations (documented, revisit with richer ETL):
- Valuation yields use the latest announced fundamentals over current
  market cap (``as_of_close * float_shares``); negative yields are
  masked to NaN per SDD 9.3 instead of reading as cheap.
- ``revenue_yoy`` compares the latest announced revenue with the
  5th-latest (~one year of quarterly reports); ``revenue_mom`` and
  ``operating_income_qoq`` compare consecutive announcements.
- ``dividend_yield`` has no ETL source yet and stays NaN (visible in
  coverage diagnostics); the column still exists per the 30-factor spec.
- Full windows are required (e.g. 121 closes for 120d momentum);
  short windows yield NaN rather than a degraded short-window value.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

FACTOR_COLUMNS: tuple[str, ...] = (
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
    "revenue_mom",
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
)

_MARKET_ID = "TAIEX"
# Equity factors use adjusted history. The raw close is needed only for
# TAIEX, an index without a stock split/dividend adjustment series.
_PRICE_KEYS = (
    "stock_id",
    "trade_date",
    "close",
    "high_adj",
    "close_adj",
    "volume",
    "traded_value",
)
_TRADING_DAYS_PER_YEAR = 252


def _safe_div(numerator: object, denominator: object) -> float:
    """Divide, returning NaN for zero/missing/non-finite inputs."""
    try:
        num = float(numerator)  # type: ignore[arg-type]
        den = float(denominator)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    result = num / den
    return result if np.isfinite(result) else float("nan")


def _num(value: object) -> float:
    """Coerce a snapshot cell to finite float, else NaN."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _strict_mean(values: np.ndarray) -> float:
    if len(values) == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.mean(values))


def _momentum(closes: np.ndarray, lookback: int, skip: int = 0) -> float:
    if len(closes) < lookback + 1 + skip:
        return float("nan")
    return _safe_div(closes[-1 - skip], closes[-1 - skip - lookback]) - 1


def _mean_tail(values: np.ndarray, window: int) -> float:
    if len(values) < window:
        return float("nan")
    tail = values[-window:].astype(float)
    if not np.all(np.isfinite(tail)):
        return float("nan")
    return float(np.mean(tail))


def _rsi14(closes: np.ndarray) -> float:
    if len(closes) < 15:
        return float("nan")
    diffs = np.diff(closes[-15:].astype(float))
    if not np.all(np.isfinite(diffs)):
        return float("nan")
    avg_gain = float(np.mean(np.clip(diffs, 0, None)))
    avg_loss = float(np.mean(np.clip(-diffs, 0, None)))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _log_returns(closes: np.ndarray, window: int) -> np.ndarray | None:
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1) :].astype(float)
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0):
        return None
    return np.diff(np.log(tail))


def calculate_raw_features(
    snapshot: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
    financials: pd.DataFrame | None = None,
    institutional: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate the 30 raw factors for every snapshot row.

    ``financials``/``institutional`` are optional trailing histories
    (only rows with ``available_date``/``trade_date`` <= ``as_of`` are
    used); without them the trailing-window chip and growth factors
    stay NaN with ``missing_flag`` set.
    """
    if not isinstance(as_of, date):
        raise ValueError(f"invalid as_of: must be a date, got {as_of!r}")
    if not isinstance(snapshot, pd.DataFrame) or "stock_id" not in snapshot.columns:
        raise ValueError("invalid snapshot: must be a DataFrame with 'stock_id'")
    if snapshot["stock_id"].duplicated().any():
        raise ValueError("invalid snapshot: duplicate stock_id")
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("invalid prices: must be a DataFrame")
    missing = [c for c in _PRICE_KEYS if c not in prices.columns]
    if missing:
        raise ValueError(f"invalid prices: missing columns {missing}")

    as_of_str = as_of.isoformat()
    target_stocks = {str(s) for s in snapshot["stock_id"]}
    target_filter = target_stocks | {int(s) for s in target_stocks if s.isdigit()}

    # Filter before sorting the full price history, then retain only the
    # longest factor window (120-day momentum needs 121 closes) per stock.
    past_prices = prices.loc[
        (prices["trade_date"] <= as_of_str) & prices["stock_id"].isin(target_filter | {_MARKET_ID})
    ].sort_values("trade_date")
    market_rets_series = _aligned_market_returns(
        past_prices.loc[past_prices["stock_id"] == _MARKET_ID]
    )
    market_rets = market_rets_series.to_dict() if market_rets_series is not None else None
    past_prices_target = past_prices.loc[past_prices["stock_id"].isin(target_filter)]
    past_prices_target = past_prices_target.groupby("stock_id", sort=False).tail(121)
    prices_by_stock: dict[str, pd.DataFrame] = {
        str(sid): df for sid, df in past_prices_target.groupby("stock_id", sort=False)
    }

    # Pre-filter and index financials by stock_id
    fin_by_stock: dict[str, pd.DataFrame] = {}
    if financials is not None and not financials.empty and "stock_id" in financials.columns:
        fin_filtered = financials.loc[
            financials["stock_id"].isin(target_filter) & (financials["available_date"] <= as_of_str)
        ].sort_values("available_date")
        fin_filtered = fin_filtered.groupby("stock_id", sort=False).tail(5)
        fin_by_stock = {str(sid): df for sid, df in fin_filtered.groupby("stock_id", sort=False)}

    # Pre-filter and index institutional history by stock_id
    inst_by_stock: dict[str, pd.DataFrame] = {}
    if (
        institutional is not None
        and not institutional.empty
        and "stock_id" in institutional.columns
    ):
        inst_filtered = institutional.loc[
            institutional["stock_id"].isin(target_filter)
            & (institutional["trade_date"] <= as_of_str)
        ].sort_values("trade_date")
        inst_filtered = inst_filtered.groupby("stock_id", sort=False).tail(21)
        inst_by_stock = {str(sid): df for sid, df in inst_filtered.groupby("stock_id", sort=False)}

    empty_price = pd.DataFrame(columns=_PRICE_KEYS)
    rows: list[dict] = []
    for snap in snapshot.to_dict("records"):
        stock_id = str(snap["stock_id"])
        hist = prices_by_stock.get(stock_id, empty_price)
        closes = hist["close_adj"].to_numpy(dtype=float, na_value=np.nan)
        row: dict[str, object] = {"stock_id": stock_id}
        row.update(_price_factors(hist, closes, market_rets))
        row.update(_fundamental_factors(snap, fin_by_stock.get(stock_id)))
        row.update(_chip_factors(snap, inst_by_stock.get(stock_id)))
        row["turnover_60d"] = _turnover(hist, snap)
        values = np.array([row[c] for c in FACTOR_COLUMNS], dtype=float)
        row["missing_flag"] = int(bool(np.isnan(values).any()))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["stock_id", *FACTOR_COLUMNS, "missing_flag"])
    frame[list(FACTOR_COLUMNS)] = frame[list(FACTOR_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return frame


def _aligned_market_returns(market: pd.DataFrame) -> pd.Series | None:
    if market.empty:
        return None
    # TAIEX is an index, so its raw close is the intended benchmark series.
    closes = market.drop_duplicates("trade_date").set_index("trade_date")["close"]
    closes = closes.apply(pd.to_numeric, errors="coerce")
    if closes.le(0).any() or closes.isna().any():
        return None
    return np.log(closes).diff()


def _price_factors(
    hist: pd.DataFrame, closes: np.ndarray, market_rets: dict[object, float] | None
) -> dict[str, float]:
    vols = hist["volume"].to_numpy(dtype=float, na_value=np.nan)
    traded = hist["traded_value"].to_numpy(dtype=float, na_value=np.nan)
    highs = hist["high_adj"].to_numpy(dtype=float, na_value=np.nan)

    ma20 = _mean_tail(closes, 20)
    ma60 = _mean_tail(closes, 60)
    rets60 = _log_returns(closes, 60)
    out = {
        "momentum_20d": _momentum(closes, 20),
        "momentum_60d": _momentum(closes, 60),
        "momentum_120d": _momentum(closes, 120),
        "momentum_20d_ex_5d": _momentum(closes, 20, skip=5),
        "ma20_ma60_gap": _safe_div(ma20, ma60) - 1,
        "rsi14": _rsi14(closes),
        "price_ma20_gap": _safe_div(closes[-1] if len(closes) else np.nan, ma20) - 1,
        "volume_ma20_ma60": _safe_div(_mean_tail(vols, 20), _mean_tail(vols, 60)) - 1,
        "volatility_60d": float(np.std(rets60, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if rets60 is not None and np.all(np.isfinite(rets60))
        else float("nan"),
        "beta_60d": _beta(hist, rets60, market_rets),
        "max_drawdown_120d": _drawdown(closes),
        "close_60d_high": _safe_div(
            closes[-1] if len(closes) >= 60 else np.nan,
            np.max(highs[-60:]) if len(highs) >= 60 else np.nan,
        ),
        "amihud_illiquidity": _amihud(rets60, traded),
    }
    return out


def _turnover(hist: pd.DataFrame, snap: Mapping[str, object]) -> float:
    """Mean 60d volume over float shares; NaN when either leg is missing."""
    float_shares = _num(snap.get("float_shares"))
    if not np.isfinite(float_shares) or float_shares <= 0:
        return float("nan")
    vols = hist["volume"].to_numpy(dtype=float, na_value=np.nan)
    if len(vols) < 60:
        return float("nan")
    return _strict_mean(vols[-60:] / float_shares)


def _beta(
    hist: pd.DataFrame, rets60: np.ndarray | None, market_rets: dict[object, float] | None
) -> float:
    if rets60 is None or market_rets is None:
        return float("nan")
    dates = hist["trade_date"].to_numpy()
    if len(dates) < 61:
        return float("nan")
    aligned_market = np.fromiter(
        (market_rets.get(day, np.nan) for day in dates[-60:]), dtype=float, count=60
    )
    if not np.all(np.isfinite(aligned_market)) or np.std(aligned_market, ddof=1) == 0:
        return float("nan")
    covariance = float(np.cov(rets60, aligned_market, ddof=1)[0, 1])
    return _safe_div(covariance, float(np.var(aligned_market, ddof=1)))


def _drawdown(closes: np.ndarray) -> float:
    if len(closes) < 120:
        return float("nan")
    tail = closes[-120:].astype(float)
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0):
        return float("nan")
    return float(np.min(tail / np.maximum.accumulate(tail) - 1))


def _amihud(rets60: np.ndarray | None, traded: np.ndarray) -> float:
    if rets60 is None or len(traded) < 61:
        return float("nan")
    t60 = traded[-60:].astype(float)
    if not np.all(np.isfinite(t60)) or np.any(t60 <= 0):
        return float("nan")
    impact = np.abs(rets60) / t60
    if not np.all(np.isfinite(impact)):
        return float("nan")
    return float(np.mean(impact))


def _fundamental_factors(
    snap: Mapping[str, object],
    financials: pd.DataFrame | None,
    stock_id: str | None = None,
    as_of_str: str | None = None,
) -> dict[str, float]:
    # The share count is nominal, so market cap must use the nominal close.
    # Technical return factors above use close_adj exclusively.
    close = _num(snap.get("as_of_close"))
    float_shares = _num(snap.get("float_shares"))
    market_cap = close * float_shares if close > 0 and float_shares > 0 else float("nan")
    net_income = _num(snap.get("net_income"))
    equity = _num(snap.get("equity"))
    revenue = _num(snap.get("revenue"))
    assets = _num(snap.get("assets"))
    operating_cf = _num(snap.get("operating_cash_flow"))
    operating_income = _num(snap.get("operating_income"))

    earnings_yield = _safe_div(net_income, market_cap)
    book_to_market = _safe_div(equity, market_cap)
    sales_yield = _safe_div(revenue, market_cap)

    if (
        stock_id is not None
        and as_of_str is not None
        and financials is not None
        and not financials.empty
    ):
        eligible = financials.loc[
            (financials["stock_id"] == stock_id) & (financials["available_date"] <= as_of_str)
        ].sort_values("available_date")
    else:
        eligible = financials

    out = {
        "earnings_yield": earnings_yield if earnings_yield > 0 else float("nan"),
        "book_to_market": book_to_market if book_to_market > 0 else float("nan"),
        "sales_yield": sales_yield if sales_yield > 0 else float("nan"),
        "dividend_yield": float("nan"),
        "roe": _safe_div(net_income, equity),
        "roa": _safe_div(net_income, assets),
        "revenue_yoy": _trailing_change(eligible, "revenue", 5),
        "revenue_mom": _trailing_change(eligible, "revenue", 2),
        "operating_income_qoq": _trailing_change(eligible, "operating_income", 2),
        "accrual_assets": _safe_div(net_income - operating_cf, assets)
        if np.isfinite(net_income) and np.isfinite(operating_cf)
        else float("nan"),
        "log_market_cap": float(np.log(market_cap)) if np.isfinite(market_cap) else float("nan"),
        "operating_margin": _safe_div(operating_income, revenue),
    }
    return out


def _trailing_change(
    financials: pd.DataFrame | None,
    column: str,
    depth: int,
    stock_id: str | None = None,
    as_of_str: str | None = None,
) -> float:
    if financials is None or financials.empty or column not in financials.columns:
        return float("nan")
    if stock_id is not None and as_of_str is not None:
        eligible = financials.loc[
            (financials["stock_id"] == stock_id) & (financials["available_date"] <= as_of_str)
        ].sort_values("available_date")
    else:
        eligible = financials
    values = eligible[column].to_numpy(dtype=float, na_value=np.nan)
    if len(values) < depth or not np.all(np.isfinite(values[-depth:])):
        return float("nan")
    return _safe_div(values[-1], values[-depth]) - 1


def _chip_factors(
    snap: Mapping[str, object],
    institutional: pd.DataFrame | None,
    stock_id: str | None = None,
    as_of_str: str | None = None,
) -> dict[str, float]:
    out = {
        "foreign_net_buy_float": float("nan"),
        "trust_net_buy_float": float("nan"),
        "margin_balance_change": float("nan"),
        "short_margin_ratio": _safe_div(
            _num(snap.get("short_balance")), _num(snap.get("margin_balance"))
        ),
    }
    if institutional is None or institutional.empty:
        return out
    if stock_id is not None and as_of_str is not None:
        eligible = institutional.loc[
            (institutional["stock_id"] == stock_id) & (institutional["trade_date"] <= as_of_str)
        ].sort_values("trade_date")
    else:
        eligible = institutional

    if len(eligible) < 20:
        return out
    window = eligible.tail(20)
    float_shares = _num(snap.get("float_shares"))
    for factor, column in (
        ("foreign_net_buy_float", "foreign_net_buy"),
        ("trust_net_buy_float", "trust_net_buy"),
    ):
        if column in window.columns:
            total = window[column].to_numpy(dtype=float, na_value=np.nan)
            out[factor] = (
                _safe_div(float(np.sum(total)), float_shares)
                if np.all(np.isfinite(total))
                else float("nan")
            )
    if "margin_balance" in eligible.columns and len(eligible) >= 21:
        balances = eligible["margin_balance"].to_numpy(dtype=float, na_value=np.nan)
        if np.all(np.isfinite(balances[-21:])):
            out["margin_balance_change"] = _safe_div(balances[-1], balances[-21]) - 1
    return out
