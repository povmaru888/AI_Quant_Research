"""P5-03: data quality audit (SDD section 16 operations).

Read-only audit over the SQLite store for one ``as_of`` date: missing
prices, duplicate keys, invalid prices, financial PIT violations, and
feature coverage. Blockers fail
``assert_no_blockers`` so the signal job refuses to run on a broken
dataset; warnings only surface in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models.market import Financial, Institutional, Price
from models.research import Feature
from models.security import Stock

COVERAGE_WARNING_THRESHOLD = 0.9
_REPORT_PERIOD_RE = re.compile(r"^\d{4}Q[1-4]$")


@dataclass(frozen=True)
class DataQualityIssue:
    """One audit finding; severity is 'blocker' or 'warning'."""

    check: str
    severity: str
    detail: str
    count: int


@dataclass(frozen=True)
class DataQualityReport:
    """Audit outcome for one as_of date; passed means no blockers."""

    as_of: str
    issues: tuple[DataQualityIssue, ...] = field(default_factory=tuple)
    passed: bool = True


def audit_data_quality(as_of: str, session: Session) -> DataQualityReport:
    """Audit prices, financials, and features up to ``as_of`` (read-only)."""
    _require_as_of(as_of)
    issues: list[DataQualityIssue] = []
    issues.extend(_check_missing_prices(as_of, session))
    issues.extend(_check_duplicate_keys(as_of, session))
    issues.extend(_check_invalid_prices(as_of, session))
    issues.extend(_check_financial_pit(as_of, session))
    issues.extend(_check_feature_coverage(as_of, session))
    passed = all(issue.severity != "blocker" for issue in issues)
    return DataQualityReport(as_of=as_of, issues=tuple(issues), passed=passed)


def assert_no_blockers(report: DataQualityReport) -> None:
    """Raise if the report contains blockers; the signal job calls this."""
    blockers = [issue for issue in report.issues if issue.severity == "blocker"]
    if blockers:
        details = "; ".join(f"{issue.check}: {issue.detail}" for issue in blockers)
        raise ValueError(f"data quality blockers: {details}")


def _require_as_of(as_of: str) -> None:
    if not isinstance(as_of, str):
        raise ValueError(f"invalid as_of: must be a string, got {as_of!r}")
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(f"invalid as_of: not a calendar date: {as_of!r}") from exc


def _trading_days(as_of: str, session: Session) -> list[str]:
    rows = session.execute(
        select(Price.trade_date)
        .where(Price.trade_date <= as_of)
        .distinct()
        .order_by(Price.trade_date)
    ).all()
    return [row[0] for row in rows]


def _check_missing_prices(as_of: str, session: Session) -> list[DataQualityIssue]:
    stocks = [row[0] for row in session.execute(select(Stock.stock_id).order_by(Stock.stock_id))]
    days = _trading_days(as_of, session)
    if not stocks or not days:
        return [
            DataQualityIssue(
                check="missing_prices",
                severity="blocker",
                detail="no stocks or no trading days to audit",
                count=0,
            )
        ]
    present = {
        (row[0], row[1])
        for row in session.execute(
            select(Price.stock_id, Price.trade_date).where(Price.trade_date <= as_of)
        )
    }
    missing = [(s, d) for s in stocks for d in days if (s, d) not in present]
    if not missing:
        return []
    empty_stocks = sorted({s for s in stocks if all((s, d) in missing for d in days)})
    issues = []
    if empty_stocks:
        issues.append(
            DataQualityIssue(
                check="missing_prices",
                severity="blocker",
                detail=f"stocks with zero bars: {empty_stocks}",
                count=len(empty_stocks),
            )
        )
    rest = len(missing) - len(empty_stocks) * len(days)
    if rest:
        issues.append(
            DataQualityIssue(
                check="missing_prices",
                severity="warning",
                detail=f"{rest} missing stock-day bars on or before {as_of}",
                count=rest,
            )
        )
    return issues


def _check_duplicate_keys(as_of: str, session: Session) -> list[DataQualityIssue]:
    """Probe PK groups; the DDL guarantees zero, so any hit is a warning.

    ``as_of`` scopes the daily tables; financials carry their own
    available dates and are probed in full (PIT versioning by
    available_date is legal, exact key duplicates are not).
    """
    probes = [
        ("prices", (Price.trade_date, Price.stock_id), Price.trade_date <= as_of),
        (
            "institutional",
            (Institutional.trade_date, Institutional.stock_id),
            Institutional.trade_date <= as_of,
        ),
        (
            "financials",
            (Financial.stock_id, Financial.report_period, Financial.available_date),
            None,
        ),
    ]
    issues = []
    for name, keys, scope in probes:
        statement = select(*keys, func.count().label("n")).group_by(*keys).having(func.count() > 1)
        if scope is not None:
            statement = statement.where(scope)
        hits = session.execute(statement).all()
        if hits:
            sample = [tuple(hit[:-1]) for hit in hits[:5]]
            issues.append(
                DataQualityIssue(
                    check="duplicate_keys",
                    severity="warning",
                    detail=f"{name}: {len(hits)} duplicated keys, e.g. {sample}",
                    count=len(hits),
                )
            )
    return issues


def _check_invalid_prices(as_of: str, session: Session) -> list[DataQualityIssue]:
    rows = session.execute(
        select(Price.stock_id, Price.trade_date)
        .where(Price.trade_date <= as_of)
        .where(
            (Price.open <= 0)
            | (Price.high <= 0)
            | (Price.low <= 0)
            | (Price.close <= 0)
            | (Price.low > Price.high)
            | (Price.high < Price.open)
            | (Price.high < Price.close)
            | (Price.low > Price.open)
            | (Price.low > Price.close)
        )
    ).all()
    if not rows:
        return []
    sample = [(row[0], row[1]) for row in rows[:5]]
    return [
        DataQualityIssue(
            check="invalid_prices",
            severity="blocker",
            detail=f"{len(rows)} bars violate OHLC positivity/range, e.g. {sample}",
            count=len(rows),
        )
    ]


def _check_financial_pit(as_of: str, session: Session) -> list[DataQualityIssue]:
    rows = session.execute(
        select(Financial.stock_id, Financial.report_period, Financial.available_date).where(
            Financial.available_date < Financial.announcement_date
        )
    ).all()
    issues = []
    if rows:
        sample = [(row[0], row[1]) for row in rows[:5]]
        issues.append(
            DataQualityIssue(
                check="financial_pit_violation",
                severity="blocker",
                detail=f"{len(rows)} reports available before announcement, e.g. {sample}",
                count=len(rows),
            )
        )
    bad_periods = session.execute(
        select(Financial.stock_id, Financial.report_period).distinct()
    ).all()
    malformed = sorted(
        {(row[0], row[1]) for row in bad_periods if not _REPORT_PERIOD_RE.match(row[1] or "")}
    )
    if malformed:
        issues.append(
            DataQualityIssue(
                check="financial_report_period",
                severity="warning",
                detail=f"malformed report periods, e.g. {malformed[:5]}",
                count=len(malformed),
            )
        )
    return issues


def _check_feature_coverage(as_of: str, session: Session) -> list[DataQualityIssue]:
    latest = session.execute(
        select(func.max(Feature.rebalance_date)).where(Feature.rebalance_date <= as_of)
    ).scalar()
    if latest is None:
        return [
            DataQualityIssue(
                check="feature_coverage",
                severity="blocker",
                detail=f"no features on or before {as_of}",
                count=0,
            )
        ]
    factor_columns = [
        col.name
        for col in Feature.__table__.columns
        if col.name not in ("rebalance_date", "stock_id", "feature_version", "missing_flag")
    ]
    rows = session.execute(
        select(*[getattr(Feature, col) for col in factor_columns]).where(
            Feature.rebalance_date == latest
        )
    ).all()
    if not rows:
        return [
            DataQualityIssue(
                check="feature_coverage",
                severity="blocker",
                detail=f"no feature rows for latest date {latest}",
                count=0,
            )
        ]
    filled = sum(1 for row in rows for value in row if value is not None)
    coverage = filled / (len(rows) * len(factor_columns))
    if coverage < COVERAGE_WARNING_THRESHOLD:
        return [
            DataQualityIssue(
                check="feature_coverage",
                severity="warning",
                detail=f"coverage {coverage:.3f} below {COVERAGE_WARNING_THRESHOLD} for {latest}",
                count=len(rows),
            )
        ]
    return []
