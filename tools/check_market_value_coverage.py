"""Check direct PIT market-value coverage before building factor_adj_pit_v3.

The denominator contains securities which pass listing status, exact-day
adjusted-price availability, nominal minimum price, and 20-row liquidity. The
market-cap gate is deliberately bypassed for this check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.dotenv import load_dotenv  # noqa: E402
from runtime.panel_data import PreparedPanelData  # noqa: E402
from services.feature_service import is_pit_v3_feature_version  # noqa: E402
from services.universe_service import build_universe  # noqa: E402
from settings import load_settings  # noqa: E402


_NEAR_CAP_LOW = 4_000_000_000.0
_NEAR_CAP_HIGH = 6_000_000_000.0
_DEFAULT_NONFORMAL_EXCLUSIONS = Path(
    "data/pit_v3_nonformal_mainboard_exclusions.csv"
)


def _load_nonformal_exclusions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load exchange-verified stock-date observations outside the formal mainboard."""
    required = {
        "signal_date",
        "stock_id",
        "market",
        "reason",
        "formal_listing_date",
        "official_source_url",
        "evidence",
    }
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read nonformal exclusions file {path}: {exc}") from exc
    exclusions: dict[tuple[str, str], dict[str, str]] = {}
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"invalid nonformal exclusions file {path}: missing columns "
                f"{sorted(required - set(reader.fieldnames or []))}"
            )
        for line_number, row in enumerate(reader, start=2):
            signal_date = (row.get("signal_date") or "").strip()
            stock_id = (row.get("stock_id") or "").strip()
            reason = (row.get("reason") or "").strip()
            source = (row.get("official_source_url") or "").strip()
            evidence = (row.get("evidence") or "").strip()
            try:
                date.fromisoformat(signal_date)
            except ValueError as exc:
                raise ValueError(
                    f"invalid signal_date on line {line_number} of {path}"
                ) from exc
            if not stock_id or not reason or not source.startswith("https://") or not evidence:
                raise ValueError(f"incomplete nonformal exclusion on line {line_number} of {path}")
            key = (signal_date, stock_id)
            if key in exclusions:
                raise ValueError(f"duplicate nonformal exclusion {key} in {path}")
            exclusions[key] = {k: (v or "").strip() for k, v in row.items() if k}
    return exclusions


def _legacy_cap_estimate(data: PreparedPanelData, stock_id: str, signal_day: str) -> float | None:
    """Estimate only for the 4-6bn missing-data diagnostic, never for v3 panels."""
    entry = data.shares.get(stock_id)
    if not isinstance(entry, dict):
        return None
    history = data.price_by_stock.get(stock_id)
    close = None
    if history is not None:
        indices = (history["trade_date"] == signal_day).nonzero()[0]
        if len(indices):
            close = float(history["close"][int(indices[0])])
    shares = entry.get("shares")
    if close is not None and isinstance(shares, (int, float)) and shares > 0:
        return close * float(shares)
    cached_cap = entry.get("market_cap")
    return float(cached_cap) if isinstance(cached_cap, (int, float)) and cached_cap > 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.pit_v3.yaml")
    parser.add_argument("--start", default="2019-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--out", default="reports/pit_v3_market_value_coverage.json")
    parser.add_argument(
        "--nonformal-exclusions",
        default=str(_DEFAULT_NONFORMAL_EXCLUSIONS),
        help="exchange-verified stock-date rows outside the formal mainboard",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    settings = load_settings(args.config)
    if not is_pit_v3_feature_version(settings.features.feature_version):
        raise ValueError("coverage check requires a factor_adj_pit_v3 feature version")
    data = PreparedPanelData(settings, args.start, args.end, f"coverage-{args.start}-{args.end}")
    try:
        nonformal_exclusions = _load_nonformal_exclusions(Path(args.nonformal_exclusions))
        months = data.month_ends()
        if not months:
            raise ValueError("no signal months found")
        data._load_prices(months)
        data._load_market_values(months)
        monthly: list[dict] = []
        total_candidates = 0
        total_covered = 0
        total_nonformal_excluded = 0
        near_candidates = 0
        near_missing = 0
        near_nonformal_excluded = 0
        nonformal_excluded_details: list[dict] = []
        near_missing_details: list[dict] = []
        missing_details: list[dict] = []
        for signal_day in months:
            stocks = data._stocks_as_of(signal_day)
            # Bypass only the market-cap gate. Keep all other universe logic
            # identical to production, including nominal price/liquidity/PIT dates.
            stocks["market_cap"] = max(settings.universe.min_market_cap_twd + 1.0, 1.0)
            pre_cap = build_universe(
                data._universe_prices(signal_day),
                stocks,
                date.fromisoformat(signal_day),
                settings,
                f"coverage-{signal_day}",
            )
            all_candidates = list(pre_cap.included_ids)
            values = data.market_values_by_day.get(signal_day, {})
            excluded_ids = [
                stock_id
                for stock_id in all_candidates
                if (signal_day, stock_id) in nonformal_exclusions
            ]
            conflicts = [stock_id for stock_id in excluded_ids if stock_id in values]
            if conflicts:
                raise RuntimeError(
                    f"nonformal exclusions conflict with exact-day market values on "
                    f"{signal_day}: {conflicts}"
                )
            excluded_set = set(excluded_ids)
            candidates = [stock_id for stock_id in all_candidates if stock_id not in excluded_set]
            missing = [stock_id for stock_id in candidates if stock_id not in values]
            total_candidates += len(candidates)
            total_covered += len(candidates) - len(missing)
            total_nonformal_excluded += len(excluded_ids)
            near_nonformal_excluded += sum(
                1
                for stock_id in excluded_ids
                if (estimated := _legacy_cap_estimate(data, stock_id, signal_day)) is not None
                and _NEAR_CAP_LOW <= estimated <= _NEAR_CAP_HIGH
            )
            nonformal_excluded_details.extend(
                {
                    "signal_date": signal_day,
                    **nonformal_exclusions[(signal_day, stock_id)],
                }
                for stock_id in excluded_ids
            )
            for stock_id in candidates:
                estimated = _legacy_cap_estimate(data, stock_id, signal_day)
                actual = values.get(stock_id)
                in_band = (
                    actual is not None and _NEAR_CAP_LOW <= actual <= _NEAR_CAP_HIGH
                ) or (
                    actual is None
                    and estimated is not None
                    and _NEAR_CAP_LOW <= estimated <= _NEAR_CAP_HIGH
                )
                if in_band:
                    near_candidates += 1
                    if actual is None:
                        near_missing += 1
                        near_missing_details.append(
                            {
                                "signal_date": signal_day,
                                "stock_id": stock_id,
                                "shares_json_estimated_market_cap": estimated,
                            }
                        )
            monthly.append(
                {
                    "signal_date": signal_day,
                    "pre_market_cap_candidates": len(candidates),
                    "market_value_covered": len(candidates) - len(missing),
                    "market_value_missing": len(missing),
                    "nonformal_mainboard_excluded": len(excluded_ids),
                    "nonformal_mainboard_excluded_stock_ids": excluded_ids,
                    "missing_stock_ids": missing,
                }
            )
            missing_details.extend(
                {"signal_date": signal_day, "stock_id": stock_id} for stock_id in missing
            )
        coverage = total_covered / total_candidates if total_candidates else 0.0
        result = {
            "feature_version": settings.features.feature_version,
            "start": args.start,
            "end": args.end,
            "months": len(months),
            "pre_market_cap_candidates": total_candidates,
            "market_value_covered": total_covered,
            "market_value_missing": total_candidates - total_covered,
            "coverage": coverage,
            "coverage_threshold": 0.99,
            "nonformal_mainboard_excluded": total_nonformal_excluded,
            "nonformal_exclusions_file": str(args.nonformal_exclusions),
            "near_4_to_6_billion_candidates": near_candidates,
            "near_4_to_6_billion_missing": near_missing,
            "near_4_to_6_billion_nonformal_mainboard_excluded": near_nonformal_excluded,
            "near_band_estimation_note": (
                "Exchange-verified stock-date observations outside the formal mainboard are excluded "
                "before checking near-band estimates. For remaining rows, shares.json is used "
                "only as a diagnostic estimate and never in v3 panels."
            ),
            "monthly": monthly,
            "missing_details": missing_details,
            "nonformal_mainboard_excluded_details": nonformal_excluded_details,
            "near_4_to_6_billion_missing_details": near_missing_details,
            "passed": coverage >= 0.99 and near_missing == 0,
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in (
            "pre_market_cap_candidates", "market_value_covered", "market_value_missing",
            "coverage", "nonformal_mainboard_excluded", "near_4_to_6_billion_candidates",
            "near_4_to_6_billion_missing", "passed"
        )}, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    finally:
        data.close()


if __name__ == "__main__":
    raise SystemExit(main())
