"""Check direct PIT market-value coverage before building factor_adj_pit_v3.

The denominator contains securities which pass listing status, exact-day
adjusted-price availability, nominal minimum price, and 20-row liquidity. The
market-cap gate is deliberately bypassed for this check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.dotenv import load_dotenv  # noqa: E402
from runtime.panel_data import PreparedPanelData  # noqa: E402
from services.universe_service import build_universe  # noqa: E402
from settings import load_settings  # noqa: E402


_NEAR_CAP_LOW = 4_000_000_000.0
_NEAR_CAP_HIGH = 6_000_000_000.0


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
    args = parser.parse_args(argv)
    load_dotenv()
    settings = load_settings(args.config)
    if settings.features.feature_version != "factor_adj_pit_v3":
        raise ValueError("coverage check requires feature_version=factor_adj_pit_v3")
    data = PreparedPanelData(settings, args.start, args.end, f"coverage-{args.start}-{args.end}")
    try:
        months = data.month_ends()
        if not months:
            raise ValueError("no signal months found")
        data._load_prices(months)
        data._load_market_values(months)
        monthly: list[dict] = []
        total_candidates = 0
        total_covered = 0
        near_candidates = 0
        near_missing = 0
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
            candidates = list(pre_cap.included_ids)
            values = data.market_values_by_day.get(signal_day, {})
            missing = [stock_id for stock_id in candidates if stock_id not in values]
            total_candidates += len(candidates)
            total_covered += len(candidates) - len(missing)
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
            "near_4_to_6_billion_candidates": near_candidates,
            "near_4_to_6_billion_missing": near_missing,
            "near_band_estimation_note": (
                "When the PIT market value is missing, shares.json is used only to flag likely "
                "4-6bn candidates for this coverage diagnostic; it is never used in v3 panels."
            ),
            "monthly": monthly,
            "missing_details": missing_details,
            "near_4_to_6_billion_missing_details": near_missing_details,
            "passed": coverage >= 0.99 and near_missing == 0,
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in (
            "pre_market_cap_candidates", "market_value_covered", "market_value_missing",
            "coverage", "near_4_to_6_billion_candidates", "near_4_to_6_billion_missing", "passed"
        )}, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    finally:
        data.close()


if __name__ == "__main__":
    raise SystemExit(main())
