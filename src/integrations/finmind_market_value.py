"""FinMind TaiwanStockMarketValue endpoint wrapper."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import requests

from repositories.market_values import canonical_day_hash


class FinMindMarketValueError(RuntimeError):
    """A market-value request failed or returned an invalid payload."""

    def __init__(self, message: str, *, temporary: bool = False) -> None:
        super().__init__(message)
        self.temporary = temporary


def fetch_market_value_day(
    trade_day: str,
    token: str,
    requester: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch all securities' nominal market values for one exchange day.

    Omitting ``data_id`` requests the all-market daily result and requires a
    FinMind Backer or Sponsor token.
    """
    try:
        parsed_day = date.fromisoformat(trade_day).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid trade_day: {trade_day!r}") from exc
    if not isinstance(token, str) or not token.strip():
        raise ValueError("missing FinMind token")
    try:
        response = requester(
            "https://api.finmindtrade.com/api/v4/data",
            headers={"Authorization": f"Bearer {token}"},
            params={"dataset": "TaiwanStockMarketValue", "start_date": parsed_day},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FinMindMarketValueError(
            f"market value request failed for {parsed_day}: {exc}", temporary=True
        ) from exc
    if response.status_code in (401, 403):
        raise PermissionError(
            f"FinMind rejected the TaiwanStockMarketValue request for {parsed_day} "
            f"(HTTP {response.status_code}); verify Backer/Sponsor access"
        )
    if response.status_code == 429 or response.status_code >= 500:
        error = FinMindMarketValueError(
            f"temporary FinMind error for {parsed_day}: HTTP {response.status_code}",
            temporary=True,
        )
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                error.retry_after = max(0.0, float(retry_after))  # type: ignore[attr-defined]
            except ValueError:
                pass
        raise error
    if response.status_code >= 400:
        raise FinMindMarketValueError(
            f"FinMind request failed for {parsed_day}: HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise FinMindMarketValueError(f"FinMind returned invalid JSON for {parsed_day}") from exc
    if not isinstance(payload, dict) or payload.get("status") not in ("success", 200, "200"):
        message = payload.get("msg", "unexpected response") if isinstance(payload, dict) else "invalid response"
        raise FinMindMarketValueError(f"FinMind response error for {parsed_day}: {message}")
    raw_rows = payload.get("data")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise FinMindMarketValueError(f"FinMind returned no market values for trading day {parsed_day}")
    frame = pd.DataFrame(raw_rows)
    required = {"date", "stock_id", "market_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FinMindMarketValueError(
            f"FinMind schema changed for {parsed_day}; missing columns {missing}"
        )
    frame = frame.loc[:, ["date", "stock_id", "market_value"]].rename(
        columns={"date": "trade_date"}
    )
    if frame["trade_date"].isna().any() or frame["stock_id"].isna().any():
        raise FinMindMarketValueError(f"FinMind response contains missing date/ID for {parsed_day}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    if frame["trade_date"].ne(parsed_day).any():
        raise FinMindMarketValueError(f"FinMind response contains an unexpected date for {parsed_day}")
    if frame["stock_id"].isna().any() or frame["stock_id"].duplicated().any():
        raise FinMindMarketValueError(f"FinMind response contains missing/duplicate IDs for {parsed_day}")
    invalid = frame["market_value"].isna() | frame["market_value"].lt(0)
    if invalid.any():
        examples = frame.loc[invalid, ["stock_id", "market_value"]].head(3).to_dict("records")
        raise FinMindMarketValueError(
            f"FinMind response contains {int(invalid.sum())} missing/negative values for {parsed_day}; "
            f"examples={examples}"
        )
    source_hash = canonical_day_hash(frame)
    zero_rows = int(frame["market_value"].eq(0).sum())
    frame = frame.loc[frame["market_value"].gt(0)].reset_index(drop=True)
    if frame.empty:
        raise FinMindMarketValueError(
            f"FinMind returned no positive market values for trading day {parsed_day}"
        )
    # Zero-value instruments (primarily warrants and rights) have no usable
    # market capitalization. Preserve the full raw-source hash while excluding
    # these rows from PIT market-cap snapshots.
    frame.attrs["source_content_hash"] = source_hash
    frame.attrs["source_row_count"] = len(raw_rows)
    frame.attrs["zero_market_value_rows"] = zero_rows
    return frame
