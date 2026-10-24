from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import services.price_gap_service as gap_service
from models import Base
from models.market import Price
from models.security import Stock
from services.price_gap_service import plan_price_gaps


def _price(day: str, stock_id: str, adjusted: bool = True) -> Price:
    values = {"open_adj": 10.0, "high_adj": 11.0, "low_adj": 9.0, "close_adj": 10.0}
    if not adjusted:
        values = {key: None for key in values}
    return Price(
        trade_date=day,
        stock_id=stock_id,
        open=10,
        high=11,
        low=9,
        close=10,
        volume=1000,
        traded_value=10000,
        source="test",
        **values,
    )


def test_plan_groups_raw_gaps_by_market_day_and_adjusted_keys(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'gap.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Stock(stock_id="TAIEX", market="INDEX"),
                Stock(stock_id="2330", market="TWSE", listed_date="2000-01-01"),
                Stock(stock_id="2317", market="TWSE", listed_date="2000-01-01"),
                Stock(stock_id="6488", market="TPEX", listed_date="2000-01-01"),
                _price("2026-10-21", "TAIEX", adjusted=False),
                _price("2026-10-21", "2330"),
                _price("2026-10-21", "2317"),
                _price("2026-10-21", "6488"),
                _price("2026-10-22", "TAIEX", adjusted=False),
                _price("2026-10-22", "2330"),
                _price("2026-10-22", "6488", adjusted=False),
                _price("2026-10-23", "TAIEX", adjusted=False),
                _price("2026-10-23", "2330"),
                _price("2026-10-23", "2317"),
                _price("2026-10-23", "6488"),
            ]
        )
    plan = plan_price_gaps(engine, "2026-10-21", "2026-10-23", resume=False)
    assert len(plan.raw_tasks) == 1
    assert plan.raw_tasks[0].market == "TWSE"
    assert plan.raw_tasks[0].missing_symbols == ("2317",)
    assert plan.adjusted_keys == (("6488", "2026-10-22"),)
    assert plan.official_request_count == 1


def test_successful_day_checkpoint_skips_repeat_request(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'resume.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Stock(stock_id="TAIEX", market="INDEX"),
                Stock(stock_id="2317", market="TWSE", listed_date="2000-01-01"),
                _price("2026-10-22", "TAIEX", adjusted=False),
            ]
        )
        session.execute(
            __import__("sqlalchemy").text(
                "INSERT INTO data_sync_days "
                "(source,dataset,market,trade_date,status,row_count,synced_at) "
                "VALUES ('official','raw_prices','TWSE','2026-10-22','succeeded',1,'now')"
            )
        )
    plan = plan_price_gaps(engine, "2026-10-22", "2026-10-22", resume=True)
    assert plan.raw_tasks == ()


def test_finmind_adjusted_gaps_are_grouped_into_one_request_per_stock(
    tmp_path: Path, monkeypatch
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'adjusted.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(Stock(stock_id="2330", market="TWSE"))
        session.add_all(
            [
                _price("2026-10-21", "2330", adjusted=False),
                _price("2026-10-23", "2330", adjusted=False),
                _price("2026-10-16", "2330"),
                _price("2026-10-17", "2330"),
                _price("2026-10-18", "2330"),
            ]
        )
    calls: list[tuple[str, str, str]] = []

    def fetch(start: str, end: str, token: str, *, stock_id: str):
        calls.append((stock_id, start, end))
        return pd.DataFrame(
            [
                {
                    "stock_id": stock_id,
                    "trade_date": day,
                    "open_adj": 10.0,
                    "high_adj": 11.0,
                    "low_adj": 9.0,
                    "close_adj": 10.0,
                }
                for day in (
                    "2026-10-16",
                    "2026-10-17",
                    "2026-10-18",
                    "2026-10-21",
                    "2026-10-23",
                )
            ]
        )

    monkeypatch.setattr(gap_service, "fetch_price_adj", fetch)
    written, unresolved = gap_service._repair_adjusted_with_finmind(
        engine,
        {("2330", "2026-10-21"), ("2330", "2026-10-23")},
        "token",
    )
    assert len(calls) == 1
    assert calls[0][0] == "2330"
    assert calls[0][1] < "2026-10-21"
    assert calls[0][2] > "2026-10-23"
    assert written == 2
    assert unresolved == []
