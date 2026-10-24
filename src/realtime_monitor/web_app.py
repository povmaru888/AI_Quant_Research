"""FastAPI application serving the local fund monitor and live WebSocket."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from realtime_monitor import DEFAULT_RUN_ID, INITIAL_CAPITAL
from realtime_monitor.engine import FundEngine
from realtime_monitor.intraday import intraday_payload, intraday_point, merge_intraday_payload
from realtime_monitor.monitor_store import MonitorStore
from realtime_monitor.oos_bridge import (
    load_fund_seed,
    load_holding_history,
    load_order_history,
)
from realtime_monitor.shioaji_feed import DatabaseQuoteFeed, ShioajiQuoteFeed
from runtime.dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DB = ROOT / "database" / "quant.db"
MONITOR_DB = ROOT / "database" / "realtime_monitor.db"
STATIC_DIR = ROOT / "realtime_web"


def create_app(
    *,
    research_db: str | Path = RESEARCH_DB,
    monitor_db: str | Path = MONITOR_DB,
    run_id: str = DEFAULT_RUN_ID,
) -> FastAPI:
    load_dotenv()
    seed = load_fund_seed(research_db, run_id, initial_capital=INITIAL_CAPITAL)
    engine = FundEngine(seed)
    store = MonitorStore(monitor_db)
    stop_event = threading.Event()

    async def reload_intraday_backfill(feed) -> None:
        if not isinstance(feed, ShioajiQuoteFeed):
            return
        snapshot = engine.snapshot()
        fallback = {str(row["stock_id"]): float(row["last_price"]) for row in snapshot["holdings"]}
        payload = await asyncio.to_thread(
            feed.load_intraday_history,
            engine.seed,
            fallback,
        )
        if payload.get("points"):
            payload["previous_close_nav"] = snapshot["cash"] + sum(
                row["shares"] * (row.get("previous_close") or row["last_price"])
                for row in snapshot["holdings"]
            )
            payload["previous_close_taiex"] = (snapshot.get("market") or {}).get("reference")
            app.state.intraday_backfill = payload

    async def persist_loop() -> None:
        while not stop_event.is_set():
            store.save_snapshot(run_id, engine.snapshot())
            await asyncio.sleep(60)

    async def refresh_book_loop(feed) -> None:
        """Pick up a newly materialized monthly signal without restarting the UI."""
        while not stop_event.is_set():
            await asyncio.sleep(60)
            try:
                refreshed = await asyncio.to_thread(
                    load_fund_seed,
                    research_db,
                    run_id,
                    initial_capital=INITIAL_CAPITAL,
                )
                current = engine.seed
                revision = (
                    refreshed.signal_date,
                    refreshed.execution_date,
                    refreshed.anchor_date,
                    refreshed.anchor_nav,
                )
                current_revision = (
                    current.signal_date,
                    current.execution_date,
                    current.anchor_date,
                    current.anchor_nav,
                )
                if revision != current_revision:
                    engine.replace_seed(refreshed)
                    symbols = [holding.stock_id for holding in refreshed.holdings]
                    message = await asyncio.to_thread(feed.reconfigure, symbols)
                    engine.set_connection(isinstance(feed, ShioajiQuoteFeed), message)
                    await reload_intraday_backfill(feed)
            except Exception as exc:  # noqa: BLE001 - retain the last valid live book.
                store.log_event("warning", "book-refresh", f"{type(exc).__name__}: {exc}")

    async def refresh_intraday_loop(feed) -> None:
        """Discover today's first bars after market open without redrawing old points."""
        while not stop_event.is_set():
            await asyncio.sleep(300)
            if not isinstance(feed, ShioajiQuoteFeed):
                continue
            try:
                from datetime import datetime, time
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo("Asia/Taipei"))
                current_session = app.state.intraday_backfill.get("session_date")
                if (
                    now.weekday() < 5
                    and time(9) <= now.time() <= time(13, 35)
                    and current_session != now.date().isoformat()
                ):
                    await reload_intraday_backfill(feed)
            except Exception as exc:  # noqa: BLE001
                store.log_event("warning", "intraday-refresh", f"{type(exc).__name__}: {exc}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        symbols = [holding.stock_id for holding in seed.holdings]
        simulation = os.getenv("SHIOAJI_SIMULATION", "true").lower() != "false"
        if ShioajiQuoteFeed.available():
            feed = ShioajiQuoteFeed(engine.update_quote, simulation=simulation)
            try:
                message = await asyncio.to_thread(feed.start, symbols)
                engine.set_connection(True, message)
            except Exception as exc:  # noqa: BLE001 - dashboard must retain delayed mode.
                feed = DatabaseQuoteFeed(research_db, engine.update_quote)
                message = feed.start(symbols)
                engine.set_connection(False, f"Shioaji failed: {type(exc).__name__}; {message}")
        else:
            feed = DatabaseQuoteFeed(research_db, engine.update_quote)
            engine.set_connection(False, feed.start(symbols))
        app.state.feed = feed
        try:
            await reload_intraday_backfill(feed)
        except Exception as exc:  # noqa: BLE001 - live quotes remain usable.
            store.log_event("warning", "intraday-backfill", f"{type(exc).__name__}: {exc}")
        tasks = [
            asyncio.create_task(persist_loop()),
            asyncio.create_task(refresh_book_loop(feed)),
            asyncio.create_task(refresh_intraday_loop(feed)),
        ]
        yield
        stop_event.set()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        await asyncio.to_thread(feed.stop)

    app = FastAPI(title="XGBoost Inverse Volatility Fund Monitor", lifespan=lifespan)
    app.state.engine = engine
    app.state.seed = seed
    app.state.store = store
    app.state.intraday_backfill = {
        "session_date": None,
        "market_open": "09:00",
        "market_close": "13:30",
        "points": [],
    }

    @app.get("/api/health")
    def health():
        snap = engine.snapshot()
        backfill = app.state.intraday_backfill
        return {
            "status": "ok",
            "run_id": run_id,
            "quote_connected": snap["quote_connected"],
            "valuation_quality": snap["valuation_quality"],
            "intraday_session": backfill.get("session_date"),
            "intraday_points": len(backfill.get("points") or ()),
        }

    @app.get("/api/fund/overview")
    def overview():
        return engine.snapshot()

    @app.get("/api/fund/holdings")
    def holdings():
        return engine.snapshot()["holdings"]

    @app.get("/api/fund/performance")
    def performance():
        current = engine.seed
        return {
            "history": current.history,
            "benchmark": current.benchmark_history,
            "monthly_returns": current.monthly_returns,
            "metrics": current.metrics,
            "intraday": store.recent_snapshots(run_id),
        }

    @app.get("/api/fund/intraday")
    def intraday():
        backfill = app.state.intraday_backfill
        if backfill.get("points"):
            return merge_intraday_payload(backfill, engine.snapshot())
        snapshots = store.recent_snapshots(run_id, limit=2000)
        snapshots.append(engine.snapshot())
        return intraday_payload(snapshots)

    @app.get("/api/fund/holding-history")
    def holding_history():
        return load_holding_history(research_db, run_id)

    @app.get("/api/fund/order-history")
    def order_history():
        return load_order_history(research_db, run_id)

    @app.get("/api/fund/risk")
    def risk():
        snapshot = engine.snapshot()
        current = engine.seed
        weights = sorted((row["live_weight"] or 0 for row in snapshot["holdings"]), reverse=True)
        return {
            "equity_exposure": snapshot["equity_exposure"],
            "cash_ratio": snapshot["cash"] / snapshot["nav"],
            "largest_weight": weights[0] if weights else 0,
            "top5_weight": sum(weights[:5]),
            "quote_coverage": snapshot["quote_coverage"],
            "valuation_quality": snapshot["valuation_quality"],
            "historical_max_drawdown": current.metrics.get("max_drawdown"),
            "historical_realized_volatility": current.metrics.get("realized_volatility"),
            "turnover": current.metrics.get("turnover"),
            "taiex_ma60": snapshot["taiex_ma60"],
            "taiex_above_ma60": snapshot["taiex_above_ma60"],
        }

    @app.websocket("/ws/live")
    async def live_socket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                snapshot = engine.snapshot()
                session_date = app.state.intraday_backfill.get("session_date")
                snapshot["intraday_point"] = intraday_point(
                    snapshot, session_date_override=session_date
                )
                await websocket.send_json(snapshot)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            return

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/{path:path}")
    def shell(path: str):  # noqa: ARG001
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
