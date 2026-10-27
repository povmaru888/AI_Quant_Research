"""FastAPI application serving the local fund monitor and live WebSocket."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from realtime_monitor import DEFAULT_RUN_ID, INITIAL_CAPITAL
from realtime_monitor.engine import FundEngine
from realtime_monitor.intraday import current_session_point, merge_intraday_payload
from realtime_monitor.monitor_store import MonitorStore
from realtime_monitor.oos_bridge import (
    load_fund_seed,
    load_holding_history,
    load_order_history,
)
from realtime_monitor.shioaji_feed import DatabaseQuoteFeed, ShioajiQuoteFeed, TwseIndexQuoteFeed
from realtime_monitor.trading_session import SessionState, TradingCalendar
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
    calendar = TradingCalendar(store)
    twse_index_feed = TwseIndexQuoteFeed(engine.update_quote)
    stop_event = threading.Event()
    backfill_lock = asyncio.Lock()

    def empty_intraday(state: SessionState) -> dict:
        snapshot = engine.snapshot()
        market = snapshot.get("market") or {}
        return {
            "session_date": state.target_date,
            "market_open": "09:00",
            "market_close": "13:30",
            "phase": state.phase,
            "data_status": state.data_status,
            "revision": f"{state.phase}:{state.target_date or 'unknown'}",
            "last_market_time": None,
            "previous_close_nav": snapshot["cash"]
            + sum(
                row["shares"] * (row.get("previous_close") or row["last_price"])
                for row in snapshot["holdings"]
            ),
            "previous_close_taiex": market.get("reference"),
            "points": [],
        }

    async def reload_intraday_backfill(
        feed, state: SessionState | None = None, *, force: bool = False
    ) -> bool:
        """Load exactly the session selected by the exchange calendar."""
        state = state or calendar.state()
        async with backfill_lock:
            refreshed_at = app.state.last_backfill_at
            if (
                not force
                and refreshed_at is not None
                and app.state.last_backfill_date == state.target_date
                and (state.now - refreshed_at).total_seconds() < 5
            ):
                return bool(app.state.intraday_backfill.get("points"))
            app.state.last_backfill_at = state.now
            app.state.last_backfill_date = state.target_date
            if state.phase in {"clearing", "unknown"} or not state.target_date:
                app.state.intraday_backfill = empty_intraday(state)
                return False
            cached = store.load_intraday_session(run_id, state.target_date)
            if not isinstance(feed, ShioajiQuoteFeed):
                app.state.intraday_backfill = cached or empty_intraday(state)
                return bool(cached and cached.get("points"))
            snapshot = engine.snapshot()
            fallback = {
                str(row["stock_id"]): float(row.get("previous_close") or row["last_price"])
                for row in snapshot["holdings"]
            }
            payload = await asyncio.to_thread(
                feed.load_intraday_history,
                engine.seed,
                fallback,
                target_date=state.target_date,
                allow_previous=False,
            )
            if not payload.get("points"):
                if cached:
                    cached.update(
                        phase=state.phase,
                        data_status="delayed" if state.phase == "live" else state.data_status,
                    )
                    app.state.intraday_backfill = cached
                else:
                    app.state.intraday_backfill = empty_intraday(state)
                return False
            payload["previous_close_nav"] = snapshot["cash"] + sum(
                row["shares"] * (row.get("previous_close") or row["last_price"])
                for row in snapshot["holdings"]
            )
            payload["previous_close_taiex"] = (snapshot.get("market") or {}).get("reference")
            payload["phase"] = state.phase
            payload["data_status"] = (
                "live"
                if state.phase == "live"
                else "complete"
                if state.phase == "closed"
                else "historical"
            )
            payload["last_market_time"] = payload["points"][-1]["time"]
            payload["revision"] = (
                f"{state.target_date}:{len(payload['points'])}:{payload['last_market_time']}"
            )
            store.save_intraday_session(run_id, payload)
            app.state.intraday_backfill = payload
            return True

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
                    await reload_intraday_backfill(feed, force=True)
            except Exception as exc:  # noqa: BLE001 - retain the last valid live book.
                store.log_event("warning", "book-refresh", f"{type(exc).__name__}: {exc}")

    async def refresh_intraday_loop(feed) -> None:
        """Apply phase transitions and retry incomplete current-day backfills."""
        signature = None
        failures = 0
        next_retry = datetime.min.replace(tzinfo=calendar.state().now.tzinfo)
        while not stop_event.is_set():
            try:
                state = calendar.state()
                current_signature = (state.phase, state.target_date)
                payload = app.state.intraday_backfill
                last_time = str(payload.get("last_market_time") or "")
                closing_incomplete = state.phase == "closed" and "T13:30:" not in last_time
                needs_retry = (
                    state.phase == "live" and not payload.get("points")
                ) or closing_incomplete
                if current_signature != signature:
                    await reload_intraday_backfill(feed, state)
                    signature = current_signature
                    failures = 0
                elif needs_retry and state.now >= next_retry:
                    success = await reload_intraday_backfill(feed, state)
                    failures = 0 if success else failures + 1
                    delays = (5, 15, 30, 60)
                    if closing_incomplete and state.now.time().hour >= 14:
                        delay = 1800
                    elif closing_incomplete and state.now.time().minute >= 40:
                        delay = 300
                    else:
                        delay = delays[min(failures, len(delays) - 1)]
                    next_retry = state.now + timedelta(seconds=delay)
            except Exception as exc:  # noqa: BLE001
                store.log_event("warning", "intraday-refresh", f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(1)

    async def refresh_market_loop(feed) -> None:
        """Keep TAIEX current when Shioaji simulation does not stream index quotes."""
        while not stop_event.is_set():
            delay = 30
            try:
                state = calendar.state()
                delay = 2 if state.phase == "live" else 30
                if state.phase in {"live", "closed"} and state.target_date:
                    quote = None
                    if isinstance(feed, ShioajiQuoteFeed):
                        quote = await asyncio.to_thread(
                            feed.refresh_index_quote, state.target_date
                        )
                    market = engine.snapshot().get("market") or {}
                    market_time = FundEngine._quote_time(  # noqa: SLF001
                        str(market.get("timestamp") or "1970-01-01T00:00:00+00:00")
                    )
                    if (
                        quote is None
                        or (state.now.astimezone(market_time.tzinfo) - market_time).total_seconds()
                        > 10
                    ):
                        quote = await asyncio.to_thread(twse_index_feed.refresh)
                    app.state.last_taiex_refresh = datetime.now(state.now.tzinfo).isoformat()
                    app.state.last_taiex_refresh_price = quote.price if quote else None
            except Exception as exc:  # noqa: BLE001 - retry without stopping live stocks.
                store.log_event("warning", "taiex-refresh", f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(delay)

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
            await asyncio.to_thread(calendar.refresh_year, datetime.now().year)
            await reload_intraday_backfill(feed)
        except Exception as exc:  # noqa: BLE001 - live quotes remain usable.
            store.log_event("warning", "intraday-backfill", f"{type(exc).__name__}: {exc}")
        tasks = [
            asyncio.create_task(persist_loop()),
            asyncio.create_task(refresh_book_loop(feed)),
            asyncio.create_task(refresh_intraday_loop(feed)),
            asyncio.create_task(refresh_market_loop(feed)),
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
    app.state.last_backfill_at = None
    app.state.last_backfill_date = None
    app.state.last_taiex_refresh = None
    app.state.last_taiex_refresh_price = None
    app.state.intraday_backfill = {
        "session_date": None,
        "market_open": "09:00",
        "market_close": "13:30",
        "points": [],
    }

    def current_intraday() -> dict:
        state = calendar.state()
        if state.phase == "clearing":
            return empty_intraday(state)
        backfill = app.state.intraday_backfill
        if backfill.get("session_date") != state.target_date:
            cached = (
                store.load_intraday_session(run_id, state.target_date)
                if state.target_date
                else None
            )
            backfill = cached or empty_intraday(state)
        backfill = dict(backfill)
        backfill["phase"] = state.phase
        if state.phase == "live" and backfill.get("data_status") not in {"live", "delayed"}:
            backfill["data_status"] = "backfilling"
        if state.phase == "closed" and not str(backfill.get("last_market_time") or "").startswith(
            f"{state.target_date}T13:30:"
        ):
            backfill["data_status"] = "closing"
        if state.phase in {"live", "closed"}:
            return merge_intraday_payload(backfill, engine.snapshot())
        return backfill

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
            "market_phase": calendar.state().phase,
            "intraday_data_status": backfill.get("data_status"),
            "last_taiex_refresh": app.state.last_taiex_refresh,
            "last_taiex_refresh_price": app.state.last_taiex_refresh_price,
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
    async def intraday():
        state = calendar.state()
        if state.phase in {"live", "closed"}:
            await reload_intraday_backfill(app.state.feed, state)
        return current_intraday()

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
            initial_state = calendar.state()
            if initial_state.phase in {"live", "closed"}:
                await reload_intraday_backfill(app.state.feed, initial_state)
            first_message = True
            while True:
                snapshot = engine.snapshot()
                state = calendar.state()
                snapshot["market_phase"] = state.phase
                snapshot["session_date"] = state.target_date
                snapshot["data_status"] = current_intraday().get("data_status")
                snapshot["intraday_clear"] = state.phase == "clearing"
                snapshot["intraday_sync_required"] = first_message
                snapshot["intraday_point"] = (
                    current_session_point(snapshot, state.target_date)
                    if state.phase in {"live", "closed"} and state.target_date
                    else None
                )
                await websocket.send_json(snapshot)
                first_message = False
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            return

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/{path:path}")
    def shell(path: str):  # noqa: ARG001
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
