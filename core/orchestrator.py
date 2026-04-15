from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from charts.chart_service import ChartService
from core.config import settings
from crt.scanner import CRTScanner
from data.data_engine import CandleEvent, DataEngine
from data.exchange_client import ExchangeClient
from data.symbol_registry import SymbolRegistry
from entry.tbs_engine import TBSEntryEngine
from execution.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from execution.trade_monitor import TradeMonitor
from setup.engine import SetupEngine5M
from storage.database import Database
from telegram.bot_service import TelegramService
from watchlist.manager import WatchlistManager


class Orchestrator:
    def __init__(self) -> None:
        self.db = Database(settings.db_path)
        self.exchange = ExchangeClient(settings.binance_api_key, settings.binance_api_secret)
        self.registry = SymbolRegistry(self.exchange)
        self.crt = CRTScanner()
        self.setup_engine = SetupEngine5M()
        self.tbs = TBSEntryEngine()
        self.watchlist = WatchlistManager(self.db, expiry_hours=settings.watchlist_expiry_hours)
        self.risk = RiskManager(self.exchange, risk_ratio=settings.risk_per_trade)
        self.executor = TradeExecutor(
            self.exchange,
            self.db,
            self.risk,
            max_active=settings.max_active_trades,
            cooldown_min=settings.symbol_cooldown_min,
        )
        self.chart = ChartService()
        self.telegram = TelegramService(settings.telegram_token, settings.telegram_chat_id, self.db)
        self.data_engine: DataEngine | None = None
        self.valid_setups: dict[str, object] = {}
        self.last_4h: dict[str, int] = {}
        self.last_5m: dict[str, int] = {}

    async def start(self) -> None:
        symbols = await self.registry.refresh()
        self.db.upsert_symbols(symbols)
        self.data_engine = DataEngine(self.exchange, symbols)
        await self.telegram.setup_commands(self.status_report)

        monitor = TradeMonitor(self.exchange, self.db, self.executor, self.telegram)
        await asyncio.gather(
            self.data_engine.run(),
            self._event_loop(),
            monitor.run(interval_seconds=settings.price_monitor_seconds),
        )

    async def _event_loop(self) -> None:
        assert self.data_engine is not None
        while True:
            event: CandleEvent = await self.data_engine.events.get()
            if not self.telegram.running:
                continue
            try:
                if event.timeframe == "4h":
                    await self._on_4h_close(event.symbol)
                elif event.timeframe == "5m":
                    await self._on_5m_close(event.symbol)
                elif event.timeframe == "1m":
                    await self._on_1m_close(event.symbol)
                self.watchlist.expire()
            except Exception as exc:
                self.db.log("ERROR", "Event loop error", symbol=event.symbol, timeframe=event.timeframe, error=str(exc))
                await self.telegram.send_error(f"{event.symbol} {event.timeframe}: {exc}")

    async def _on_4h_close(self, symbol: str) -> None:
        assert self.data_engine is not None
        candles = self.data_engine.candles(symbol, "4h")
        if len(candles) < 10:
            return
        close_ts = int(candles[-1][0])
        if self.last_4h.get(symbol) == close_ts:
            return
        self.last_4h[symbol] = close_ts
        item = self.crt.evaluate(symbol, candles)
        if item and self.watchlist.upsert(item):
            chart = self.chart.render(symbol, candles[-120:], item)
            await self.telegram.send_watchlist_alert(item, chart)

    async def _on_5m_close(self, symbol: str) -> None:
        assert self.data_engine is not None
        item = self.watchlist.items.get(symbol)
        if not item:
            return
        candles = self.data_engine.candles(symbol, "5m")
        if len(candles) < 200:
            return
        close_ts = int(candles[-1][0])
        if self.last_5m.get(symbol) == close_ts:
            return
        self.last_5m[symbol] = close_ts
        setup = self.setup_engine.validate(item, candles)
        if setup:
            self.valid_setups[symbol] = setup
            self.db.insert_setup(setup)
            self.db.update_watchlist_status(symbol, "SETUP_VALID_5M")
            await self.telegram.send_setup_ready(setup)

    async def _on_1m_close(self, symbol: str) -> None:
        assert self.data_engine is not None
        setup = self.valid_setups.get(symbol)
        item = self.watchlist.items.get(symbol)
        if not setup or not item:
            return
        candles = self.data_engine.candles(symbol, "1m")
        if len(candles) < 200:
            return
        signal = self.tbs.detect(setup, item, candles)
        if not signal:
            return
        trade = await self.executor.execute(signal, order_mode="limit" if settings.use_limit_retest else "market")
        if trade:
            chart = self.chart.render(symbol, candles[-160:], item, setup=setup, entry=signal)
            await self.telegram.send_photo(chart, caption=f"🚀 {symbol} {signal.side} TBS ENTRY")
            await self.telegram.send_trade_entry(trade)
            self.db.update_watchlist_status(symbol, "TRADE_OPEN")
            self.valid_setups.pop(symbol, None)

    def status_report(self, section: str | None = None) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        open_trades = self.db.get_open_trades()
        if section == "watchlist":
            ranked = self.watchlist.ranked()[:20]
            lines = [f"{w.symbol} {w.bias} score={w.score}" for w in ranked] or ["(empty)"]
            return "📋 Watchlist\n" + "\n".join(lines)
        if section == "positions":
            lines = [f"{t.symbol} {t.side} qty={t.qty} entry={t.entry_price}" for t in open_trades] or ["(none)"]
            return "💼 Positions\n" + "\n".join(lines)
        if section == "signals":
            lines = [f"{s.symbol} {s.bias} zone={s.entry_zone_low:.4f}-{s.entry_zone_high:.4f}" for s in self.valid_setups.values()] or ["(none)"]
            return "📡 Last Signals\n" + "\n".join(lines)
        return (
            f"🤖 System Status\nTime: {now}\nSymbols: {len(self.registry.symbols)}\n"
            f"Watchlist: {len(self.watchlist.items)}\nOpen Trades: {len(open_trades)}\n"
            f"Setups Ready: {len(self.valid_setups)}"
        )
