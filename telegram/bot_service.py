from __future__ import annotations

from pathlib import Path

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from core.models import OpenTrade, SetupSignal, WatchlistItem
from storage.database import Database


class TelegramService:
    def __init__(self, token: str, chat_id: str, db: Database) -> None:
        self.chat_id = chat_id
        self.db = db
        self.bot = Bot(token=token) if token else None
        self.application = Application.builder().token(token).build() if token else None
        self.running = True

    async def setup_commands(self, status_provider):
        if not self.application:
            return

        async def start(_: Update, context: ContextTypes.DEFAULT_TYPE):
            self.running = True
            await context.bot.send_message(chat_id=self.chat_id, text="✅ Trading system started")

        async def stop(_: Update, context: ContextTypes.DEFAULT_TYPE):
            self.running = False
            await context.bot.send_message(chat_id=self.chat_id, text="🛑 Trading system paused")

        async def status(_: Update, context: ContextTypes.DEFAULT_TYPE):
            await context.bot.send_message(chat_id=self.chat_id, text=status_provider())

        async def balance(_: Update, context: ContextTypes.DEFAULT_TYPE):
            await context.bot.send_message(chat_id=self.chat_id, text="Use /status for current balance snapshot")

        async def watchlist(_: Update, context: ContextTypes.DEFAULT_TYPE):
            await context.bot.send_message(chat_id=self.chat_id, text=status_provider(section="watchlist"))

        async def positions(_: Update, context: ContextTypes.DEFAULT_TYPE):
            await context.bot.send_message(chat_id=self.chat_id, text=status_provider(section="positions"))

        async def lastsignals(_: Update, context: ContextTypes.DEFAULT_TYPE):
            await context.bot.send_message(chat_id=self.chat_id, text=status_provider(section="signals"))

        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("stop", stop))
        self.application.add_handler(CommandHandler("status", status))
        self.application.add_handler(CommandHandler("balance", balance))
        self.application.add_handler(CommandHandler("watchlist", watchlist))
        self.application.add_handler(CommandHandler("positions", positions))
        self.application.add_handler(CommandHandler("lastsignals", lastsignals))
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def send_text(self, text: str) -> None:
        if self.bot:
            await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_photo(self, path: Path, caption: str) -> None:
        if self.bot:
            with path.open("rb") as f:
                await self.bot.send_photo(chat_id=self.chat_id, photo=f, caption=caption)

    async def send_watchlist_alert(self, item: WatchlistItem, chart_path: Path | None = None) -> None:
        msg = (
            f"📋 WATCHLIST\nSymbol: {item.symbol}\nBias: {item.bias}\nScore: {item.score:.2f}\n"
            f"Range: {item.range_low:.6f} - {item.range_high:.6f}\nSweep: {item.sweep_side}"
        )
        if chart_path:
            await self.send_photo(chart_path, caption=msg)
        else:
            await self.send_text(msg)

    async def send_setup_ready(self, setup: SetupSignal) -> None:
        await self.send_text(
            f"🧩 SETUP READY\n{setup.symbol} {setup.bias}\nZone: {setup.entry_zone_low:.6f}-{setup.entry_zone_high:.6f}\n{setup.reason}"
        )

    async def send_trade_entry(self, trade: OpenTrade) -> None:
        await self.send_text(
            f"🚀 TRADE ENTRY\n{trade.symbol} {trade.side}\nQty: {trade.qty}\nEntry: {trade.entry_price:.6f}\nSL: {trade.sl_price:.6f}"
        )

    async def send_trade_closed(self, trade: OpenTrade, close_price: float, pnl: float, reason: str) -> None:
        await self.send_text(
            f"🏁 TRADE CLOSED ({reason})\n{trade.symbol} {trade.side}\nClose: {close_price:.6f}\nPnL: ${pnl:.2f}"
        )

    async def send_error(self, message: str) -> None:
        await self.send_text(f"❌ SYSTEM ERROR\n{message}")
