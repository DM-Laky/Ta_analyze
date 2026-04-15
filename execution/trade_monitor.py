from __future__ import annotations

import asyncio

from data.exchange_client import ExchangeClient
from execution.trade_executor import TradeExecutor
from storage.database import Database
from telegram.bot_service import TelegramService


class TradeMonitor:
    def __init__(self, exchange: ExchangeClient, db: Database, executor: TradeExecutor, telegram: TelegramService) -> None:
        self.exchange = exchange
        self.db = db
        self.executor = executor
        self.telegram = telegram

    async def run(self, interval_seconds: int = 2) -> None:
        while True:
            try:
                await self.check_all()
            except Exception as exc:
                await self.telegram.send_error(f"Trade monitor error: {exc}")
                self.db.log("ERROR", "Trade monitor failure", error=str(exc))
            await asyncio.sleep(interval_seconds)

    async def check_all(self) -> None:
        for trade in self.db.get_open_trades():
            ticker = await self.exchange.watch_ticker(trade.symbol)
            mark = float(ticker.get("last") or ticker.get("close"))
            pnl = (mark - trade.entry_price) * trade.qty
            if trade.side == "SELL":
                pnl *= -1
            if pnl >= trade.tp_usd:
                realized = await self.executor.close_trade(trade, close_price=mark, status="TP_HIT")
                await self.telegram.send_trade_closed(trade, mark, realized, "TP")
                continue
            if (trade.side == "BUY" and mark <= trade.sl_price) or (trade.side == "SELL" and mark >= trade.sl_price):
                realized = await self.executor.close_trade(trade, close_price=mark, status="SL_HIT")
                await self.telegram.send_trade_closed(trade, mark, realized, "SL")
