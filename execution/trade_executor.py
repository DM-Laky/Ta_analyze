from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.models import EntrySignal, OpenTrade
from data.exchange_client import ExchangeClient
from execution.risk_manager import RiskManager
from storage.database import Database


class TradeExecutor:
    def __init__(self, exchange: ExchangeClient, db: Database, risk: RiskManager, max_active: int = 2, cooldown_min: int = 30) -> None:
        self.exchange = exchange
        self.db = db
        self.risk = risk
        self.max_active = max_active
        self.cooldown = timedelta(minutes=cooldown_min)
        self.recent_closed: dict[str, datetime] = {}

    def can_open(self, symbol: str) -> bool:
        open_trades = self.db.get_open_trades()
        if len(open_trades) >= self.max_active:
            return False
        if any(t.symbol == symbol for t in open_trades):
            return False
        last_closed = self.recent_closed.get(symbol)
        return not last_closed or (datetime.now(timezone.utc) - last_closed > self.cooldown)

    async def execute(self, signal: EntrySignal, order_mode: str = "market") -> OpenTrade | None:
        if not self.can_open(signal.symbol):
            return None
        sizing = self.risk.size_position(signal.symbol, signal.entry_price, signal.sl_price)
        if sizing is None:
            return None
        order = await self.exchange.create_order(
            symbol=signal.symbol,
            side=signal.side,
            amount=sizing.qty,
            order_type=order_mode,
            price=signal.entry_price if order_mode == "limit" else None,
        )
        trade_id = self.db.insert_trade(signal, qty=sizing.qty, order_id=order.get("id"))
        return OpenTrade(
            id=trade_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=sizing.qty,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_usd=0.40,
            opened_at=signal.created_at,
            exchange_order_id=order.get("id"),
        )

    async def close_trade(self, trade: OpenTrade, close_price: float, status: str) -> float:
        side = "SELL" if trade.side == "BUY" else "BUY"
        await self.exchange.create_order(symbol=trade.symbol, side=side, amount=trade.qty, order_type="market")
        pnl = (close_price - trade.entry_price) * trade.qty
        if trade.side == "SELL":
            pnl *= -1
        self.db.close_trade(trade.id, close_price=close_price, pnl_usd=pnl, status=status)
        self.recent_closed[trade.symbol] = datetime.now(timezone.utc)
        return pnl
