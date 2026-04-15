from __future__ import annotations

import asyncio
from typing import Any

import ccxt
import ccxt.pro as ccxtpro


class ExchangeClient:
    def __init__(self, api_key: str, api_secret: str) -> None:
        params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        self.ws = ccxtpro.binanceusdm(params)
        self.rest = ccxt.binanceusdm(params)
        self.rest.set_sandbox_mode(False)

    async def load_markets(self) -> dict[str, Any]:
        return await self.ws.load_markets()

    async def watch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[list[float]]:
        return await self.ws.watch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self.ws.watch_ticker(symbol)

    async def fetch_balance(self) -> dict[str, Any]:
        return await self.ws.fetch_balance()

    def fetch_balance_sync(self) -> dict[str, Any]:
        return self.rest.fetch_balance()

    def market(self, symbol: str) -> dict[str, Any]:
        return self.rest.market(symbol)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(self.rest.amount_to_precision(symbol, amount))

    def price_to_precision(self, symbol: str, price: float) -> float:
        return float(self.rest.price_to_precision(symbol, price))

    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = "market", price: float | None = None) -> dict[str, Any]:
        params = {"marginType": "ISOLATED"}
        return await self.ws.create_order(symbol, order_type, side.lower(), amount, price, params)

    async def close(self) -> None:
        await self.ws.close()
        await asyncio.to_thread(self.rest.close)
