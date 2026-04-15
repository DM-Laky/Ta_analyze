from __future__ import annotations

from data.exchange_client import ExchangeClient


class SymbolRegistry:
    def __init__(self, exchange: ExchangeClient) -> None:
        self.exchange = exchange
        self.symbols: list[str] = []

    async def refresh(self) -> list[str]:
        markets = await self.exchange.load_markets()
        out: list[str] = []
        for sym, market in markets.items():
            if market.get("contract") and market.get("quote") == "USDT" and market.get("active"):
                out.append(sym)
        self.symbols = sorted(out)
        return self.symbols
