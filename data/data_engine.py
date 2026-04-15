from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass

from data.exchange_client import ExchangeClient


@dataclass(slots=True)
class CandleEvent:
    symbol: str
    timeframe: str
    closed_ts: int


class DataEngine:
    TF_LIMITS = {"4h": 20, "5m": 260, "1m": 260}

    def __init__(self, exchange: ExchangeClient, symbols: list[str]) -> None:
        self.exchange = exchange
        self.symbols = symbols
        self.buffers: dict[str, dict[str, deque[list[float]]]] = defaultdict(
            lambda: {tf: deque(maxlen=maxlen) for tf, maxlen in self.TF_LIMITS.items()}
        )
        self.events: asyncio.Queue[CandleEvent] = asyncio.Queue()
        self.last_seen_ts: dict[tuple[str, str], int] = {}

    def candles(self, symbol: str, timeframe: str) -> list[list[float]]:
        return list(self.buffers[symbol][timeframe])

    async def run(self) -> None:
        tasks = []
        for symbol in self.symbols:
            for timeframe, limit in self.TF_LIMITS.items():
                tasks.append(asyncio.create_task(self._stream_symbol_tf(symbol, timeframe, limit)))
        await asyncio.gather(*tasks)

    async def _stream_symbol_tf(self, symbol: str, timeframe: str, limit: int) -> None:
        while True:
            try:
                candles = await self.exchange.watch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if len(candles) < 3:
                    continue
                closed = candles[-2]
                key = (symbol, timeframe)
                if self.last_seen_ts.get(key) == int(closed[0]):
                    continue
                self.last_seen_ts[key] = int(closed[0])
                self._sync_buffer(symbol, timeframe, candles[:-1])
                await self.events.put(CandleEvent(symbol=symbol, timeframe=timeframe, closed_ts=int(closed[0])))
            except Exception:
                await asyncio.sleep(2)

    def _sync_buffer(self, symbol: str, timeframe: str, candles: list[list[float]]) -> None:
        buf = self.buffers[symbol][timeframe]
        for row in candles[-buf.maxlen :]:
            ts = int(row[0])
            if buf and int(buf[-1][0]) == ts:
                buf[-1] = row
            elif not buf or ts > int(buf[-1][0]):
                buf.append(row)
