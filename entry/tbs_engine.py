from __future__ import annotations

from core.models import EntrySignal, SetupSignal, WatchlistItem


class TBSEntryEngine:
    def detect(self, setup: SetupSignal, watch_item: WatchlistItem, candles_1m: list[list[float]]) -> EntrySignal | None:
        if len(candles_1m) < 30:
            return None
        c = candles_1m[-1]
        prev = candles_1m[-2]
        lookback = candles_1m[-10:-1]

        if setup.bias == "BUY":
            structure_high = max(k[2] for k in lookback)
            bearish_before = lookback[-1][4] < lookback[0][4]
            strong_bull = c[4] > c[1] and (c[4] - c[1]) > (prev[2] - prev[3]) * 0.8
            broke = c[4] > structure_high
            if bearish_before and strong_bull and broke:
                sl = watch_item.range_low * (1 - 0.0008)
                return EntrySignal(symbol=setup.symbol, side="BUY", entry_price=c[4], sl_price=sl)

        if setup.bias == "SELL":
            structure_low = min(k[3] for k in lookback)
            bullish_before = lookback[-1][4] > lookback[0][4]
            strong_bear = c[4] < c[1] and (c[1] - c[4]) > (prev[2] - prev[3]) * 0.8
            broke = c[4] < structure_low
            if bullish_before and strong_bear and broke:
                sl = watch_item.range_high * (1 + 0.0008)
                return EntrySignal(symbol=setup.symbol, side="SELL", entry_price=c[4], sl_price=sl)

        return None
