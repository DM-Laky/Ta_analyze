from __future__ import annotations

from core.models import SetupSignal, WatchlistItem


class SetupEngine5M:
    def validate(self, item: WatchlistItem, candles_5m: list[list[float]]) -> SetupSignal | None:
        if len(candles_5m) < 20:
            return None
        last = candles_5m[-1]
        close = last[4]
        low = min(c[3] for c in candles_5m[-5:])
        high = max(c[2] for c in candles_5m[-5:])

        if item.bias == "BUY":
            near_low = close <= item.range_low + (item.range_high - item.range_low) * 0.25
            bullish_reaction = close > candles_5m[-2][4] and low >= item.range_low * 0.999
            if near_low and bullish_reaction:
                return SetupSignal(
                    symbol=item.symbol,
                    bias="BUY",
                    entry_zone_low=low,
                    entry_zone_high=close,
                    reason="5M reclaim/bounce at 4H range low",
                )
        if item.bias == "SELL":
            near_high = close >= item.range_high - (item.range_high - item.range_low) * 0.25
            bearish_reject = close < candles_5m[-2][4] and high <= item.range_high * 1.001
            if near_high and bearish_reject:
                return SetupSignal(
                    symbol=item.symbol,
                    bias="SELL",
                    entry_zone_low=close,
                    entry_zone_high=high,
                    reason="5M rejection/failure at 4H range high",
                )
        return None
