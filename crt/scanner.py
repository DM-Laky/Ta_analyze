from __future__ import annotations

from core.models import WatchlistItem


def _body_quality(c):
    o, h, l, cl = c[1], c[2], c[3], c[4]
    body = abs(cl - o)
    wick = (h - l) - body
    if h == l:
        return 0.0
    return max(0.0, min(1.0, body / max(1e-8, body + wick)))


class CRTScanner:
    def evaluate(self, symbol: str, candles_4h: list[list[float]]) -> WatchlistItem | None:
        if len(candles_4h) < 4:
            return None
        c1, c2, c3 = candles_4h[-3], candles_4h[-2], candles_4h[-1]
        highs = [c[2] for c in (c1, c2, c3)]
        lows = [c[3] for c in (c1, c2, c3)]
        range_high = max(highs)
        range_low = min(lows)
        midpoint = (range_high + range_low) / 2

        sweep_high = c3[2] > max(c1[2], c2[2]) and c3[4] < max(c1[2], c2[2])
        sweep_low = c3[3] < min(c1[3], c2[3]) and c3[4] > min(c1[3], c2[3])
        if not (sweep_high or sweep_low):
            return None

        bias = "BUY" if sweep_low else "SELL"
        sweep_side = "LOW" if sweep_low else "HIGH"

        clear_sweep = 20 if (sweep_high or sweep_low) else 0
        reclaim_strength = 20 if (sweep_low and c3[4] > midpoint) or (sweep_high and c3[4] < midpoint) else 12
        body_quality = _body_quality(c3) * 15
        bias_clarity = 15 if abs(c3[4] - midpoint) / max(1e-8, range_high - range_low) > 0.2 else 7
        if bias == "BUY":
            room_ratio = (range_high - c3[4]) / max(1e-8, range_high - range_low)
        else:
            room_ratio = (c3[4] - range_low) / max(1e-8, range_high - range_low)
        room_score = max(0, min(15, room_ratio * 15))

        ranges = [c[2] - c[3] for c in (c1, c2, c3)]
        avg_range = sum(ranges) / 3
        variance = sum((r - avg_range) ** 2 for r in ranges) / 3
        noise = max(0, 1 - (variance**0.5 / max(1e-8, avg_range)))
        noise_score = noise * 15

        score = clear_sweep + reclaim_strength + body_quality + bias_clarity + room_score + noise_score
        if score < 60:
            return None

        return WatchlistItem(
            symbol=symbol,
            bias=bias,
            range_high=range_high,
            range_low=range_low,
            midpoint=midpoint,
            sweep_side=sweep_side,
            score=round(score, 2),
        )
