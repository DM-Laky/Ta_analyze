"""
analysis/market_structure.py
==============================
Detects:
  - Swing Highs / Swing Lows (ZigZag-style, confirmed by N bars either side)
  - Market Trend (bullish / bearish / ranging)
  - BOS  (Break of Structure) — continuation
  - CHoCH (Change of Character) — reversal warning
  - MSB  (Market Structure Break) — strong reversal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from utils.logger import log


# ── Enums ─────────────────────────────────────────────────────────────────────

class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


class StructureEvent(Enum):
    BOS_BULL   = "BOS_BULL"    # Bullish BOS — higher high confirmed
    BOS_BEAR   = "BOS_BEAR"    # Bearish BOS — lower low confirmed
    CHOCH_BULL = "CHOCH_BULL"  # Bullish CHoCH — first break up after downtrend
    CHOCH_BEAR = "CHOCH_BEAR"  # Bearish CHoCH — first break down after uptrend
    MSB_BULL   = "MSB_BULL"    # Major structure break to upside
    MSB_BEAR   = "MSB_BEAR"    # Major structure break to downside


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    time: pd.Timestamp
    is_high: bool      # True = swing high, False = swing low

    @property
    def label(self) -> str:
        return "SH" if self.is_high else "SL"

    def __repr__(self) -> str:
        return f"SwingPoint({self.label} @ {self.price:.5f} idx={self.index})"


@dataclass
class StructurePoint:
    event: StructureEvent
    broken_price: float        # The swing level that was broken
    break_index: int           # Candle index that caused the break
    break_time: pd.Timestamp
    prior_trend: Trend
    strength: float            # 0–1 based on candle body size / ATR


@dataclass
class MarketStructure:
    trend: Trend
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    events: List[StructurePoint]
    last_high: Optional[SwingPoint]
    last_low: Optional[SwingPoint]
    higher_high: bool          # Is latest high higher than previous high?
    higher_low: bool
    lower_high: bool
    lower_low: bool


# ── Core Detector ─────────────────────────────────────────────────────────────

class MarketStructureAnalyzer:
    """
    Runs on H1 candles to determine the dominant trend and key structure events.
    Also usable on M15 / M1 for lower-timeframe analysis.
    """

    def __init__(
        self,
        left_bars: int = None,
        right_bars: int = None,
    ):
        self.left = left_bars or config.SWING_LEFT_BARS
        self.right = right_bars or config.SWING_RIGHT_BARS

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> MarketStructure:
        """
        Analyse a OHLCV DataFrame.
        `df` must have columns: open, high, low, close, time.
        """
        if len(df) < self.left + self.right + 2:
            log.warning("Not enough candles for structure analysis (%d)", len(df))
            return self._empty_structure()

        highs, lows = self._detect_swings(df)
        events = self._detect_events(highs, lows, df)
        trend = self._determine_trend(highs, lows)
        last_high = highs[-1] if highs else None
        last_low = lows[-1] if lows else None

        # HH/HL/LL/LH flags
        hh = hl = ll = lh = False
        if len(highs) >= 2:
            hh = highs[-1].price > highs[-2].price
            lh = highs[-1].price < highs[-2].price
        if len(lows) >= 2:
            hl = lows[-1].price > lows[-2].price
            ll = lows[-1].price < lows[-2].price

        return MarketStructure(
            trend=trend,
            swing_highs=highs,
            swing_lows=lows,
            events=events,
            last_high=last_high,
            last_low=last_low,
            higher_high=hh,
            higher_low=hl,
            lower_high=lh,
            lower_low=ll,
        )

    # ── Swing Detection ───────────────────────────────────────────────────────

    def _detect_swings(
        self, df: pd.DataFrame
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Classic ZigZag swing detection:
        A candle at index i is a swing HIGH if its high is the highest
        among [i-left .. i+right]. Same logic for swing LOW with low prices.
        Only confirmed swings (right_bars already formed) are returned.
        """
        n = len(df)
        highs: List[SwingPoint] = []
        lows: List[SwingPoint] = []

        # We only analyse up to n - right_bars so swings are confirmed
        for i in range(self.left, n - self.right):
            window_h = df["high"].iloc[i - self.left: i + self.right + 1]
            window_l = df["low"].iloc[i - self.left: i + self.right + 1]

            if df["high"].iloc[i] == window_h.max():
                highs.append(SwingPoint(
                    index=i,
                    price=df["high"].iloc[i],
                    time=df["time"].iloc[i],
                    is_high=True,
                ))

            if df["low"].iloc[i] == window_l.min():
                lows.append(SwingPoint(
                    index=i,
                    price=df["low"].iloc[i],
                    time=df["time"].iloc[i],
                    is_high=False,
                ))

        return highs, lows

    # ── Trend ─────────────────────────────────────────────────────────────────

    def _determine_trend(
        self,
        highs: List[SwingPoint],
        lows: List[SwingPoint],
    ) -> Trend:
        """
        Bullish  = HH + HL
        Bearish  = LH + LL
        Ranging  = mixed
        """
        if len(highs) < 2 or len(lows) < 2:
            return Trend.RANGING

        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price

        if hh and hl:
            return Trend.BULLISH
        if lh and ll:
            return Trend.BEARISH
        return Trend.RANGING

    # ── BOS / CHoCH Detection ─────────────────────────────────────────────────

    def _detect_events(
        self,
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        df: pd.DataFrame,
    ) -> List[StructurePoint]:
        """
        Scan for:
        - BOS_BULL  : close > previous swing HIGH (in uptrend)
        - BOS_BEAR  : close < previous swing LOW  (in downtrend)
        - CHOCH_BULL: close > previous swing HIGH (in downtrend) → reversal
        - CHOCH_BEAR: close < previous swing LOW  (in uptrend)  → reversal
        """
        events: List[StructurePoint] = []
        if len(highs) < 2 or len(lows) < 2:
            return events

        atr = self._atr(df)
        n = len(df)

        # Pre-compute trend checkpoints to avoid O(n²) _determine_trend calls.
        # We only recompute when the set of prior swings changes.
        _trend_cache: dict = {}

        def _cached_trend(h_count: int, l_count: int) -> Trend:
            key = (h_count, l_count)
            if key not in _trend_cache:
                _trend_cache[key] = self._determine_trend(
                    highs[:h_count], lows[:l_count]
                )
            return _trend_cache[key]

        last_event_idx = -1

        for i in range(1, n):
            close = df["close"].iloc[i]
            idx_time = df["time"].iloc[i]

            # Count how many confirmed swings precede index i
            h_count = sum(1 for h in highs if h.index < i)
            l_count = sum(1 for l in lows  if l.index < i)

            if h_count < 2 or l_count < 2:
                continue

            last_h = highs[h_count - 1]
            last_l = lows[l_count - 1]

            if i <= last_event_idx:
                continue  # one event per candle maximum

            trend = _cached_trend(h_count, l_count)
            strength = min(abs(close - last_h.price) / (atr + 1e-9), 1.0)

            # ── Bullish break ────────────────────────────────────────────────
            if close > last_h.price:
                if trend == Trend.BULLISH:
                    evt = StructureEvent.BOS_BULL
                elif trend == Trend.BEARISH:
                    evt = StructureEvent.CHOCH_BULL
                else:
                    evt = StructureEvent.BOS_BULL

                events.append(StructurePoint(
                    event=evt,
                    broken_price=last_h.price,
                    break_index=i,
                    break_time=idx_time,
                    prior_trend=trend,
                    strength=strength,
                ))
                last_event_idx = i

            # ── Bearish break ────────────────────────────────────────────────
            elif close < last_l.price:
                if trend == Trend.BEARISH:
                    evt = StructureEvent.BOS_BEAR
                elif trend == Trend.BULLISH:
                    evt = StructureEvent.CHOCH_BEAR
                else:
                    evt = StructureEvent.BOS_BEAR

                events.append(StructurePoint(
                    event=evt,
                    broken_price=last_l.price,
                    break_index=i,
                    break_time=idx_time,
                    prior_trend=trend,
                    strength=strength,
                ))
                last_event_idx = i

        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range (simple implementation)."""
        high = df["high"]
        low  = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _empty_structure(self) -> MarketStructure:
        return MarketStructure(
            trend=Trend.RANGING,
            swing_highs=[],
            swing_lows=[],
            events=[],
            last_high=None,
            last_low=None,
            higher_high=False,
            higher_low=False,
            lower_high=False,
            lower_low=False,
        )


# Convenience function
def analyze_structure(df: pd.DataFrame, left: int = None, right: int = None) -> MarketStructure:
    return MarketStructureAnalyzer(left, right).analyze(df)
