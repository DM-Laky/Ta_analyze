"""
analysis/fvg_detector.py
=========================
Detects Fair Value Gaps (FVG) — also called Imbalances.

SMC Definition:
  • Bullish FVG: candle[i].low > candle[i-2].high
      → Gap between bottom of candle[i] and top of candle[i-2]
      → Price may return to fill this gap before continuing up.

  • Bearish FVG: candle[i].high < candle[i-2].low
      → Gap between top of candle[i] and bottom of candle[i-2]
      → Price may return to fill this gap before continuing down.

Quality scoring:
  - Size (larger = higher quality)
  - Freshness (recently formed = higher quality)
  - Volume (higher volume on impulse = stronger FVG)
  - Whether FVG overlaps with an OB (confluence)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from config import config
from utils.logger import log


class FVGType(Enum):
    BULLISH = "BULLISH"   # Demand imbalance — expect price to fill from above
    BEARISH = "BEARISH"   # Supply imbalance — expect price to fill from below


@dataclass
class FVG:
    fvg_type: FVGType
    top: float          # Upper boundary of the gap
    bottom: float       # Lower boundary of the gap
    index: int          # Index of the 3rd candle (candle[i])
    time: pd.Timestamp  # Time of the 3rd candle
    quality: float      # 0–1
    filled: bool = False          # True once price has touched the zone
    fully_filled: bool = False    # True once price closed through the gap
    fill_percent: float = 0.0     # How much of the gap has been filled (0–1)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def label(self) -> str:
        return f"{'Bull' if self.fvg_type == FVGType.BULLISH else 'Bear'} FVG"

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.bottom - tolerance) <= price <= (self.top + tolerance)

    def __repr__(self) -> str:
        return (
            f"FVG({self.label} | {self.bottom:.5f}–{self.top:.5f} "
            f"| size={self.size:.5f} | Q={self.quality:.2f} | filled={self.filled})"
        )


class FVGDetector:
    """Detect Fair Value Gaps from OHLCV data."""

    def __init__(self):
        self.min_pips = config.FVG_MIN_PIPS

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
    ) -> List[FVG]:
        """
        Detect all FVGs in `df`. Returns list sorted by quality desc.
        Filters out filled FVGs and those smaller than min_pips.
        """
        pip = config.SYMBOL_PIP.get(symbol, 0.0001)
        min_size = self.min_pips * pip
        atr = self._atr(df)
        fvgs: List[FVG] = []

        for i in range(2, len(df)):
            c0 = df.iloc[i - 2]   # First candle
            # c1 = df.iloc[i - 1] # Middle candle (impulse)
            c2 = df.iloc[i]       # Third candle

            # ── Bullish FVG ──────────────────────────────────────────────────
            # Gap between c2.low and c0.high (c2.low > c0.high)
            if c2["low"] > c0["high"]:
                size = c2["low"] - c0["high"]
                if size >= min_size:
                    # Freshness: more recent = closer to end = higher score
                    recency = (i / len(df))
                    size_score = min(size / (atr + 1e-9), 1.0)
                    quality = 0.5 * recency + 0.5 * size_score

                    fvg = FVG(
                        fvg_type=FVGType.BULLISH,
                        top=c2["low"],
                        bottom=c0["high"],
                        index=i,
                        time=df["time"].iloc[i],
                        quality=quality,
                    )
                    fvgs.append(fvg)

            # ── Bearish FVG ──────────────────────────────────────────────────
            # Gap between c0.low and c2.high (c2.high < c0.low)
            if c2["high"] < c0["low"]:
                size = c0["low"] - c2["high"]
                if size >= min_size:
                    recency = (i / len(df))
                    size_score = min(size / (atr + 1e-9), 1.0)
                    quality = 0.5 * recency + 0.5 * size_score

                    fvg = FVG(
                        fvg_type=FVGType.BEARISH,
                        top=c0["low"],
                        bottom=c2["high"],
                        index=i,
                        time=df["time"].iloc[i],
                        quality=quality,
                    )
                    fvgs.append(fvg)

        # Update fill status
        self._update_fills(fvgs, df)

        # Remove fully filled
        fvgs = [f for f in fvgs if not f.fully_filled]

        # Sort by quality
        fvgs.sort(key=lambda f: f.quality, reverse=True)

        log.debug("Detected %d FVGs", len(fvgs))
        return fvgs

    # ── Fill detection ────────────────────────────────────────────────────────

    def _update_fills(self, fvgs: List[FVG], df: pd.DataFrame):
        for fvg in fvgs:
            post = df.iloc[fvg.index + 1:] if fvg.index + 1 < len(df) else pd.DataFrame()
            if post.empty:
                continue

            for _, row in post.iterrows():
                if fvg.fvg_type == FVGType.BULLISH:
                    # Price dips into the gap from above
                    if row["low"] <= fvg.top:
                        fvg.filled = True
                        # How deep into the gap did price go?
                        penetration = fvg.top - max(row["low"], fvg.bottom)
                        fvg.fill_percent = min(penetration / (fvg.size + 1e-9), 1.0)
                    if row["close"] < fvg.bottom:
                        fvg.fully_filled = True
                        break

                elif fvg.fvg_type == FVGType.BEARISH:
                    # Price rallies into the gap from below
                    if row["high"] >= fvg.bottom:
                        fvg.filled = True
                        penetration = min(row["high"], fvg.top) - fvg.bottom
                        fvg.fill_percent = min(penetration / (fvg.size + 1e-9), 1.0)
                    if row["close"] > fvg.top:
                        fvg.fully_filled = True
                        break

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low  = df["low"]
        prev = df["close"].shift(1)
        tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]


def detect_fvg(df: pd.DataFrame, symbol: str = "XAUUSD") -> List[FVG]:
    return FVGDetector().detect(df, symbol)
