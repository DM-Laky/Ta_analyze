"""
analysis/liquidity.py
======================
Detects liquidity pools in the market:

  1. Equal Highs (EQH) — cluster of swing highs at similar price
                         → Lots of stop losses sitting above → ripe for sweep
  2. Equal Lows  (EQL) — cluster of swing lows at similar price
                         → Lots of stop losses sitting below
  3. Previous Day High / Low (PDH/PDL) — major liquidity targets
  4. Swing Highs / Lows as liquidity targets
  5. Liquidity Sweep detection — price spikes beyond EQH/EQL then reverses

After a sweep, the "next" liquidity becomes the TP target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.market_structure import SwingPoint
from config import config
from utils.logger import log


class LiqType(Enum):
    EQH      = "EQH"      # Equal Highs
    EQL      = "EQL"      # Equal Lows
    PDH      = "PDH"      # Previous Day High
    PDL      = "PDL"      # Previous Day Low
    SWING_H  = "SWING_H"  # Swing High (major)
    SWING_L  = "SWING_L"  # Swing Low (major)


@dataclass
class LiquidityLevel:
    liq_type: LiqType
    price: float
    time: pd.Timestamp
    strength: float        # 0–1 (how many touches / how significant)
    swept: bool = False    # Has this been swept (stop hunt occurred)?
    sweep_time: Optional[pd.Timestamp] = None

    @property
    def label(self) -> str:
        return self.liq_type.value

    @property
    def is_above(self) -> bool:
        """True if this is a high-side liquidity (target for bearish sweep)."""
        return self.liq_type in (LiqType.EQH, LiqType.PDH, LiqType.SWING_H)

    def __repr__(self) -> str:
        return f"Liquidity({self.label} @ {self.price:.5f} | str={self.strength:.2f} | swept={self.swept})"


@dataclass
class LiquiditySweep:
    """Represents a detected stop hunt."""
    level: LiquidityLevel
    sweep_high: float      # Wick that swept beyond level
    sweep_low: float
    sweep_candle_idx: int
    sweep_time: pd.Timestamp
    reversal_confirmed: bool   # Did price quickly reverse after sweep?
    direction: str             # "BULL_SWEEP" or "BEAR_SWEEP"

    def __repr__(self) -> str:
        return (
            f"LiqSweep({self.direction} @ {self.level.price:.5f} "
            f"| reversal={self.reversal_confirmed})"
        )


class LiquidityAnalyzer:
    """Detect liquidity pools and sweeps."""

    def __init__(self):
        self.tol = config.LIQ_CLUSTER_TOLERANCE

    def detect_levels(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        symbol: str = "XAUUSD",
    ) -> List[LiquidityLevel]:
        """
        Detect all liquidity levels in `df`.
        Returns list sorted by strength desc.
        """
        levels: List[LiquidityLevel] = []

        # 1. Equal Highs / Lows (swing clustering)
        levels.extend(self._equal_highs(swing_highs, df))
        levels.extend(self._equal_lows(swing_lows, df))

        # 2. Previous Day H/L
        levels.extend(self._prev_day_hl(df))

        # 3. Prominent swing points as liquidity
        levels.extend(self._swing_liquidity(swing_highs, swing_lows))

        # Sort and deduplicate
        levels = self._deduplicate(levels, symbol)
        levels.sort(key=lambda l: l.strength, reverse=True)

        log.debug("Detected %d liquidity levels", len(levels))
        return levels

    def detect_sweeps(
        self,
        df: pd.DataFrame,
        levels: List[LiquidityLevel],
    ) -> List[LiquiditySweep]:
        """
        Check if any liquidity level was recently swept.
        A sweep = wick beyond level BUT candle closes back on the other side.
        """
        sweeps: List[LiquiditySweep] = []
        n = len(df)
        # Reset index so positional iloc matches iteration order
        df_pos = df.reset_index(drop=True)

        for level in levels:
            if level.swept:
                continue

            check_range = df_pos.iloc[max(0, n - 10):]

            for pos_idx, row in check_range.iterrows():
                # High-side sweep (bearish sweep of EQH/PDH)
                if level.is_above:
                    if row["high"] > level.price and row["close"] < level.price:
                        reversal = self._check_reversal(df_pos, pos_idx, direction="bear")
                        sweep = LiquiditySweep(
                            level=level,
                            sweep_high=row["high"],
                            sweep_low=row["low"],
                            sweep_candle_idx=pos_idx,
                            sweep_time=row["time"],
                            reversal_confirmed=reversal,
                            direction="BEAR_SWEEP",
                        )
                        sweeps.append(sweep)
                        level.swept = True
                        level.sweep_time = row["time"]
                        break

                else:
                    if row["low"] < level.price and row["close"] > level.price:
                        reversal = self._check_reversal(df_pos, pos_idx, direction="bull")
                        sweep = LiquiditySweep(
                            level=level,
                            sweep_high=row["high"],
                            sweep_low=row["low"],
                            sweep_candle_idx=pos_idx,
                            sweep_time=row["time"],
                            reversal_confirmed=reversal,
                            direction="BULL_SWEEP",
                        )
                        sweeps.append(sweep)
                        level.swept = True
                        level.sweep_time = row["time"]
                        break

        return sweeps

    def nearest_target(
        self,
        levels: List[LiquidityLevel],
        current_price: float,
        direction: str,  # "BUY" or "SELL"
        min_distance_pct: float = 0.001,
    ) -> Optional[LiquidityLevel]:
        """
        Find the nearest unswept liquidity level in the direction of the trade.
        Used for TP targeting.
        """
        candidates = [
            l for l in levels
            if not l.swept
            and (
                (direction == "BUY"  and l.price > current_price * (1 + min_distance_pct)) or
                (direction == "SELL" and l.price < current_price * (1 - min_distance_pct))
            )
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda l: abs(l.price - current_price)
        )
        return candidates[0]

    def tp_targets(
        self,
        levels: List[LiquidityLevel],
        current_price: float,
        direction: str,
        n: int = 3,
    ) -> List[float]:
        """Return up to n TP price targets from liquidity levels."""
        if direction == "BUY":
            above = sorted(
                [l.price for l in levels if l.price > current_price and not l.swept]
            )
            return above[:n]
        else:
            below = sorted(
                [l.price for l in levels if l.price < current_price and not l.swept],
                reverse=True,
            )
            return below[:n]

    # ── Internal detection methods ────────────────────────────────────────────

    def _equal_highs(
        self, swing_highs: List[SwingPoint], df: pd.DataFrame
    ) -> List[LiquidityLevel]:
        """Cluster swing highs within tolerance → Equal Highs."""
        if len(swing_highs) < 2:
            return []

        prices = [s.price for s in swing_highs]
        levels = []
        used = set()

        for i, s in enumerate(swing_highs):
            if i in used:
                continue
            cluster = [s]
            for j, s2 in enumerate(swing_highs[i + 1:], start=i + 1):
                if j not in used and abs(s.price - s2.price) / s.price < self.tol:
                    cluster.append(s2)
                    used.add(j)
            if len(cluster) >= 2:
                avg_price = sum(c.price for c in cluster) / len(cluster)
                strength = min(len(cluster) / 5, 1.0)
                levels.append(LiquidityLevel(
                    liq_type=LiqType.EQH,
                    price=avg_price,
                    time=cluster[-1].time,
                    strength=strength,
                ))
        return levels

    def _equal_lows(
        self, swing_lows: List[SwingPoint], df: pd.DataFrame
    ) -> List[LiquidityLevel]:
        if len(swing_lows) < 2:
            return []

        levels = []
        used = set()

        for i, s in enumerate(swing_lows):
            if i in used:
                continue
            cluster = [s]
            for j, s2 in enumerate(swing_lows[i + 1:], start=i + 1):
                if j not in used and abs(s.price - s2.price) / s.price < self.tol:
                    cluster.append(s2)
                    used.add(j)
            if len(cluster) >= 2:
                avg_price = sum(c.price for c in cluster) / len(cluster)
                strength = min(len(cluster) / 5, 1.0)
                levels.append(LiquidityLevel(
                    liq_type=LiqType.EQL,
                    price=avg_price,
                    time=cluster[-1].time,
                    strength=strength,
                ))
        return levels

    def _prev_day_hl(self, df: pd.DataFrame) -> List[LiquidityLevel]:
        """Extract previous day high and low."""
        levels = []
        if "time" not in df.columns:
            return levels

        df2 = df.copy()
        df2["date"] = df2["time"].dt.date
        daily = df2.groupby("date").agg({"high": "max", "low": "min", "time": "last"}).reset_index()

        if len(daily) >= 2:
            prev = daily.iloc[-2]
            levels.append(LiquidityLevel(
                liq_type=LiqType.PDH,
                price=prev["high"],
                time=prev["time"],
                strength=0.85,
            ))
            levels.append(LiquidityLevel(
                liq_type=LiqType.PDL,
                price=prev["low"],
                time=prev["time"],
                strength=0.85,
            ))
        return levels

    def _swing_liquidity(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[LiquidityLevel]:
        """Add major swing points as liquidity targets."""
        levels = []
        # Only the last 5 significant swings
        for s in swing_highs[-5:]:
            levels.append(LiquidityLevel(
                liq_type=LiqType.SWING_H,
                price=s.price,
                time=s.time,
                strength=0.6,
            ))
        for s in swing_lows[-5:]:
            levels.append(LiquidityLevel(
                liq_type=LiqType.SWING_L,
                price=s.price,
                time=s.time,
                strength=0.6,
            ))
        return levels

    def _deduplicate(
        self, levels: List[LiquidityLevel], symbol: str
    ) -> List[LiquidityLevel]:
        pip = config.SYMBOL_PIP.get(symbol, 0.0001)
        tol_price = pip * 20
        unique = []
        for l in levels:
            if not any(abs(u.price - l.price) < tol_price for u in unique):
                unique.append(l)
        return unique

    def _check_reversal(
        self, df: pd.DataFrame, pos_idx: int, direction: str
    ) -> bool:
        """Check if the 1–2 candles after sweep show reversal. df must be positionally indexed."""
        post = df.iloc[pos_idx + 1: pos_idx + 3]
        if post.empty:
            return False
        if direction == "bull":
            return all(row["close"] > row["open"] for _, row in post.iterrows())
        else:
            return all(row["close"] < row["open"] for _, row in post.iterrows())


def detect_liquidity(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    symbol: str = "XAUUSD",
) -> Tuple[List[LiquidityLevel], List[LiquiditySweep]]:
    analyzer = LiquidityAnalyzer()
    levels = analyzer.detect_levels(df, swing_highs, swing_lows, symbol)
    sweeps = analyzer.detect_sweeps(df, levels)
    return levels, sweeps
