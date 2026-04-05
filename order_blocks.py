"""
analysis/order_blocks.py
=========================
Detects Bullish and Bearish Order Blocks (OBs).

SMC Definition:
  • Bullish OB — the LAST BEARISH candle immediately before a significant bullish
                  impulse (BOS up). Price likely returns here to continue up.
  • Bearish OB — the LAST BULLISH candle immediately before a significant bearish
                  impulse (BOS down). Price likely returns here to continue down.

Quality scoring considers:
  - Size of the impulse that followed
  - Whether OB overlaps with FVG (confluence)
  - Freshness (untested OBs score higher)
  - Position relative to current price
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.market_structure import MarketStructure, StructureEvent, Trend
from config import config
from utils.logger import log


class OBType(Enum):
    BULLISH = "BULLISH"   # Demand zone — expect price to bounce UP
    BEARISH = "BEARISH"   # Supply zone — expect price to drop DOWN


@dataclass
class OrderBlock:
    ob_type: OBType
    top: float              # High of OB candle
    bottom: float           # Low of OB candle
    body_top: float         # Open or close (whichever is higher)
    body_bottom: float      # Open or close (whichever is lower)
    index: int              # Candle index in DataFrame
    time: pd.Timestamp
    impulse_size: float     # How large was the move that created this OB
    quality: float          # 0–1 quality score
    tested: bool = False    # Has price returned to this OB?
    test_count: int = 0     # How many times tested
    invalidated: bool = False  # Closed through → OB no longer valid

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def label(self) -> str:
        return f"{'Bull' if self.ob_type == OBType.BULLISH else 'Bear'} OB"

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.bottom - tolerance) <= price <= (self.top + tolerance)

    def __repr__(self) -> str:
        return (
            f"OrderBlock({self.label} | {self.bottom:.5f}–{self.top:.5f} "
            f"| Q={self.quality:.2f} | tested={self.tested})"
        )


class OrderBlockDetector:
    """
    Detect Order Blocks from M15 (or any) OHLCV DataFrame plus structure info.
    """

    def __init__(self):
        self.min_pips = config.OB_MIN_PIPS

    def detect(
        self,
        df: pd.DataFrame,
        structure: MarketStructure,
        symbol: str = "XAUUSD",
    ) -> List[OrderBlock]:
        """
        Primary entry point.
        Returns a list of valid (non-invalidated) OBs sorted by quality desc.
        """
        pip = config.SYMBOL_PIP.get(symbol, 0.0001)
        min_size = self.min_pips * pip

        obs: List[OrderBlock] = []

        # Method 1: OBs from confirmed BOS events
        obs.extend(self._obs_from_events(df, structure, pip))

        # Method 2: Structural OBs — last opposing candle before impulse
        obs.extend(self._structural_obs(df, pip))

        # Deduplicate and filter
        obs = self._deduplicate(obs, pip)
        obs = [ob for ob in obs if ob.size >= min_size and not ob.invalidated]

        # Update tested status
        self._update_tested(obs, df)

        # Re-sort by quality
        obs.sort(key=lambda o: o.quality, reverse=True)

        log.debug("Detected %d Order Blocks", len(obs))
        return obs

    # ── OBs from structure events ─────────────────────────────────────────────

    def _obs_from_events(
        self,
        df: pd.DataFrame,
        structure: MarketStructure,
        pip: float,
    ) -> List[OrderBlock]:
        obs = []
        for event in structure.events:
            if event.event in (StructureEvent.BOS_BULL, StructureEvent.CHOCH_BULL):
                # Look left for the last bearish candle before break
                ob = self._find_last_bearish_before(df, event.break_index, pip)
                if ob:
                    ob.quality = self._score(ob, event.strength, False)
                    obs.append(ob)

            elif event.event in (StructureEvent.BOS_BEAR, StructureEvent.CHOCH_BEAR):
                # Look left for the last bullish candle before break
                ob = self._find_last_bullish_before(df, event.break_index, pip)
                if ob:
                    ob.quality = self._score(ob, event.strength, True)
                    obs.append(ob)
        return obs

    def _find_last_bearish_before(
        self, df: pd.DataFrame, break_idx: int, pip: float
    ) -> Optional[OrderBlock]:
        for i in range(break_idx - 1, max(0, break_idx - 20), -1):
            row = df.iloc[i]
            if row["close"] < row["open"]:  # Bearish candle
                impulse = df["high"].iloc[i + 1: break_idx + 1].max() - row["high"]
                return OrderBlock(
                    ob_type=OBType.BULLISH,
                    top=row["high"],
                    bottom=row["low"],
                    body_top=row["open"],
                    body_bottom=row["close"],
                    index=i,
                    time=df["time"].iloc[i],
                    impulse_size=max(impulse, 0),
                    quality=0.5,
                )
        return None

    def _find_last_bullish_before(
        self, df: pd.DataFrame, break_idx: int, pip: float
    ) -> Optional[OrderBlock]:
        for i in range(break_idx - 1, max(0, break_idx - 20), -1):
            row = df.iloc[i]
            if row["close"] > row["open"]:  # Bullish candle
                impulse = row["low"] - df["low"].iloc[i + 1: break_idx + 1].min()
                return OrderBlock(
                    ob_type=OBType.BEARISH,
                    top=row["high"],
                    bottom=row["low"],
                    body_top=row["close"],
                    body_bottom=row["open"],
                    index=i,
                    time=df["time"].iloc[i],
                    impulse_size=max(impulse, 0),
                    quality=0.5,
                )
        return None

    # ── Structural OBs (scan whole DataFrame) ─────────────────────────────────

    def _structural_obs(self, df: pd.DataFrame, pip: float) -> List[OrderBlock]:
        """
        Scan for impulse moves (≥ ATR) preceded by opposing candle.
        Less strict than event-based — catches missed OBs.
        """
        obs = []
        atr = self._atr(df)
        impulse_threshold = atr * 1.5

        for i in range(1, len(df) - 2):
            # Bullish impulse candle
            bull_body = df["close"].iloc[i] - df["open"].iloc[i]
            if bull_body > impulse_threshold:
                # Look for last bearish candle at i-1
                prev = df.iloc[i - 1]
                if prev["close"] < prev["open"]:
                    ob = OrderBlock(
                        ob_type=OBType.BULLISH,
                        top=prev["high"],
                        bottom=prev["low"],
                        body_top=prev["open"],
                        body_bottom=prev["close"],
                        index=i - 1,
                        time=df["time"].iloc[i - 1],
                        impulse_size=bull_body,
                        quality=0.4,
                    )
                    obs.append(ob)

            # Bearish impulse candle
            bear_body = df["open"].iloc[i] - df["close"].iloc[i]
            if bear_body > impulse_threshold:
                prev = df.iloc[i - 1]
                if prev["close"] > prev["open"]:
                    ob = OrderBlock(
                        ob_type=OBType.BEARISH,
                        top=prev["high"],
                        bottom=prev["low"],
                        body_top=prev["close"],
                        body_bottom=prev["open"],
                        index=i - 1,
                        time=df["time"].iloc[i - 1],
                        impulse_size=bear_body,
                        quality=0.4,
                    )
                    obs.append(ob)

        return obs

    # ── Post-processing ───────────────────────────────────────────────────────

    def _update_tested(self, obs: List[OrderBlock], df: pd.DataFrame):
        """Mark OBs that price has returned to (tested) or closed through (invalidated)."""
        df_reset = df.reset_index(drop=True)  # ensure positional indexing
        for ob in obs:
            start = ob.index + 1
            if start >= len(df_reset):
                continue
            post = df_reset.iloc[start:]

            for _, row in post.iterrows():
                if ob.contains(row["low"]) or ob.contains(row["high"]):
                    ob.tested = True
                    ob.test_count += 1

                # Invalidation: full candle close through OB
                if ob.ob_type == OBType.BULLISH and row["close"] < ob.bottom:
                    ob.invalidated = True
                    break
                if ob.ob_type == OBType.BEARISH and row["close"] > ob.top:
                    ob.invalidated = True
                    break

    def _deduplicate(
        self, obs: List[OrderBlock], pip: float, tolerance_pips: float = 5.0
    ) -> List[OrderBlock]:
        """Remove overlapping OBs of the same type, keeping the one with highest quality."""
        tol = tolerance_pips * pip
        unique: List[OrderBlock] = []
        for ob in obs:
            replaced = False
            for j, u in enumerate(unique):
                if (u.ob_type == ob.ob_type and abs(u.mid - ob.mid) < tol):
                    if ob.quality > u.quality:
                        unique[j] = ob   # replace in-place — no list.remove()
                    replaced = True
                    break
            if not replaced:
                unique.append(ob)
        return unique

    def _score(self, ob: OrderBlock, event_strength: float, is_bearish_ob: bool) -> float:
        """
        Quality score 0–1:
        - 40% from impulse event strength
        - 30% from candle body ratio (bigger body = cleaner OB)
        - 20% from test status (untested = better)
        - 10% bonus if this is the most recent OB
        """
        body_ratio = min(ob.size / (ob.impulse_size + 1e-9), 1.0)
        tested_penalty = 0.2 if ob.tested else 0.0
        return max(
            0.4 * event_strength
            + 0.3 * body_ratio
            + 0.2 * (1 - tested_penalty)
            + 0.1,
            0.0
        )

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low  = df["low"]
        prev = df["close"].shift(1)
        tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]


def detect_order_blocks(
    df: pd.DataFrame,
    structure: MarketStructure,
    symbol: str = "XAUUSD",
) -> List[OrderBlock]:
    return OrderBlockDetector().detect(df, structure, symbol)
