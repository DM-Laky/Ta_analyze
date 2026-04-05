"""
analysis/smc_engine.py
=======================
Master SMC Orchestrator.

For each symbol, every analysis cycle:
  1. Fetch H1 candles → Market Structure (BOS, CHoCH, trend)
  2. Fetch M15 candles → Order Blocks + FVGs
  3. Merge → find confluent POI zones (OB + FVG overlap = gold setup)
  4. Detect liquidity sweeps
  5. Score each potential setup
  6. Return SetupCandidate list for watchlist injection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

from analysis.fvg_detector import FVG, FVGType, detect_fvg
from analysis.liquidity import (
    LiquidityLevel, LiquiditySweep, LiquidityAnalyzer, detect_liquidity
)
from analysis.market_structure import (
    MarketStructure, StructureEvent, Trend, analyze_structure
)
from analysis.order_blocks import OrderBlock, OBType, detect_order_blocks
from config import config
from core.data_fetcher import fetcher
from core.session_manager import session_manager, SessionInfo
from utils.logger import log


# ── Setup Candidate ───────────────────────────────────────────────────────────

@dataclass
class SetupCandidate:
    """A potential trading setup waiting for entry confirmation."""
    symbol: str
    direction: str                 # "BUY" or "SELL"
    poi_top: float                 # POI zone top
    poi_bottom: float              # POI zone bottom
    poi_mid: float                 # POI zone midpoint
    poi_type: str                  # "OB", "FVG", "OB+FVG" (confluence)
    
    # Context
    h1_trend: Trend
    h1_structure: MarketStructure
    m15_obs: List[OrderBlock]
    m15_fvgs: List[FVG]
    liq_levels: List[LiquidityLevel]
    liq_sweeps: List[LiquiditySweep]
    session: SessionInfo
    
    # Scoring
    confluence_score: float        # 0–1
    setup_score: float             # Overall score 0–1
    
    # TP targets from next liquidity
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_hours: int = field(default_factory=lambda: config.WATCHLIST_EXPIRY_HOURS)
    
    # Supporting OB and FVG (may be None if single-reason)
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FVG] = None
    
    # Reason string for alert
    reason: str = ""
    
    @property
    def is_expired(self) -> bool:
        from datetime import timedelta
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() > self.expiry_hours * 3600

    @property
    def rr_estimate(self) -> float:
        """Estimate R:R using TP1 and a 1-ATR SL."""
        if self.tp1 is None:
            return 0.0
        risk = abs(self.poi_mid - self.poi_bottom) if self.direction == "BUY" else abs(self.poi_top - self.poi_mid)
        reward = abs(self.tp1 - self.poi_mid)
        return reward / (risk + 1e-9)

    def __repr__(self) -> str:
        return (
            f"SetupCandidate({self.symbol} {self.direction} "
            f"| POI={self.poi_bottom:.5f}–{self.poi_top:.5f} "
            f"| type={self.poi_type} | score={self.setup_score:.2f})"
        )


# ── SMC Engine ────────────────────────────────────────────────────────────────

class SMCEngine:
    """
    Runs full SMC analysis for one symbol.
    Call .analyze(symbol) → List[SetupCandidate]
    """

    def analyze(self, symbol: str) -> List[SetupCandidate]:
        log.info("🔬 Deep SMC analysis starting: %s", symbol)
        session = session_manager.get_current_session()

        # ── 1. Fetch candles ──────────────────────────────────────────────────
        df_h1  = fetcher.get_candles(symbol, "H1",  config.HTF_CANDLES)
        df_m15 = fetcher.get_candles(symbol, "M15", config.MTF_CANDLES)

        if df_h1 is None or df_m15 is None:
            log.error("Cannot fetch candles for %s", symbol)
            return []

        log.debug("%s | H1=%d bars, M15=%d bars", symbol, len(df_h1), len(df_m15))

        # ── 2. H1 Market Structure ────────────────────────────────────────────
        h1_structure = analyze_structure(df_h1)
        log.info(
            "%s | H1 Trend: %s | Events: %d | HH=%s HL=%s",
            symbol,
            h1_structure.trend.value,
            len(h1_structure.events),
            h1_structure.higher_high,
            h1_structure.higher_low,
        )

        # ── 3. M15 OB + FVG ──────────────────────────────────────────────────
        m15_structure = analyze_structure(df_m15, left=2, right=2)
        obs   = detect_order_blocks(df_m15, m15_structure, symbol)
        fvgs  = detect_fvg(df_m15, symbol)
        liq_levels, liq_sweeps = detect_liquidity(
            df_m15,
            m15_structure.swing_highs,
            m15_structure.swing_lows,
            symbol,
        )

        log.info(
            "%s | M15 OBs=%d, FVGs=%d, LiqLevels=%d, Sweeps=%d",
            symbol, len(obs), len(fvgs), len(liq_levels), len(liq_sweeps),
        )

        # ── 4. Build setup candidates ─────────────────────────────────────────
        candidates: List[SetupCandidate] = []

        # Determine eligible trade direction from H1 trend + recent events
        eligible_dirs = self._eligible_directions(h1_structure, liq_sweeps)
        log.info("%s | Eligible directions: %s", symbol, eligible_dirs)

        for direction in eligible_dirs:
            ob_type  = OBType.BULLISH if direction == "BUY" else OBType.BEARISH
            fvg_type = FVGType.BULLISH if direction == "BUY" else FVGType.BEARISH

            # Filter OBs and FVGs to matching direction
            dir_obs  = [o for o in obs  if o.ob_type  == ob_type  and not o.invalidated]
            dir_fvgs = [f for f in fvgs if f.fvg_type == fvg_type and not f.fully_filled]

            # ── POI from OB + FVG confluence ─────────────────────────────────
            for ob in dir_obs[:3]:  # Top-3 OBs
                for fvg in dir_fvgs[:3]:
                    overlap = self._zone_overlap(
                        ob.bottom, ob.top, fvg.bottom, fvg.top
                    )
                    if overlap is not None:
                        poi_b, poi_t = overlap
                        cand = self._build_candidate(
                            symbol, direction, poi_b, poi_t,
                            poi_type="OB+FVG",
                            ob=ob, fvg=fvg,
                            h1_structure=h1_structure,
                            m15_obs=obs, m15_fvgs=fvgs,
                            liq_levels=liq_levels, liq_sweeps=liq_sweeps,
                            session=session,
                        )
                        if cand:
                            candidates.append(cand)

            # ── POI from OB alone ─────────────────────────────────────────────
            for ob in dir_obs[:5]:
                cand = self._build_candidate(
                    symbol, direction, ob.bottom, ob.top,
                    poi_type="OB",
                    ob=ob, fvg=None,
                    h1_structure=h1_structure,
                    m15_obs=obs, m15_fvgs=fvgs,
                    liq_levels=liq_levels, liq_sweeps=liq_sweeps,
                    session=session,
                )
                if cand:
                    candidates.append(cand)

            # ── POI from FVG alone ────────────────────────────────────────────
            for fvg in dir_fvgs[:5]:
                cand = self._build_candidate(
                    symbol, direction, fvg.bottom, fvg.top,
                    poi_type="FVG",
                    ob=None, fvg=fvg,
                    h1_structure=h1_structure,
                    m15_obs=obs, m15_fvgs=fvgs,
                    liq_levels=liq_levels, liq_sweeps=liq_sweeps,
                    session=session,
                )
                if cand:
                    candidates.append(cand)

        # Deduplicate and sort
        candidates = self._deduplicate_candidates(candidates, symbol)
        candidates.sort(key=lambda c: c.setup_score, reverse=True)

        # Keep only top-N per symbol
        candidates = candidates[:config.MAX_WATCHLIST_PER_SYMBOL]

        log.info(
            "✅ %s analysis complete | %d setup candidates found",
            symbol, len(candidates),
        )
        return candidates

    # ── Direction Eligibility ─────────────────────────────────────────────────

    def _eligible_directions(
        self,
        structure: MarketStructure,
        sweeps: List[LiquiditySweep],
    ) -> List[str]:
        """
        Determine valid trade directions based on:
        - H1 trend (trade with trend = higher probability)
        - Recent CHoCH (possible reversal = opposite direction)
        - Liquidity sweeps (sweep of lows → possible BUY reversal)
        """
        dirs = []

        # Recent events (last 3)
        recent = structure.events[-3:] if structure.events else []
        recent_types = {e.event for e in recent}

        bull_events = {
            StructureEvent.BOS_BULL,
            StructureEvent.CHOCH_BULL,
            StructureEvent.MSB_BULL,
        }
        bear_events = {
            StructureEvent.BOS_BEAR,
            StructureEvent.CHOCH_BEAR,
            StructureEvent.MSB_BEAR,
        }

        if structure.trend == Trend.BULLISH:
            dirs.append("BUY")          # Trend trades
            if recent_types & bear_events:
                dirs.append("SELL")    # Possible reversal

        elif structure.trend == Trend.BEARISH:
            dirs.append("SELL")
            if recent_types & bull_events:
                dirs.append("BUY")

        else:  # RANGING
            dirs = ["BUY", "SELL"]     # Both sides valid in range

        # If there are recent sweeps, add the reversal direction
        for sweep in sweeps[-3:]:
            if sweep.reversal_confirmed:
                if sweep.direction == "BULL_SWEEP" and "BUY" not in dirs:
                    dirs.append("BUY")
                elif sweep.direction == "BEAR_SWEEP" and "SELL" not in dirs:
                    dirs.append("SELL")

        return dirs or ["BUY", "SELL"]

    # ── Candidate Builder ─────────────────────────────────────────────────────

    def _build_candidate(
        self,
        symbol: str,
        direction: str,
        poi_bottom: float,
        poi_top: float,
        poi_type: str,
        ob: Optional[OrderBlock],
        fvg: Optional[FVG],
        h1_structure: MarketStructure,
        m15_obs: List[OrderBlock],
        m15_fvgs: List[FVG],
        liq_levels: List[LiquidityLevel],
        liq_sweeps: List[LiquiditySweep],
        session: SessionInfo,
    ) -> Optional[SetupCandidate]:
        poi_mid = (poi_top + poi_bottom) / 2

        # TP targets from next liquidity
        liq_analyzer = LiquidityAnalyzer()
        tp_targets = liq_analyzer.tp_targets(liq_levels, poi_mid, direction, n=3)
        if not tp_targets:
            # Fallback: use a 2R / 3R projection
            risk = (poi_top - poi_bottom)
            if direction == "BUY":
                tp_targets = [poi_mid + risk * r for r in (2, 3, 5)]
            else:
                tp_targets = [poi_mid - risk * r for r in (2, 3, 5)]

        # Check minimum R:R
        risk = poi_top - poi_bottom
        reward = abs(tp_targets[0] - poi_mid) if tp_targets else 0
        rr = reward / (risk + 1e-9)
        if rr < config.MIN_RR:
            return None

        # Confluence score
        confluence = self._confluence_score(
            poi_type, ob, fvg, h1_structure, liq_sweeps, session
        )

        # Build reason string
        reason = self._build_reason(
            direction, poi_type, ob, fvg, h1_structure, liq_sweeps, rr
        )

        return SetupCandidate(
            symbol=symbol,
            direction=direction,
            poi_top=poi_top,
            poi_bottom=poi_bottom,
            poi_mid=poi_mid,
            poi_type=poi_type,
            h1_trend=h1_structure.trend,
            h1_structure=h1_structure,
            m15_obs=m15_obs,
            m15_fvgs=m15_fvgs,
            liq_levels=liq_levels,
            liq_sweeps=liq_sweeps,
            session=session,
            confluence_score=confluence,
            setup_score=confluence,  # single source of truth
            tp1=tp_targets[0] if len(tp_targets) > 0 else None,
            tp2=tp_targets[1] if len(tp_targets) > 1 else None,
            tp3=tp_targets[2] if len(tp_targets) > 2 else None,
            order_block=ob,
            fvg=fvg,
            reason=reason,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _confluence_score(
        self,
        poi_type: str,
        ob: Optional[OrderBlock],
        fvg: Optional[FVG],
        structure: MarketStructure,
        sweeps: List[LiquiditySweep],
        session: SessionInfo,
    ) -> float:
        score = 0.0

        # POI type weight
        if poi_type == "OB+FVG":   score += 0.35
        elif poi_type == "OB":     score += 0.20
        elif poi_type == "FVG":    score += 0.15

        # Trend alignment
        if structure.trend != Trend.RANGING:
            score += 0.15

        # H1 structure quality (HH+HL or LL+LH)
        if (structure.higher_high and structure.higher_low) or \
           (structure.lower_high and structure.lower_low):
            score += 0.10

        # Recent liquidity sweep (confirmation of institutional intent)
        if sweeps:
            recent_confirmed = any(s.reversal_confirmed for s in sweeps[-3:])
            score += 0.20 if recent_confirmed else 0.10

        # Session quality
        score += 0.10 * session_manager.session_quality()

        # OB quality bonus
        if ob and ob.quality > 0.7:
            score += 0.10

        return min(score, 1.0)

    def _build_reason(
        self,
        direction: str,
        poi_type: str,
        ob: Optional[OrderBlock],
        fvg: Optional[FVG],
        structure: MarketStructure,
        sweeps: List[LiquiditySweep],
        rr: float,
    ) -> str:
        parts = []
        emoji = "🟢" if direction == "BUY" else "🔴"
        parts.append(f"{emoji} {direction} setup on {poi_type}")
        parts.append(f"H1 Trend: {structure.trend.value}")

        if ob:
            parts.append(
                f"{'Bullish' if ob.ob_type.value == 'BULLISH' else 'Bearish'} OB "
                f"({ob.bottom:.5f}–{ob.top:.5f})"
            )
        if fvg:
            parts.append(
                f"{'Bullish' if fvg.fvg_type.value == 'BULLISH' else 'Bearish'} FVG "
                f"({fvg.bottom:.5f}–{fvg.top:.5f})"
            )

        recent = structure.events[-1] if structure.events else None
        if recent:
            parts.append(f"Last event: {recent.event.value}")

        if sweeps:
            last_sweep = sweeps[-1]
            parts.append(
                f"Liquidity sweep detected: {last_sweep.direction} "
                f"(reversal={'✅' if last_sweep.reversal_confirmed else '⏳'})"
            )

        parts.append(f"Est. R:R = {rr:.1f}R")
        return " | ".join(parts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _zone_overlap(
        self,
        a_bot: float, a_top: float,
        b_bot: float, b_top: float,
    ) -> Optional[Tuple[float, float]]:
        """Return the overlap zone of two price ranges, or None if no overlap."""
        overlap_bot = max(a_bot, b_bot)
        overlap_top = min(a_top, b_top)
        if overlap_bot < overlap_top:
            return overlap_bot, overlap_top
        return None

    def _deduplicate_candidates(
        self,
        candidates: List[SetupCandidate],
        symbol: str,
    ) -> List[SetupCandidate]:
        pip = config.SYMBOL_PIP.get(symbol, 0.0001)
        tol = pip * 50  # 50 pips tolerance for deduplication
        unique: List[SetupCandidate] = []
        for c in candidates:
            replaced = False
            for j, u in enumerate(unique):
                if (u.direction == c.direction and
                        abs(u.poi_mid - c.poi_mid) < tol):
                    if c.setup_score > u.setup_score:
                        unique[j] = c   # replace in-place — no list.remove()
                    replaced = True
                    break
            if not replaced:
                unique.append(c)
        return unique


# Module-level singleton
smc_engine = SMCEngine()
