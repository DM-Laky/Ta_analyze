"""
signals/entry_engine.py
========================
Entry Confirmation Engine (LTF — 1M).

When a watchlist entry is triggered (price enters POI):
  1. Fetch last N 1M candles
  2. Scan for CHoCH (Change of Character) on 1M — structural confirmation
  3. Scan for V-Shape movement — momentum confirmation
  4. If either is confirmed → build the full entry signal
  5. Generate 5-split entry zone, SL, TP1/2/3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.market_structure import (
    MarketStructureAnalyzer, StructureEvent, Trend
)
from config import config
from core.data_fetcher import fetcher
from signals.watchlist import WatchlistEntry
from utils.logger import log


# ── Confirmation result ───────────────────────────────────────────────────────

@dataclass
class ConfirmationResult:
    confirmed: bool
    method: str              # "CHOCH" | "VSHAPE" | "BOTH" | "NONE"
    confidence: float        # 0–1
    details: str             # Human-readable explanation


# ── Entry Signal ──────────────────────────────────────────────────────────────

@dataclass
class EntrySignal:
    uid: str                       # Link back to watchlist entry UID
    symbol: str
    direction: str                 # "BUY" | "SELL"

    # Entry zone (5 laddered entries)
    entries: List[float]           # 5 price levels
    entry_zone_top: float
    entry_zone_bottom: float

    # Risk levels
    stop_loss: float
    tp1: float
    tp2: Optional[float]
    tp3: Optional[float]

    # Confirmation details
    confirmation: ConfirmationResult
    setup_score: float
    rr1: float                     # R:R to TP1
    rr2: Optional[float]           # R:R to TP2

    # Context
    poi_type: str
    h1_trend: Trend
    session_name: str

    # Timing
    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Chart data
    df_m15: Optional[pd.DataFrame] = None
    df_m1: Optional[pd.DataFrame] = None

    @property
    def avg_entry(self) -> float:
        return sum(self.entries) / len(self.entries)

    @property
    def risk_pips(self) -> float:
        pip = config.SYMBOL_PIP.get(self.symbol, 0.0001)
        return abs(self.avg_entry - self.stop_loss) / pip

    def summary_line(self) -> str:
        emoji = "🟢 BUY" if self.direction == "BUY" else "🔴 SELL"
        return (
            f"{emoji} {self.symbol} "
            f"| Entry: {self.avg_entry:.5f} "
            f"| SL: {self.stop_loss:.5f} "
            f"| TP1: {self.tp1:.5f} "
            f"| R:R {self.rr1:.1f}R"
        )

    def __repr__(self) -> str:
        return f"EntrySignal({self.summary_line()})"


# ── Confirmation Engine ───────────────────────────────────────────────────────

class EntryConfirmationEngine:

    def __init__(self):
        self.lookback = config.CONFIRMATION_LOOKBACK
        self.vshape_min = config.VSHAPE_RETRACE_MIN

    def check(
        self, entry: WatchlistEntry, current_price: float,
        df_m15: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[EntrySignal], ConfirmationResult]:
        """
        Main entry point.
        Fetches 1M candles and checks for CHoCH or V-shape confirmation.
        Returns (signal, confirmation_result).
        """
        symbol = entry.symbol
        direction = entry.direction

        # Fetch 1M candles
        df_m1 = fetcher.get_candles(symbol, "M1", self.lookback)
        if df_m1 is None or len(df_m1) < 5:
            log.warning("Insufficient 1M data for %s", symbol)
            return None, ConfirmationResult(False, "NONE", 0, "No 1M data")

        # ── Run confirmation checks ───────────────────────────────────────────
        choch_result = self._check_choch(df_m1, direction)
        vshape_result = self._check_vshape(df_m1, direction, entry)

        # Combine results
        confirmed = choch_result.confirmed or vshape_result.confirmed
        if choch_result.confirmed and vshape_result.confirmed:
            method = "BOTH"
            confidence = min((choch_result.confidence + vshape_result.confidence) / 2 + 0.1, 1.0)
            details = f"CHoCH ✅ + V-Shape ✅ | {choch_result.details} | {vshape_result.details}"
        elif choch_result.confirmed:
            method = "CHOCH"
            confidence = choch_result.confidence
            details = choch_result.details
        elif vshape_result.confirmed:
            method = "VSHAPE"
            confidence = vshape_result.confidence
            details = vshape_result.details
        else:
            confirmation = ConfirmationResult(False, "NONE", 0, "No confirmation found")
            return None, confirmation

        confirmation = ConfirmationResult(confirmed, method, confidence, details)

        # ── Build entry signal ────────────────────────────────────────────────
        signal = self._build_signal(entry, confirmation, current_price, df_m1, df_m15)
        return signal, confirmation

    # ── CHoCH Detection on 1M ─────────────────────────────────────────────────

    def _check_choch(
        self, df_m1: pd.DataFrame, direction: str
    ) -> ConfirmationResult:
        """
        Detect CHoCH on 1M:
        - BUY  confirmation: after touching bullish POI, price makes HH on 1M
                              (breaks a recent swing high → CHoCH bullish)
        - SELL confirmation: after touching bearish POI, price makes LL on 1M
        """
        try:
            analyzer = MarketStructureAnalyzer(left_bars=2, right_bars=2)
            structure = analyzer.analyze(df_m1)

            bull_events = {StructureEvent.BOS_BULL, StructureEvent.CHOCH_BULL}
            bear_events = {StructureEvent.BOS_BEAR, StructureEvent.CHOCH_BEAR}

            if not structure.events:
                return ConfirmationResult(False, "CHOCH", 0, "No structure events on 1M")

            # Only check the most recent events
            recent = structure.events[-3:]
            recent_types = {e.event for e in recent}

            if direction == "BUY" and recent_types & bull_events:
                last = [e for e in recent if e.event in bull_events][-1]
                conf = 0.6 + last.strength * 0.4
                return ConfirmationResult(
                    True, "CHOCH", conf,
                    f"1M {last.event.value} detected at {last.broken_price:.5f}"
                )

            if direction == "SELL" and recent_types & bear_events:
                last = [e for e in recent if e.event in bear_events][-1]
                conf = 0.6 + last.strength * 0.4
                return ConfirmationResult(
                    True, "CHOCH", conf,
                    f"1M {last.event.value} detected at {last.broken_price:.5f}"
                )

        except Exception as exc:
            log.warning("CHoCH check failed: %s", exc)

        return ConfirmationResult(False, "CHOCH", 0, "No 1M CHoCH detected")

    # ── V-Shape Detection ─────────────────────────────────────────────────────

    def _check_vshape(
        self,
        df_m1: pd.DataFrame,
        direction: str,
        entry: WatchlistEntry,
    ) -> ConfirmationResult:
        """
        V-Shape:
        - BUY : price drops into POI (creating a wick/low), then rallies back
                 closing significantly above the low (retrace ≥ 50% of the drop)
        - SELL: price spikes into POI, then drops back

        We scan the last 5 candles.
        """
        last5 = df_m1.iloc[-max(self.lookback // 2, 5):].reset_index(drop=True)
        if len(last5) < 3:
            return ConfirmationResult(False, "VSHAPE", 0, "< 3 candles")

        try:
            if direction == "BUY":
                # Find the lowest point in last 5 candles
                low_idx = last5["low"].idxmin()
                low_price = last5["low"].iloc[low_idx]
                # Candles after the low
                after = last5.iloc[low_idx + 1:]
                if after.empty:
                    return ConfirmationResult(False, "VSHAPE", 0, "Low is last candle")

                # High point after the low
                high_after = after["high"].max()
                recovery = high_after - low_price

                # Reference move (previous drop to the low)
                before = last5.iloc[:low_idx + 1]
                drop = before["high"].max() - low_price if not before.empty else recovery

                if drop < 1e-9:
                    return ConfirmationResult(False, "VSHAPE", 0, "No meaningful drop")

                retrace_ratio = recovery / drop
                if retrace_ratio >= self.vshape_min:
                    conf = min(0.5 + retrace_ratio * 0.5, 1.0)
                    return ConfirmationResult(
                        True, "VSHAPE", conf,
                        f"V-Shape ↑ | Low={low_price:.5f} | Recovery={recovery:.5f} | "
                        f"Retrace={retrace_ratio:.0%}"
                    )

            elif direction == "SELL":
                high_idx = last5["high"].idxmax()
                high_price = last5["high"].iloc[high_idx]
                after = last5.iloc[high_idx + 1:]
                if after.empty:
                    return ConfirmationResult(False, "VSHAPE", 0, "High is last candle")

                low_after = after["low"].min()
                drop = high_price - low_after

                before = last5.iloc[:high_idx + 1]
                spike = high_price - before["low"].min() if not before.empty else drop

                if spike < 1e-9:
                    return ConfirmationResult(False, "VSHAPE", 0, "No meaningful spike")

                retrace_ratio = drop / spike
                if retrace_ratio >= self.vshape_min:
                    conf = min(0.5 + retrace_ratio * 0.5, 1.0)
                    return ConfirmationResult(
                        True, "VSHAPE", conf,
                        f"V-Shape ↓ | High={high_price:.5f} | Drop={drop:.5f} | "
                        f"Retrace={retrace_ratio:.0%}"
                    )

        except Exception as exc:
            log.warning("V-Shape check failed: %s", exc)

        return ConfirmationResult(False, "VSHAPE", 0, "No V-Shape detected")

    # ── Signal Builder ────────────────────────────────────────────────────────

    def _build_signal(
        self,
        entry: WatchlistEntry,
        confirmation: ConfirmationResult,
        current_price: float,
        df_m1: pd.DataFrame,
        df_m15: Optional[pd.DataFrame] = None,
    ) -> Optional[EntrySignal]:
        """
        Build the full EntrySignal with:
        - 5 laddered entry prices across the POI zone
        - Stop Loss below/above the POI (with buffer)
        - TP1, TP2, TP3 from liquidity levels
        """
        candidate = entry.candidate
        direction = entry.direction
        symbol = entry.symbol
        pip = config.SYMBOL_PIP.get(symbol, 0.0001)
        sl_buffer = config.SL_BUFFER_PIPS * pip

        poi_top = candidate.poi_top
        poi_bot = candidate.poi_bottom
        poi_range = poi_top - poi_bot
        n_splits = config.ENTRY_SPLITS

        # ── Entry zone (5 splits) ─────────────────────────────────────────────
        if direction == "BUY":
            # Entries spread from poi_mid down to poi_bottom
            entry_top = (poi_top + poi_bot) / 2   # Upper half of POI
            entry_bot = poi_bot
            entries = [
                entry_bot + (entry_top - entry_bot) * (i / (n_splits - 1))
                for i in range(n_splits)
            ]
            stop_loss = poi_bot - sl_buffer

        else:  # SELL
            entry_bot = (poi_top + poi_bot) / 2   # Lower half of POI
            entry_top = poi_top
            entries = [
                entry_top - (entry_top - entry_bot) * (i / (n_splits - 1))
                for i in range(n_splits)
            ]
            stop_loss = poi_top + sl_buffer

        # Round entries to symbol precision
        digits = 5 if pip < 0.001 else 2
        entries = [round(e, digits) for e in entries]
        stop_loss = round(stop_loss, digits)

        avg_entry = round(sum(entries) / len(entries), digits)

        # ── TP levels ─────────────────────────────────────────────────────────
        tp1 = candidate.tp1
        tp2 = candidate.tp2
        tp3 = candidate.tp3

        if tp1 is None:
            risk = abs(avg_entry - stop_loss)
            if direction == "BUY":
                tp1 = round(avg_entry + risk * 2, digits)
                tp2 = round(avg_entry + risk * 3, digits)
                tp3 = round(avg_entry + risk * 5, digits)
            else:
                tp1 = round(avg_entry - risk * 2, digits)
                tp2 = round(avg_entry - risk * 3, digits)
                tp3 = round(avg_entry - risk * 5, digits)

        # ── R:R ──────────────────────────────────────────────────────────────
        risk = abs(avg_entry - stop_loss)
        rr1 = abs(tp1 - avg_entry) / (risk + 1e-9) if tp1 else 0
        rr2 = abs(tp2 - avg_entry) / (risk + 1e-9) if tp2 else None

        if rr1 < config.MIN_RR:
            log.info(
                "Signal rejected — R:R %.2f < minimum %.1f for %s",
                rr1, config.MIN_RR, symbol,
            )
            return None

        # Use pre-fetched M15 data if available, otherwise fetch
        if df_m15 is None:
            df_m15 = fetcher.get_candles(symbol, "M15", config.CHART_CANDLES_SHOWN)

        session = candidate.session

        return EntrySignal(
            uid=entry.uid,
            symbol=symbol,
            direction=direction,
            entries=entries,
            entry_zone_top=max(entries),
            entry_zone_bottom=min(entries),
            stop_loss=stop_loss,
            tp1=round(tp1, digits),
            tp2=round(tp2, digits) if tp2 else None,
            tp3=round(tp3, digits) if tp3 else None,
            confirmation=confirmation,
            setup_score=candidate.setup_score * confirmation.confidence,
            rr1=round(rr1, 2),
            rr2=round(rr2, 2) if rr2 else None,
            poi_type=candidate.poi_type,
            h1_trend=candidate.h1_trend,
            session_name=str(session),
            df_m15=df_m15,
            df_m1=df_m1,
        )


entry_engine = EntryConfirmationEngine()
