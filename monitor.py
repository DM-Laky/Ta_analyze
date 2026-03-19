"""
monitor.py — APEX V10 Trade Monitor
5-minute rules-based monitoring loop.
Gemini is called ONLY when a hard trigger fires.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from engine import ExchangeManager, analyze_timeframe
from state import (
    load_trade_state, update_trade_state,
    get_pnl_pct, get_pnl_usdt
)

logger = logging.getLogger("monitor")


# ─────────────────────────────────────────────
# Trigger Detection
# ─────────────────────────────────────────────

def check_hard_triggers(state: dict, price: float,
                          analysis_1h: dict) -> Optional[dict]:
    """
    Returns trigger dict if a hard condition is met, else None.
    Hard triggers → Gemini is called for final verdict.
    """
    direction  = state.get("direction", "long")
    hard_inv   = state.get("hard_invalidation")
    entry      = state.get("entry", price)
    rsi_1h     = analysis_1h.get("rsi", 50)
    bos_1h     = analysis_1h.get("structure", {}).get("bos", False)
    bos_dir    = analysis_1h.get("structure", {}).get("bos_direction")

    # Trigger 1: Price beyond hard invalidation
    if hard_inv:
        if direction == "long"  and price < hard_inv:
            return {"trigger": "hard_invalidation_breached",
                    "trend": analysis_1h.get("structure", {}).get("trend", "unknown"),
                    "rsi": rsi_1h, "bos": bos_1h, "pattern": analysis_1h.get("candlestick_pattern"),
                    "detail": f"Price {price} < invalidation {hard_inv}"}
        if direction == "short" and price > hard_inv:
            return {"trigger": "hard_invalidation_breached",
                    "trend": analysis_1h.get("structure", {}).get("trend", "unknown"),
                    "rsi": rsi_1h, "bos": bos_1h, "pattern": analysis_1h.get("candlestick_pattern"),
                    "detail": f"Price {price} > invalidation {hard_inv}"}

    # Trigger 2: Opposite BOS on 1H
    if bos_1h:
        if direction == "long"  and bos_dir == "bearish":
            return {"trigger": "opposite_bos_1h",
                    "trend": "bearish", "rsi": rsi_1h, "bos": True, "pattern": analysis_1h.get("candlestick_pattern"),
                    "detail": "Bearish BOS on 1H confirmed — long bias invalidated"}
        if direction == "short" and bos_dir == "bullish":
            return {"trigger": "opposite_bos_1h",
                    "trend": "bullish", "rsi": rsi_1h, "bos": True, "pattern": analysis_1h.get("candlestick_pattern"),
                    "detail": "Bullish BOS on 1H confirmed — short bias invalidated"}

    # Trigger 3: RSI extreme against trade on 1H
    if direction == "long"  and rsi_1h > 78:
        return {"trigger": "rsi_overbought_1h",
                "trend": analysis_1h.get("structure", {}).get("trend"),
                "rsi": rsi_1h, "bos": False, "pattern": analysis_1h.get("candlestick_pattern"),
                "detail": f"RSI 1H overbought ({rsi_1h:.0f}) — long exhaustion risk"}
    if direction == "short" and rsi_1h < 22:
        return {"trigger": "rsi_oversold_1h",
                "trend": analysis_1h.get("structure", {}).get("trend"),
                "rsi": rsi_1h, "bos": False, "pattern": analysis_1h.get("candlestick_pattern"),
                "detail": f"RSI 1H oversold ({rsi_1h:.0f}) — short exhaustion risk"}

    return None


def check_soft_triggers(state: dict, price: float,
                          analysis_1h: dict, analysis_15m: dict) -> list[str]:
    """
    Soft triggers → warnings displayed, no Gemini call.
    """
    warnings   = []
    direction  = state.get("direction", "long")
    soft_warn  = state.get("soft_warning")
    rsi_1h     = analysis_1h.get("rsi", 50)
    pat_15     = analysis_15m.get("candlestick_pattern", "none")

    if soft_warn:
        if direction == "long"  and price < soft_warn:
            warnings.append(f"⚠️ Price approaching soft warning level {soft_warn}")
        if direction == "short" and price > soft_warn:
            warnings.append(f"⚠️ Price approaching soft warning level {soft_warn}")

    # Counter-trend pattern on 15m
    bearish_15m = {"shooting_star", "bearish_pinbar", "bearish_engulfing", "evening_star"}
    bullish_15m = {"hammer", "bullish_pinbar", "bullish_engulfing", "morning_star"}
    if direction == "long"  and pat_15 in bearish_15m:
        warnings.append(f"⚠️ Counter-trend 15m pattern: {pat_15}")
    if direction == "short" and pat_15 in bullish_15m:
        warnings.append(f"⚠️ Counter-trend 15m pattern: {pat_15}")

    # RSI approaching extreme
    if direction == "long"  and 68 < rsi_1h < 78:
        warnings.append(f"⚠️ RSI 1H approaching overbought ({rsi_1h:.0f})")
    if direction == "short" and 22 < rsi_1h < 32:
        warnings.append(f"⚠️ RSI 1H approaching oversold ({rsi_1h:.0f})")

    return warnings


def check_tp_hits(state: dict, price: float) -> list[str]:
    """Check if any TP levels have been hit."""
    hits      = []
    direction = state.get("direction", "long")
    tp1, tp2, tp3 = state.get("tp1"), state.get("tp2"), state.get("tp3")

    if direction == "long":
        if tp1 and price >= tp1 and not state.get("tp1_hit"): hits.append("TP1")
        if tp2 and price >= tp2:                               hits.append("TP2")
        if tp3 and price >= tp3:                               hits.append("TP3")
    else:
        if tp1 and price <= tp1 and not state.get("tp1_hit"): hits.append("TP1")
        if tp2 and price <= tp2:                               hits.append("TP2")
        if tp3 and price <= tp3:                               hits.append("TP3")

    return hits


# ─────────────────────────────────────────────
# Monitor Loop
# ─────────────────────────────────────────────

class TradeMonitor:
    def __init__(self, exchange: ExchangeManager, ai_analyst, log_fn: Callable,
                 broadcast_fn: Callable, interval: int = 300):
        self.exchange     = exchange
        self.ai           = ai_analyst
        self.log          = log_fn
        self.broadcast    = broadcast_fn
        self.interval     = interval
        self._running     = False
        self._task        = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task    = asyncio.create_task(self._loop())
        await self.log("[MONITOR] 🔁 5-minute monitoring loop started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.log("[MONITOR] Monitoring loop stopped.")

    async def _loop(self):
        while self._running:
            await asyncio.sleep(self.interval)
            try:
                await self._check_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.log(f"[MONITOR] ❌ Error in monitor cycle: {e}")

    async def _check_cycle(self):
        state = load_trade_state()

        if not state.get("in_trade"):
            return

        pair      = state.get("pair")
        direction = state.get("direction", "long")
        entry     = state.get("entry", 0)
        cycle     = state.get("monitor_cycles", 0) + 1

        await self.log(f"[MONITOR] ─── Monitor Cycle #{cycle} — {pair} ───")

        # ── Fetch current price ──
        try:
            ticker = await self.exchange.fetch_ticker(pair)
            price  = float(ticker.get("last", 0))
        except Exception as e:
            await self.log(f"[MONITOR] ❌ Ticker fetch failed: {e}")
            return

        # ── PnL ──
        pnl_pct  = get_pnl_pct(price)
        pnl_usdt = get_pnl_usdt(price)
        pnl_sign = "+" if pnl_pct >= 0 else ""
        await self.log(
            f"[MONITOR] {pair} | CMP: {price} | Entry: {entry} | "
            f"PnL: {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${pnl_usdt:.2f})"
        )

        # ── TP hit check ──
        tp_hits = check_tp_hits(state, price)
        if "TP1" in tp_hits and not state.get("tp1_hit"):
            await self.log(f"[MONITOR] 🎯 TP1 HIT! Moving SL to break-even @ {entry}")
            state = update_trade_state(tp1_hit=True, be_moved=True)

        if tp_hits:
            await self.log(f"[MONITOR] 🎯 TP levels hit: {', '.join(tp_hits)}")

        # ── Fetch 1H + 15m for rule checks ──
        try:
            from engine import analyze_timeframe as atf
            df1h  = await self.exchange.fetch_ohlcv(pair, "1h",  limit=50)
            df15  = await self.exchange.fetch_ohlcv(pair, "15m", limit=50)
            a1h   = atf(df1h, "1h")
            a15   = atf(df15, "15m")
        except Exception as e:
            await self.log(f"[MONITOR] ❌ Analysis fetch failed: {e}")
            update_trade_state(monitor_cycles=cycle, last_monitor=datetime.utcnow().isoformat())
            await self.broadcast("monitor", {
                "pair": pair, "price": price, "pnl_pct": round(pnl_pct, 3),
                "pnl_usdt": round(pnl_usdt, 4), "status": "DATA_ERROR",
                "tp_hits": tp_hits, "be_moved": state.get("be_moved", False)
            })
            return

        # ── Soft triggers ──
        soft_warnings = check_soft_triggers(state, price, a1h, a15)
        for w in soft_warnings:
            await self.log(f"[MONITOR] {w}")

        # ── Hard triggers → Gemini escalation ──
        hard_trigger = check_hard_triggers(state, price, a1h)
        status       = "HOLD"
        gemini_verdict = None

        if hard_trigger:
            await self.log(
                f"[MONITOR] 🚨 HARD TRIGGER: {hard_trigger['trigger']} — "
                f"{hard_trigger['detail']}"
            )
            await self.log("[MONITOR] Escalating to Gemini for final verdict...")

            gemini_verdict = await self.ai.evaluate_structure(state, price, hard_trigger)
            action = gemini_verdict.get("action", "HOLD")
            reason = gemini_verdict.get("reason", "")

            await self.log(f"[MONITOR] Gemini verdict: {action} — {reason}")

            if action == "CLOSE_NOW":
                status = "CLOSE_NOW"
                await self.log(f"[MONITOR] ⛔ CLOSE TRADE NOW — {reason}")
            elif action == "CLOSE_WARNING":
                status = "CLOSE_WARNING"
                await self.log(f"[MONITOR] ⚠️ CLOSE WARNING — {reason}")
            else:
                status = "HOLD"
                await self.log(f"[MONITOR] ✅ Gemini says HOLD — {reason}")
        else:
            # Check structure health for HOLD confirmation
            trend_1h = a1h.get("structure", {}).get("trend", "ranging")
            if (direction == "long"  and trend_1h == "bullish") or \
               (direction == "short" and trend_1h == "bearish"):
                status = "HOLD"
                await self.log(f"[MONITOR] ✅ Structure intact — HOLD | 1H trend: {trend_1h.upper()}")
            else:
                status = "HOLD"
                await self.log(f"[MONITOR] ✅ No hard triggers — HOLD")

        # ── Update state ──
        update_trade_state(
            monitor_cycles=cycle,
            last_monitor=datetime.utcnow().isoformat(),
            last_status=status,
            gemini_verdict=gemini_verdict,
        )

        # ── Broadcast monitor update ──
        await self.broadcast("monitor", {
            "pair":          pair,
            "price":         price,
            "pnl_pct":       round(pnl_pct, 3),
            "pnl_usdt":      round(pnl_usdt, 4),
            "status":        status,
            "tp_hits":       tp_hits,
            "be_moved":      state.get("be_moved", False) or ("TP1" in tp_hits),
            "soft_warnings": soft_warnings,
            "rsi_1h":        a1h.get("rsi", 50),
            "trend_1h":      a1h.get("structure", {}).get("trend", "ranging"),
            "cycle":         cycle,
        })
