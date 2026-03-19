"""
state.py — APEX V10 Trade State Manager
Handles trade_state.json persistence, analysis cache, and full reset
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("state")

TRADE_STATE_FILE  = "trade_state.json"
ANALYSIS_CACHE_FILE = "analysis_cache.json"


# ─────────────────────────────────────────────
# Trade State Schema
# ─────────────────────────────────────────────

DEFAULT_TRADE_STATE = {
    "active": False,
    "pair": None,
    "mode": None,                  # "scalp" | "swing"
    "direction": None,             # "long" | "short"
    "entry": None,
    "stop_loss": None,
    "tp1": None,
    "tp2": None,
    "tp3": None,
    "leverage": None,
    "margin_usdt": None,
    "rr": None,
    "confidence": None,
    "confluence_score": None,
    "entry_type": None,            # "sniper" | "strong" | "standard"
    "hard_invalidation": None,     # price level → mandatory close
    "soft_warning": None,          # price level → caution
    "ob_zone": None,               # [low, high] of entry OB
    "liquidity_above": None,
    "liquidity_below": None,
    "reasoning": None,
    "signal_timestamp": None,
    "entry_timestamp": None,       # when user clicked "Enter Trade"
    "in_trade": False,             # True after user confirms entry
    "be_moved": False,             # SL moved to break-even
    "tp1_hit": False,
    "monitor_cycles": 0,
    "last_monitor": None,
    "last_status": None,           # "HOLD" | "CLOSE WARNING" | "CLOSE NOW"
    "gemini_verdict": None,        # last Gemini re-evaluation result
}


# ─────────────────────────────────────────────
# Read / Write
# ─────────────────────────────────────────────

def load_trade_state() -> dict:
    if not os.path.exists(TRADE_STATE_FILE):
        return dict(DEFAULT_TRADE_STATE)
    try:
        with open(TRADE_STATE_FILE, "r") as f:
            data = json.load(f)
        # Merge with defaults for any missing keys
        merged = dict(DEFAULT_TRADE_STATE)
        merged.update(data)
        return merged
    except Exception as e:
        logger.error(f"[STATE] Failed to load trade state: {e}")
        return dict(DEFAULT_TRADE_STATE)


def save_trade_state(state: dict) -> bool:
    try:
        state["last_updated"] = datetime.utcnow().isoformat()
        with open(TRADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"[STATE] Failed to save trade state: {e}")
        return False


def update_trade_state(**kwargs) -> dict:
    state = load_trade_state()
    for key, value in kwargs.items():
        state[key] = value
    save_trade_state(state)
    return state


# ─────────────────────────────────────────────
# Analysis Cache
# ─────────────────────────────────────────────

def save_analysis_cache(data: dict) -> bool:
    try:
        data["cached_at"] = datetime.utcnow().isoformat()
        with open(ANALYSIS_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"[STATE] Failed to save analysis cache: {e}")
        return False


def load_analysis_cache() -> Optional[dict]:
    if not os.path.exists(ANALYSIS_CACHE_FILE):
        return None
    try:
        with open(ANALYSIS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[STATE] Failed to load analysis cache: {e}")
        return None


# ─────────────────────────────────────────────
# Signal → State
# ─────────────────────────────────────────────

def apply_signal_to_state(signal: dict) -> dict:
    """
    Takes Gemini signal dict and writes it into trade_state.json.
    Returns the saved state.
    """
    tp_levels = signal.get("tp_levels", [])
    state = dict(DEFAULT_TRADE_STATE)
    state.update({
        "active":            True,
        "pair":              signal.get("pair"),
        "mode":              signal.get("mode"),
        "direction":         signal.get("direction"),
        "entry":             signal.get("entry"),
        "stop_loss":         signal.get("stop_loss"),
        "tp1":               tp_levels[0] if len(tp_levels) > 0 else None,
        "tp2":               tp_levels[1] if len(tp_levels) > 1 else None,
        "tp3":               tp_levels[2] if len(tp_levels) > 2 else None,
        "leverage":          signal.get("leverage"),
        "margin_usdt":       signal.get("margin_usdt"),
        "rr":                signal.get("rr"),
        "confidence":        signal.get("confidence"),
        "confluence_score":  signal.get("confluence_score"),
        "entry_type":        signal.get("entry_type"),
        "hard_invalidation": signal.get("hard_invalidation"),
        "soft_warning":      signal.get("soft_warning"),
        "ob_zone":           signal.get("ob_zone"),
        "liquidity_above":   signal.get("liquidity_above"),
        "liquidity_below":   signal.get("liquidity_below"),
        "reasoning":         signal.get("reasoning"),
        "signal_timestamp":  datetime.utcnow().isoformat(),
        "in_trade":          False,
        "be_moved":          False,
        "tp1_hit":           False,
        "monitor_cycles":    0,
        "last_status":       "SIGNAL_READY",
    })
    save_trade_state(state)
    return state


# ─────────────────────────────────────────────
# Enter Trade
# ─────────────────────────────────────────────

def mark_trade_entered() -> dict:
    return update_trade_state(
        in_trade=True,
        entry_timestamp=datetime.utcnow().isoformat(),
        last_status="IN_TRADE"
    )


# ─────────────────────────────────────────────
# Reset (New Day)
# ─────────────────────────────────────────────

def full_reset() -> bool:
    """Delete trade state and analysis cache. Clean slate."""
    removed = []
    for filepath in [TRADE_STATE_FILE, ANALYSIS_CACHE_FILE]:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                removed.append(filepath)
            except Exception as e:
                logger.error(f"[STATE] Could not remove {filepath}: {e}")
    logger.info(f"[STATE] Reset complete. Removed: {removed}")
    return True


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def is_trade_active() -> bool:
    state = load_trade_state()
    return state.get("active", False)


def is_in_trade() -> bool:
    state = load_trade_state()
    return state.get("in_trade", False)


def get_pnl_pct(current_price: float) -> float:
    state = load_trade_state()
    entry = state.get("entry")
    direction = state.get("direction")
    if not entry or not direction:
        return 0.0
    if direction == "long":
        return (current_price - entry) / entry * 100
    else:
        return (entry - current_price) / entry * 100


def get_pnl_usdt(current_price: float) -> float:
    state = load_trade_state()
    margin = state.get("margin_usdt", 0) or 0
    leverage = state.get("leverage", 1) or 1
    pnl_pct = get_pnl_pct(current_price)
    return round((pnl_pct / 100) * margin * leverage, 4)
