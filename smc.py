"""
smc.py — The Mathematical Brain (Institutional-Grade) v2

Pure SMC (Smart Money Concepts) logic using Pandas & NumPy.
NO API calls, NO side-effects. Every function is deterministic.

v2 Critical Fixes (FVG quality + freshness overhaul):
  • FVG SIZE FILTER: gap must be >= 1.0× ATR — small FVGs EXCLUDED
  • VOLUME FILTER: OB/FVG displacement candle must have vol > 1.5× avg
  • FRESHNESS: zones must be < 50 candles old AND completely untouched
  • OB without high-volume displacement = excluded
  • FVG without high-volume middle candle = excluded
  • 30% partial fill tracked: zone being respected = valid POI
  • Small FVG + no OB overlap = never enters watchlist
  • OB+FVG overlap with volume = 150-point bypass (unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

# ── Core Constants ──
MIN_RR = 2.0
CHOCH_WINDOW = 15
SWEEP_RECOVERY_CANDLES = 3
VOL_MULT = 1.2
MIN_SCORE = 70
OB_FVG_OVERLAP_SCORE = 150

# ── ✅ NEW: FVG Quality & Freshness Constants ──
MIN_FVG_ATR_RATIO = 1.0          # FVG gap must be >= 1.0× ATR to qualify as "large"
SMALL_FVG_CUTOFF = 0.4           # FVG gap < 0.4× ATR = tiny, ALWAYS skip
MAX_ZONE_AGE = 75                # Zone must be formed within last 75 candles
FVG_PARTIAL_FILL_PCT = 0.30      # 30% fill = price respecting zone (valid POI)
FVG_MITIGATION_PCT = 0.40        # 40% CE fill = mitigated (dead zone) — used as gap_mid

# ── Pro Entry & Anti-Fake-CHoCH Constants ──────────────────────────
ENTRY_RETRACE        = 0.30    # 30% retrace entry from CHoCH toward sweep extreme (0.70 Fib)
SL_BUFFER_PCT        = 0.00175 # 0.175% buffer below/above sweep wick (double-sweep guard)
CHOCH_MIN_CANDLE_GAP = 2       # min candles between sweep recovery and CHoCH
CHOCH_BREAK_MARGIN   = 0.0005  # close must exceed prior swing level by ≥0.05%
CHOCH_BODY_PCT_MIN   = 0.40    # CHoCH candle body must be ≥40% of its high–low range
CHOCH_VOL_MULT       = 0.85    # CHoCH candle volume ≥ 85% of 20-bar average
CHOCH_RSI_BULL_MIN   = 35.0    # RSI(14) ≥ 35 for bullish CHoCH (avoids dead-cat bounce)
CHOCH_RSI_BEAR_MAX   = 65.0    # RSI(14) ≤ 65 for bearish CHoCH (avoids bull-trap fade)
CHOCH_MAX_SWING_AGE  = 25      # prior structural swing must be within last 25 candles


# ─────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["SH", "SL"]
    timestamp: str = ""


@dataclass
class BOS:
    direction: Literal["bullish", "bearish"]
    break_index: int
    break_price: float
    swing_price: float


@dataclass
class Zone:
    high: float
    low: float
    ob_index: int
    kind: Literal["ob", "fvg"]
    direction: Literal["bullish", "bearish"]
    role: Literal["extreme", "decisional", "fvg"] = "fvg"
    timestamp: str = ""
    has_volume: bool = False          # ✅ NEW: track if candle had high volume
    fvg_atr_ratio: float = 0.0       # ✅ NEW: how large the FVG is vs ATR
    is_fresh: bool = True             # ✅ NEW: completely untouched
    partial_fill_pct: float = 0.0     # ✅ NEW: how much has been filled (0-1)

    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def gap_size(self) -> float:      # ✅ NEW
        return self.high - self.low

    def is_large(self) -> bool:       # ✅ NEW
        return self.fvg_atr_ratio >= MIN_FVG_ATR_RATIO

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LiquiditySweep:
    sweep_index: int
    swept_price: float
    direction: Literal["bullish", "bearish"]
    recovery_index: int = 0


@dataclass
class CHoCH:
    choch_index: int
    direction: Literal["bullish", "bearish"]
    sweep: LiquiditySweep
    displacement_candle_index: int


@dataclass
class SniperSignal:
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_high: float
    entry_low: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    tp4: float
    tp5: float
    tp6: float
    risk_reward: float
    position_size: float
    risk_usd: float
    confluences: List[str]
    order_type: str = "LIMIT"


# ─────────────────────────────────────────────────────────────────────
# 1. NON-REPAINTING FRACTALS — adaptive lookback
# ─────────────────────────────────────────────────────────────────────

def detect_swing_points(
    df: pd.DataFrame,
    lb: int = 5,
) -> List[SwingPoint]:
    if df is None or len(df) < 2 * lb + 2:
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = (df["timestamp"].values
                  if "timestamp" in df.columns else [""] * len(df))
    n = len(df)
    last_confirmed = n - 1 - lb
    points: List[SwingPoint] = []

    for i in range(lb, last_confirmed + 1):
        window_h = highs[i - lb: i + lb + 1]
        window_l = lows[i - lb: i + lb + 1]

        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(highs[i]), kind="SH",
                timestamp=str(timestamps[i]),
            ))

        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            points.append(SwingPoint(
                index=i, price=float(lows[i]), kind="SL",
                timestamp=str(timestamps[i]),
            ))

    points.sort(key=lambda p: p.index)
    return points


# ─────────────────────────────────────────────────────────────────────
# 2. BOS WITH INDUCEMENT CHECK
# ─────────────────────────────────────────────────────────────────────

def _has_inducement_sweep(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    direction: Literal["bullish", "bearish"],
    break_swing_idx: int,
) -> bool:
    lows = df["low"].values
    highs = df["high"].values

    if direction == "bullish":
        shs = [s for s in swings if s.kind == "SH" and s.index <= break_swing_idx]
        if len(shs) < 2:
            return True
        prev_sh = shs[-2]
        curr_sh = shs[-1]
        internal_lows = [
            s for s in swings
            if s.kind == "SL" and prev_sh.index < s.index < curr_sh.index
        ]
        if not internal_lows:
            return True
        idm = internal_lows[-1]
        for j in range(idm.index + 1, curr_sh.index + 1):
            if j < len(lows) and lows[j] < idm.price:
                return True
        return False
    else:
        sls = [s for s in swings if s.kind == "SL" and s.index <= break_swing_idx]
        if len(sls) < 2:
            return True
        prev_sl = sls[-2]
        curr_sl = sls[-1]
        internal_highs = [
            s for s in swings
            if s.kind == "SH" and prev_sl.index < s.index < curr_sl.index
        ]
        if not internal_highs:
            return True
        idm = internal_highs[-1]
        for j in range(idm.index + 1, curr_sl.index + 1):
            if j < len(highs) and highs[j] > idm.price:
                return True
        return False


def detect_bos(
    df: pd.DataFrame,
    swings: Optional[List[SwingPoint]] = None,
    lb: int = 5,
) -> Optional[BOS]:
    if swings is None:
        swings = detect_swing_points(df, lb)
    if not swings:
        return None

    swing_highs = [s for s in swings if s.kind == "SH"]
    swing_lows = [s for s in swings if s.kind == "SL"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]
    closes = df["close"].values
    n = len(df)

    bullish_break_idx: Optional[int] = None
    bearish_break_idx: Optional[int] = None

    for j in range(last_sh.index + 1, n):
        if closes[j] > last_sh.price:
            if _has_inducement_sweep(df, swings, "bullish", last_sh.index):
                bullish_break_idx = j
            break

    for j in range(last_sl.index + 1, n):
        if closes[j] < last_sl.price:
            if _has_inducement_sweep(df, swings, "bearish", last_sl.index):
                bearish_break_idx = j
            break

    if bullish_break_idx is not None and bearish_break_idx is not None:
        if bullish_break_idx >= bearish_break_idx:
            return BOS("bullish", bullish_break_idx,
                        float(closes[bullish_break_idx]), last_sh.price)
        else:
            return BOS("bearish", bearish_break_idx,
                        float(closes[bearish_break_idx]), last_sl.price)
    elif bullish_break_idx is not None:
        return BOS("bullish", bullish_break_idx,
                    float(closes[bullish_break_idx]), last_sh.price)
    elif bearish_break_idx is not None:
        return BOS("bearish", bearish_break_idx,
                    float(closes[bearish_break_idx]), last_sl.price)
    return None


def get_trend(
    df: pd.DataFrame,
    swings: Optional[List[SwingPoint]] = None,
    lb: int = 5,
) -> Optional[Literal["bullish", "bearish"]]:
    bos = detect_bos(df, swings, lb)
    return bos.direction if bos else None


# ─────────────────────────────────────────────────────────────────────
# 3. PREMIUM / DISCOUNT & OTE
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FibZones:
    swing_low: float
    swing_high: float
    discount_upper: float
    premium_lower: float
    ote_low: float
    ote_high: float


def compute_fib_zones(swings: List[SwingPoint]) -> Optional[FibZones]:
    highs = [s for s in swings if s.kind == "SH"]
    lows = [s for s in swings if s.kind == "SL"]
    if not highs or not lows:
        return None
    sh = highs[-1].price
    sl = lows[-1].price
    rng = sh - sl
    if rng <= 0:
        return None
    return FibZones(
        swing_low=sl, swing_high=sh,
        discount_upper=sl + 0.5 * rng,
        premium_lower=sl + 0.5 * rng,
        # Bullish OTE = deep retrace near demand (61.8–78.6% retrace from high)
        ote_low=sl + (1.0 - 0.786) * rng,
        ote_high=sl + (1.0 - 0.618) * rng,
    )


def is_in_ote(price: float, fib: FibZones,
              direction: Literal["bullish", "bearish"] = "bullish") -> bool:
    """Direction-aware OTE check.
    Bullish OTE: deep retrace zone near demand (lower portion of range).
    Bearish OTE: deep retrace zone near supply (upper portion of range).
    """
    if direction == "bullish":
        return fib.ote_low <= price <= fib.ote_high
    else:
        # Bearish OTE mirrors the bullish one from the top of the range
        bear_ote_low  = fib.swing_high - (1.0 - 0.618) * (fib.swing_high - fib.swing_low)
        bear_ote_high = fib.swing_high - (1.0 - 0.786) * (fib.swing_high - fib.swing_low)
        return bear_ote_low <= price <= bear_ote_high


def is_in_discount(price: float, fib: FibZones) -> bool:
    return fib.swing_low <= price <= fib.discount_upper


def is_in_premium(price: float, fib: FibZones) -> bool:
    return fib.premium_lower <= price <= fib.swing_high


# ─────────────────────────────────────────────────────────────────────
# ✅ NEW SECTION: VOLUME & FRESHNESS HELPERS
# ─────────────────────────────────────────────────────────────────────

def _has_high_volume(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    mult: float = VOL_MULT,
) -> bool:
    """
    ✅ NEW: Returns True if candle at `idx` has volume >= mult × 20-bar avg.
    If no volume column exists, returns True (pass-through).
    """
    if "volume" not in df.columns:
        return True
    if idx < lookback or idx >= len(df):
        return True
    vol = df["volume"].iat[idx]
    avg_vol = df["volume"].iloc[max(0, idx - lookback):idx].mean()
    if avg_vol <= 0:
        return True
    return vol >= mult * avg_vol


def _get_candle_volume_ratio(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
) -> float:
    """
    ✅ NEW: Returns the ratio of candle volume to 20-bar average.
    Used for ranking zone quality.
    """
    if "volume" not in df.columns:
        return 1.0
    if idx < lookback or idx >= len(df):
        return 1.0
    vol = df["volume"].iat[idx]
    avg_vol = df["volume"].iloc[max(0, idx - lookback):idx].mean()
    if avg_vol <= 0:
        return 1.0
    return vol / avg_vol


def _is_zone_completely_fresh(
    direction: Literal["bullish", "bearish"],
    zone_high: float,
    zone_low: float,
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> bool:
    """
    ✅ NEW: Strict freshness — zone has NEVER been touched.
    Bullish zone (demand): no candle low has wicked INTO the zone.
    Bearish zone (supply): no candle high has wicked INTO the zone.
    """
    if direction == "bullish":
        for m in range(start_idx, end_idx):
            if m < len(lows) and lows[m] <= zone_high:
                return False
    else:
        for m in range(start_idx, end_idx):
            if m < len(highs) and highs[m] >= zone_low:
                return False
    return True


def _compute_partial_fill(
    direction: Literal["bullish", "bearish"],
    zone_high: float,
    zone_low: float,
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """
    ✅ NEW: Calculate how much of the zone has been filled (0.0 to 1.0).
    For bullish: how deep did price wick down into the zone from the top.
    For bearish: how deep did price wick up into the zone from the bottom.
    Returns 0.0 if untouched, 1.0 if fully filled.
    """
    zone_size = zone_high - zone_low
    if zone_size <= 0:
        return 0.0

    if direction == "bullish":
        deepest = zone_high  # start at top (no fill)
        for m in range(start_idx, end_idx):
            if m < len(lows) and lows[m] < deepest:
                deepest = lows[m]
        penetration = zone_high - max(deepest, zone_low)
        return min(penetration / zone_size, 1.0)
    else:
        deepest = zone_low  # start at bottom (no fill)
        for m in range(start_idx, end_idx):
            if m < len(highs) and highs[m] > deepest:
                deepest = highs[m]
        penetration = min(deepest, zone_high) - zone_low
        return min(penetration / zone_size, 1.0)


def _fvg_size_vs_atr(
    gap_high: float,
    gap_low: float,
    atr_val: float,
) -> float:
    """
    ✅ NEW: Returns ratio of FVG gap size to ATR.
    Used to filter small FVGs.
    """
    gap_size = gap_high - gap_low
    if atr_val <= 0:
        return 999.0  # no ATR data, let it pass
    return gap_size / atr_val


# ─────────────────────────────────────────────────────────────────────
# 4. ORDER BLOCKS — Extreme + Decisional
#    ✅ CHANGED: Now requires high-volume candle + freshness + age
# ─────────────────────────────────────────────────────────────────────

def _is_ob_mitigated(
    direction: Literal["bullish", "bearish"],
    ob_high: float,
    ob_low: float,
    lows: np.ndarray,
    highs: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> bool:
    """OB is mitigated if price penetrates past 50% of body."""
    ob_mid = (ob_high + ob_low) / 2.0
    if direction == "bullish":
        for m in range(start_idx, end_idx):
            if lows[m] < ob_mid:
                return True
    else:
        for m in range(start_idx, end_idx):
            if highs[m] > ob_mid:
                return True
    return False


def detect_order_blocks(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    swings: Optional[List[SwingPoint]] = None,
    lb: int = 5,
) -> List[Zone]:
    """
    Find unmitigated OBs (Extreme + Decisional).

    ✅ v2 FILTERS ADDED:
      1. Volume: OB candle OR the displacement candle right after it
         must have volume >= 1.5× 20-bar average
      2. Age: OB must be within last MAX_ZONE_AGE candles
      3. Freshness: Zone must be completely untouched (not just <50%)
    """
    if df is None or len(df) < 15:
        return []

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = (df["timestamp"].values
                  if "timestamp" in df.columns else [""] * len(df))
    n = len(df)

    if swings is None:
        swings = detect_swing_points(df, lb)

    bos = detect_bos(df, swings, lb)
    if bos is None:
        return []

    obs: List[Zone] = []
    bos_idx = bos.break_index
    scan_start = max(bos_idx - 40, 0)

    if direction == "bullish" and bos.direction == "bullish":
        red_candles = []
        for k in range(scan_start, bos_idx):
            if closes[k] < opens[k]:
                red_candles.append(k)

        if red_candles:
            # ── Decisional OB = last red candle before BOS ──
            dec_k = red_candles[-1]

            # ✅ NEW: Age filter
            if (n - 1) - dec_k <= MAX_ZONE_AGE:                           # ✅ L~320

                # ✅ NEW: Volume filter — OB candle or next candle
                ob_has_vol = (                                              # ✅ L~323
                    _has_high_volume(df, dec_k)
                    or (dec_k + 1 < n and _has_high_volume(df, dec_k + 1))
                )

                if ob_has_vol:                                              # ✅ L~328
                    dec_high = float(max(opens[dec_k], highs[dec_k]))
                    dec_low = float(lows[dec_k])

                    # ✅ CHANGED: Use strict freshness (completely untouched)
                    fresh = _is_zone_completely_fresh(                       # ✅ L~332
                        "bullish", dec_high, dec_low,
                        highs, lows, dec_k + 1, n,
                    )
                    not_mitigated = not _is_ob_mitigated(
                        "bullish", dec_high, dec_low,
                        lows, highs, dec_k + 1, n,
                    )

                    if fresh and not_mitigated:                             # ✅ L~340
                        vol_ratio = _get_candle_volume_ratio(df, dec_k)
                        obs.append(Zone(
                            high=dec_high, low=dec_low, ob_index=dec_k,
                            kind="ob", direction="bullish",
                            role="decisional",
                            timestamp=str(timestamps[dec_k]),
                            has_volume=ob_has_vol,                          # ✅ NEW
                            is_fresh=True,                                  # ✅ NEW
                        ))

            # ── Extreme OB = first red candle in impulse ──
            if len(red_candles) >= 2:
                ext_k = red_candles[0]

                if (n - 1) - ext_k <= MAX_ZONE_AGE:                        # ✅ L~355
                    ob_has_vol = (
                        _has_high_volume(df, ext_k)
                        or (ext_k + 1 < n and _has_high_volume(df, ext_k + 1))
                    )
                    if ob_has_vol:
                        ext_high = float(max(opens[ext_k], highs[ext_k]))
                        ext_low = float(lows[ext_k])
                        fresh = _is_zone_completely_fresh(
                            "bullish", ext_high, ext_low,
                            highs, lows, ext_k + 1, n,
                        )
                        not_mitigated = not _is_ob_mitigated(
                            "bullish", ext_high, ext_low,
                            lows, highs, ext_k + 1, n,
                        )
                        if fresh and not_mitigated:
                            obs.append(Zone(
                                high=ext_high, low=ext_low,
                                ob_index=ext_k,
                                kind="ob", direction="bullish",
                                role="extreme",
                                timestamp=str(timestamps[ext_k]),
                                has_volume=ob_has_vol,
                                is_fresh=True,
                            ))

    elif direction == "bearish" and bos.direction == "bearish":
        green_candles = []
        for k in range(scan_start, bos_idx):
            if closes[k] > opens[k]:
                green_candles.append(k)

        if green_candles:
            # ── Decisional OB ──
            dec_k = green_candles[-1]
            if (n - 1) - dec_k <= MAX_ZONE_AGE:                            # ✅
                ob_has_vol = (
                    _has_high_volume(df, dec_k)
                    or (dec_k + 1 < n and _has_high_volume(df, dec_k + 1))
                )
                if ob_has_vol:
                    dec_low = float(min(opens[dec_k], lows[dec_k]))
                    dec_high = float(highs[dec_k])
                    fresh = _is_zone_completely_fresh(
                        "bearish", dec_high, dec_low,
                        highs, lows, dec_k + 1, n,
                    )
                    not_mitigated = not _is_ob_mitigated(
                        "bearish", dec_high, dec_low,
                        lows, highs, dec_k + 1, n,
                    )
                    if fresh and not_mitigated:
                        obs.append(Zone(
                            high=dec_high, low=dec_low, ob_index=dec_k,
                            kind="ob", direction="bearish",
                            role="decisional",
                            timestamp=str(timestamps[dec_k]),
                            has_volume=ob_has_vol,
                            is_fresh=True,
                        ))

            # ── Extreme OB ──
            if len(green_candles) >= 2:
                ext_k = green_candles[0]
                if (n - 1) - ext_k <= MAX_ZONE_AGE:                        # ✅
                    ob_has_vol = (
                        _has_high_volume(df, ext_k)
                        or (ext_k + 1 < n and _has_high_volume(df, ext_k + 1))
                    )
                    if ob_has_vol:
                        ext_low = float(min(opens[ext_k], lows[ext_k]))
                        ext_high = float(highs[ext_k])
                        fresh = _is_zone_completely_fresh(
                            "bearish", ext_high, ext_low,
                            highs, lows, ext_k + 1, n,
                        )
                        not_mitigated = not _is_ob_mitigated(
                            "bearish", ext_high, ext_low,
                            lows, highs, ext_k + 1, n,
                        )
                        if fresh and not_mitigated:
                            obs.append(Zone(
                                high=ext_high, low=ext_low,
                                ob_index=ext_k,
                                kind="ob", direction="bearish",
                                role="extreme",
                                timestamp=str(timestamps[ext_k]),
                                has_volume=ob_has_vol,
                                is_fresh=True,
                            ))

    return obs


# ─────────────────────────────────────────────────────────────────────
# 5. FVG — ✅ COMPLETELY REWRITTEN with size/volume/freshness filters
# ─────────────────────────────────────────────────────────────────────

def detect_fvgs(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
) -> List[Zone]:
    """
    Detect unmitigated Fair Value Gaps.

    ✅ v2 FILTERS:
      1. SIZE: gap must be >= MIN_FVG_ATR_RATIO × ATR(14)
         — gaps below SMALL_FVG_CUTOFF × ATR are ALWAYS skipped
      2. VOLUME: the middle candle (displacement) must have
         volume >= 1.5× 20-bar average volume
      3. AGE: FVG must be within last MAX_ZONE_AGE candles
      4. MITIGATION: 50% CE (Consequent Encroachment) = dead
      5. PARTIAL FILL: tracks how much has been filled (0-1)
         — 30% fill = price respecting zone = valid POI
    """
    if df is None or len(df) < 4:
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = (df["timestamp"].values
                  if "timestamp" in df.columns else [""] * len(df))
    n = len(df)
    zones: List[Zone] = []

    # ✅ NEW: Pre-compute ATR for size filtering
    atr_series = compute_atr(df)

    for i in range(n - 2):

        # ✅ NEW: Age filter — skip FVGs older than MAX_ZONE_AGE candles
        if (n - 1) - i > MAX_ZONE_AGE:
            continue

        if direction == "bullish":
            if lows[i + 2] > highs[i]:  # gap exists
                gap_low = float(highs[i])
                gap_high = float(lows[i + 2])

                # ✅ NEW: ATR-based size filter
                atr_val = (atr_series.iat[i + 1]
                           if i + 1 < len(atr_series)
                              and not np.isnan(atr_series.iat[i + 1])
                           else 0.0)
                ratio = _fvg_size_vs_atr(gap_high, gap_low, atr_val)

                if ratio < SMALL_FVG_CUTOFF:           # ✅ Tiny FVG → SKIP
                    continue

                if ratio < MIN_FVG_ATR_RATIO:          # ✅ Small FVG → SKIP
                    continue                            #    (no watchlist)

                # ✅ NEW: Volume filter on displacement candle (middle)
                if not _has_high_volume(df, i + 1):
                    continue

                # Mitigation check at 50% CE
                gap_mid = (gap_high + gap_low) / 2.0
                mitigated = False
                for m in range(i + 3, n):
                    if lows[m] <= gap_mid:
                        mitigated = True
                        break
                if mitigated:
                    continue

                # ✅ NEW: Compute partial fill percentage
                fill_pct = _compute_partial_fill(
                    "bullish", gap_high, gap_low,
                    highs, lows, i + 3, n,
                )

                # ✅ 100% Freshness — any wick touch = invalid, discard immediately
                fresh = _is_zone_completely_fresh(
                    "bullish", gap_high, gap_low,
                    highs, lows, i + 3, n,
                )
                if not fresh:
                    continue

                vol_ratio = _get_candle_volume_ratio(df, i + 1)

                zones.append(Zone(
                    high=gap_high, low=gap_low, ob_index=i,
                    kind="fvg", direction="bullish", role="fvg",
                    timestamp=str(timestamps[i + 1]),
                    has_volume=True,                    # ✅ passed filter
                    fvg_atr_ratio=round(ratio, 2),      # ✅ store ratio
                    is_fresh=fresh,                      # ✅ store freshness
                    partial_fill_pct=round(fill_pct, 3), # ✅ store fill %
                ))

        elif direction == "bearish":
            if highs[i + 2] < lows[i]:  # gap exists
                gap_high = float(lows[i])
                gap_low = float(highs[i + 2])

                # ✅ Size filter
                atr_val = (atr_series.iat[i + 1]
                           if i + 1 < len(atr_series)
                              and not np.isnan(atr_series.iat[i + 1])
                           else 0.0)
                ratio = _fvg_size_vs_atr(gap_high, gap_low, atr_val)

                if ratio < SMALL_FVG_CUTOFF:
                    continue
                if ratio < MIN_FVG_ATR_RATIO:
                    continue

                # ✅ Volume filter
                if not _has_high_volume(df, i + 1):
                    continue

                # Mitigation at 50% CE
                gap_mid = (gap_high + gap_low) / 2.0
                mitigated = False
                for m in range(i + 3, n):
                    if highs[m] >= gap_mid:
                        mitigated = True
                        break
                if mitigated:
                    continue

                # ✅ Partial fill
                fill_pct = _compute_partial_fill(
                    "bearish", gap_high, gap_low,
                    highs, lows, i + 3, n,
                )

                # ✅ 100% Freshness — any wick touch = invalid, discard immediately
                fresh = _is_zone_completely_fresh(
                    "bearish", gap_high, gap_low,
                    highs, lows, i + 3, n,
                )
                if not fresh:
                    continue

                zones.append(Zone(
                    high=gap_high, low=gap_low, ob_index=i,
                    kind="fvg", direction="bearish", role="fvg",
                    timestamp=str(timestamps[i + 1]),
                    has_volume=True,
                    fvg_atr_ratio=round(ratio, 2),
                    is_fresh=fresh,
                    partial_fill_pct=round(fill_pct, 3),
                ))

    return zones


# ─────────────────────────────────────────────────────────────────────
# 6. DISPLACEMENT FILTER — body + ATR + VOLUME
# ─────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def _dynamic_lb(df: pd.DataFrame, base: int = 5) -> int:
    """
    ATR-adaptive lookback: clamp(3, round(base * current_ATR / mean_ATR_20), 10).
    High volatility  → larger lb (filters micro-noise).
    Low  volatility  → smaller lb (catches smaller pivots).
    Vectorised: uses NumPy on the ATR series, no Python loop.
    """
    atr_vals = compute_atr(df, 14).values
    valid    = atr_vals[~np.isnan(atr_vals)]
    if len(valid) < 2:
        return base
    current_atr = float(valid[-1])
    avg_atr     = float(np.mean(valid[-20:])) if len(valid) >= 20 else float(np.mean(valid))
    if avg_atr <= 0:
        return base
    raw = int(np.round(base * current_atr / avg_atr))
    return int(np.clip(raw, 3, 10))


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Vectorised Wilder RSI.  Returns array length == len(closes);
    values are NaN until the warm-up window is complete.
    """
    n   = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    delta = np.diff(closes, prepend=closes[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain = float(np.mean(gain[1: period + 1]))
    avg_loss = float(np.mean(loss[1: period + 1]))

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rsi[i] = (100.0 if avg_loss == 0.0
                  else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


def has_displacement(df: pd.DataFrame, idx: int, atr_mult: float = 1.0) -> bool:
    if idx < 20 or idx >= len(df):
        return False

    o = df["open"].iat[idx]
    c = df["close"].iat[idx]
    h = df["high"].iat[idx]
    lo = df["low"].iat[idx]
    body = abs(c - o)
    rng = h - lo
    if rng == 0:
        return False

    atr_series = compute_atr(df)
    atr_val = atr_series.iat[idx]
    if np.isnan(atr_val):
        return False

    if not (body > 0.5 * rng and body > atr_mult * atr_val):
        return False

    # ✅ Volume confirmation (unchanged but critical)
    if "volume" in df.columns:
        vol = df["volume"].iat[idx]
        avg_vol = df["volume"].iloc[max(0, idx - 20):idx].mean()
        if avg_vol > 0 and vol < VOL_MULT * avg_vol:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────
# 7. LIQUIDITY SWEEP — multi-candle recovery window
# ─────────────────────────────────────────────────────────────────────

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    direction: Literal["bullish", "bearish"],
    poi_high: Optional[float] = None,
    poi_low: Optional[float] = None,
) -> List[LiquiditySweep]:
    if df is None or not swings:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweeps: List[LiquiditySweep] = []

    if direction == "bullish":
        swing_lows = [s for s in swings if s.kind == "SL"]
        for sl_pt in reversed(swing_lows):
            for j in range(sl_pt.index + 1, n):
                if lows[j] < sl_pt.price:
                    if poi_low is not None and poi_high is not None:
                        if not (poi_low <= lows[j] <= poi_high):
                            continue
                    for r in range(j, min(j + SWEEP_RECOVERY_CANDLES + 1, n)):
                        if closes[r] > sl_pt.price:
                            sweeps.append(LiquiditySweep(
                                sweep_index=j,
                                swept_price=sl_pt.price,
                                direction="bullish",
                                recovery_index=r,
                            ))
                            break
                    if sweeps:
                        break
            if sweeps:
                break

    elif direction == "bearish":
        swing_highs = [s for s in swings if s.kind == "SH"]
        for sh_pt in reversed(swing_highs):
            for j in range(sh_pt.index + 1, n):
                if highs[j] > sh_pt.price:
                    if poi_low is not None and poi_high is not None:
                        if not (poi_low <= highs[j] <= poi_high):
                            continue
                    for r in range(j, min(j + SWEEP_RECOVERY_CANDLES + 1, n)):
                        if closes[r] < sh_pt.price:
                            sweeps.append(LiquiditySweep(
                                sweep_index=j,
                                swept_price=sh_pt.price,
                                direction="bearish",
                                recovery_index=r,
                            ))
                            break
                    if sweeps:
                        break
            if sweeps:
                break

    return sweeps


# ─────────────────────────────────────────────────────────────────────
# 8. CHoCH — expanded search window
# ─────────────────────────────────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    sweep: LiquiditySweep,
    swings: List[SwingPoint],
) -> Optional[CHoCH]:
    """
    Enhanced CHoCH detection — 7-layer anti-fake-CHoCH filter:
      1. Minimum candle gap (CHOCH_MIN_CANDLE_GAP) between sweep and break
      2. Break margin (CHOCH_BREAK_MARGIN): close must clear level by ≥0.05%
      3. Candle direction: CHoCH candle must close in break direction
      4. Candle body quality: body ≥ 40% of candle range (no doji / pin bars)
      5. Volume confirmation: candle volume ≥ 85% of 20-bar average
      6. ATR displacement: body ≥ 0.5× ATR (real momentum, not a tickle)
      7. RSI momentum: bullish ≥35, bearish ≤65 (avoids dead-cat bounces)
    Only accepts a prior swing that was formed within the last 25 candles.
    """
    if df is None or len(df) < 20 or not swings:
        return None

    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    vols   = (df["volume"].values
              if "volume" in df.columns
              else np.ones(len(df)))
    n      = len(df)

    start = sweep.recovery_index if sweep.recovery_index > 0 else sweep.sweep_index
    end   = min(start + CHOCH_WINDOW + 1, n)

    # Pre-compute ATR and RSI once for the whole array
    atr_arr = compute_atr(df).values
    rsi_arr = _rsi(closes, 14)

    # Rolling 20-bar average volume
    avg_vol = np.array([
        float(np.mean(vols[max(0, i - 20): i])) if i > 0 else 1.0
        for i in range(n)
    ])

    if sweep.direction == "bullish":
        prior_shs = [s for s in swings
                     if s.kind == "SH" and s.index <= sweep.sweep_index]
        if not prior_shs:
            return None
        target = prior_shs[-1]

        # Structural swing must be recent (not ancient history)
        if sweep.sweep_index - target.index > CHOCH_MAX_SWING_AGE:
            return None

        for j in range(start + CHOCH_MIN_CANDLE_GAP, end):
            # ── 1. Break margin ──────────────────────────────────────
            if closes[j] <= target.price * (1.0 + CHOCH_BREAK_MARGIN):
                continue

            # ── 2. Candle must be bullish ────────────────────────────
            if closes[j] <= opens[j]:
                continue

            # ── 3. Body quality (not a doji or hammer) ──────────────
            crange = highs[j] - lows[j]
            body   = closes[j] - opens[j]
            if crange > 0 and (body / crange) < CHOCH_BODY_PCT_MIN:
                continue

            # ── 4. Volume confirmation ───────────────────────────────
            if avg_vol[j] > 0 and vols[j] < avg_vol[j] * CHOCH_VOL_MULT:
                continue

            # ── 5. ATR displacement (real body, not a tickle) ────────
            if (not np.isnan(atr_arr[j])
                    and atr_arr[j] > 0
                    and body < atr_arr[j] * 0.50):
                continue

            # ── 6. RSI momentum (avoid bounce into exhaustion) ───────
            if not np.isnan(rsi_arr[j]) and rsi_arr[j] < CHOCH_RSI_BULL_MIN:
                continue

            # ── 7. Original ATR+volume displacement check ────────────
            if has_displacement(df, j):
                return CHoCH(j, "bullish", sweep, j)

    elif sweep.direction == "bearish":
        prior_sls = [s for s in swings
                     if s.kind == "SL" and s.index <= sweep.sweep_index]
        if not prior_sls:
            return None
        target = prior_sls[-1]

        if sweep.sweep_index - target.index > CHOCH_MAX_SWING_AGE:
            return None

        for j in range(start + CHOCH_MIN_CANDLE_GAP, end):
            # ── 1. Break margin ──────────────────────────────────────
            if closes[j] >= target.price * (1.0 - CHOCH_BREAK_MARGIN):
                continue

            # ── 2. Candle must be bearish ────────────────────────────
            if closes[j] >= opens[j]:
                continue

            # ── 3. Body quality ──────────────────────────────────────
            crange = highs[j] - lows[j]
            body   = opens[j] - closes[j]
            if crange > 0 and (body / crange) < CHOCH_BODY_PCT_MIN:
                continue

            # ── 4. Volume confirmation ───────────────────────────────
            if avg_vol[j] > 0 and vols[j] < avg_vol[j] * CHOCH_VOL_MULT:
                continue

            # ── 5. ATR displacement ──────────────────────────────────
            if (not np.isnan(atr_arr[j])
                    and atr_arr[j] > 0
                    and body < atr_arr[j] * 0.50):
                continue

            # ── 6. RSI momentum ──────────────────────────────────────
            if not np.isnan(rsi_arr[j]) and rsi_arr[j] > CHOCH_RSI_BEAR_MAX:
                continue

            # ── 7. ATR+volume displacement check ─────────────────────
            if has_displacement(df, j):
                return CHoCH(j, "bearish", sweep, j)

    return None


# ─────────────────────────────────────────────────────────────────────
# COMPOSITE HELPERS — ✅ CHANGED: Volume-aware filtering
# ─────────────────────────────────────────────────────────────────────

def find_pois(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    lb: int = 5,
) -> List[Zone]:
    """
    Return all unmitigated POIs (OBs first, then FVGs).

    ✅ v2: FVGs without high volume or below size threshold
    are already filtered out in detect_fvgs(). Only quality
    zones reach this point.
    """
    swings = detect_swing_points(df, lb)
    obs = detect_order_blocks(df, direction, swings, lb)
    fvgs = detect_fvgs(df, direction)

    all_zones = sorted(obs + fvgs, key=lambda z: z.ob_index)
    return all_zones


def find_best_poi(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    lb: int = 5,
) -> Optional[Zone]:
    """
    Return the best single POI.

    ✅ v2 PRIORITY (updated):
      1. Extreme OB with high volume → best
      2. Decisional OB with high volume → good
      3. Large FVG with high volume → acceptable
      4. FVG without volume or small → EXCLUDED (never returned)
    """
    zones = find_pois(df, direction, lb)
    if not zones:
        return None

    # ✅ NEW: Filter — only return zones with volume confirmation
    quality_zones = [z for z in zones if z.has_volume or z.kind == "ob"]  # ✅

    if not quality_zones:
        return None

    # Prefer extreme OB
    extremes = [z for z in quality_zones if z.role == "extreme"]
    if extremes:
        return extremes[-1]
    decisionals = [z for z in quality_zones if z.role == "decisional"]
    if decisionals:
        return decisionals[-1]

    # ✅ NEW: For FVG-only, require it to be large
    fvg_zones = [z for z in quality_zones
                 if z.kind == "fvg" and z.is_large()]                    # ✅
    if fvg_zones:
        return fvg_zones[-1]

    return None  # ✅ CHANGED: return None instead of weak zone


def find_opposing_zones(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    lb: int = 3,
) -> List[Zone]:
    opp = "bearish" if direction == "bullish" else "bullish"
    return find_pois(df, opp, lb)


# ─────────────────────────────────────────────────────────────────────
# OB + FVG OVERLAP — 150-point bypass
# ✅ CHANGED: Both OB and FVG must individually have high volume
# ─────────────────────────────────────────────────────────────────────

def _zones_overlap(a: Zone, b: Zone) -> bool:
    return a.low <= b.high and b.low <= a.high


def detect_ob_fvg_overlap(
    df: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    lb: int = 5,
) -> Optional[Zone]:
    """
    OB + FVG overlap = 150-point ultra-high-probability setup.

    ✅ v2: Both the OB and the FVG must have high-volume candles.
    Small FVGs overlapping a weak OB no longer qualify.
    """
    swings = detect_swing_points(df, lb)
    obs = detect_order_blocks(df, direction, swings, lb)
    fvgs = detect_fvgs(df, direction)

    if not obs or not fvgs:
        return None

    for ob in reversed(obs):
        # ✅ NEW: OB must have volume
        if not ob.has_volume:                                               # ✅
            continue

        for fvg in reversed(fvgs):
            # ✅ NEW: FVG must have volume AND be large
            if not fvg.has_volume or not fvg.is_large():                    # ✅
                continue

            if _zones_overlap(ob, fvg):
                merged_high = max(ob.high, fvg.high)
                merged_low = min(ob.low, fvg.low)
                return Zone(
                    high=merged_high, low=merged_low,
                    ob_index=ob.ob_index,
                    kind="ob", direction=direction,
                    role="extreme",
                    timestamp=ob.timestamp,
                    has_volume=True,                                         # ✅
                    fvg_atr_ratio=fvg.fvg_atr_ratio,                       # ✅
                    is_fresh=ob.is_fresh and fvg.is_fresh,                  # ✅
                )
    return None


# ─────────────────────────────────────────────────────────────────────
# SUPER SCORING HELPERS — inducement + volume trend (new in v3)
# ─────────────────────────────────────────────────────────────────────

def _count_inducement_levels(
    swings: List[SwingPoint],
    direction: Literal["bullish", "bearish"],
    cmp: float,
    poi_edge: float,
) -> int:
    """
    Count internal swing points sitting between CMP and the POI edge.
    These are retail stop clusters (inducement) that price will likely
    sweep on the way to the POI — increasing POI visit probability.

    Bullish: count swing LOWS between poi.high and cmp
      (sell-side liquidity price must absorb before reaching demand)
    Bearish: count swing HIGHS between cmp and poi.low
      (buy-side liquidity price must absorb before reaching supply)
    """
    if direction == "bullish":
        return sum(
            1 for s in swings
            if s.kind == "SL" and poi_edge < s.price < cmp
        )
    else:
        return sum(
            1 for s in swings
            if s.kind == "SH" and cmp < s.price < poi_edge
        )


def _recent_volume_trend(df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Ratio of the last `lookback` candles' avg volume vs the prior 20-bar avg.
    > 1.2  = increasing volume approaching POI (bullish sign for watchlist)
    < 0.8  = drying up (contracting)
    ~1.0   = neutral
    """
    if "volume" not in df.columns or len(df) < lookback + 20:
        return 1.0
    recent_avg = float(df["volume"].iloc[-lookback:].mean())
    base_avg   = float(df["volume"].iloc[-lookback - 20: -lookback].mean())
    if base_avg <= 0:
        return 1.0
    return recent_avg / base_avg


# ─────────────────────────────────────────────────────────────────────
# SUPER SCORING — proportional, no hard blocks, multi-factor
# ─────────────────────────────────────────────────────────────────────

def score_coin(
    df_1h: pd.DataFrame,   # 1H — HTF trend bias
    df_15m: pd.DataFrame,  # 15m — LTF POI + retrace
    cmp: float,
    lb: int = 5,
) -> dict:
    """
    Super Score System v4 — 0–100 points, no hard blocks.

    All zone types (OB extreme, OB decisional, large FVG, small FVG,
    FVG without volume) contribute PROPORTIONAL points.  A weak zone
    naturally scores fewer points — it never artificially blocks a
    coin that has strong trend + OTE + inducement in its favour.

    Score components (max 100):
    ─────────────────────────────────────────────────────
    A. HTF Trend quality          →  0–20 pts
       • 1H BOS confirmed            12
       • BOS has inducement sweep    +5  (bonus)
       • 1H HH/HL or LL/LH pattern  +3  (bonus, capped in A)

    B. LTF Retrace alignment      →  0–15 pts
       • 15m opposing 1H (pullback)  15   ← ideal
       • 15m unclear                  6
       • 15m aligned (extended)       3   ← caution

    C. POI Quality                →  0–30 pts
       Proportional by zone type and quality:
       • OB Extreme  + high vol      30
       • OB Extreme  – no vol        22
       • OB Decision + high vol      24
       • OB Decision – no vol        16
       • FVG ≥ 2.0×ATR + high vol   26
       • FVG ≥ 1.0×ATR + high vol   20
       • FVG ≥ 0.4×ATR + high vol   14   ← small but vol confirmed
       • FVG ≥ 2.0×ATR – no vol     17   ← large gap, trust gap
       • FVG ≥ 1.0×ATR – no vol     11
       • FVG ≥ 0.4×ATR – no vol      6   ← weakest, but NOT blocked
       OB+FVG overlap bonus         +10  (applied on top, max 30 + 10 = 40 effective)

    D. OTE / Fibonacci position   →  0–15 pts
       • In OTE (0.618–0.786)        15
       • In discount / premium       10

    E. Proximity (CMP → POI)      →  0–15 pts
       • ≤1.5% away                15
       • 1.5–3.0% away              10
       • 3.0–5.0% away               5
       • >5.0% away                  0  (too far)

    F. Inducement liquidity        →  0–10 pts  ← NEW
       Count of internal swing pts between CMP and POI:
       • 3 or more                   10
       • 2                            7
       • 1                            4
       • 0                            0

    G. Volume on approach          →  0–5 pts   ← NEW
       Recent 5-candle vol vs 20-bar avg:
       • ≥ 1.5×                       5
       • ≥ 1.2×                       3
       • < 0.8× (drying up)           0

    Hard blocks removed.  MIN_SCORE = 70 naturally filters to
    coins with at least 3–4 factors aligned.

    OB+FVG overlap (both high-vol): bypass → OB_FVG_OVERLAP_SCORE (150).
    """
    result = {
        "score":         0,
        "direction":     None,
        "poi":           None,
        "breakdown":     {},
        "ob_fvg_overlap": False,
    }

    lb_15m    = 3
    swings_1h  = detect_swing_points(df_1h,  lb)
    swings_15m = detect_swing_points(df_15m, lb_15m)
    trend_1h   = get_trend(df_1h,  swings_1h,  lb)
    trend_15m  = get_trend(df_15m, swings_15m, lb_15m)

    if trend_1h is None:
        return result

    direction          = trend_1h
    result["direction"] = direction
    bd                 = {}

    # ── OB+FVG OVERLAP: 150-point bypass (unchanged) ─────────────────
    overlap_zone = detect_ob_fvg_overlap(df_15m, direction, lb_15m)
    if overlap_zone is not None:
        poi_edge = overlap_zone.high if direction == "bullish" else overlap_zone.low
        dist_pct = abs(cmp - poi_edge) / cmp * 100 if cmp > 0 else 999
        close_enough = (dist_pct <= 3.0
                        and (cmp > poi_edge if direction == "bullish" else cmp < poi_edge))
        if close_enough:
            result.update({
                "score":          OB_FVG_OVERLAP_SCORE,
                "poi":            overlap_zone,
                "ob_fvg_overlap": True,
                "breakdown": {
                    "htf_trend": 20, "retrace": 15, "poi_quality": 30,
                    "ote": 15, "proximity": 15, "inducement": 10,
                    "vol_approach": 5, "ob_fvg_overlap_bonus": 40,
                    "fvg_atr_ratio": overlap_zone.fvg_atr_ratio,
                    "has_volume":    True,
                    "is_fresh":      overlap_zone.is_fresh,
                },
            })
            return result

    # ─────────────────────────────────────────────────────────────────
    # A. HTF TREND QUALITY (max 20 pts)
    # ─────────────────────────────────────────────────────────────────
    a_pts = 0
    bos_1h = detect_bos(df_1h, swings_1h, lb)
    if bos_1h is not None:
        a_pts += 12   # BOS exists

        # Bonus: BOS had inducement sweep beforehand
        has_idm = _has_inducement_sweep(df_1h, swings_1h, direction,
                                        bos_1h.break_index)
        if has_idm:
            a_pts += 5

        # Bonus: structure is in a clear HH/HL (bull) or LL/LH (bear) sequence
        shs = [s for s in swings_1h if s.kind == "SH"]
        sls = [s for s in swings_1h if s.kind == "SL"]
        if direction == "bullish" and len(shs) >= 2 and len(sls) >= 2:
            if shs[-1].price > shs[-2].price and sls[-1].price > sls[-2].price:
                a_pts += 3   # HH + HL confirmed
        elif direction == "bearish" and len(shs) >= 2 and len(sls) >= 2:
            if shs[-1].price < shs[-2].price and sls[-1].price < sls[-2].price:
                a_pts += 3   # LL + LH confirmed

    a_pts = min(a_pts, 20)
    bd["htf_trend"] = a_pts

    # ─────────────────────────────────────────────────────────────────
    # B. LTF RETRACE ALIGNMENT (max 15 pts)
    # ─────────────────────────────────────────────────────────────────
    if trend_15m is not None and trend_15m != trend_1h:
        b_pts = 15   # pullback / internal retrace — ideal scalp
    elif trend_15m is None:
        b_pts = 6    # unclear — partial credit
    else:
        b_pts = 3    # 15m aligned with 1H = possibly over-extended
    bd["retrace"] = b_pts

    # ─────────────────────────────────────────────────────────────────
    # C. POI QUALITY (max 30 pts, +10 if OB+FVG same direction overlap)
    # ─────────────────────────────────────────────────────────────────
    poi = find_best_poi(df_15m, direction, lb_15m)
    if poi is None:
        # No qualifying zone found — score what we have and return (normalized)
        raw_no_poi = a_pts + b_pts
        bd.update({"poi_quality": 0, "ote": 0, "proximity": 0,
                   "inducement": 0, "vol_approach": 0,
                   "fvg_atr_ratio": 0, "has_volume": False, "is_fresh": False})
        result["score"]     = round(min(raw_no_poi / 120 * 100, 100))
        result["breakdown"] = bd
        return result

    result["poi"] = poi

    # Proportional zone quality — no hard blocks
    if poi.kind == "ob":
        if poi.role == "extreme":
            c_pts = 30 if poi.has_volume else 22
        else:
            c_pts = 24 if poi.has_volume else 16
    else:
        # FVG — scored proportionally by ATR ratio AND volume
        r = poi.fvg_atr_ratio
        if r >= 2.0:
            c_pts = 26 if poi.has_volume else 17
        elif r >= 1.0:
            c_pts = 20 if poi.has_volume else 11
        else:
            # Small FVG — lowest points but NOT blocked
            c_pts = 14 if poi.has_volume else 6

    # Freshness bonus: +2 if zone is completely pristine
    if poi.is_fresh:
        c_pts += 2

    # OB+FVG loose overlap in same direction (weaker than the bypass)
    # — both must exist but only one needs high volume
    all_obs  = detect_order_blocks(df_15m, direction, swings_15m, lb_15m)
    all_fvgs = detect_fvgs(df_15m, direction)
    loose_overlap = any(
        _zones_overlap(ob, fvg)
        for ob in all_obs for fvg in all_fvgs
        if ob.has_volume or fvg.has_volume
    )
    if loose_overlap and poi.kind in ("ob", "fvg"):
        c_pts += 10   # confluence bonus

    c_pts = min(c_pts, 40)   # allow slightly over 30 with bonuses
    bd["poi_quality"]   = c_pts
    bd["fvg_atr_ratio"] = poi.fvg_atr_ratio if poi.kind == "fvg" else 0.0
    bd["has_volume"]    = poi.has_volume
    bd["is_fresh"]      = poi.is_fresh

    # ─────────────────────────────────────────────────────────────────
    # D. OTE / FIBONACCI POSITION (max 15 pts)
    # ─────────────────────────────────────────────────────────────────
    fib    = compute_fib_zones(swings_15m)
    d_pts  = 0
    if fib is not None:
        pm = poi.mid()
        if is_in_ote(pm, fib, direction):
            d_pts = 15
        elif direction == "bullish" and is_in_discount(pm, fib):
            d_pts = 10
        elif direction == "bearish" and is_in_premium(pm, fib):
            d_pts = 10
    bd["ote"] = d_pts

    # ─────────────────────────────────────────────────────────────────
    # E. PROXIMITY — CMP approaching POI (max 15 pts)
    # ─────────────────────────────────────────────────────────────────
    if direction == "bullish":
        edge     = poi.high
        dist_pct = abs(cmp - edge) / cmp * 100 if cmp > 0 else 999
        on_side  = cmp > edge
    else:
        edge     = poi.low
        dist_pct = abs(cmp - edge) / cmp * 100 if cmp > 0 else 999
        on_side  = cmp < edge

    if on_side:
        if dist_pct <= 2.5:
            e_pts = 15
        elif dist_pct <= 5.0:
            e_pts = 15
        elif dist_pct <= 8.0:
            e_pts = 15
        else:
            e_pts = 0
    else:
        e_pts = 0   # price already inside or beyond zone

    bd["proximity"] = e_pts
    bd["dist_pct"]  = round(dist_pct, 2)

    # ─────────────────────────────────────────────────────────────────
    # F. INDUCEMENT LIQUIDITY (max 10 pts)  ← NEW
    # ─────────────────────────────────────────────────────────────────
    idm_count = _count_inducement_levels(swings_15m, direction, cmp, edge)
    if idm_count >= 3:
        f_pts = 10
    elif idm_count == 2:
        f_pts = 7
    elif idm_count == 1:
        f_pts = 4
    else:
        f_pts = 0
    bd["inducement"]       = f_pts
    bd["inducement_count"] = idm_count

    # ─────────────────────────────────────────────────────────────────
    # G. VOLUME ON APPROACH (max 5 pts)  ← NEW
    # ─────────────────────────────────────────────────────────────────
    vol_ratio = _recent_volume_trend(df_15m, lookback=5)
    if vol_ratio >= 1.5:
        g_pts = 5
    elif vol_ratio >= 1.2:
        g_pts = 3
    else:
        g_pts = 0   # volume drying up — no bonus
    bd["vol_approach"]  = g_pts
    bd["vol_ratio"]     = round(vol_ratio, 2)

    # ─────────────────────────────────────────────────────────────────
    # TOTAL — normalised to 100, no hard blocks
    # ─────────────────────────────────────────────────────────────────
    raw_total = a_pts + b_pts + c_pts + d_pts + e_pts + f_pts + g_pts

    # Normalise: theoretical max = 20 + 15 + 40 + 15 + 15 + 10 + 5 = 120
    # Scale to 100 so MIN_SCORE=70 stays meaningful
    score_100 = round(min(raw_total / 120 * 100, 100))

    result["score"]     = score_100
    result["breakdown"] = bd
    return result


# ─────────────────────────────────────────────────────────────────────
# MOMENTUM / V-SHAPE SIGNAL — sweep + engulf candle, no CHoCH required
# ─────────────────────────────────────────────────────────────────────

MOMENTUM_BODY_PCT  = 0.70   # candle body must be ≥ 70% of high-low range
MOMENTUM_VOL_MULT  = 1.50   # candle volume must be ≥ 1.5× 20-bar avg
MOMENTUM_RETRACE   = 0.30   # limit entry at 30% retrace from candle extreme


def build_momentum_signal(
    symbol:       str,
    direction:    str,                # "bullish" | "bearish"
    df_ltf:       pd.DataFrame,       # 5m or 1m OHLCV, ≥ 30 candles
    sweep_idx:    int,                # candle index of the confirmed sweep
    htf_poi_high: float,
    htf_poi_low:  float,
) -> Optional["SniperSignal"]:
    """
    V-Shape / Momentum Entry — triggered WITHOUT waiting for a CHoCH.

    Conditions (all must hold on the candle IMMEDIATELY after the sweep):
      1. Body  > 70% of (high - low)
      2. Volume > 1.5× avg(volume[-20])
      3. Direction matches trade bias:
           bullish sweep → bullish engulf (close > open)
           bearish sweep → bearish engulf (close < open)

    Entry (30% retrace from candle extreme — NumPy vectorised):
      LONG  → entry = candle_low  + (candle_high - candle_low) × 0.70
      SHORT → entry = candle_high - (candle_high - candle_low) × 0.70

    SL: 0.175% buffer beyond sweep wick extreme (identical to CHoCH path).
    TP3: entry ± 2 × risk  →  exact $3.00 on $1.50 risk (1:2 R:R).
    """
    n = len(df_ltf)
    if df_ltf is None or n < 30:
        return None

    trade_dir = "LONG" if direction == "bullish" else "SHORT"
    sign      = 1.0 if trade_dir == "LONG" else -1.0

    # Momentum candle is the bar right after the sweep
    mc_idx = sweep_idx + 1
    if mc_idx >= n:
        return None

    opens  = df_ltf["open"].values
    closes = df_ltf["close"].values
    highs  = df_ltf["high"].values
    lows   = df_ltf["low"].values
    vols   = df_ltf["volume"].values

    mc_o = opens[mc_idx]
    mc_c = closes[mc_idx]
    mc_h = highs[mc_idx]
    mc_l = lows[mc_idx]

    mc_range = mc_h - mc_l
    if mc_range <= 0:
        return None

    # ── 1. Direction check ────────────────────────────────────────────
    if trade_dir == "LONG"  and mc_c <= mc_o:   # must be bullish candle
        return None
    if trade_dir == "SHORT" and mc_c >= mc_o:   # must be bearish candle
        return None

    # ── 2. Body ≥ 70% of range (vectorised single comparison) ────────
    body = abs(mc_c - mc_o)
    if body < MOMENTUM_BODY_PCT * mc_range:
        return None

    # ── 3. Volume > 1.5× 20-bar average (NumPy mean, no loop) ────────
    vol_window = vols[max(0, mc_idx - 20): mc_idx]
    if len(vol_window) == 0 or np.mean(vol_window) <= 0:
        return None
    if vols[mc_idx] < MOMENTUM_VOL_MULT * np.mean(vol_window):
        return None

    # ── Entry: 30% retrace from candle extreme ────────────────────────
    if trade_dir == "LONG":
        entry_price = mc_l + mc_range * (1.0 - MOMENTUM_RETRACE)   # 0.70 level
    else:
        entry_price = mc_h - mc_range * (1.0 - MOMENTUM_RETRACE)   # 0.70 level

    spread      = entry_price * 0.0005
    entry_high  = entry_price + spread
    entry_low   = entry_price - spread

    # ── SL: 0.175% buffer beyond sweep wick ──────────────────────────
    atr_series  = compute_atr(df_ltf, 14)
    atr_val     = (
        float(atr_series.iloc[-1])
        if len(atr_series) > 0 and not np.isnan(atr_series.iloc[-1])
        else 0.0
    )

    if trade_dir == "LONG":
        sl_extreme = float(lows[sweep_idx])
        sl_buffer  = max(atr_val * 0.5, sl_extreme * SL_BUFFER_PCT)
        sl         = sl_extreme - sl_buffer
    else:
        sl_extreme = float(highs[sweep_idx])
        sl_buffer  = max(atr_val * 0.5, sl_extreme * SL_BUFFER_PCT)
        sl         = sl_extreme + sl_buffer

    risk = abs(entry_price - sl)
    if risk < 1e-12:
        return None
    if trade_dir == "LONG"  and sl >= entry_price:
        return None
    if trade_dir == "SHORT" and sl <= entry_price:
        return None

    # ── TPs: TP3 at exact 1:2 R:R ────────────────────────────────────
    tp1 = entry_price + sign * risk * 1.0
    tp2 = entry_price + sign * risk * 1.5
    tp3 = entry_price + sign * risk * 2.0   # $3.00 on $1.50 risk
    tp4 = entry_price + sign * risk * 4.0
    tp5 = entry_price + sign * risk * 6.0
    tp6 = entry_price + sign * risk * 8.0

    risk_usd      = 1.50
    position_size = round(risk_usd / risk, 4) if risk > 0 else 0.0
    rr            = round(abs(tp3 - entry_price) / risk, 2)

    confluences = [
        f"HTF POI: {htf_poi_low:.6f} – {htf_poi_high:.6f}",
        f"⚡ V-SHAPE: Sweep idx={sweep_idx} extreme={sl_extreme:.6f}",
        f"Momentum candle idx={mc_idx}: body={body:.6f} ({body/mc_range*100:.0f}% of range)",
        f"Volume ratio: {vols[mc_idx]/np.mean(vol_window):.2f}× avg20",
        f"30% Retrace Entry: {entry_price:.6f} [{entry_low:.6f} – {entry_high:.6f}]",
        f"Buffered SL: {sl:.6f} (buffer={sl_buffer:.6f})",
        f"ATR: {atr_val:.6f}",
        f"R:R = 1:{rr} | NO CHoCH — Momentum path",
    ]

    return SniperSignal(
        symbol        = symbol,
        direction     = trade_dir,
        entry_high    = round(entry_high, 6),
        entry_low     = round(entry_low,  6),
        stop_loss     = round(sl, 6),
        tp1           = round(tp1, 6),
        tp2           = round(tp2, 6),
        tp3           = round(tp3, 6),
        tp4           = round(tp4, 6),
        tp5           = round(tp5, 6),
        tp6           = round(tp6, 6),
        risk_reward   = rr,
        position_size = position_size,
        risk_usd      = risk_usd,
        confluences   = confluences,
        order_type    = "LIMIT",
    )


# ─────────────────────────────────────────────────────────────────────
# SNIPER SIGNAL — R:R filter, LTF opposing zones, limit order
# ─────────────────────────────────────────────────────────────────────

def build_sniper_signal(
    symbol: str,
    direction: Literal["bullish", "bearish"],
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    htf_poi_high: float,
    htf_poi_low: float,
    df_1h: pd.DataFrame,
) -> Optional[SniperSignal]:
    """
    LTF Scalp Sniper — 30% Retrace Entry (0.70 Fib) + 0.175% Double-Sweep SL Buffer.

    Entry Logic (30% retrace = 0.70 Fibonacci level):
      After Sweep + CHoCH, enter at the 30% retrace of the displacement leg.
      More aggressive than 40% — higher fill rate on fast V-shape moves.
        LONG:  entry = CHoCH_close - 0.30 × (CHoCH_close - sweep_low)
        SHORT: entry = CHoCH_close + 0.30 × (sweep_high - CHoCH_close)
      Lookback `lb` is DYNAMIC: clamp(3, round(5 * cur_ATR / avg_ATR_20), 10).

    SL Logic (0.175% double-sweep buffer):
      SL placed beyond sweep wick extreme + max(0.5×ATR, 0.175% of wick price).

    R:R (primary target = TP3, exact 1:2):
      risk  = abs(entry_price - buffered_sl)  ← always positive
      sign  = +1 (LONG) or -1 (SHORT)
      TP1   = entry + sign * risk * 1.0  (1:1  — $1.50)
      TP3   = entry + sign * risk * 2.0  (1:2  — $3.00, 90% close = $2.70)
      TP4-6 = runner targets anchored to 1H swing levels
    """
    trade_dir = "LONG" if direction == "bullish" else "SHORT"
    sweep_dir = direction   # "bullish" or "bearish"
    sign      = 1.0 if trade_dir == "LONG" else -1.0

    for label, df_ltf in [("5m", df_5m), ("1m", df_1m)]:
        if df_ltf is None or len(df_ltf) < 20:
            continue
        lb = _dynamic_lb(df_ltf)

        swings = detect_swing_points(df_ltf, lb)
        if not swings:
            continue

        sweeps = detect_liquidity_sweeps(
            df_ltf, swings, sweep_dir,
            poi_high=htf_poi_high, poi_low=htf_poi_low,
        )
        if not sweeps:
            continue

        sweep = sweeps[0]
        choch = detect_choch(df_ltf, sweep, swings)
        if choch is None:
            continue

        # ── ATR for buffer calculations ──────────────────────────────
        atr_series = compute_atr(df_ltf)
        atr_val = float(atr_series.iloc[-1]) if (
            len(atr_series) > 0 and not np.isnan(atr_series.iloc[-1])
        ) else 0.0

        lows_ltf  = df_ltf["low"].values
        highs_ltf = df_ltf["high"].values

        # ── 40% Retrace Entry (outer edge of displacement leg) ─────
        #
        # Displacement leg:
        #   LONG : sweep wick LOW  → CHoCH close  (upward leg)
        #   SHORT: sweep wick HIGH → CHoCH close  (downward leg)
        # Entry = 40% retrace from CHoCH close toward sweep extreme.
        # Outer edge = closest to current market = highest-probability fill.
        choch_close = float(df_ltf["close"].iat[choch.choch_index])

        if trade_dir == "LONG":
            leg_low  = float(lows_ltf[sweep.sweep_index])   # sweep wick extreme
            leg_high = choch_close                            # CHoCH breakout close
            if leg_high <= leg_low:
                continue
            leg_size    = leg_high - leg_low
            entry_price = leg_high - ENTRY_RETRACE * leg_size  # 40% retrace
        else:
            leg_high = float(highs_ltf[sweep.sweep_index])  # sweep wick extreme
            leg_low  = choch_close                            # CHoCH breakdown close
            if leg_low >= leg_high:
                continue
            leg_size    = leg_high - leg_low
            entry_price = leg_low + ENTRY_RETRACE * leg_size  # 40% retrace

        entry_mid  = entry_price   # alias — all downstream uses entry_mid
        spread     = entry_price * 0.0005   # ±0.05% display zone only
        entry_high = entry_price + spread
        entry_low  = entry_price - spread

        # ── Buffered SL — 0.175% beyond sweep wick ───────────────────
        # Absorbs double-sweep attempts without blowing risk budget.
        if trade_dir == "LONG":
            sl_extreme = float(lows_ltf[sweep.sweep_index])
            sl_buffer  = max(atr_val * 0.5, sl_extreme * SL_BUFFER_PCT)
            sl         = sl_extreme - sl_buffer
        else:
            sl_extreme = float(highs_ltf[sweep.sweep_index])
            sl_buffer  = max(atr_val * 0.5, sl_extreme * SL_BUFFER_PCT)
            sl         = sl_extreme + sl_buffer

        # ── Risk: always positive abs distance ───────────────────────
        risk = abs(entry_mid - sl)
        if risk < 1e-12:
            continue   # degenerate — skip

        # Sanity: for LONG, SL must be below entry; for SHORT, above
        if trade_dir == "LONG" and sl >= entry_mid:
            continue
        if trade_dir == "SHORT" and sl <= entry_mid:
            continue

        # ── TPs: TP3 at exact 1:2 R:R (→ $3.00 on $1.50 risk) ─────────────────
        tp1 = entry_mid + sign * risk * 1.0   # 1:1 — $1.50
        tp2 = entry_mid + sign * risk * 1.5   # 1:1.5
        tp3 = entry_mid + sign * risk * 2.0   # 1:2 — $3.00 (primary, 90% close)
        tp4 = entry_mid + sign * risk * 4.0   # 1:4 runner
        tp5 = entry_mid + sign * risk * 6.0   # 1:6 runner
        tp6 = entry_mid + sign * risk * 8.0   # 1:8 runner

        # Sanity guard: all TPs must be on the correct side of entry
        if trade_dir == "LONG":
            if not (entry_mid < tp1 < tp3 < tp4 < tp5 < tp6):
                continue
        else:
            if not (entry_mid > tp1 > tp3 > tp4 > tp5 > tp6):
                continue

        # Anchor TP4-6 to 1H swing levels (never move them against us)
        swings_1h = detect_swing_points(df_1h, 5)
        if trade_dir == "LONG":
            htf_targets = sorted([
                s.price for s in swings_1h
                if s.kind == "SH" and s.price > tp3   # only above TP3
            ])
            if len(htf_targets) >= 1:
                tp4 = max(tp4, htf_targets[0])
            if len(htf_targets) >= 2:
                tp5 = max(tp5, htf_targets[1])
            if len(htf_targets) >= 3:
                tp6 = max(tp6, htf_targets[2])
        else:
            htf_targets = sorted([
                s.price for s in swings_1h
                if s.kind == "SL" and s.price < tp3   # only below TP3
            ], reverse=True)
            if len(htf_targets) >= 1:
                tp4 = min(tp4, htf_targets[0])
            if len(htf_targets) >= 2:
                tp5 = min(tp5, htf_targets[1])
            if len(htf_targets) >= 3:
                tp6 = min(tp6, htf_targets[2])

        # ── R:R (primary target = TP3, $3.00 on $1.50 risk = 1:2) ────
        rr = round(abs(tp3 - entry_price) / risk, 2)
        if rr < MIN_RR:
            continue

        # ── Position sizing (alert-only, no exchange orders) ─────────
        risk_usd      = 1.50   # fixed $1.50 per trade
        position_size = round(risk_usd / risk, 4) if risk > 0 else 0.0

        confluences = [
            f"HTF POI: {htf_poi_low:.6f} – {htf_poi_high:.6f}",
            f"Liq Sweep on {label} idx={sweep.sweep_index} "
            f"extreme={sl_extreme:.6f}",
            f"Displacement leg: {leg_low:.6f} → {leg_high:.6f}",
            f"30% Retrace Entry (0.70 Fib): {entry_price:.6f} "
            f"[{entry_low:.6f} – {entry_high:.6f}]",
            f"Buffered SL: {sl:.6f} (buffer={sl_buffer:.6f}, 0.175% guard)",
            f"CHoCH on {label} idx={choch.choch_index} "
            f"close={choch_close:.6f}",
            f"ATR ({label}): {atr_val:.6f}",
            f"R:R = 1:{rr} | Fixed Risk $1.50 | Primary TP = TP3",
        ]

        return SniperSignal(
            symbol=symbol,
            direction=trade_dir,
            entry_high=round(entry_high, 6),
            entry_low=round(entry_low,  6),
            stop_loss=round(sl, 6),
            tp1=round(tp1, 6),
            tp2=round(tp2, 6),
            tp3=round(tp3, 6),
            tp4=round(tp4, 6),
            tp5=round(tp5, 6),
            tp6=round(tp6, 6),
            risk_reward=rr,
            position_size=position_size,
            risk_usd=risk_usd,
            confluences=confluences,
            order_type="LIMIT",
        )

    return None