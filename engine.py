"""
engine.py — APEX V10 Analysis Engine
Full 5-timeframe SMC/ICT analysis with confluence scoring,
pre-signal filters, sniper entry detection, and indicator suite.
"""

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

logger = logging.getLogger("engine")

# ─────────────────────────────────────────────
# Exchange Manager
# ─────────────────────────────────────────────

class ExchangeManager:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.exchange = ccxt.binanceusdm({
            "apiKey":        api_key,
            "secret":        api_secret,
            "enableRateLimit": True,
            "options":       {"defaultType": "future"},
        })
        self._markets_loaded = False

    async def load_markets(self):
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def get_top_volume_pairs(self, n: int = 10) -> list[dict]:
        await self.load_markets()
        tickers = await self.exchange.fetch_tickers()
        pairs = [
            {"pair": sym, "volume": t.get("quoteVolume", 0), "price": t.get("last", 0),
             "change": round(t.get("percentage", 0), 2)}
            for sym, t in tickers.items()
            if sym.endswith("/USDT") and t.get("quoteVolume", 0) > 0
        ]
        pairs.sort(key=lambda x: x["volume"], reverse=True)
        return pairs[:n]

    async def fetch_ohlcv(self, pair: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        raw = await self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.reset_index(drop=True)

    async def fetch_ticker(self, pair: str) -> dict:
        return await self.exchange.fetch_ticker(pair)

    async def fetch_funding_rate(self, pair: str) -> float:
        try:
            fr = await self.exchange.fetch_funding_rate(pair)
            return float(fr.get("fundingRate", 0))
        except Exception:
            return 0.0

    async def fetch_balance(self) -> dict:
        try:
            bal = await self.exchange.fetch_balance()
            u = bal.get("USDT", {})
            return {"total": float(u.get("total", 0)),
                    "free":  float(u.get("free", 0)),
                    "used":  float(u.get("used", 0))}
        except Exception:
            return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────
# Indicator Functions
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 10) -> str:
    """
    Returns "bullish_div", "bearish_div", or "none"
    Bullish: price makes lower low but RSI makes higher low
    Bearish: price makes higher high but RSI makes lower high
    """
    if len(df) < lookback + 2:
        return "none"
    prices = df["close"].values[-lookback:]
    rsis   = rsi.values[-lookback:]
    p_low_idx  = int(np.argmin(prices))
    p_high_idx = int(np.argmax(prices))

    # Bullish divergence
    if p_low_idx < len(prices) - 3:
        later_low_price = prices[-1]
        later_low_rsi   = rsis[-1]
        if later_low_price < prices[p_low_idx] and later_low_rsi > rsis[p_low_idx]:
            return "bullish_div"

    # Bearish divergence
    if p_high_idx < len(prices) - 3:
        later_high_price = prices[-1]
        later_high_rsi   = rsis[-1]
        if later_high_price > prices[p_high_idx] and later_high_rsi < rsis[p_high_idx]:
            return "bearish_div"

    return "none"


# ─────────────────────────────────────────────
# SMC / ICT Functions
# ─────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lb: int = 5) -> tuple[list, list]:
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        w_h = df["high"].iloc[i - lb:i + lb + 1]
        w_l = df["low"].iloc[i  - lb:i + lb + 1]
        if df["high"].iloc[i] == w_h.max():
            highs.append({"i": i, "price": float(df["high"].iloc[i]),
                          "ts": str(df["timestamp"].iloc[i])})
        if df["low"].iloc[i] == w_l.min():
            lows.append({"i": i, "price": float(df["low"].iloc[i]),
                         "ts": str(df["timestamp"].iloc[i])})
    return highs, lows


def detect_structure(df: pd.DataFrame, highs: list, lows: list) -> dict:
    """BOS + CHoCH detection. Returns structured dict."""
    result = {
        "trend": "ranging",
        "bos": False, "bos_level": None, "bos_direction": None,
        "choch": False, "choch_level": None,
        "swing_highs": [], "swing_lows": [],
        "hh_hl": False, "ll_lh": False,
    }
    if len(highs) < 2 or len(lows) < 2:
        return result

    rh = sorted(highs, key=lambda x: x["i"])[-4:]
    rl = sorted(lows,  key=lambda x: x["i"])[-4:]
    result["swing_highs"] = [h["price"] for h in rh]
    result["swing_lows"]  = [l["price"] for l in rl]

    price = float(df["close"].iloc[-1])

    # HH/HL = bullish sequence
    if len(rh) >= 2 and len(rl) >= 2:
        result["hh_hl"] = rh[-1]["price"] > rh[-2]["price"] and rl[-1]["price"] > rl[-2]["price"]
        result["ll_lh"] = rh[-1]["price"] < rh[-2]["price"] and rl[-1]["price"] < rl[-2]["price"]

    # BOS: price closed beyond last swing
    if price > rh[-1]["price"]:
        result["bos"] = True
        result["bos_level"] = rh[-1]["price"]
        result["bos_direction"] = "bullish"
        result["trend"] = "bullish"
    elif price < rl[-1]["price"]:
        result["bos"] = True
        result["bos_level"] = rl[-1]["price"]
        result["bos_direction"] = "bearish"
        result["trend"] = "bearish"
    elif result["hh_hl"]:
        result["trend"] = "bullish"
    elif result["ll_lh"]:
        result["trend"] = "bearish"

    # CHoCH: last swing opposes current trend
    if result["trend"] == "bullish" and len(rh) >= 2:
        if rh[-1]["price"] < rh[-2]["price"]:
            result["choch"] = True
            result["choch_level"] = rh[-1]["price"]
    elif result["trend"] == "bearish" and len(rl) >= 2:
        if rl[-1]["price"] > rl[-2]["price"]:
            result["choch"] = True
            result["choch_level"] = rl[-1]["price"]

    return result


def detect_order_blocks(df: pd.DataFrame, timeframe: str) -> list[dict]:
    obs = []
    if len(df) < 10:
        return obs
    avg_range = float((df["high"] - df["low"]).mean())
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(3, len(df) - 3):
        move_up   = closes[i + 1] - opens[i + 1]
        move_down = opens[i + 1]  - closes[i + 1]

        # Bullish OB: last bearish candle before strong up impulse
        if opens[i] > closes[i] and move_up > avg_range * 1.5:
            tested = any(
                lows[j] <= opens[i] for j in range(i + 2, min(i + 15, len(df)))
            )
            fresh = not tested
            obs.append({
                "type": "bullish", "high": float(opens[i]), "low": float(lows[i]),
                "mid":  round((opens[i] + lows[i]) / 2, 6),
                "strength": round(min(move_up / (avg_range * 1.5), 1.0), 2),
                "fresh": fresh, "timeframe": timeframe,
                "ts": str(df["timestamp"].iloc[i]),
            })

        # Bearish OB: last bullish candle before strong down impulse
        if closes[i] > opens[i] and move_down > avg_range * 1.5:
            tested = any(
                highs[j] >= closes[i] for j in range(i + 2, min(i + 15, len(df)))
            )
            fresh = not tested
            obs.append({
                "type": "bearish", "high": float(highs[i]), "low": float(closes[i]),
                "mid":  round((highs[i] + closes[i]) / 2, 6),
                "strength": round(min(move_down / (avg_range * 1.5), 1.0), 2),
                "fresh": fresh, "timeframe": timeframe,
                "ts": str(df["timestamp"].iloc[i]),
            })

    # Sort by strength, return top 5, preferring fresh OBs
    obs.sort(key=lambda x: (int(x["fresh"]), x["strength"]), reverse=True)
    return obs[:5]


def detect_fvgs(df: pd.DataFrame) -> list[dict]:
    fvgs = []
    price = float(df["close"].iloc[-1])
    for i in range(1, len(df) - 1):
        ph = float(df["high"].iloc[i - 1])
        pl = float(df["low"].iloc[i - 1])
        nh = float(df["high"].iloc[i + 1])
        nl = float(df["low"].iloc[i + 1])
        ts = str(df["timestamp"].iloc[i])

        if ph < nl:  # Bullish FVG
            filled = price <= nl
            fvgs.append({"type": "bullish", "top": nl, "bottom": ph,
                         "mid": round((nl + ph) / 2, 6), "filled": filled, "ts": ts})
        if pl > nh:  # Bearish FVG
            filled = price >= nh
            fvgs.append({"type": "bearish", "top": pl, "bottom": nh,
                         "mid": round((pl + nh) / 2, 6), "filled": filled, "ts": ts})

    unfilled = [f for f in fvgs if not f["filled"]]
    unfilled.sort(key=lambda x: abs(x["mid"] - price))
    return unfilled[:4]


def detect_liquidity(df: pd.DataFrame, tol: float = 0.002) -> dict:
    highs  = df["high"].values
    lows   = df["low"].values
    price  = float(df["close"].iloc[-1])
    eq_highs, eq_lows = [], []

    for i in range(5, len(df) - 2):
        for j in range(i + 3, len(df) - 1):
            if abs(highs[i] - highs[j]) / highs[i] < tol:
                eq_highs.append(float(round((highs[i] + highs[j]) / 2, 6)))
                break
            if abs(lows[i] - lows[j]) / lows[i] < tol:
                eq_lows.append(float(round((lows[i] + lows[j]) / 2, 6)))
                break

    nearest_high = min(eq_highs, key=lambda x: abs(x - price)) if eq_highs else None
    nearest_low  = min(eq_lows,  key=lambda x: abs(x - price)) if eq_lows  else None

    return {
        "equal_highs":    eq_highs[-5:],
        "equal_lows":     eq_lows[-5:],
        "nearest_above":  nearest_high,
        "nearest_below":  nearest_low,
        "buy_side_liq":   nearest_high,   # resting above
        "sell_side_liq":  nearest_low,    # resting below
    }


def detect_candlestick_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    c, p, pp = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    body  = abs(c["close"] - c["open"])
    uw    = c["high"] - max(c["close"], c["open"])
    lw    = min(c["close"], c["open"]) - c["low"]
    rng   = c["high"] - c["low"]
    if rng < 1e-10:
        return "doji"

    body_r = body / rng

    if body_r < 0.08:                        return "doji"
    if lw > body * 2.2 and uw < body * 0.4: return "hammer"
    if uw > body * 2.2 and lw < body * 0.4: return "shooting_star"
    if lw > rng * 0.62:                      return "bullish_pinbar"
    if uw > rng * 0.62:                      return "bearish_pinbar"

    if (c["close"] > c["open"] and p["close"] < p["open"]
            and c["open"] <= p["close"] and c["close"] >= p["open"]):
        return "bullish_engulfing"
    if (c["close"] < c["open"] and p["close"] > p["open"]
            and c["open"] >= p["close"] and c["close"] <= p["open"]):
        return "bearish_engulfing"

    pb = abs(p["close"] - p["open"])
    ppb = abs(pp["close"] - pp["open"])
    if (pp["close"] < pp["open"] and pb < ppb * 0.3
            and c["close"] > c["open"] and c["close"] > (pp["open"] + pp["close"]) / 2):
        return "morning_star"
    if (pp["close"] > pp["open"] and pb < ppb * 0.3
            and c["close"] < c["open"] and c["close"] < (pp["open"] + pp["close"]) / 2):
        return "evening_star"

    if body_r > 0.88 and c["close"] > c["open"]: return "bullish_marubozu"
    if body_r > 0.88 and c["close"] < c["open"]: return "bearish_marubozu"

    return "none"


def detect_chart_pattern(df: pd.DataFrame) -> str:
    if len(df) < 30:
        return "none"
    h = df["high"].values[-30:]
    l = df["low"].values[-30:]
    hi, li = [], []
    for i in range(2, len(h) - 2):
        if h[i] == max(h[i-2:i+3]): hi.append(i)
        if l[i] == min(l[i-2:i+3]): li.append(i)

    if len(hi) >= 2:
        if abs(h[hi[-1]] - h[hi[-2]]) / h[hi[-1]] < 0.006: return "double_top"
    if len(li) >= 2:
        if abs(l[li[-1]] - l[li[-2]]) / l[li[-1]] < 0.006: return "double_bottom"

    if len(hi) >= 3 and len(li) >= 3:
        ht = np.polyfit(hi[-3:], [h[i] for i in hi[-3:]], 1)[0]
        lt = np.polyfit(li[-3:], [l[i] for i in li[-3:]], 1)[0]
        if ht > 0 and lt > 0:   return "ascending_channel"
        if ht < 0 and lt < 0:   return "descending_channel"
        if ht < 0 and lt > 0:   return "symmetrical_triangle"
        if abs(ht) < 0.0001 and lt > 0: return "ascending_triangle"
        if ht < 0 and abs(lt) < 0.0001: return "descending_triangle"

    return "none"


def volume_analysis(df: pd.DataFrame) -> dict:
    avg20 = float(df["volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
    last  = float(df["volume"].iloc[-1])
    avg5  = float(df["volume"].iloc[-6:-1].mean()) if len(df) > 6 else 1.0
    return {
        "last_volume":    round(last, 2),
        "avg20_volume":   round(avg20, 2),
        "spike":          last > avg20 * 2.0,
        "increasing":     avg5 > avg20 * 1.1,
        "decreasing":     avg5 < avg20 * 0.8,
        "ratio":          round(last / avg20, 2) if avg20 > 0 else 0,
    }


def ema_alignment(df: pd.DataFrame) -> dict:
    c = df["close"]
    e20  = float(ema(c, 20).iloc[-1])
    e50  = float(ema(c, 50).iloc[-1])
    e200 = float(ema(c, 200).iloc[-1])
    s20  = float(sma(c, 20).iloc[-1])
    s50  = float(sma(c, 50).iloc[-1])
    price = float(c.iloc[-1])

    bullish_stack = price > e20 > e50 > e200
    bearish_stack = price < e20 < e50 < e200
    return {
        "ema20": round(e20, 6), "ema50": round(e50, 6), "ema200": round(e200, 6),
        "sma20": round(s20, 6), "sma50": round(s50, 6),
        "price_above_ema20": price > e20,
        "price_above_ema50": price > e50,
        "price_above_ema200": price > e200,
        "bullish_stack": bullish_stack,
        "bearish_stack": bearish_stack,
        "ema20_cross_50": "golden" if e20 > e50 else "death",
    }


# ─────────────────────────────────────────────
# Sniper Conditions (5m)
# ─────────────────────────────────────────────

def evaluate_sniper_conditions(df5m: pd.DataFrame, direction: str) -> dict:
    """
    Evaluate 5 sniper entry conditions on 5m candles.
    Returns score 0–5 and which conditions passed.
    """
    rsi5  = compute_rsi(df5m["close"], 14)
    rsi_v = float(rsi5.iloc[-1])
    div   = detect_rsi_divergence(df5m, rsi5)
    vol   = volume_analysis(df5m)
    pat   = detect_candlestick_pattern(df5m)
    e20   = float(ema(df5m["close"], 20).iloc[-1])
    price = float(df5m["close"].iloc[-1])

    bullish_candles = {"hammer", "bullish_pinbar", "bullish_engulfing", "morning_star", "bullish_marubozu"}
    bearish_candles = {"shooting_star", "bearish_pinbar", "bearish_engulfing", "evening_star", "bearish_marubozu"}

    c1_rejection = (
        (direction == "long"  and pat in bullish_candles) or
        (direction == "short" and pat in bearish_candles)
    )
    c2_rsi_extreme = (
        (direction == "long"  and rsi_v < 42) or
        (direction == "short" and rsi_v > 58)
    )
    c3_divergence = (
        (direction == "long"  and div == "bullish_div") or
        (direction == "short" and div == "bearish_div")
    )
    c4_volume_spike = vol["spike"]
    c5_ema_direction = (
        (direction == "long"  and price > e20) or
        (direction == "short" and price < e20)
    )

    conditions = {
        "rejection_candle": c1_rejection,
        "rsi_extreme":      c2_rsi_extreme,
        "rsi_divergence":   c3_divergence,
        "volume_spike":     c4_volume_spike,
        "ema_alignment":    c5_ema_direction,
    }
    score = sum(conditions.values())
    entry_type = (
        "sniper"   if score == 5 else
        "strong"   if score >= 3 else
        "standard" if score >= 2 else
        "weak"
    )
    return {"score": score, "conditions": conditions, "entry_type": entry_type,
            "rsi_5m": round(rsi_v, 1), "pattern_5m": pat}


# ─────────────────────────────────────────────
# Confluence Scorer
# ─────────────────────────────────────────────

def compute_confluence_score(analyses: dict, sniper: dict) -> dict:
    score = 0
    breakdown = {}

    # ── Structure (30 pts) ──
    s = 0
    a1d = analyses.get("1d", {})
    a4h = analyses.get("4h", {})
    a1h = analyses.get("1h", {})
    a15 = analyses.get("15m", {})

    struct_1d = a1d.get("structure", {})
    struct_4h = a4h.get("structure", {})
    struct_1h = a1h.get("structure", {})
    struct_15 = a15.get("structure", {})

    if struct_1d.get("trend") != "ranging":          s += 8
    if struct_4h.get("bos"):                          s += 8
    if struct_1h.get("choch") or struct_1h.get("bos"): s += 7
    if struct_15.get("bos") or struct_15.get("choch"): s += 7
    breakdown["structure"] = min(s, 30)
    score += breakdown["structure"]

    # ── Order Blocks (25 pts) ──
    ob = 0
    for tf, pts, key in [("4h", 10, a4h), ("1h", 8, a1h), ("15m", 7, a15)]:
        obs = key.get("order_blocks", [])
        fresh_obs = [o for o in obs if o.get("fresh")]
        if fresh_obs:
            ob += pts
    breakdown["order_blocks"] = min(ob, 25)
    score += breakdown["order_blocks"]

    # ── Indicators (20 pts) ──
    ind = 0
    rsi_1h = a1h.get("rsi", 50)
    div_1h = a1h.get("rsi_divergence", "none")
    ema_1h = a1h.get("ema_alignment", {})
    vol_1h = a1h.get("volume", {})

    if div_1h in ("bullish_div", "bearish_div"):       ind += 7
    if 30 < rsi_1h < 45 or 55 < rsi_1h < 70:          ind += 5
    if ema_1h.get("bullish_stack") or ema_1h.get("bearish_stack"): ind += 5
    if vol_1h.get("spike"):                             ind += 3
    breakdown["indicators"] = min(ind, 20)
    score += breakdown["indicators"]

    # ── Patterns (15 pts) ──
    pat = 0
    chart_4h = a4h.get("chart_pattern", "none")
    chart_1h = a1h.get("chart_pattern", "none")
    candle_15 = a15.get("candlestick_pattern", "none")
    candle_1h = a1h.get("candlestick_pattern", "none")

    if chart_4h != "none" or chart_1h != "none":        pat += 8
    if candle_15 != "none" or candle_1h != "none":      pat += 7
    breakdown["patterns"] = min(pat, 15)
    score += breakdown["patterns"]

    # ── Sniper (10 pts) ──
    sniper_score = sniper.get("score", 0)
    sniper_pts = min(sniper_score * 2, 10)
    breakdown["sniper_5m"] = sniper_pts
    score += sniper_pts

    # Grade
    if score >= 95:   grade = "ELITE"
    elif score >= 85: grade = "STRONG"
    elif score >= 75: grade = "STANDARD"
    elif score >= 60: grade = "WEAK"
    else:             grade = "NO_SIGNAL"

    return {"total": min(score, 100), "breakdown": breakdown, "grade": grade}


# ─────────────────────────────────────────────
# Pre-Signal Filters
# ─────────────────────────────────────────────

def apply_pre_filters(analyses: dict, direction: str, price: float,
                       funding_rate: float) -> dict:
    """
    Returns {"passed": bool, "reason": str, "warnings": list}
    These filters KILL bad setups before Gemini is called.
    """
    warnings = []

    a1d = analyses.get("1d", {})
    a4h = analyses.get("4h", {})
    a1h = analyses.get("1h", {})

    trend_1d = a1d.get("structure", {}).get("trend", "ranging")
    trend_4h = a4h.get("structure", {}).get("trend", "ranging")
    rsi_1h   = a1h.get("rsi", 50)
    liq      = a1h.get("liquidity", {})
    atr_1h   = a1h.get("atr", 1.0)
    vol_1h   = a1h.get("volume", {})

    # F1: 1D and 4H trend must agree
    if trend_1d != "ranging" and trend_4h != "ranging" and trend_1d != trend_4h:
        return {"passed": False,
                "reason": f"HTF conflict: 1D={trend_1d} vs 4H={trend_4h}. No trade.",
                "warnings": warnings}

    # F2: RSI extremes (chasing)
    if direction == "long" and rsi_1h > 75:
        return {"passed": False,
                "reason": f"RSI 1H overbought ({rsi_1h:.0f}) for long. Risk of reversal.",
                "warnings": warnings}
    if direction == "short" and rsi_1h < 25:
        return {"passed": False,
                "reason": f"RSI 1H oversold ({rsi_1h:.0f}) for short. Risk of reversal.",
                "warnings": warnings}

    # F3: Price in liquidity trap (between equal highs and equal lows)
    eq_h = liq.get("nearest_above")
    eq_l = liq.get("nearest_below")
    if eq_h and eq_l:
        range_sz = (eq_h - eq_l) / price
        if range_sz < 0.008:
            return {"passed": False,
                    "reason": f"Price squeezed between liq zones ({eq_l}–{eq_h}). Sweep risk.",
                    "warnings": warnings}

    # F4: Extreme funding rate against direction
    if direction == "long" and funding_rate > 0.002:
        warnings.append(f"High funding rate {funding_rate*100:.3f}% — crowded longs")
    if direction == "short" and funding_rate < -0.002:
        warnings.append(f"Negative funding {funding_rate*100:.3f}% — crowded shorts")

    # F5: No active fresh OB within 1.5% of price
    obs_1h = a1h.get("order_blocks", [])
    fresh_nearby = [
        o for o in obs_1h
        if o.get("fresh") and abs(o.get("mid", price) - price) / price < 0.015
    ]
    if not fresh_nearby:
        warnings.append("No fresh 1H OB within 1.5% of price — entry may lack structure")

    # F6: Volume dying (no institutional interest)
    if vol_1h.get("decreasing") and not vol_1h.get("spike"):
        warnings.append("Volume declining on 1H — low conviction move")

    return {"passed": True, "reason": "All filters passed", "warnings": warnings}


# ─────────────────────────────────────────────
# Per-Timeframe Analysis
# ─────────────────────────────────────────────

def analyze_timeframe(df: pd.DataFrame, tf: str) -> dict:
    """Full SMC + ICT + indicator analysis for one timeframe."""
    highs, lows = detect_swing_points(df, lb=5 if tf in ("5m", "15m") else 7)
    structure   = detect_structure(df, highs, lows)
    obs         = detect_order_blocks(df, tf)
    fvgs        = detect_fvgs(df)
    liquidity   = detect_liquidity(df)
    candle_pat  = detect_candlestick_pattern(df)
    chart_pat   = detect_chart_pattern(df)
    vol         = volume_analysis(df)
    ema_info    = ema_alignment(df)

    rsi_series  = compute_rsi(df["close"], 14)
    rsi_val     = float(rsi_series.iloc[-1])
    rsi_div     = detect_rsi_divergence(df, rsi_series)
    atr_val     = float(compute_atr(df, 14).iloc[-1])
    price       = float(df["close"].iloc[-1])

    # OB tap: is price inside the best fresh OB?
    best_ob = next((o for o in obs if o.get("fresh")), None)
    ob_tap  = False
    if best_ob:
        ob_tap = best_ob["low"] <= price <= best_ob["high"]

    return {
        "timeframe":           tf,
        "price":               round(price, 6),
        "structure":           structure,
        "order_blocks":        obs,
        "fvgs":                fvgs,
        "liquidity":           liquidity,
        "candlestick_pattern": candle_pat,
        "chart_pattern":       chart_pat,
        "volume":              vol,
        "ema_alignment":       ema_info,
        "rsi":                 round(rsi_val, 2),
        "rsi_divergence":      rsi_div,
        "atr":                 round(atr_val, 6),
        "ob_tap":              ob_tap,
        "best_ob":             best_ob,
    }


# ─────────────────────────────────────────────
# Trading Session
# ─────────────────────────────────────────────

def get_session() -> str:
    hour = datetime.utcnow().hour
    if 22 <= hour or hour < 8:  return "asian"
    if 7  <= hour < 12:          return "london_open"
    if 12 <= hour < 17:          return "new_york_open"
    return "london_ny_overlap"


# ─────────────────────────────────────────────
# Risk Calculator
# ─────────────────────────────────────────────

def calculate_risk(entry: float, stop_loss: float,
                   account: float = 100.0, risk_pct: float = 0.015,
                   atr: float = 1.0) -> dict:
    """
    Returns full risk management data.
    Leverage based on SL distance + ATR volatility cap.
    """
    risk_amount   = account * risk_pct
    sl_dist_pct   = abs(entry - stop_loss) / entry
    if sl_dist_pct < 1e-6:
        return {}

    position_notional = risk_amount / sl_dist_pct
    raw_leverage      = position_notional / account

    # ATR-based volatility cap
    atr_pct    = atr / entry
    vol_cap    = max(1, min(20, int(0.02 / atr_pct)))   # tighter ATR → lower cap
    leverage   = min(int(raw_leverage) + 1, vol_cap, 20)
    leverage   = max(leverage, 1)

    margin     = round(position_notional / leverage, 2)
    margin     = min(margin, account * 0.3)              # never use >30% of account as margin

    # Recalc position with capped margin
    position_notional = margin * leverage

    tp1_dist = sl_dist_pct * 1.5
    tp2_dist = sl_dist_pct * 2.5
    tp3_dist = sl_dist_pct * 4.0

    direction_sign = 1 if entry > stop_loss else -1  # 1=long, -1=short... actually:
    # If entry > SL → long
    if stop_loss < entry:  # long
        tp1 = round(entry * (1 + tp1_dist), 6)
        tp2 = round(entry * (1 + tp2_dist), 6)
        tp3 = round(entry * (1 + tp3_dist), 6)
        hard_inv = round(stop_loss * 0.998, 6)
        soft_warn = round(stop_loss * 1.003, 6)
    else:  # short
        tp1 = round(entry * (1 - tp1_dist), 6)
        tp2 = round(entry * (1 - tp2_dist), 6)
        tp3 = round(entry * (1 - tp3_dist), 6)
        hard_inv = round(stop_loss * 1.002, 6)
        soft_warn = round(stop_loss * 0.997, 6)

    rr = round((tp2_dist / sl_dist_pct), 2)

    return {
        "risk_amount_usdt":  round(risk_amount, 2),
        "position_notional": round(position_notional, 2),
        "margin_usdt":       margin,
        "leverage":          leverage,
        "sl_dist_pct":       round(sl_dist_pct * 100, 3),
        "rr":                rr,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "hard_invalidation": hard_inv,
        "soft_warning":      soft_warn,
    }


# ─────────────────────────────────────────────
# Main Analysis Orchestrator
# ─────────────────────────────────────────────

async def run_full_analysis(exchange: ExchangeManager, pair: str,
                             account_size: float = 100.0,
                             risk_pct: float = 0.015,
                             log_fn=None) -> dict:
    """
    Fetches 5 timeframes, runs full SMC/ICT analysis, scores confluence,
    applies filters, and returns a structured payload ready for Gemini.
    """
    async def log(msg: str):
        if log_fn:
            await log_fn(msg)
        logger.info(msg)

    await log(f"[ENGINE] Starting full 5-TF analysis for {pair}...")

    # ── Fetch all timeframes in parallel ──
    await log("[ENGINE] Fetching OHLCV data: 1D · 4H · 1H · 15m · 5m")
    try:
        df1d, df4h, df1h, df15, df5m = await asyncio.gather(
            exchange.fetch_ohlcv(pair, "1d",  limit=200),
            exchange.fetch_ohlcv(pair, "4h",  limit=200),
            exchange.fetch_ohlcv(pair, "1h",  limit=200),
            exchange.fetch_ohlcv(pair, "15m", limit=200),
            exchange.fetch_ohlcv(pair, "5m",  limit=200),
        )
    except Exception as e:
        await log(f"[ENGINE] ❌ OHLCV fetch failed: {e}")
        return {"error": str(e)}

    await log("[ENGINE] Data fetched. Running SMC/ICT analysis per timeframe...")

    price = float(df1h["close"].iloc[-1])

    # ── Analyze each timeframe ──
    a1d  = analyze_timeframe(df1d,  "1d")
    await log(f"[ENGINE] 1D → Trend: {a1d['structure']['trend'].upper()} | RSI: {a1d['rsi']} | OBs: {len(a1d['order_blocks'])}")

    a4h  = analyze_timeframe(df4h,  "4h")
    await log(f"[ENGINE] 4H → Trend: {a4h['structure']['trend'].upper()} | BOS: {'✓' if a4h['structure']['bos'] else '✗'} | CHoCH: {'✓' if a4h['structure']['choch'] else '✗'} | FVGs: {len(a4h['fvgs'])}")

    a1h  = analyze_timeframe(df1h,  "1h")
    await log(f"[ENGINE] 1H → Trend: {a1h['structure']['trend'].upper()} | RSI: {a1h['rsi']} | Pattern: {a1h['candlestick_pattern']} | OB Tap: {'✓' if a1h['ob_tap'] else '✗'}")

    a15  = analyze_timeframe(df15,  "15m")
    await log(f"[ENGINE] 15m → Trend: {a15['structure']['trend'].upper()} | RSI: {a15['rsi']} | Pattern: {a15['candlestick_pattern']}")

    a5m  = analyze_timeframe(df5m,  "5m")
    await log(f"[ENGINE] 5m → RSI: {a5m['rsi']} | Pattern: {a5m['candlestick_pattern']} | Vol spike: {'✓' if a5m['volume']['spike'] else '✗'}")

    analyses = {"1d": a1d, "4h": a4h, "1h": a1h, "15m": a15, "5m": a5m}

    # ── Determine preliminary direction from HTF ──
    trend_1d = a1d["structure"]["trend"]
    trend_4h = a4h["structure"]["trend"]
    if trend_1d == "bullish" or trend_4h == "bullish":
        prelim_direction = "long"
    elif trend_1d == "bearish" or trend_4h == "bearish":
        prelim_direction = "short"
    else:
        prelim_direction = "long"  # Gemini will decide

    await log(f"[ENGINE] Preliminary direction: {prelim_direction.upper()} (HTF bias)")

    # ── Sniper conditions on 5m ──
    sniper = evaluate_sniper_conditions(df5m, prelim_direction)
    await log(f"[ENGINE] 5m Sniper score: {sniper['score']}/5 → {sniper['entry_type'].upper()} | Conditions: {sniper['conditions']}")

    # ── Confluence score ──
    confluence = compute_confluence_score(analyses, sniper)
    await log(f"[ENGINE] Confluence score: {confluence['total']}/100 → {confluence['grade']}")
    await log(f"[ENGINE] Breakdown → Structure:{confluence['breakdown']['structure']} | OBs:{confluence['breakdown']['order_blocks']} | Indicators:{confluence['breakdown']['indicators']} | Patterns:{confluence['breakdown']['patterns']} | Sniper:{confluence['breakdown']['sniper_5m']}")

    if confluence["grade"] in ("NO_SIGNAL", "WEAK"):
        await log(f"[ENGINE] ⚠️ Confluence too low ({confluence['total']}/100). Skipping Gemini call.")
        return {
            "pair": pair,
            "price": price,
            "confluence": confluence,
            "analyses": analyses,
            "no_signal": True,
            "no_signal_reason": f"Confluence score {confluence['total']}/100 below threshold (75)",
        }

    # ── Funding rate ──
    funding_rate = await exchange.fetch_funding_rate(pair)
    await log(f"[ENGINE] Funding rate: {funding_rate*100:.4f}%")

    # ── Pre-signal filters ──
    await log("[ENGINE] Applying pre-signal filters...")
    filters = apply_pre_filters(analyses, prelim_direction, price, funding_rate)
    if filters["warnings"]:
        for w in filters["warnings"]:
            await log(f"[ENGINE] ⚠️ Warning: {w}")
    if not filters["passed"]:
        await log(f"[ENGINE] ❌ Filter REJECTED: {filters['reason']}")
        return {
            "pair": pair, "price": price,
            "confluence": confluence, "analyses": analyses,
            "no_signal": True, "no_signal_reason": filters["reason"],
        }
    await log("[ENGINE] ✅ All filters passed.")

    session = get_session()
    await log(f"[ENGINE] Trading session: {session.upper()}")

    # ── Build Gemini payload ──
    def safe_obs(obs_list):
        return [
            {"type": o["type"], "high": o["high"], "low": o["low"],
             "mid": o["mid"], "strength": o["strength"], "fresh": o["fresh"]}
            for o in obs_list[:3]
        ]

    def safe_fvgs(fvg_list):
        return [
            {"type": f["type"], "top": f["top"], "bottom": f["bottom"],
             "mid": f["mid"]}
            for f in fvg_list[:2]
        ]

    gemini_payload = {
        "pair":           pair,
        "current_price":  price,
        "account_size":   account_size,
        "risk_pct":       risk_pct,
        "confluence_score": confluence["total"],
        "confluence_grade": confluence["grade"],
        "confluence_breakdown": confluence["breakdown"],
        "entry_type_hint":  sniper["entry_type"],
        "session":          session,
        "funding_rate":     funding_rate,
        "filter_warnings":  filters["warnings"],

        "timeframes": {
            "1d": {
                "trend":            a1d["structure"]["trend"],
                "bos":              a1d["structure"]["bos"],
                "choch":            a1d["structure"]["choch"],
                "hh_hl_sequence":   a1d["structure"]["hh_hl"],
                "ll_lh_sequence":   a1d["structure"]["ll_lh"],
                "swing_highs":      a1d["structure"]["swing_highs"][-3:],
                "swing_lows":       a1d["structure"]["swing_lows"][-3:],
                "order_blocks":     safe_obs(a1d["order_blocks"]),
                "fvgs":             safe_fvgs(a1d["fvgs"]),
                "ema_stack":        "bullish" if a1d["ema_alignment"]["bullish_stack"] else
                                    "bearish" if a1d["ema_alignment"]["bearish_stack"] else "mixed",
                "ema200":           a1d["ema_alignment"]["ema200"],
                "rsi":              a1d["rsi"],
                "volume_spike":     a1d["volume"]["spike"],
                "chart_pattern":    a1d["chart_pattern"],
            },
            "4h": {
                "trend":            a4h["structure"]["trend"],
                "bos":              a4h["structure"]["bos"],
                "bos_level":        a4h["structure"]["bos_level"],
                "choch":            a4h["structure"]["choch"],
                "choch_level":      a4h["structure"]["choch_level"],
                "order_blocks":     safe_obs(a4h["order_blocks"]),
                "fvgs":             safe_fvgs(a4h["fvgs"]),
                "liquidity_above":  a4h["liquidity"]["nearest_above"],
                "liquidity_below":  a4h["liquidity"]["nearest_below"],
                "rsi":              a4h["rsi"],
                "rsi_divergence":   a4h["rsi_divergence"],
                "ema_alignment":    a4h["ema_alignment"]["ema20_cross_50"],
                "volume_spike":     a4h["volume"]["spike"],
                "chart_pattern":    a4h["chart_pattern"],
                "atr":              a4h["atr"],
            },
            "1h": {
                "trend":            a1h["structure"]["trend"],
                "bos":              a1h["structure"]["bos"],
                "choch":            a1h["structure"]["choch"],
                "ob_tap":           a1h["ob_tap"],
                "best_ob":          {"high": a1h["best_ob"]["high"], "low": a1h["best_ob"]["low"]}
                                    if a1h["best_ob"] else None,
                "fvgs":             safe_fvgs(a1h["fvgs"]),
                "rsi":              a1h["rsi"],
                "rsi_divergence":   a1h["rsi_divergence"],
                "ema20":            a1h["ema_alignment"]["ema20"],
                "ema50":            a1h["ema_alignment"]["ema50"],
                "ema_bullish_stack": a1h["ema_alignment"]["bullish_stack"],
                "candlestick":      a1h["candlestick_pattern"],
                "chart_pattern":    a1h["chart_pattern"],
                "volume_spike":     a1h["volume"]["spike"],
                "atr":              a1h["atr"],
            },
            "15m": {
                "trend":            a15["structure"]["trend"],
                "bos":              a15["structure"]["bos"],
                "choch":            a15["structure"]["choch"],
                "ob_tap":           a15["ob_tap"],
                "best_ob":          {"high": a15["best_ob"]["high"], "low": a15["best_ob"]["low"]}
                                    if a15["best_ob"] else None,
                "rsi":              a15["rsi"],
                "rsi_divergence":   a15["rsi_divergence"],
                "candlestick":      a15["candlestick_pattern"],
                "volume":           "spike" if a15["volume"]["spike"] else
                                    "above_avg" if a15["volume"]["increasing"] else "normal",
            },
            "5m": {
                "sniper_score":           sniper["score"],
                "sniper_conditions":      sniper["conditions"],
                "entry_type":             sniper["entry_type"],
                "candlestick":            sniper["pattern_5m"],
                "rsi":                    sniper["rsi_5m"],
                "rsi_divergence":         a5m["rsi_divergence"],
                "volume_spike":           a5m["volume"]["spike"],
                "ema20_direction":        "up" if a5m["ema_alignment"]["price_above_ema20"] else "down",
            },
        },

        "liquidity_map": {
            "equal_highs_above": a1h["liquidity"]["nearest_above"],
            "equal_lows_below":  a1h["liquidity"]["nearest_below"],
            "buy_side_liq":      a4h["liquidity"]["buy_side_liq"],
            "sell_side_liq":     a4h["liquidity"]["sell_side_liq"],
        },
    }

    await log("[ENGINE] ✅ Analysis complete. Payload built for Gemini.")
    return {
        "pair": pair, "price": price,
        "confluence": confluence, "analyses": analyses,
        "sniper": sniper, "session": session,
        "funding_rate": funding_rate,
        "gemini_payload": gemini_payload,
        "no_signal": False,
        "atr_1h": a1h["atr"],
    }
