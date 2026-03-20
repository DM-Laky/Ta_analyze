"""
engine.py — APEX V10 Engine (Rewrite)
Correct ICT/SMC logic:
  PRIMARY  : Liquidity Sweep → BOS/CHoCH → Entry Zone (FVG or OB)
  SECONDARY: RSI, EMA, Volume, Patterns = confidence adjusters only
  RULE     : SMC structure alone can fire a signal. Indicators never block.
"""

import asyncio
import logging
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
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options":         {"defaultType": "future"},
        })
        self._markets_loaded = False

    async def load_markets(self):
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def get_top_volume_pairs(self, n: int = 10) -> list[dict]:
        """Fetch top N Binance Futures USDT pairs by 24h volume."""
        await self.load_markets()
        try:
            tickers = await self.exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"[SCREENER] fetch_tickers failed: {e}")
            return []

        pairs = []
        for sym, t in tickers.items():
            # Accept both spot-style and futures-style symbols
            if not sym.endswith("/USDT") and not sym.endswith("USDT"):
                continue
            vol = t.get("quoteVolume") or t.get("baseVolume") or 0
            price = t.get("last") or 0
            change = round(t.get("percentage") or 0, 2)
            if vol > 0 and price > 0:
                # Normalise to PAIR/USDT format
                display = sym if "/" in sym else sym.replace("USDT", "/USDT")
                pairs.append({
                    "pair":   display,
                    "volume": float(vol),
                    "price":  float(price),
                    "change": change,
                })

        pairs.sort(key=lambda x: x["volume"], reverse=True)
        return pairs[:n]

    async def fetch_ohlcv(self, pair: str, timeframe: str,
                           limit: int = 200) -> pd.DataFrame:
        raw = await self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
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
            u   = bal.get("USDT", {})
            return {
                "total": float(u.get("total", 0)),
                "free":  float(u.get("free",  0)),
                "used":  float(u.get("used",  0)),
            }
        except Exception:
            return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────
# Indicator Helpers
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi_divergence(df: pd.DataFrame, rsi: pd.Series,
                   lookback: int = 12) -> str:
    if len(df) < lookback + 2:
        return "none"
    p  = df["close"].values[-lookback:]
    r  = rsi.values[-lookback:]
    pi = int(np.argmin(p))
    pk = int(np.argmax(p))
    if pi < len(p) - 3 and p[-1] < p[pi] and r[-1] > r[pi]:
        return "bullish_div"
    if pk < len(p) - 3 and p[-1] > p[pk] and r[-1] < r[pk]:
        return "bearish_div"
    return "none"


# ─────────────────────────────────────────────
# SMC / ICT — PRIMARY DETECTION
# ─────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lb: int = 5) -> tuple[list, list]:
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        wh = df["high"].iloc[i - lb: i + lb + 1]
        wl = df["low"].iloc[i  - lb: i + lb + 1]
        if df["high"].iloc[i] == wh.max():
            highs.append({"i": i, "price": float(df["high"].iloc[i]),
                          "ts": str(df["timestamp"].iloc[i])})
        if df["low"].iloc[i] == wl.min():
            lows.append({"i": i, "price": float(df["low"].iloc[i]),
                         "ts": str(df["timestamp"].iloc[i])})
    return highs, lows


def detect_bos_choch(df: pd.DataFrame,
                     highs: list, lows: list) -> dict:
    """
    Detect Break of Structure and Change of Character.
    Returns direction, BOS level, CHoCH level.
    """
    result = {
        "trend":       "ranging",
        "bos":         False,  "bos_level":   None, "bos_dir": None,
        "choch":       False,  "choch_level": None,
        "hh_hl":       False,  "ll_lh":       False,
        "swing_highs": [],     "swing_lows":  [],
    }
    if len(highs) < 2 or len(lows) < 2:
        return result

    rh = sorted(highs, key=lambda x: x["i"])[-6:]
    rl = sorted(lows,  key=lambda x: x["i"])[-6:]
    result["swing_highs"] = [h["price"] for h in rh]
    result["swing_lows"]  = [l["price"] for l in rl]

    price = float(df["close"].iloc[-1])

    # HH/HL = bullish; LL/LH = bearish
    result["hh_hl"] = (rh[-1]["price"] > rh[-2]["price"] and
                       rl[-1]["price"] > rl[-2]["price"])
    result["ll_lh"] = (rh[-1]["price"] < rh[-2]["price"] and
                       rl[-1]["price"] < rl[-2]["price"])

    # BOS: price closed beyond last swing point
    if price > rh[-1]["price"]:
        result["bos"]    = True
        result["bos_level"] = rh[-1]["price"]
        result["bos_dir"]   = "bullish"
        result["trend"]     = "bullish"
    elif price < rl[-1]["price"]:
        result["bos"]    = True
        result["bos_level"] = rl[-1]["price"]
        result["bos_dir"]   = "bearish"
        result["trend"]     = "bearish"
    elif result["hh_hl"]:
        result["trend"] = "bullish"
    elif result["ll_lh"]:
        result["trend"] = "bearish"

    # CHoCH: swing structure flips against current trend
    if result["trend"] == "bullish" and len(rh) >= 2:
        if rh[-1]["price"] < rh[-2]["price"]:
            result["choch"]       = True
            result["choch_level"] = rh[-1]["price"]
    elif result["trend"] == "bearish" and len(rl) >= 2:
        if rl[-1]["price"] > rl[-2]["price"]:
            result["choch"]       = True
            result["choch_level"] = rl[-1]["price"]

    return result


def detect_liquidity_sweep(df: pd.DataFrame,
                            highs: list, lows: list,
                            tol: float = 0.002) -> dict:
    """
    CORE ICT DETECTION:
    Bullish sweep: price wicks BELOW an equal low then CLOSES back above
    Bearish sweep: price wicks ABOVE an equal high then CLOSES back below
    
    Returns the most recent confirmed sweep.
    """
    result = {
        "swept":          False,
        "direction":      None,   # "bullish" (swept lows) or "bearish" (swept highs)
        "sweep_level":    None,   # the liquidity level that was swept
        "sweep_low":      None,   # actual wick low (for SL placement)
        "sweep_high":     None,   # actual wick high (for SL placement)
        "sweep_index":    None,
        "candles_ago":    None,
        "equal_highs":    [],
        "equal_lows":     [],
    }

    if len(df) < 10:
        return result

    highs_arr  = df["high"].values
    lows_arr   = df["low"].values
    closes_arr = df["close"].values
    opens_arr  = df["open"].values

    # Build equal highs and equal lows
    eq_highs, eq_lows = [], []
    swing_h_prices = [h["price"] for h in highs]
    swing_l_prices = [l["price"] for l in lows]

    for i in range(len(swing_h_prices)):
        for j in range(i + 1, len(swing_h_prices)):
            if abs(swing_h_prices[i] - swing_h_prices[j]) / swing_h_prices[i] < tol:
                level = round((swing_h_prices[i] + swing_h_prices[j]) / 2, 6)
                if level not in eq_highs:
                    eq_highs.append(level)

    for i in range(len(swing_l_prices)):
        for j in range(i + 1, len(swing_l_prices)):
            if abs(swing_l_prices[i] - swing_l_prices[j]) / swing_l_prices[i] < tol:
                level = round((swing_l_prices[i] + swing_l_prices[j]) / 2, 6)
                if level not in eq_lows:
                    eq_lows.append(level)

    result["equal_highs"] = eq_highs[-4:]
    result["equal_lows"]  = eq_lows[-4:]

    # Scan recent candles for sweeps (most recent first)
    scan_range = min(30, len(df) - 1)
    for i in range(len(df) - 1, len(df) - scan_range, -1):
        low_i   = lows_arr[i]
        high_i  = highs_arr[i]
        close_i = closes_arr[i]
        open_i  = opens_arr[i]

        # BULLISH SWEEP: wick below equal low, close back above it
        for eq_low in eq_lows:
            if low_i < eq_low * (1 - tol * 0.5) and close_i > eq_low:
                result.update({
                    "swept":       True,
                    "direction":   "bullish",
                    "sweep_level": eq_low,
                    "sweep_low":   float(low_i),
                    "sweep_index": i,
                    "candles_ago": len(df) - 1 - i,
                })
                return result

        # BEARISH SWEEP: wick above equal high, close back below it
        for eq_high in eq_highs:
            if high_i > eq_high * (1 + tol * 0.5) and close_i < eq_high:
                result.update({
                    "swept":        True,
                    "direction":    "bearish",
                    "sweep_level":  eq_high,
                    "sweep_high":   float(high_i),
                    "sweep_index":  i,
                    "candles_ago":  len(df) - 1 - i,
                })
                return result

        # Also detect single candle sweeps (pin bar / hammer style)
        # Bullish: strong wick below recent low, closed above
        if i >= 5:
            recent_low = min(lows_arr[max(0, i-10):i])
            recent_high = max(highs_arr[max(0, i-10):i])
            wick_down = open_i - low_i if open_i > close_i else close_i - low_i
            wick_up   = high_i - close_i if close_i > open_i else high_i - open_i
            body      = abs(close_i - open_i)
            candle_range = high_i - low_i

            if (low_i < recent_low and close_i > recent_low and
                    candle_range > 0 and wick_down > candle_range * 0.55):
                result.update({
                    "swept":       True,
                    "direction":   "bullish",
                    "sweep_level": float(recent_low),
                    "sweep_low":   float(low_i),
                    "sweep_index": i,
                    "candles_ago": len(df) - 1 - i,
                })
                return result

            if (high_i > recent_high and close_i < recent_high and
                    candle_range > 0 and wick_up > candle_range * 0.55):
                result.update({
                    "swept":        True,
                    "direction":    "bearish",
                    "sweep_level":  float(recent_high),
                    "sweep_high":   float(high_i),
                    "sweep_index":  i,
                    "candles_ago":  len(df) - 1 - i,
                })
                return result

    return result


def detect_order_blocks(df: pd.DataFrame, direction: str) -> list[dict]:
    """
    ICT Order Block: Last candle of OPPOSITE color before an impulse move.
    For bullish signal: last BEARISH candle before strong up move.
    For bearish signal: last BULLISH candle before strong down move.
    Prioritises FRESH (untested) OBs.
    """
    obs      = []
    avg_rng  = float((df["high"] - df["low"]).mean())
    closes   = df["close"].values
    opens    = df["open"].values
    highs    = df["high"].values
    lows     = df["low"].values
    price    = float(closes[-1])

    for i in range(2, len(df) - 3):
        next_move_up   = closes[i + 1] - opens[i + 1]
        next_move_down = opens[i + 1]  - closes[i + 1]

        if direction == "long" or direction == "bullish":
            # Last bearish candle before strong bullish impulse
            if opens[i] > closes[i] and next_move_up > avg_rng * 1.2:
                tested = any(lows[j] <= closes[i]
                             for j in range(i + 2, min(i + 20, len(df))))
                fresh  = not tested
                dist   = abs(price - ((opens[i] + lows[i]) / 2)) / price
                obs.append({
                    "type":     "bullish",
                    "high":     float(opens[i]),
                    "low":      float(lows[i]),
                    "mid":      round((opens[i] + lows[i]) / 2, 6),
                    "strength": round(min(next_move_up / (avg_rng * 1.2), 1.5), 2),
                    "fresh":    fresh,
                    "dist_pct": round(dist * 100, 3),
                    "index":    i,
                })

        if direction == "short" or direction == "bearish":
            # Last bullish candle before strong bearish impulse
            if closes[i] > opens[i] and next_move_down > avg_rng * 1.2:
                tested = any(highs[j] >= opens[i]
                             for j in range(i + 2, min(i + 20, len(df))))
                fresh  = not tested
                dist   = abs(price - ((closes[i] + highs[i]) / 2)) / price
                obs.append({
                    "type":     "bearish",
                    "high":     float(highs[i]),
                    "low":      float(closes[i]),
                    "mid":      round((closes[i] + highs[i]) / 2, 6),
                    "strength": round(min(next_move_down / (avg_rng * 1.2), 1.5), 2),
                    "fresh":    fresh,
                    "dist_pct": round(dist * 100, 3),
                    "index":    i,
                })

    # Sort: fresh first, then by distance (closest to price)
    obs.sort(key=lambda x: (not x["fresh"], x["dist_pct"]))
    return obs[:5]


def detect_fvgs(df: pd.DataFrame, direction: str) -> list[dict]:
    """
    Fair Value Gap (3-candle imbalance):
    Bullish FVG: candle[i-1].high < candle[i+1].low  → gap = unfilled imbalance
    Bearish FVG: candle[i-1].low  > candle[i+1].high → gap = unfilled imbalance
    Prioritise unfilled, closest to current price.
    """
    fvgs  = []
    price = float(df["close"].iloc[-1])

    for i in range(1, len(df) - 1):
        ph = float(df["high"].iloc[i - 1])
        pl = float(df["low"].iloc[i  - 1])
        nh = float(df["high"].iloc[i + 1])
        nl = float(df["low"].iloc[i  + 1])
        ts = str(df["timestamp"].iloc[i])

        if direction in ("long", "bullish") and ph < nl:
            # Bullish FVG: gap between prev high and next low
            top    = nl
            bottom = ph
            mid    = round((top + bottom) / 2, 6)
            size   = round((top - bottom) / price * 100, 3)
            filled = price <= bottom  # price has already gone below
            if size > 0:
                fvgs.append({
                    "type": "bullish", "top": top, "bottom": bottom,
                    "mid": mid, "size_pct": size, "filled": filled, "ts": ts,
                    "dist_pct": round(abs(price - mid) / price * 100, 3),
                })

        if direction in ("short", "bearish") and pl > nh:
            # Bearish FVG
            top    = pl
            bottom = nh
            mid    = round((top + bottom) / 2, 6)
            size   = round((top - bottom) / price * 100, 3)
            filled = price >= top
            if size > 0:
                fvgs.append({
                    "type": "bearish", "top": top, "bottom": bottom,
                    "mid": mid, "size_pct": size, "filled": filled, "ts": ts,
                    "dist_pct": round(abs(price - mid) / price * 100, 3),
                })

    unfilled = [f for f in fvgs if not f["filled"]]
    unfilled.sort(key=lambda x: x["dist_pct"])
    return unfilled[:4]


def find_best_entry_zone(obs: list[dict], fvgs: list[dict],
                          price: float, direction: str) -> dict:
    """
    Select the BEST entry zone between OBs and FVGs.
    Best = freshest + closest to current price + largest zone (more buffer).
    Returns a single entry dict with zone, mid, and source.
    """
    candidates = []

    for ob in obs[:3]:
        if ob["fresh"]:
            score = 100 - ob["dist_pct"] * 10  # closer = better
            score += ob["strength"] * 20
            candidates.append({
                "source":   "OB",
                "high":     ob["high"],
                "low":      ob["low"],
                "entry":    ob["mid"],
                "dist_pct": ob["dist_pct"],
                "score":    score,
            })

    for fvg in fvgs[:3]:
        score = 100 - fvg["dist_pct"] * 8
        score += fvg["size_pct"] * 5
        candidates.append({
            "source":   "FVG",
            "high":     fvg["top"],
            "low":      fvg["bottom"],
            "entry":    fvg["mid"],
            "dist_pct": fvg["dist_pct"],
            "score":    score,
        })

    if not candidates:
        # Fallback: use price itself (market order territory)
        return {
            "source": "MARKET",
            "high":   price * 1.001,
            "low":    price * 0.999,
            "entry":  price,
            "dist_pct": 0,
            "score":  50,
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────
# Secondary Indicators (confidence adjusters)
# ─────────────────────────────────────────────

def secondary_indicators(df: pd.DataFrame, direction: str) -> dict:
    """
    RSI, EMA, Volume, Candlestick, Chart Pattern.
    These ONLY adjust confidence. They NEVER block a signal.
    """
    close  = df["close"]
    price  = float(close.iloc[-1])

    # RSI
    rsi_s  = compute_rsi(close, 14)
    rsi_v  = float(rsi_s.iloc[-1])
    rsi_div = rsi_divergence(df, rsi_s)

    # EMA
    e20  = float(ema(close, 20).iloc[-1])
    e50  = float(ema(close, 50).iloc[-1])
    e200 = float(ema(close, 200).iloc[-1])
    bull_ema = price > e20 > e50
    bear_ema = price < e20 < e50

    # Volume
    avg20 = float(df["volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
    last_v = float(df["volume"].iloc[-1])
    vol_spike = last_v > avg20 * 1.8
    vol_ratio = round(last_v / avg20, 2) if avg20 > 0 else 1.0

    # ATR
    atr_v = float(compute_atr(df, 14).iloc[-1])

    # Candlestick pattern
    candle_pat = _candlestick_pattern(df)

    # Chart pattern
    chart_pat = _chart_pattern(df)

    # Confidence contribution (max 30 pts)
    conf_pts = 0
    if direction in ("long", "bullish"):
        if rsi_div == "bullish_div":       conf_pts += 8
        if rsi_v < 45:                     conf_pts += 5
        if bull_ema:                       conf_pts += 6
        if vol_spike:                      conf_pts += 6
        if candle_pat in {"hammer","bullish_pinbar","bullish_engulfing",
                           "morning_star","bullish_marubozu"}: conf_pts += 5
    else:
        if rsi_div == "bearish_div":       conf_pts += 8
        if rsi_v > 55:                     conf_pts += 5
        if bear_ema:                       conf_pts += 6
        if vol_spike:                      conf_pts += 6
        if candle_pat in {"shooting_star","bearish_pinbar","bearish_engulfing",
                           "evening_star","bearish_marubozu"}: conf_pts += 5

    return {
        "rsi":            round(rsi_v, 2),
        "rsi_divergence": rsi_div,
        "ema20":          round(e20, 6),
        "ema50":          round(e50, 6),
        "ema200":         round(e200, 6),
        "bull_ema_stack": bull_ema,
        "bear_ema_stack": bear_ema,
        "vol_spike":      vol_spike,
        "vol_ratio":      vol_ratio,
        "atr":            round(atr_v, 6),
        "candle_pattern": candle_pat,
        "chart_pattern":  chart_pat,
        "conf_pts":       min(conf_pts, 30),
    }


def _candlestick_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    c, p, pp = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body = abs(c["close"] - c["open"])
    uw   = c["high"] - max(c["close"], c["open"])
    lw   = min(c["close"], c["open"]) - c["low"]
    rng  = c["high"] - c["low"]
    if rng < 1e-10: return "doji"
    if body / rng < 0.08:          return "doji"
    if lw > body * 2.2 and uw < body * 0.4: return "hammer"
    if uw > body * 2.2 and lw < body * 0.4: return "shooting_star"
    if lw > rng * 0.62:            return "bullish_pinbar"
    if uw > rng * 0.62:            return "bearish_pinbar"
    if (c["close"] > c["open"] and p["close"] < p["open"]
            and c["open"] <= p["close"] and c["close"] >= p["open"]):
        return "bullish_engulfing"
    if (c["close"] < c["open"] and p["close"] > p["open"]
            and c["open"] >= p["close"] and c["close"] <= p["open"]):
        return "bearish_engulfing"
    pb  = abs(p["close"] - p["open"])
    ppb = abs(pp["close"] - pp["open"])
    if (pp["close"] < pp["open"] and pb < ppb * 0.3 and c["close"] > c["open"]
            and c["close"] > (pp["open"] + pp["close"]) / 2):
        return "morning_star"
    if (pp["close"] > pp["open"] and pb < ppb * 0.3 and c["close"] < c["open"]
            and c["close"] < (pp["open"] + pp["close"]) / 2):
        return "evening_star"
    if body / rng > 0.88 and c["close"] > c["open"]: return "bullish_marubozu"
    if body / rng > 0.88 and c["close"] < c["open"]: return "bearish_marubozu"
    return "none"


def _chart_pattern(df: pd.DataFrame) -> str:
    if len(df) < 30: return "none"
    h = df["high"].values[-30:]
    l = df["low"].values[-30:]
    hi, li = [], []
    for i in range(2, len(h) - 2):
        if h[i] == max(h[i-2:i+3]): hi.append(i)
        if l[i] == min(l[i-2:i+3]): li.append(i)
    if len(hi) >= 2 and abs(h[hi[-1]] - h[hi[-2]]) / (h[hi[-1]] + 1e-9) < 0.006:
        return "double_top"
    if len(li) >= 2 and abs(l[li[-1]] - l[li[-2]]) / (l[li[-1]] + 1e-9) < 0.006:
        return "double_bottom"
    if len(hi) >= 3 and len(li) >= 3:
        ht = np.polyfit(hi[-3:], [h[i] for i in hi[-3:]], 1)[0]
        lt = np.polyfit(li[-3:], [l[i] for i in li[-3:]], 1)[0]
        if ht > 0 and lt > 0:   return "ascending_channel"
        if ht < 0 and lt < 0:   return "descending_channel"
        if ht < 0 and lt > 0:   return "symmetrical_triangle"
        if abs(ht) < 0.0001 and lt > 0: return "ascending_triangle"
        if ht < 0 and abs(lt) < 0.0001: return "descending_triangle"
    return "none"


# ─────────────────────────────────────────────
# Risk Calculator
# ─────────────────────────────────────────────

def calculate_risk(entry: float, stop_loss: float,
                   account: float = 100.0,
                   risk_pct: float = 0.015,
                   atr: float = 1.0) -> dict:
    risk_amount  = account * risk_pct
    sl_dist      = abs(entry - stop_loss)
    sl_dist_pct  = sl_dist / entry if entry > 0 else 0.01
    if sl_dist_pct < 1e-6:
        sl_dist_pct = 0.01

    notional     = risk_amount / sl_dist_pct
    raw_lev      = notional / account

    # ATR volatility cap
    atr_pct  = atr / entry if entry > 0 else 0.01
    vol_cap  = max(1, min(20, int(0.025 / atr_pct)))
    leverage = int(min(raw_lev + 1, vol_cap, 20))
    leverage = max(leverage, 1)

    margin   = round(notional / leverage, 2)
    margin   = min(margin, account * 0.35)  # max 35% of account as margin

    # Recalc notional with capped margin
    notional = margin * leverage

    is_long  = stop_loss < entry
    m1, m2, m3 = sl_dist_pct * 1.5, sl_dist_pct * 2.5, sl_dist_pct * 4.0

    if is_long:
        tp1 = round(entry * (1 + m1), 6)
        tp2 = round(entry * (1 + m2), 6)
        tp3 = round(entry * (1 + m3), 6)
        hard_inv  = round(stop_loss * 0.998, 6)
        soft_warn = round(stop_loss * 1.004, 6)
    else:
        tp1 = round(entry * (1 - m1), 6)
        tp2 = round(entry * (1 - m2), 6)
        tp3 = round(entry * (1 - m3), 6)
        hard_inv  = round(stop_loss * 1.002, 6)
        soft_warn = round(stop_loss * 0.996, 6)

    return {
        "risk_amount_usdt":  round(risk_amount, 2),
        "position_notional": round(notional, 2),
        "margin_usdt":       margin,
        "leverage":          leverage,
        "sl_dist_pct":       round(sl_dist_pct * 100, 3),
        "rr":                round(m2 / sl_dist_pct, 2),
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "hard_invalidation": hard_inv,
        "soft_warning":      soft_warn,
    }


# ─────────────────────────────────────────────
# Per-Timeframe Full Analysis
# ─────────────────────────────────────────────

def analyze_tf(df: pd.DataFrame, tf: str,
               direction_hint: str = "any") -> dict:
    """Full SMC + indicator analysis for one timeframe."""
    lb      = 3 if tf in ("5m", "15m") else 5
    highs, lows = detect_swing_points(df, lb=lb)
    struct  = detect_bos_choch(df, highs, lows)
    sweep   = detect_liquidity_sweep(df, highs, lows)

    # Use sweep direction or structure trend for directional detection
    dir_for_obs = direction_hint
    if direction_hint == "any":
        if sweep["swept"]:
            dir_for_obs = sweep["direction"]
        else:
            dir_for_obs = struct["trend"] if struct["trend"] != "ranging" else "long"

    obs     = detect_order_blocks(df, dir_for_obs)
    fvgs    = detect_fvgs(df, dir_for_obs)
    price   = float(df["close"].iloc[-1])
    entry_z = find_best_entry_zone(obs, fvgs, price, dir_for_obs)
    indics  = secondary_indicators(df, dir_for_obs)

    # Is price currently inside the entry zone?
    in_zone = (entry_z["low"] <= price <= entry_z["high"])

    return {
        "tf":          tf,
        "price":       round(price, 8),
        "structure":   struct,
        "sweep":       sweep,
        "obs":         obs,
        "fvgs":        fvgs,
        "entry_zone":  entry_z,
        "in_zone":     in_zone,
        "indicators":  indics,
    }


# ─────────────────────────────────────────────
# SMC Signal Decision Logic
# ─────────────────────────────────────────────

def make_smc_decision(analyses: dict) -> dict:
    """
    PRIMARY ICT DECISION ENGINE.

    Signal fires when ANY of these SMC conditions are met:
    1. Liquidity sweep + BOS/CHoCH on 1H or 4H (highest weight)
    2. Sweep on 15m confirmed by 1H structure
    3. BOS + fresh OB + FVG on 1H (no sweep needed if all three present)

    Direction is determined by sweep direction or dominant structure.
    Indicators only adjust base confidence (70%) up or down.
    """
    result = {
        "signal":          False,
        "direction":       None,
        "base_confidence": 0,
        "final_confidence": 0,
        "trigger":         None,   # what triggered the signal
        "entry_zone":      None,
        "sweep_level":     None,
        "stop_loss_level": None,
        "reasons":         [],
        "indic_bonus":     0,
    }

    a1d  = analyses.get("1d",  {})
    a4h  = analyses.get("4h",  {})
    a1h  = analyses.get("1h",  {})
    a15  = analyses.get("15m", {})
    a5m  = analyses.get("5m",  {})

    def s(a):  return a.get("structure", {})
    def sw(a): return a.get("sweep", {})
    def ez(a): return a.get("entry_zone", {})
    def ind(a): return a.get("indicators", {})

    reasons = []
    direction = None
    base_conf = 0
    trigger   = None
    entry_zone = None
    sl_level   = None
    sweep_level = None

    # ── TRIGGER 1: 4H sweep + 1H BOS (highest quality) ──
    if sw(a4h).get("swept") and (s(a1h).get("bos") or s(a1h).get("choch")):
        direction  = sw(a4h)["direction"]
        base_conf  = 80
        trigger    = "4H_sweep_1H_BOS"
        sweep_level = sw(a4h).get("sweep_level")
        entry_zone  = ez(a1h) if ez(a1h).get("entry") else ez(a4h)
        # SL: beyond the sweep wick
        if direction == "bullish":
            sl_level = sw(a4h).get("sweep_low", sweep_level * 0.998)
            sl_level = round(sl_level * 0.999, 6)
        else:
            sl_level = sw(a4h).get("sweep_high", sweep_level * 1.002)
            sl_level = round(sl_level * 1.001, 6)
        reasons.append(f"4H liquidity sweep at {sweep_level} + 1H BOS confirmed")

    # ── TRIGGER 2: 1H sweep + BOS/CHoCH ──
    elif sw(a1h).get("swept") and (s(a1h).get("bos") or s(a1h).get("choch")):
        direction  = sw(a1h)["direction"]
        base_conf  = 77
        trigger    = "1H_sweep_BOS"
        sweep_level = sw(a1h).get("sweep_level")
        entry_zone  = ez(a1h) if ez(a1h).get("entry") else ez(a15)
        if direction == "bullish":
            sl_level = round((sw(a1h).get("sweep_low") or sweep_level) * 0.999, 6)
        else:
            sl_level = round((sw(a1h).get("sweep_high") or sweep_level) * 1.001, 6)
        reasons.append(f"1H liquidity sweep at {sweep_level} + BOS/CHoCH confirmed")

    # ── TRIGGER 3: 15m sweep + 1H structure aligned ──
    elif (sw(a15).get("swept") and
          s(a1h).get("trend") == sw(a15).get("direction") and
          s(a1h).get("trend") != "ranging"):
        direction  = sw(a15)["direction"]
        base_conf  = 74
        trigger    = "15m_sweep_1H_aligned"
        sweep_level = sw(a15).get("sweep_level")
        entry_zone  = ez(a15) if ez(a15).get("entry") else ez(a1h)
        if direction == "bullish":
            sl_level = round((sw(a15).get("sweep_low") or sweep_level) * 0.999, 6)
        else:
            sl_level = round((sw(a15).get("sweep_high") or sweep_level) * 1.001, 6)
        reasons.append(f"15m sweep at {sweep_level} aligned with 1H {direction} structure")

    # ── TRIGGER 4: BOS + fresh OB + FVG on 1H (no sweep needed) ──
    elif (s(a1h).get("bos") and
          any(o.get("fresh") for o in a1h.get("obs", [])) and
          len(a1h.get("fvgs", [])) > 0):
        direction  = s(a1h).get("bos_dir", s(a1h).get("trend", "bullish"))
        base_conf  = 72
        trigger    = "1H_BOS_OB_FVG"
        entry_zone  = ez(a1h)
        # SL below OB low (long) or above OB high (short)
        fresh_obs  = [o for o in a1h.get("obs", []) if o.get("fresh")]
        if fresh_obs:
            if direction == "bullish":
                sl_level = round(fresh_obs[0]["low"] * 0.999, 6)
            else:
                sl_level = round(fresh_obs[0]["high"] * 1.001, 6)
        reasons.append(f"1H BOS ({direction}) + fresh OB + unfilled FVG — no sweep needed")

    # ── TRIGGER 5: 4H BOS + 1H CHoCH (structure cascade) ──
    elif s(a4h).get("bos") and s(a1h).get("choch"):
        direction  = s(a4h).get("bos_dir", "bullish")
        base_conf  = 70
        trigger    = "4H_BOS_1H_CHoCH"
        entry_zone  = ez(a1h)
        fresh_obs   = [o for o in a1h.get("obs", []) if o.get("fresh")]
        if fresh_obs:
            if direction == "bullish":
                sl_level = round(fresh_obs[0]["low"] * 0.999, 6)
            else:
                sl_level = round(fresh_obs[0]["high"] * 1.001, 6)
        elif entry_zone:
            price = entry_zone.get("entry", 0)
            sl_level = round(price * 0.985, 6) if direction == "bullish" else round(price * 1.015, 6)
        reasons.append(f"4H BOS + 1H CHoCH cascade → {direction}")

    if direction is None:
        result["reasons"] = ["No SMC trigger found: no sweep, no BOS+OB+FVG combination"]
        return result

    # ── Normalise direction ──
    direction = "long"  if direction in ("bullish", "long")  else "short"

    # ── Indicator bonus (max +25 pts) ──
    bonus = 0
    # Check alignment across timeframes
    for a_tf in [a1h, a15, a4h]:
        bonus += ind(a_tf).get("conf_pts", 0)
    bonus = min(bonus // 2, 25)  # average and cap

    # HTF alignment bonus
    trend_1d = s(a1d).get("trend", "ranging")
    trend_4h = s(a4h).get("trend", "ranging")
    if direction == "long":
        if trend_1d == "bullish": bonus += 5
        if trend_4h == "bullish": bonus += 5
    else:
        if trend_1d == "bearish": bonus += 5
        if trend_4h == "bearish": bonus += 5
    bonus = min(bonus, 25)

    # 5m sniper confirmation bonus
    sw5 = sw(a5m)
    ind5 = ind(a5m)
    sniper_5m = 0
    if sw5.get("swept") and sw5.get("direction") == ("bullish" if direction == "long" else "bearish"):
        sniper_5m += 5
    if ind5.get("candle_pattern") not in ("none", "doji"):
        sniper_5m += 3
    if ind5.get("vol_spike"):
        sniper_5m += 2
    bonus = min(bonus + sniper_5m, 25)

    final_conf = min(base_conf + bonus, 98)

    # Entry zone fallback
    if not entry_zone or not entry_zone.get("entry"):
        price = float(a1h.get("price", 0))
        entry_zone = {"source": "MARKET", "entry": price,
                      "high": price * 1.001, "low": price * 0.999}

    # SL fallback
    if not sl_level:
        entry = entry_zone["entry"]
        sl_level = round(entry * 0.985, 6) if direction == "long" else round(entry * 1.015, 6)

    # Build structure summary for reasons
    if s(a1h).get("bos"):  reasons.append(f"1H BOS {s(a1h).get('bos_dir','')}")
    if s(a1h).get("choch"): reasons.append("1H CHoCH")
    if s(a4h).get("bos"):  reasons.append(f"4H BOS {s(a4h).get('bos_dir','')}")
    if a1h.get("fvgs"):    reasons.append(f"{len(a1h['fvgs'])} unfilled FVG(s) on 1H")
    fresh_obs_all = [o for o in a1h.get("obs",[]) if o.get("fresh")]
    if fresh_obs_all:       reasons.append(f"{len(fresh_obs_all)} fresh OB(s) on 1H")
    if ind(a1h).get("rsi_divergence") != "none":
        reasons.append(f"RSI divergence: {ind(a1h)['rsi_divergence']}")
    if ind(a1h).get("vol_spike"):
        reasons.append("Volume spike on 1H")

    result.update({
        "signal":           True,
        "direction":        direction,
        "base_confidence":  base_conf,
        "final_confidence": final_conf,
        "trigger":          trigger,
        "entry_zone":       entry_zone,
        "sweep_level":      sweep_level,
        "stop_loss_level":  sl_level,
        "reasons":          reasons,
        "indic_bonus":      bonus,
    })
    return result


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

async def run_full_analysis(exchange: ExchangeManager, pair: str,
                             account_size: float = 100.0,
                             risk_pct: float = 0.015,
                             log_fn=None) -> dict:
    async def log(msg):
        if log_fn: await log_fn(msg)
        logger.info(msg)

    await log(f"[ENGINE] ═══ Starting full 5-TF SMC/ICT analysis: {pair} ═══")
    await log(f"[ENGINE] Fetching OHLCV: 1D · 4H · 1H · 15m · 5m")

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

    await log("[ENGINE] Data fetched. Running SMC structure analysis...")

    # Analyse all timeframes
    a1d  = analyze_tf(df1d,  "1d")
    a4h  = analyze_tf(df4h,  "4h")
    a1h  = analyze_tf(df1h,  "1h")
    a15  = analyze_tf(df15,  "15m")
    a5m  = analyze_tf(df5m,  "5m")

    analyses = {"1d": a1d, "4h": a4h, "1h": a1h, "15m": a15, "5m": a5m}

    # Log structure per timeframe
    for label, a in [("1D", a1d), ("4H", a4h), ("1H", a1h), ("15m", a15), ("5m", a5m)]:
        sw  = a["sweep"]
        st  = a["structure"]
        ind = a["indicators"]
        await log(
            f"[SMC] {label} | Trend: {st['trend'].upper()} | "
            f"BOS: {'✓' if st['bos'] else '✗'} | "
            f"CHoCH: {'✓' if st['choch'] else '✗'} | "
            f"Sweep: {'✓ ' + sw['direction'].upper() if sw['swept'] else '✗'} | "
            f"FVGs: {len(a['fvgs'])} | "
            f"OBs: {len([o for o in a['obs'] if o['fresh']])} fresh | "
            f"RSI: {ind['rsi']} | "
            f"Pattern: {ind['candle_pattern']}"
        )

    price = float(df1h["close"].iloc[-1])

    # ── SMC Decision ──
    await log("[ENGINE] Running ICT signal decision engine...")
    decision = make_smc_decision(analyses)

    if not decision["signal"]:
        await log(f"[ENGINE] ⚠️ No SMC signal: {decision['reasons']}")
        return {
            "pair": pair, "price": price,
            "analyses": analyses,
            "no_signal": True,
            "no_signal_reason": " | ".join(decision["reasons"]),
            "analyses_summary": _build_summary(analyses),
        }

    direction  = decision["direction"]
    entry_zone = decision["entry_zone"]
    sl_level   = decision["stop_loss_level"]
    trigger    = decision["trigger"]
    conf       = decision["final_confidence"]

    await log(f"[ENGINE] ✅ SMC SIGNAL: {direction.upper()} | Trigger: {trigger} | Confidence: {conf}%")
    await log(f"[ENGINE] Entry zone: {entry_zone['source']} @ {entry_zone['entry']} [{entry_zone['low']}–{entry_zone['high']}]")
    await log(f"[ENGINE] Stop Loss: {sl_level} | Sweep level: {decision.get('sweep_level')}")
    for r in decision["reasons"]:
        await log(f"[SMC] ✓ {r}")

    # ── Risk calculation ──
    atr_1h = a1h["indicators"]["atr"]
    risk   = calculate_risk(
        entry=entry_zone["entry"],
        stop_loss=sl_level,
        account=account_size,
        risk_pct=risk_pct,
        atr=atr_1h,
    )
    await log(
        f"[ORDER] Margin: ${risk['margin_usdt']} | "
        f"Leverage: {risk['leverage']}x | "
        f"RR: 1:{risk['rr']} | "
        f"Risk: ${risk['risk_amount_usdt']}"
    )

    # ── Funding rate ──
    funding_rate = await exchange.fetch_funding_rate(pair)
    await log(f"[ENGINE] Funding rate: {funding_rate*100:.4f}%")

    # ── Build Gemini payload ──
    def trim_obs(obs_list):
        return [{"type": o["type"], "high": o["high"], "low": o["low"],
                 "mid": o["mid"], "fresh": o["fresh"], "dist_pct": o["dist_pct"]}
                for o in obs_list[:3]]

    def trim_fvgs(fvg_list):
        return [{"type": f["type"], "top": f["top"], "bottom": f["bottom"],
                 "mid": f["mid"], "size_pct": f["size_pct"]}
                for f in fvg_list[:3]]

    gemini_payload = {
        "pair":                  pair,
        "current_price":         price,
        "account_size":          account_size,
        "risk_pct":              risk_pct,
        "smc_trigger":           trigger,
        "smc_direction":         direction,
        "smc_confidence":        conf,
        "smc_reasons":           decision["reasons"],
        "funding_rate":          funding_rate,
        "pre_calculated_risk":   risk,

        "entry_zone": {
            "source":  entry_zone["source"],
            "high":    entry_zone["high"],
            "low":     entry_zone["low"],
            "mid":     entry_zone["entry"],
        },
        "stop_loss_level": sl_level,
        "sweep_level":     decision.get("sweep_level"),

        "timeframes": {
            "1d": {
                "trend":    a1d["structure"]["trend"],
                "bos":      a1d["structure"]["bos"],
                "sweep":    a1d["sweep"]["swept"],
                "ema_stack": "bull" if a1d["indicators"]["bull_ema_stack"] else
                             "bear" if a1d["indicators"]["bear_ema_stack"] else "mixed",
                "rsi":      a1d["indicators"]["rsi"],
                "obs":      trim_obs(a1d["obs"]),
                "fvgs":     trim_fvgs(a1d["fvgs"]),
            },
            "4h": {
                "trend":     a4h["structure"]["trend"],
                "bos":       a4h["structure"]["bos"],
                "bos_level": a4h["structure"]["bos_level"],
                "choch":     a4h["structure"]["choch"],
                "sweep":     a4h["sweep"]["swept"],
                "sweep_dir": a4h["sweep"].get("direction"),
                "obs":       trim_obs(a4h["obs"]),
                "fvgs":      trim_fvgs(a4h["fvgs"]),
                "rsi":       a4h["indicators"]["rsi"],
                "vol_spike": a4h["indicators"]["vol_spike"],
            },
            "1h": {
                "trend":       a1h["structure"]["trend"],
                "bos":         a1h["structure"]["bos"],
                "choch":       a1h["structure"]["choch"],
                "sweep":       a1h["sweep"]["swept"],
                "sweep_level": a1h["sweep"].get("sweep_level"),
                "obs":         trim_obs(a1h["obs"]),
                "fvgs":        trim_fvgs(a1h["fvgs"]),
                "in_zone":     a1h["in_zone"],
                "rsi":         a1h["indicators"]["rsi"],
                "rsi_div":     a1h["indicators"]["rsi_divergence"],
                "vol_spike":   a1h["indicators"]["vol_spike"],
                "ema20":       a1h["indicators"]["ema20"],
                "candle":      a1h["indicators"]["candle_pattern"],
                "atr":         atr_1h,
            },
            "15m": {
                "trend":       a15["structure"]["trend"],
                "sweep":       a15["sweep"]["swept"],
                "sweep_dir":   a15["sweep"].get("direction"),
                "obs":         trim_obs(a15["obs"]),
                "fvgs":        trim_fvgs(a15["fvgs"]),
                "rsi":         a15["indicators"]["rsi"],
                "candle":      a15["indicators"]["candle_pattern"],
                "vol_spike":   a15["indicators"]["vol_spike"],
            },
            "5m": {
                "sweep":     a5m["sweep"]["swept"],
                "rsi":       a5m["indicators"]["rsi"],
                "candle":    a5m["indicators"]["candle_pattern"],
                "vol_spike": a5m["indicators"]["vol_spike"],
                "ema20_dir": "up" if a5m["indicators"]["bull_ema_stack"] else "down",
            },
        },

        "liquidity": {
            "equal_highs": a1h["sweep"].get("equal_highs", []),
            "equal_lows":  a1h["sweep"].get("equal_lows",  []),
        },
    }

    await log("[ENGINE] ✅ Payload built. Ready for Gemini.")

    return {
        "pair":         pair,
        "price":        price,
        "direction":    direction,
        "decision":     decision,
        "analyses":     analyses,
        "risk":         risk,
        "funding_rate": funding_rate,
        "gemini_payload": gemini_payload,
        "no_signal":    False,
        "atr_1h":       atr_1h,
    }


def _build_summary(analyses: dict) -> dict:
    out = {}
    for tf, a in analyses.items():
        out[tf] = {
            "trend":  a["structure"]["trend"],
            "bos":    a["structure"]["bos"],
            "choch":  a["structure"]["choch"],
            "sweep":  a["sweep"]["swept"],
            "fvgs":   len(a["fvgs"]),
            "obs":    len([o for o in a["obs"] if o["fresh"]]),
            "rsi":    a["indicators"]["rsi"],
        }
    return out
