"""
engine.py — APEX V10 Engine (Smart Screener Edition)
Scoring: 85pts SMC/ICT/Supply-Demand + 15pts Indicators = 100pts
Diamond Zone = OB + FVG + Supply/Demand overlap → sniper entry
Smart Screener: scores ALL pairs on 1H+4H, returns top > 60
"""

import asyncio
import logging
import aiohttp
from typing import Optional
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

logger = logging.getLogger("engine")

# ─────────────────────────────────────────────
# Score Grades
# ─────────────────────────────────────────────
def score_grade(score: int) -> str:
    if score >= 95: return "ELITE"
    if score >= 85: return "STRONG"
    if score >= 75: return "STANDARD"
    if score >= 60: return "WEAK"
    return "NO_SIGNAL"

def score_to_risk(score: int, account: float = 100.0) -> dict:
    """Scale margin, leverage, risk% based on score."""
    if score >= 95:
        risk_pct, lev_cap, label = 0.020, 15, "ELITE"
    elif score >= 85:
        risk_pct, lev_cap, label = 0.015, 12, "STRONG"
    elif score >= 75:
        risk_pct, lev_cap, label = 0.010, 8,  "STANDARD"
    else:
        risk_pct, lev_cap, label = 0.005, 5,  "WEAK"
    return {
        "risk_pct":  risk_pct,
        "lev_cap":   lev_cap,
        "risk_usdt": round(account * risk_pct, 2),
        "label":     label,
    }


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

    async def get_all_usdt_futures(self) -> list[dict]:
        """
        Fetch ALL Binance Futures USDT pairs via public API.
        Returns list sorted by quoteVolume descending.
        """
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    data = await resp.json()
        except Exception as e:
            logger.error(f"[SCREENER] Public API failed: {e}")
            return []

        pairs = []
        for t in data:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"): continue
            if any(x in sym for x in ["_", "BUSD", "USDC"]): continue
            try:
                vol    = float(t.get("quoteVolume", 0))
                price  = float(t.get("lastPrice", 0))
                change = round(float(t.get("priceChangePercent", 0)), 2)
            except (ValueError, TypeError):
                continue
            if vol > 0 and price > 0:
                base = sym[:-4]
                pairs.append({
                    "pair":   f"{base}/USDT",
                    "symbol": sym,
                    "volume": vol,
                    "price":  price,
                    "change": change,
                })

        pairs.sort(key=lambda x: x["volume"], reverse=True)
        return pairs

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
                "error": None,
            }
        except Exception as e:
            msg = str(e)
            if "-2015" in msg or "Invalid API" in msg:
                logger.error(
                    "[WALLET] ❌ IP not whitelisted. "
                    "Binance → API Management → Edit → add server IP."
                )
                return {"total": 0.0, "free": 0.0, "used": 0.0,
                        "error": "IP not whitelisted — add server IP to Binance API"}
            return {"total": 0.0, "free": 0.0, "used": 0.0, "error": str(e)}

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────
# Indicator Helpers
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

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

def rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lb: int = 12) -> str:
    if len(df) < lb + 2: return "none"
    p  = df["close"].values[-lb:]
    r  = rsi.values[-lb:]
    pi = int(np.argmin(p))
    pk = int(np.argmax(p))
    if pi < len(p) - 3 and p[-1] < p[pi] and r[-1] > r[pi]:
        return "bullish_div"
    if pk < len(p) - 3 and p[-1] > p[pk] and r[-1] < r[pk]:
        return "bearish_div"
    return "none"


# ─────────────────────────────────────────────
# SMC Detectors
# ─────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lb: int = 5) -> tuple[list, list]:
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        wh = df["high"].iloc[i - lb: i + lb + 1]
        wl = df["low"].iloc[i  - lb: i + lb + 1]
        if df["high"].iloc[i] == wh.max():
            highs.append({"i": i, "price": float(df["high"].iloc[i])})
        if df["low"].iloc[i] == wl.min():
            lows.append({"i": i, "price": float(df["low"].iloc[i])})
    return highs, lows


def detect_structure(df: pd.DataFrame,
                     highs: list, lows: list) -> dict:
    res = {
        "trend": "ranging", "bos": False, "bos_level": None, "bos_dir": None,
        "choch": False, "choch_level": None,
        "hh_hl": False, "ll_lh": False,
        "swing_highs": [], "swing_lows": [],
    }
    if len(highs) < 2 or len(lows) < 2: return res
    rh = sorted(highs, key=lambda x: x["i"])[-6:]
    rl = sorted(lows,  key=lambda x: x["i"])[-6:]
    res["swing_highs"] = [h["price"] for h in rh]
    res["swing_lows"]  = [l["price"] for l in rl]
    price = float(df["close"].iloc[-1])
    res["hh_hl"] = rh[-1]["price"] > rh[-2]["price"] and rl[-1]["price"] > rl[-2]["price"]
    res["ll_lh"] = rh[-1]["price"] < rh[-2]["price"] and rl[-1]["price"] < rl[-2]["price"]
    if price > rh[-1]["price"]:
        res.update({"bos": True, "bos_level": rh[-1]["price"],
                    "bos_dir": "bullish", "trend": "bullish"})
    elif price < rl[-1]["price"]:
        res.update({"bos": True, "bos_level": rl[-1]["price"],
                    "bos_dir": "bearish", "trend": "bearish"})
    elif res["hh_hl"]: res["trend"] = "bullish"
    elif res["ll_lh"]: res["trend"] = "bearish"
    if res["trend"] == "bullish" and len(rh) >= 2:
        if rh[-1]["price"] < rh[-2]["price"]:
            res.update({"choch": True, "choch_level": rh[-1]["price"]})
    elif res["trend"] == "bearish" and len(rl) >= 2:
        if rl[-1]["price"] > rl[-2]["price"]:
            res.update({"choch": True, "choch_level": rl[-1]["price"]})
    return res


def detect_liquidity_sweep(df: pd.DataFrame,
                            highs: list, lows: list,
                            tol: float = 0.0025) -> dict:
    """Detect confirmed liquidity sweep: wick beyond equal high/low + close back."""
    res = {
        "swept": False, "direction": None,
        "sweep_level": None, "sweep_low": None, "sweep_high": None,
        "candles_ago": None, "equal_highs": [], "equal_lows": [],
    }
    if len(df) < 10: return res

    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    o_arr = df["open"].values

    sh_prices = [h["price"] for h in highs]
    sl_prices = [l["price"] for l in lows]

    eq_h, eq_l = [], []
    for i in range(len(sh_prices)):
        for j in range(i + 1, len(sh_prices)):
            if abs(sh_prices[i] - sh_prices[j]) / (sh_prices[i] + 1e-10) < tol:
                level = round((sh_prices[i] + sh_prices[j]) / 2, 8)
                if level not in eq_h: eq_h.append(level)
    for i in range(len(sl_prices)):
        for j in range(i + 1, len(sl_prices)):
            if abs(sl_prices[i] - sl_prices[j]) / (sl_prices[i] + 1e-10) < tol:
                level = round((sl_prices[i] + sl_prices[j]) / 2, 8)
                if level not in eq_l: eq_l.append(level)

    res["equal_highs"] = eq_h[-5:]
    res["equal_lows"]  = eq_l[-5:]

    scan = min(40, len(df) - 1)
    for i in range(len(df) - 1, len(df) - scan, -1):
        li, hi, ci, oi = l_arr[i], h_arr[i], c_arr[i], o_arr[i]
        rng = hi - li

        # Bullish sweep: equal low swept, closed above
        for eq in eq_l:
            if li < eq * (1 - tol * 0.3) and ci > eq:
                res.update({"swept": True, "direction": "bullish",
                            "sweep_level": eq, "sweep_low": float(li),
                            "candles_ago": len(df) - 1 - i})
                return res

        # Bearish sweep: equal high swept, closed below
        for eq in eq_h:
            if hi > eq * (1 + tol * 0.3) and ci < eq:
                res.update({"swept": True, "direction": "bearish",
                            "sweep_level": eq, "sweep_high": float(hi),
                            "candles_ago": len(df) - 1 - i})
                return res

        # Pin-bar style sweep (single candle)
        if i >= 5 and rng > 0:
            rec_low  = min(l_arr[max(0, i-10):i])
            rec_high = max(h_arr[max(0, i-10):i])
            wick_dn  = min(oi, ci) - li
            wick_up  = hi - max(oi, ci)
            if li < rec_low and ci > rec_low and wick_dn > rng * 0.55:
                res.update({"swept": True, "direction": "bullish",
                            "sweep_level": float(rec_low), "sweep_low": float(li),
                            "candles_ago": len(df) - 1 - i})
                return res
            if hi > rec_high and ci < rec_high and wick_up > rng * 0.55:
                res.update({"swept": True, "direction": "bearish",
                            "sweep_level": float(rec_high), "sweep_high": float(hi),
                            "candles_ago": len(df) - 1 - i})
                return res
    return res


def detect_order_blocks(df: pd.DataFrame, direction: str) -> list[dict]:
    """ICT OB: last opposite candle before strong impulse. Fresh OBs preferred."""
    obs     = []
    avg_rng = float((df["high"] - df["low"]).mean())
    c, o    = df["close"].values, df["open"].values
    h, l    = df["high"].values,  df["low"].values
    price   = float(c[-1])

    for i in range(2, len(df) - 3):
        up   = c[i+1] - o[i+1]
        down = o[i+1] - c[i+1]
        if direction in ("long","bullish") and o[i] > c[i] and up > avg_rng * 1.2:
            tested = any(l[j] <= c[i] for j in range(i+2, min(i+20, len(df))))
            dist   = abs(price - (o[i]+l[i])/2) / price
            obs.append({"type":"bullish","high":float(o[i]),"low":float(l[i]),
                        "mid":round((o[i]+l[i])/2,8),"fresh":not tested,
                        "strength":round(min(up/(avg_rng*1.2),1.5),2),
                        "dist_pct":round(dist*100,3),"index":i})
        if direction in ("short","bearish") and c[i] > o[i] and down > avg_rng * 1.2:
            tested = any(h[j] >= o[i] for j in range(i+2, min(i+20, len(df))))
            dist   = abs(price - (c[i]+h[i])/2) / price
            obs.append({"type":"bearish","high":float(h[i]),"low":float(c[i]),
                        "mid":round((c[i]+h[i])/2,8),"fresh":not tested,
                        "strength":round(min(down/(avg_rng*1.2),1.5),2),
                        "dist_pct":round(dist*100,3),"index":i})

    obs.sort(key=lambda x: (not x["fresh"], x["dist_pct"]))
    return obs[:5]


def detect_fvgs(df: pd.DataFrame, direction: str) -> list[dict]:
    """3-candle FVG. Unfilled gaps closest to price."""
    fvgs  = []
    price = float(df["close"].iloc[-1])
    for i in range(1, len(df) - 1):
        ph = float(df["high"].iloc[i-1]); pl = float(df["low"].iloc[i-1])
        nh = float(df["high"].iloc[i+1]); nl = float(df["low"].iloc[i+1])
        ts = str(df["timestamp"].iloc[i])
        if direction in ("long","bullish") and ph < nl:
            top=nl; bot=ph; mid=round((top+bot)/2,8)
            sz=round((top-bot)/price*100,3); filled=price<=bot
            if sz > 0:
                fvgs.append({"type":"bullish","top":top,"bottom":bot,"mid":mid,
                             "size_pct":sz,"filled":filled,"ts":ts,
                             "dist_pct":round(abs(price-mid)/price*100,3)})
        if direction in ("short","bearish") and pl > nh:
            top=pl; bot=nh; mid=round((top+bot)/2,8)
            sz=round((top-bot)/price*100,3); filled=price>=top
            if sz > 0:
                fvgs.append({"type":"bearish","top":top,"bottom":bot,"mid":mid,
                             "size_pct":sz,"filled":filled,"ts":ts,
                             "dist_pct":round(abs(price-mid)/price*100,3)})
    uf = [f for f in fvgs if not f["filled"]]
    uf.sort(key=lambda x: x["dist_pct"])
    return uf[:4]


# ─────────────────────────────────────────────
# Supply & Demand Zone Detection
# ─────────────────────────────────────────────

def detect_supply_demand_zones(df: pd.DataFrame, direction: str) -> list[dict]:
    """
    Supply/Demand zones: consolidation base before a strong impulse.
    Demand zone (long): base before strong bullish move
    Supply zone (short): base before strong bearish move

    Zone strength:
      - Fresh (first touch)    → highest
      - Respected (bounced)    → strong
      - Touched multiple times → weakening
      - Broken through         → invalid
    """
    zones  = []
    price  = float(df["close"].iloc[-1])
    c, o   = df["close"].values, df["open"].values
    h, l   = df["high"].values,  df["low"].values
    avg_rng = float((df["high"] - df["low"]).mean())

    for i in range(3, len(df) - 4):
        # Detect impulse candle
        body_i   = abs(c[i] - o[i])
        impulse  = body_i > avg_rng * 1.8

        if not impulse:
            continue

        is_bull = c[i] > o[i]
        is_bear = c[i] < o[i]

        # Base = up to 5 consolidation candles before impulse
        base_start = max(0, i - 5)
        base_slice_h = h[base_start:i]
        base_slice_l = l[base_start:i]

        if len(base_slice_h) < 1:
            continue

        zone_high = float(max(base_slice_h))
        zone_low  = float(min(base_slice_l))
        zone_mid  = round((zone_high + zone_low) / 2, 8)
        zone_size = round((zone_high - zone_low) / price * 100, 3)

        if zone_size < 0.05:  # too thin, skip
            continue

        # Count how many times price returned to zone
        touch_count = 0
        broken      = False
        for j in range(i + 1, len(df)):
            if is_bull and l[j] <= zone_high and h[j] >= zone_low:
                touch_count += 1
                if c[j] < zone_low:
                    broken = True
                    break
            if is_bear and h[j] >= zone_low and l[j] <= zone_high:
                touch_count += 1
                if c[j] > zone_high:
                    broken = True
                    break

        if broken:
            continue

        # Freshness score
        if touch_count == 0:
            freshness = "fresh"
            fresh_score = 10
        elif touch_count == 1:
            freshness = "respected"
            fresh_score = 6
        elif touch_count == 2:
            freshness = "tested"
            fresh_score = 3
        else:
            freshness = "weak"
            fresh_score = 1

        # Direction match
        if direction in ("long","bullish") and is_bull:
            zone_type = "demand"
            dist      = abs(price - zone_mid) / price
            in_zone   = zone_low <= price <= zone_high
            zones.append({
                "type":        "demand",
                "high":        zone_high,
                "low":         zone_low,
                "mid":         zone_mid,
                "size_pct":    zone_size,
                "freshness":   freshness,
                "fresh_score": fresh_score,
                "touch_count": touch_count,
                "in_zone":     in_zone,
                "dist_pct":    round(dist * 100, 3),
            })

        if direction in ("short","bearish") and is_bear:
            zone_type = "supply"
            dist      = abs(price - zone_mid) / price
            in_zone   = zone_low <= price <= zone_high
            zones.append({
                "type":        "supply",
                "high":        zone_high,
                "low":         zone_low,
                "mid":         zone_mid,
                "size_pct":    zone_size,
                "freshness":   freshness,
                "fresh_score": fresh_score,
                "touch_count": touch_count,
                "in_zone":     in_zone,
                "dist_pct":    round(dist * 100, 3),
            })

    # Sort: in-zone first, then fresh, then closest
    zones.sort(key=lambda x: (not x["in_zone"], -x["fresh_score"], x["dist_pct"]))
    return zones[:4]


def detect_diamond_zone(obs: list[dict], fvgs: list[dict],
                         sd_zones: list[dict], price: float) -> Optional[dict]:
    """
    Diamond Zone ⬡ = OB + FVG + Supply/Demand overlap.
    All three zones must overlap → sniper entry zone.
    Returns overlap zone or None.
    """
    if not obs or not fvgs or not sd_zones:
        return None

    best_ob = obs[0]
    best_fvg = fvgs[0]
    best_sd  = sd_zones[0]

    # Find overlap of all three
    overlap_low  = max(best_ob["low"],  best_fvg["bottom"], best_sd["low"])
    overlap_high = min(best_ob["high"], best_fvg["top"],    best_sd["high"])

    if overlap_low >= overlap_high:
        # Try OB + FVG only (partial diamond)
        overlap_low2  = max(best_ob["low"],  best_fvg["bottom"])
        overlap_high2 = min(best_ob["high"], best_fvg["top"])
        if overlap_low2 >= overlap_high2:
            return None
        return {
            "type":         "partial_diamond",
            "high":         overlap_high2,
            "low":          overlap_low2,
            "mid":          round((overlap_high2 + overlap_low2) / 2, 8),
            "components":   ["OB", "FVG"],
            "bonus_pts":    3,
            "in_zone":      overlap_low2 <= price <= overlap_high2,
        }

    return {
        "type":         "diamond",
        "high":         overlap_high,
        "low":          overlap_low,
        "mid":          round((overlap_high + overlap_low) / 2, 8),
        "components":   ["OB", "FVG", "S/D"],
        "bonus_pts":    5,
        "in_zone":      overlap_low <= price <= overlap_high,
    }


def find_best_entry_zone(obs: list[dict], fvgs: list[dict],
                          sd_zones: list[dict],
                          diamond: Optional[dict],
                          price: float, direction: str) -> dict:
    """
    ICT Limit Order Entry Logic:
    ─────────────────────────────
    LONG  entry = BOTTOM of zone (best discount price, wait for retrace DOWN)
    SHORT entry = TOP    of zone (best premium price, wait for retrace UP)

    Priority: Diamond > FVG (tightest zone) > OB (wider buffer) > S/D > Market

    entry_condition:
      "LIMIT_WAIT"   → price not yet at zone, place limit and wait
      "LIMIT_NOW"    → price already inside zone, enter immediately
      "MARKET"       → no structure zone found, use current price (last resort)
    """
    is_long = direction in ("long", "bullish")

    def limit_price(zone: dict, zone_type: str) -> float:
        """
        ICT rule:
          OB  long  → enter at OB LOW  (bottom of the block = best value)
          OB  short → enter at OB HIGH (top of the block = best premium)
          FVG long  → enter at FVG BOTTOM (bottom of gap = discount)
          FVG short → enter at FVG TOP    (top of gap = premium)
          S/D long  → enter at ZONE LOW
          S/D short → enter at ZONE HIGH
          Diamond   → enter at overlap LOW (long) or HIGH (short)
        """
        if zone_type in ("OB",):
            return zone["low"] if is_long else zone["high"]
        if zone_type in ("FVG",):
            return zone["bottom"] if is_long else zone["top"]
        if zone_type in ("S/D", "demand", "supply"):
            return zone["low"] if is_long else zone["high"]
        if zone_type in ("DIAMOND", "PARTIAL_DIAMOND"):
            return zone["low"] if is_long else zone["high"]
        return zone.get("mid", price)

    def entry_condition(entry_price: float, zone_low: float,
                        zone_high: float) -> str:
        if zone_low <= price <= zone_high:
            return "LIMIT_NOW"   # price already inside zone
        if is_long and price > zone_high:
            return "LIMIT_WAIT"  # price above zone, wait for retrace down
        if not is_long and price < zone_low:
            return "LIMIT_WAIT"  # price below zone, wait for retrace up
        return "LIMIT_WAIT"

    candidates = []

    # ── Diamond (OB + FVG + S/D overlap) — highest priority ──
    if diamond:
        ep   = limit_price(diamond, diamond["type"].upper())
        cond = entry_condition(ep, diamond["low"], diamond["high"])
        dist = abs(price - ep) / price * 100
        # Diamond gets huge score bonus
        d_score = 150 - dist * 3
        candidates.append({
            **diamond,
            "source":          diamond["type"].upper(),
            "entry":           round(ep, 8),
            "zone_low":        diamond["low"],
            "zone_high":       diamond["high"],
            "entry_condition": cond,
            "dist_from_entry": round(dist, 3),
            "score":           d_score,
            "bonus_pts":       diamond.get("bonus_pts", 5),
        })

    # ── FVG — highest precision zone ──
    for fvg in fvgs[:3]:
        ep   = limit_price(fvg, "FVG")
        cond = entry_condition(ep, fvg["bottom"], fvg["top"])
        dist = abs(price - ep) / price * 100
        # Prefer: unfilled, close to price, larger gap (more buffer)
        score = 100 - dist * 5 + fvg.get("size_pct", 0) * 8
        if cond == "LIMIT_NOW": score += 20  # bonus if already in zone
        candidates.append({
            **fvg,
            "source":          "FVG",
            "entry":           round(ep, 8),
            "zone_low":        fvg["bottom"],
            "zone_high":       fvg["top"],
            "entry_condition": cond,
            "dist_from_entry": round(dist, 3),
            "score":           score,
            "bonus_pts":       0,
        })

    # ── OB — wider zone, strong institutional area ──
    for ob in obs[:3]:
        if not ob.get("fresh"):
            continue
        ep   = limit_price(ob, "OB")
        cond = entry_condition(ep, ob["low"], ob["high"])
        dist = abs(price - ep) / price * 100
        score = 100 - dist * 4 + ob.get("strength", 1) * 12
        if cond == "LIMIT_NOW": score += 20
        candidates.append({
            **ob,
            "source":          "OB",
            "entry":           round(ep, 8),
            "zone_low":        ob["low"],
            "zone_high":       ob["high"],
            "entry_condition": cond,
            "dist_from_entry": round(dist, 3),
            "score":           score,
            "bonus_pts":       0,
        })

    # ── Supply / Demand Zone ──
    for sd in sd_zones[:2]:
        ep   = limit_price(sd, "S/D")
        cond = entry_condition(ep, sd["low"], sd["high"])
        dist = abs(price - ep) / price * 100
        score = 90 - dist * 5 + sd.get("fresh_score", 0) * 6
        if cond == "LIMIT_NOW": score += 15
        candidates.append({
            **sd,
            "source":          "S/D",
            "entry":           round(ep, 8),
            "zone_low":        sd["low"],
            "zone_high":       sd["high"],
            "entry_condition": cond,
            "dist_from_entry": round(dist, 3),
            "score":           score,
            "bonus_pts":       0,
        })

    # ── No zone found → market entry (last resort) ──
    if not candidates:
        return {
            "source":          "MARKET",
            "entry":           round(price, 8),
            "zone_low":        round(price * 0.999, 8),
            "zone_high":       round(price * 1.001, 8),
            "entry_condition": "MARKET",
            "dist_from_entry": 0.0,
            "score":           30,
            "bonus_pts":       0,
            "high":            round(price * 1.001, 8),
            "low":             round(price * 0.999, 8),
        }

    # Sort: highest score first
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = candidates[0]

    # Ensure zone_low / zone_high keys exist (alias for HTML compatibility)
    best.setdefault("zone_low",  best.get("low",  price * 0.999))
    best.setdefault("zone_high", best.get("high", price * 1.001))
    best.setdefault("low",       best["zone_low"])
    best.setdefault("high",      best["zone_high"])

    return best


# ─────────────────────────────────────────────
# Secondary Indicators
# ─────────────────────────────────────────────

def secondary_indicators(df: pd.DataFrame, direction: str) -> dict:
    """Max 15 pts. Never block a signal."""
    close = df["close"]
    price = float(close.iloc[-1])
    rsi_s = compute_rsi(close, 14)
    rsi_v = float(rsi_s.iloc[-1])
    div   = rsi_divergence(df, rsi_s)
    e20   = float(ema(close, 20).iloc[-1])
    e50   = float(ema(close, 50).iloc[-1])
    e200  = float(ema(close, 200).iloc[-1])
    bull_ema = price > e20 > e50
    bear_ema = price < e20 < e50
    avg20    = float(df["volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
    last_v   = float(df["volume"].iloc[-1])
    vol_spike = last_v > avg20 * 1.8
    vol_ratio = round(last_v / avg20, 2) if avg20 > 0 else 1.0
    atr_v = float(compute_atr(df, 14).iloc[-1])
    candle = _candlestick_pattern(df)
    chart  = _chart_pattern(df)

    pts = 0
    if direction in ("long","bullish"):
        if div == "bullish_div":  pts += 6
        if rsi_v < 45:            pts += 3
        if bull_ema:              pts += 4
        if vol_spike:             pts += 5
        if candle in {"hammer","bullish_pinbar","bullish_engulfing",
                      "morning_star","bullish_marubozu"}: pts += 2
    else:
        if div == "bearish_div":  pts += 6
        if rsi_v > 55:            pts += 3
        if bear_ema:              pts += 4
        if vol_spike:             pts += 5
        if candle in {"shooting_star","bearish_pinbar","bearish_engulfing",
                      "evening_star","bearish_marubozu"}: pts += 2

    return {
        "rsi": round(rsi_v, 2), "rsi_divergence": div,
        "ema20": round(e20,8), "ema50": round(e50,8), "ema200": round(e200,8),
        "bull_ema_stack": bull_ema, "bear_ema_stack": bear_ema,
        "vol_spike": vol_spike, "vol_ratio": vol_ratio,
        "atr": round(atr_v, 8),
        "candle_pattern": candle, "chart_pattern": chart,
        "pts": min(pts, 15),
    }


def _candlestick_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3: return "none"
    c,p,pp = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body=abs(c["close"]-c["open"]); uw=c["high"]-max(c["close"],c["open"])
    lw=min(c["close"],c["open"])-c["low"]; rng=c["high"]-c["low"]
    if rng<1e-10: return "doji"
    if body/rng<0.08:          return "doji"
    if lw>body*2.2 and uw<body*0.4: return "hammer"
    if uw>body*2.2 and lw<body*0.4: return "shooting_star"
    if lw>rng*0.62:            return "bullish_pinbar"
    if uw>rng*0.62:            return "bearish_pinbar"
    if (c["close"]>c["open"] and p["close"]<p["open"]
            and c["open"]<=p["close"] and c["close"]>=p["open"]): return "bullish_engulfing"
    if (c["close"]<c["open"] and p["close"]>p["open"]
            and c["open"]>=p["close"] and c["close"]<=p["open"]): return "bearish_engulfing"
    pb=abs(p["close"]-p["open"]); ppb=abs(pp["close"]-pp["open"])
    if (pp["close"]<pp["open"] and pb<ppb*0.3 and c["close"]>c["open"]
            and c["close"]>(pp["open"]+pp["close"])/2): return "morning_star"
    if (pp["close"]>pp["open"] and pb<ppb*0.3 and c["close"]<c["open"]
            and c["close"]<(pp["open"]+pp["close"])/2): return "evening_star"
    if body/rng>0.88 and c["close"]>c["open"]: return "bullish_marubozu"
    if body/rng>0.88 and c["close"]<c["open"]: return "bearish_marubozu"
    return "none"


def _chart_pattern(df: pd.DataFrame) -> str:
    if len(df) < 30: return "none"
    h=df["high"].values[-30:]; l=df["low"].values[-30:]
    hi,li=[],[]
    for i in range(2,len(h)-2):
        if h[i]==max(h[i-2:i+3]): hi.append(i)
        if l[i]==min(l[i-2:i+3]): li.append(i)
    if len(hi)>=2 and abs(h[hi[-1]]-h[hi[-2]])/(h[hi[-1]]+1e-9)<0.006: return "double_top"
    if len(li)>=2 and abs(l[li[-1]]-l[li[-2]])/(l[li[-1]]+1e-9)<0.006: return "double_bottom"
    if len(hi)>=3 and len(li)>=3:
        ht=np.polyfit(hi[-3:],[h[i] for i in hi[-3:]],1)[0]
        lt=np.polyfit(li[-3:],[l[i] for i in li[-3:]],1)[0]
        if ht>0 and lt>0:   return "ascending_channel"
        if ht<0 and lt<0:   return "descending_channel"
        if ht<0 and lt>0:   return "symmetrical_triangle"
        if abs(ht)<0.0001 and lt>0: return "ascending_triangle"
        if ht<0 and abs(lt)<0.0001: return "descending_triangle"
    return "none"


# ─────────────────────────────────────────────
# 85 / 15 Scoring System
# ─────────────────────────────────────────────

def compute_full_score(struct: dict, sweep: dict,
                        obs: list, fvgs: list,
                        sd_zones: list, diamond: Optional[dict],
                        indics: dict, direction: str) -> dict:
    """
    85 pts SMC/ICT/Supply-Demand + 15 pts Indicators = 100 pts total.
    """
    breakdown = {}
    smc_total = 0

    # ── STRUCTURE (25 pts) ──
    s_pts = 0
    if struct.get("bos"):   s_pts += 10
    if struct.get("choch"): s_pts += 8
    if direction in ("long","bullish") and struct.get("hh_hl"):  s_pts += 7
    if direction in ("short","bearish") and struct.get("ll_lh"): s_pts += 7
    s_pts = min(s_pts, 25)
    breakdown["structure"] = s_pts
    smc_total += s_pts

    # ── LIQUIDITY SWEEP (20 pts) ──
    sw_pts = 0
    if sweep.get("swept"):
        ago = sweep.get("candles_ago", 99)
        if ago <= 5:   sw_pts = 20   # very recent sweep
        elif ago <= 15: sw_pts = 16
        elif ago <= 30: sw_pts = 10
        else:           sw_pts = 6
    breakdown["sweep"] = sw_pts
    smc_total += sw_pts

    # ── ORDER BLOCKS (15 pts) ──
    ob_pts = 0
    fresh_obs = [o for o in obs if o.get("fresh")]
    if fresh_obs:
        best = fresh_obs[0]
        ob_pts += 8
        if best["dist_pct"] < 1.0: ob_pts += 4
        if best["strength"] > 1.2: ob_pts += 3
    ob_pts = min(ob_pts, 15)
    breakdown["order_blocks"] = ob_pts
    smc_total += ob_pts

    # ── FVG / IMBALANCE (10 pts) ──
    fvg_pts = 0
    if fvgs:
        best = fvgs[0]
        fvg_pts += 6
        if best["dist_pct"] < 1.5: fvg_pts += 2
        if best["size_pct"] > 0.3: fvg_pts += 2
    fvg_pts = min(fvg_pts, 10)
    breakdown["fvg"] = fvg_pts
    smc_total += fvg_pts

    # ── SUPPLY / DEMAND ZONES (15 pts) ──
    sd_pts = 0
    if sd_zones:
        best_sd = sd_zones[0]
        if best_sd.get("in_zone"):
            sd_pts += 8
        else:
            if best_sd["dist_pct"] < 2.0: sd_pts += 5
            elif best_sd["dist_pct"] < 5.0: sd_pts += 3
        sd_pts += min(best_sd["fresh_score"], 4)
        if best_sd["freshness"] == "respected": sd_pts += 3
    sd_pts = min(sd_pts, 15)
    breakdown["supply_demand"] = sd_pts
    smc_total += sd_pts

    # ── DIAMOND ZONE BONUS (max +5) ──
    diamond_bonus = 0
    if diamond:
        diamond_bonus = diamond.get("bonus_pts", 0)
    breakdown["diamond_bonus"] = diamond_bonus
    smc_total = min(smc_total + diamond_bonus, 85)
    breakdown["smc_total"] = smc_total

    # ── INDICATORS (15 pts) ──
    ind_pts = indics.get("pts", 0)
    breakdown["indicators"] = ind_pts

    total = min(smc_total + ind_pts, 100)
    breakdown["total"] = total
    breakdown["grade"] = score_grade(total)

    return breakdown


# ─────────────────────────────────────────────
# Quick Screener Score (1H + 4H only, fast)
# ─────────────────────────────────────────────

def quick_score_pair(df1h: pd.DataFrame,
                     df4h: pd.DataFrame) -> dict:
    """
    Fast scoring on 1H + 4H only for screener.
    No supply/demand (too slow for batch), but includes OB, FVG, sweep, structure.
    Returns score, grade, setup_type.
    """
    score = 0
    setup_types = []

    for df, tf_weight in [(df4h, 1.2), (df1h, 1.0)]:
        if df is None or len(df) < 20:
            continue
        lb = 5
        highs, lows = detect_swing_points(df, lb=lb)
        struct      = detect_structure(df, highs, lows)
        sweep       = detect_liquidity_sweep(df, highs, lows)

        trend = struct.get("trend","ranging")
        direction = "bullish" if trend == "bullish" else "bearish"

        obs  = detect_order_blocks(df, direction)
        fvgs = detect_fvgs(df, direction)

        rsi_s = compute_rsi(df["close"], 14)
        rsi_v = float(rsi_s.iloc[-1])
        div   = rsi_divergence(df, rsi_s)
        avg20 = float(df["volume"].iloc[-21:-1].mean()) if len(df)>21 else 1.0
        last_v = float(df["volume"].iloc[-1])
        vol_spike = last_v > avg20 * 1.8

        tf_score = 0

        # Structure
        if struct.get("bos"):   tf_score += 10
        if struct.get("choch"): tf_score += 8
        if struct.get("hh_hl") or struct.get("ll_lh"): tf_score += 5

        # Sweep
        if sweep.get("swept"):
            ago = sweep.get("candles_ago", 99)
            if ago <= 5:    tf_score += 20; setup_types.append("SWEEP")
            elif ago <= 15: tf_score += 14; setup_types.append("SWEEP")
            elif ago <= 30: tf_score += 8

        # OB
        fresh_obs = [o for o in obs if o.get("fresh")]
        if fresh_obs:
            tf_score += 12
            setup_types.append("OB")

        # FVG
        if fvgs:
            tf_score += 8
            setup_types.append("FVG")

        # RSI
        if div in ("bullish_div","bearish_div"): tf_score += 6; setup_types.append("RSI_DIV")
        if vol_spike: tf_score += 5; setup_types.append("VOL")

        score += int(tf_score * tf_weight)

    score = min(score // 2, 100)  # average + cap
    grade = score_grade(score)
    unique_setups = list(dict.fromkeys(setup_types))[:3]

    # Determine direction from 4H dominant trend
    if df4h is not None and len(df4h) >= 20:
        highs4, lows4 = detect_swing_points(df4h, lb=5)
        struct4 = detect_structure(df4h, highs4, lows4)
        direction = struct4.get("trend","ranging")
    else:
        direction = "ranging"

    return {
        "score":      score,
        "grade":      grade,
        "direction":  direction,
        "setup_type": " + ".join(unique_setups) if unique_setups else "STRUCTURE",
    }


# ─────────────────────────────────────────────
# Smart Screener — Main Function
# ─────────────────────────────────────────────

async def smart_screen_pairs(exchange: ExchangeManager,
                              min_score: int = 60,
                              log_fn=None) -> list[dict]:
    """
    Fetch all Binance Futures USDT pairs,
    score each on 1H + 4H SMC/ICT,
    return all scoring > min_score, ranked by score.
    """
    async def log(msg):
        if log_fn: await log_fn(msg)
        logger.info(msg)

    await log("[SCREENER] Fetching all Binance Futures USDT pairs...")
    all_pairs = await exchange.get_all_usdt_futures()
    if not all_pairs:
        await log("[SCREENER] ❌ No pairs returned from Binance public API")
        return []

    await log(f"[SCREENER] Got {len(all_pairs)} pairs. Screening top 60 by volume...")

    # Take top 60 by volume for screening (balances speed vs coverage)
    candidates = all_pairs[:60]

    await log(f"[SCREENER] Fetching 1H + 4H candles for {len(candidates)} pairs in parallel...")

    # Fetch all candles in parallel batches of 10
    results = []
    batch_size = 10

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        await log(f"[SCREENER] Batch {batch_num}/{total_batches}: {', '.join(p['pair'] for p in batch[:3])}...")

        tasks = []
        for p in batch:
            async def fetch_pair(pair_info):
                try:
                    df1h, df4h = await asyncio.gather(
                        exchange.fetch_ohlcv(pair_info["pair"], "1h", limit=100),
                        exchange.fetch_ohlcv(pair_info["pair"], "4h", limit=100),
                    )
                    sc = quick_score_pair(df1h, df4h)
                    return {
                        **pair_info,
                        "score":      sc["score"],
                        "grade":      sc["grade"],
                        "direction":  sc["direction"],
                        "setup_type": sc["setup_type"],
                    }
                except Exception as e:
                    logger.debug(f"[SCREENER] Skip {pair_info['pair']}: {e}")
                    return None

            tasks.append(fetch_pair(p))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if r and isinstance(r, dict) and r.get("score", 0) >= min_score:
                results.append(r)

        # Small delay between batches to respect rate limits
        await asyncio.sleep(0.3)

    results.sort(key=lambda x: x["score"], reverse=True)
    await log(f"[SCREENER] ✅ Found {len(results)} pairs scoring ≥ {min_score}/100")
    for i, r in enumerate(results[:5], 1):
        await log(
            f"[SCREENER] #{i} {r['pair']} | Score: {r['score']}/100 "
            f"| {r['grade']} | {r['direction'].upper()} | {r['setup_type']}"
        )
    return results


# ─────────────────────────────────────────────
# Risk Calculator
# ─────────────────────────────────────────────

def calculate_risk(entry: float, stop_loss: float,
                   account: float = 100.0,
                   score: int = 75,
                   atr: float = 1.0) -> dict:
    """Risk scales with score. Higher score = more margin + leverage."""
    risk_profile = score_to_risk(score, account)
    risk_pct     = risk_profile["risk_pct"]
    lev_cap      = risk_profile["lev_cap"]

    risk_amount = account * risk_pct
    sl_dist_pct = abs(entry - stop_loss) / entry if entry > 0 else 0.01
    if sl_dist_pct < 1e-6: sl_dist_pct = 0.01

    notional = risk_amount / sl_dist_pct
    raw_lev  = notional / account

    atr_pct = atr / entry if entry > 0 else 0.01
    vol_cap = max(1, min(20, int(0.025 / atr_pct)))
    leverage = int(min(raw_lev + 1, lev_cap, vol_cap, 20))
    leverage = max(leverage, 1)

    margin = round(notional / leverage, 2)
    margin = min(margin, account * 0.4)
    notional = margin * leverage

    is_long = stop_loss < entry
    d1, d2, d3 = sl_dist_pct*1.5, sl_dist_pct*2.5, sl_dist_pct*4.0

    if is_long:
        tp1=round(entry*(1+d1),8); tp2=round(entry*(1+d2),8); tp3=round(entry*(1+d3),8)
        hard_inv=round(stop_loss*0.998,8); soft_warn=round(stop_loss*1.004,8)
    else:
        tp1=round(entry*(1-d1),8); tp2=round(entry*(1-d2),8); tp3=round(entry*(1-d3),8)
        hard_inv=round(stop_loss*1.002,8); soft_warn=round(stop_loss*0.996,8)

    return {
        "risk_amount_usdt":  round(risk_amount, 2),
        "position_notional": round(notional, 2),
        "margin_usdt":       margin,
        "leverage":          leverage,
        "lev_cap":           lev_cap,
        "sl_dist_pct":       round(sl_dist_pct*100, 3),
        "rr":                round(d2/sl_dist_pct, 2),
        "risk_pct":          risk_pct,
        "risk_label":        risk_profile["label"],
        "tp1":tp1, "tp2":tp2, "tp3":tp3,
        "hard_invalidation": hard_inv,
        "soft_warning":      soft_warn,
    }


# ─────────────────────────────────────────────
# Per-TF Full Analysis
# ─────────────────────────────────────────────

def analyze_tf(df: pd.DataFrame, tf: str, direction_hint: str = "any") -> dict:
    lb = 3 if tf in ("5m","15m") else 5
    highs, lows = detect_swing_points(df, lb=lb)
    struct  = detect_structure(df, highs, lows)
    sweep   = detect_liquidity_sweep(df, highs, lows)
    price   = float(df["close"].iloc[-1])

    dir_use = direction_hint
    if dir_use == "any":
        if sweep.get("swept"):   dir_use = sweep["direction"]
        elif struct["trend"] != "ranging": dir_use = struct["trend"]
        else: dir_use = "bullish"

    obs      = detect_order_blocks(df, dir_use)
    fvgs     = detect_fvgs(df, dir_use)
    sd_zones = detect_supply_demand_zones(df, dir_use)
    diamond  = detect_diamond_zone(obs, fvgs, sd_zones, price)
    indics   = secondary_indicators(df, dir_use)
    score_bd = compute_full_score(struct, sweep, obs, fvgs, sd_zones,
                                   diamond, indics, dir_use)
    in_zone  = bool(
        any(o["fresh"] and o["low"] <= price <= o["high"] for o in obs) or
        any(f["bottom"] <= price <= f["top"] for f in fvgs) or
        any(z["in_zone"] for z in sd_zones)
    )

    return {
        "tf": tf, "price": round(price,8),
        "structure": struct, "sweep": sweep,
        "obs": obs, "fvgs": fvgs,
        "sd_zones": sd_zones, "diamond": diamond,
        "entry_zone": find_best_entry_zone(obs,fvgs,sd_zones,diamond,price,dir_use),
        "in_zone": in_zone,
        "indicators": indics,
        "score": score_bd,
        "direction": dir_use,
    }


# ─────────────────────────────────────────────
# SMC Signal Decision
# ─────────────────────────────────────────────

def make_smc_decision(analyses: dict) -> dict:
    """
    5 signal triggers ranked by quality.
    Score scales position sizing automatically.
    """
    result = {
        "signal": False, "direction": None,
        "base_confidence": 0, "final_confidence": 0,
        "trigger": None, "entry_zone": None,
        "sweep_level": None, "stop_loss_level": None,
        "score": 0, "score_breakdown": {},
        "reasons": [],
    }

    a1d = analyses.get("1d",{})
    a4h = analyses.get("4h",{})
    a1h = analyses.get("1h",{})
    a15 = analyses.get("15m",{})
    a5m = analyses.get("5m",{})

    def s(a):   return a.get("structure",{})
    def sw(a):  return a.get("sweep",{})
    def ez(a):  return a.get("entry_zone",{})
    def sc(a):  return a.get("score",{})

    direction = conf = trigger = entry_zone = sl_level = sweep_level = None
    reasons   = []

    # ── T1: 4H sweep + 1H BOS ──
    if sw(a4h).get("swept") and (s(a1h).get("bos") or s(a1h).get("choch")):
        direction   = sw(a4h)["direction"]
        conf        = 82; trigger = "4H_SWEEP_1H_BOS"
        sweep_level = sw(a4h).get("sweep_level")
        entry_zone  = ez(a1h) if ez(a1h).get("entry") else ez(a4h)
        sl_level    = round((sw(a4h).get("sweep_low") or sweep_level)*0.999,8) \
                      if direction=="bullish" else \
                      round((sw(a4h).get("sweep_high") or sweep_level)*1.001,8)
        reasons.append(f"4H liquidity sweep @ {sweep_level} + 1H BOS confirmed")

    # ── T2: 1H sweep + BOS/CHoCH ──
    elif sw(a1h).get("swept") and (s(a1h).get("bos") or s(a1h).get("choch")):
        direction   = sw(a1h)["direction"]
        conf        = 78; trigger = "1H_SWEEP_BOS"
        sweep_level = sw(a1h).get("sweep_level")
        entry_zone  = ez(a1h) if ez(a1h).get("entry") else ez(a15)
        sl_level    = round((sw(a1h).get("sweep_low") or sweep_level)*0.999,8) \
                      if direction=="bullish" else \
                      round((sw(a1h).get("sweep_high") or sweep_level)*1.001,8)
        reasons.append(f"1H sweep @ {sweep_level} + BOS/CHoCH")

    # ── T3: 15m sweep + 1H structure aligned ──
    elif (sw(a15).get("swept") and
          s(a1h).get("trend") == sw(a15).get("direction") and
          s(a1h).get("trend") != "ranging"):
        direction   = sw(a15)["direction"]
        conf        = 74; trigger = "15m_SWEEP_1H_ALIGNED"
        sweep_level = sw(a15).get("sweep_level")
        entry_zone  = ez(a15) if ez(a15).get("entry") else ez(a1h)
        sl_level    = round((sw(a15).get("sweep_low") or sweep_level)*0.999,8) \
                      if direction=="bullish" else \
                      round((sw(a15).get("sweep_high") or sweep_level)*1.001,8)
        reasons.append(f"15m sweep + 1H {direction} structure aligned")

    # ── T4: BOS + fresh OB + FVG on 1H ──
    elif (s(a1h).get("bos") and
          any(o.get("fresh") for o in a1h.get("obs",[])) and
          len(a1h.get("fvgs",[])) > 0):
        direction   = s(a1h).get("bos_dir","bullish")
        conf        = 72; trigger = "1H_BOS_OB_FVG"
        entry_zone  = ez(a1h)
        fresh_obs   = [o for o in a1h.get("obs",[]) if o.get("fresh")]
        if fresh_obs:
            sl_level = round(fresh_obs[0]["low"]*0.999,8) \
                       if direction=="bullish" else \
                       round(fresh_obs[0]["high"]*1.001,8)
        reasons.append(f"1H BOS ({direction}) + fresh OB + unfilled FVG")

    # ── T5: 4H BOS + 1H CHoCH (structure cascade) ──
    elif s(a4h).get("bos") and s(a1h).get("choch"):
        direction  = s(a4h).get("bos_dir","bullish")
        conf       = 70; trigger = "4H_BOS_1H_CHoCH"
        entry_zone = ez(a1h)
        fresh_obs  = [o for o in a1h.get("obs",[]) if o.get("fresh")]
        if fresh_obs:
            sl_level = round(fresh_obs[0]["low"]*0.999,8) \
                       if direction=="bullish" else \
                       round(fresh_obs[0]["high"]*1.001,8)
        reasons.append(f"4H BOS → 1H CHoCH cascade ({direction})")

    if direction is None:
        result["reasons"] = ["No SMC trigger: no sweep, no BOS+OB+FVG found"]
        return result

    direction = "long"  if direction in ("bullish","long")  else "short"

    # ── Aggregate score from 1H (primary) ──
    score_bd = sc(a1h)
    total_score = score_bd.get("total", conf)

    # Diamond zone bonus
    diamond_1h = a1h.get("diamond")
    if diamond_1h:
        bonus = diamond_1h.get("bonus_pts", 0)
        total_score = min(total_score + bonus, 100)
        reasons.append(f"⬡ Diamond Zone ({'+'.join(diamond_1h.get('components',[]))}) +{bonus}pts")

    # HTF alignment bonus
    if direction=="long":
        if s(a1d).get("trend")=="bullish": total_score=min(total_score+3,100); reasons.append("1D bullish aligned")
        if s(a4h).get("trend")=="bullish": total_score=min(total_score+2,100)
    else:
        if s(a1d).get("trend")=="bearish": total_score=min(total_score+3,100); reasons.append("1D bearish aligned")
        if s(a4h).get("trend")=="bearish": total_score=min(total_score+2,100)

    # Supply/demand at zone
    sd_1h = a1h.get("sd_zones",[])
    if any(z.get("in_zone") for z in sd_1h):
        total_score = min(total_score + 5, 100)
        reasons.append("Price inside S/D zone")

    # Structure reasons
    if s(a1h).get("bos"):   reasons.append(f"1H BOS {s(a1h).get('bos_dir','')}")
    if s(a1h).get("choch"): reasons.append("1H CHoCH")
    if s(a4h).get("bos"):   reasons.append(f"4H BOS {s(a4h).get('bos_dir','')}")
    if a1h.get("fvgs"):     reasons.append(f"{len(a1h['fvgs'])} FVG(s) unfilled on 1H")
    fresh_count = len([o for o in a1h.get("obs",[]) if o.get("fresh")])
    if fresh_count:         reasons.append(f"{fresh_count} fresh OB(s) on 1H")
    if a1h.get("indicators",{}).get("rsi_divergence") != "none":
        reasons.append(f"RSI div: {a1h['indicators']['rsi_divergence']}")
    if a1h.get("indicators",{}).get("vol_spike"):
        reasons.append("Volume spike on 1H")
    sd_zones_1h = [z for z in sd_1h if z["freshness"] in ("fresh","respected")]
    if sd_zones_1h:
        reasons.append(f"{len(sd_zones_1h)} S/D zone(s) ({sd_zones_1h[0]['freshness']})")

    # SL fallback
    if not sl_level:
        ep = entry_zone.get("entry", 0) if entry_zone else 0
        sl_level = round(ep*0.985,8) if direction=="long" else round(ep*1.015,8)

    # Entry fallback
    if not entry_zone or not entry_zone.get("entry"):
        p = float(a1h.get("price",0))
        entry_zone = {"source":"MARKET","entry":p,"high":p*1.001,"low":p*0.999}

    result.update({
        "signal": True, "direction": direction,
        "base_confidence": conf, "final_confidence": min(total_score, 98),
        "trigger": trigger, "entry_zone": entry_zone,
        "sweep_level": sweep_level, "stop_loss_level": sl_level,
        "score": total_score, "score_breakdown": score_bd,
        "diamond": diamond_1h,
        "reasons": reasons,
    })
    return result


# ─────────────────────────────────────────────
# Main Full Analysis Orchestrator
# ─────────────────────────────────────────────

async def run_full_analysis(exchange: ExchangeManager, pair: str,
                             account_size: float = 100.0,
                             risk_pct: float = 0.015,
                             log_fn=None) -> dict:
    async def log(msg):
        if log_fn: await log_fn(msg)
        logger.info(msg)

    await log(f"[ENGINE] ═══ Full 5-TF SMC/ICT Analysis: {pair} ═══")
    await log("[ENGINE] Fetching 1D · 4H · 1H · 15m · 5m candles...")

    try:
        df1d,df4h,df1h,df15,df5m = await asyncio.gather(
            exchange.fetch_ohlcv(pair,"1d",limit=200),
            exchange.fetch_ohlcv(pair,"4h",limit=200),
            exchange.fetch_ohlcv(pair,"1h",limit=200),
            exchange.fetch_ohlcv(pair,"15m",limit=200),
            exchange.fetch_ohlcv(pair,"5m",limit=200),
        )
    except Exception as e:
        await log(f"[ENGINE] ❌ Fetch failed: {e}")
        return {"error": str(e)}

    await log("[ENGINE] Candles fetched. Running full SMC/ICT + Supply/Demand analysis...")

    a1d = analyze_tf(df1d,"1d")
    a4h = analyze_tf(df4h,"4h")
    a1h = analyze_tf(df1h,"1h")
    a15 = analyze_tf(df15,"15m")
    a5m = analyze_tf(df5m,"5m")
    analyses = {"1d":a1d,"4h":a4h,"1h":a1h,"15m":a15,"5m":a5m}
    price = float(df1h["close"].iloc[-1])

    for label, a in [("1D",a1d),("4H",a4h),("1H",a1h),("15m",a15),("5m",a5m)]:
        sw = a["sweep"]; st = a["structure"]; sc = a["score"]
        diamond = a.get("diamond")
        await log(
            f"[SMC] {label} | Trend:{st['trend'].upper()} | "
            f"BOS:{'✓' if st['bos'] else '✗'} CHoCH:{'✓' if st['choch'] else '✗'} | "
            f"Sweep:{'✓ '+sw['direction'].upper() if sw['swept'] else '✗'} | "
            f"OBs:{len([o for o in a['obs'] if o['fresh']])}fresh | "
            f"FVGs:{len(a['fvgs'])} | S/D:{len(a['sd_zones'])} | "
            f"Score:{sc.get('total',0)} | "
            f"{'⬡ DIAMOND' if diamond else ''}"
        )

    await log("[ENGINE] Running ICT signal decision engine...")
    decision = make_smc_decision(analyses)

    if not decision["signal"]:
        await log(f"[ENGINE] ⚠️ No signal: {' | '.join(decision['reasons'])}")
        return {
            "pair":pair,"price":price,"analyses":analyses,
            "no_signal":True,
            "no_signal_reason":" | ".join(decision["reasons"]),
        }

    direction  = decision["direction"]
    entry_zone = decision["entry_zone"]
    sl_level   = decision["stop_loss_level"]
    trigger    = decision["trigger"]
    score      = decision["score"]
    conf       = decision["final_confidence"]

    await log(f"[ENGINE] ✅ SIGNAL: {direction.upper()} | Trigger:{trigger} | Score:{score}/100 | Conf:{conf}%")

    entry_cond = entry_zone.get("entry_condition", "LIMIT_WAIT")
    dist_from  = entry_zone.get("dist_from_entry", 0)
    await log(
        f"[ENGINE] Entry zone: {entry_zone.get('source','?')} | "
        f"Limit @ {entry_zone.get('entry')} | "
        f"Zone: [{entry_zone.get('zone_low')}–{entry_zone.get('zone_high')}] | "
        f"Condition: {entry_cond} | "
        f"Distance from CMP: {dist_from:.3f}%"
    )
    if entry_cond == "LIMIT_WAIT":
        await log(
            f"[ENGINE] ⏳ WAIT — price must retrace to {entry_zone.get('entry')} "
            f"({'DOWN' if direction=='long' else 'UP'} from current {price})"
        )
    elif entry_cond == "LIMIT_NOW":
        await log(f"[ENGINE] ✅ Price already INSIDE zone — enter limit now @ {entry_zone.get('entry')}")
    await log(f"[ENGINE] SL: {sl_level} | Sweep level: {decision.get('sweep_level')}")
    for r in decision["reasons"]:
        await log(f"[SMC] ✓ {r}")

    atr_1h = a1h["indicators"]["atr"]
    risk   = calculate_risk(
        entry=entry_zone["entry"], stop_loss=sl_level,
        account=account_size, score=score, atr=atr_1h,
    )
    await log(
        f"[ORDER] Score-scaled risk → {risk['risk_label']} | "
        f"Margin:${risk['margin_usdt']} | Lev:{risk['leverage']}x | "
        f"RR:1:{risk['rr']} | Risk:${risk['risk_amount_usdt']}"
    )

    funding_rate = await exchange.fetch_funding_rate(pair)
    await log(f"[ENGINE] Funding: {funding_rate*100:.4f}%")

    def trim_obs(obs_list):
        return [{"type":o["type"],"high":o["high"],"low":o["low"],
                 "mid":o["mid"],"fresh":o["fresh"],"dist_pct":o["dist_pct"]}
                for o in obs_list[:3]]
    def trim_fvgs(f_list):
        return [{"type":f["type"],"top":f["top"],"bottom":f["bottom"],
                 "mid":f["mid"],"size_pct":f["size_pct"]}
                for f in f_list[:3]]
    def trim_sd(sd_list):
        return [{"type":z["type"],"high":z["high"],"low":z["low"],
                 "mid":z["mid"],"freshness":z["freshness"],"in_zone":z["in_zone"]}
                for z in sd_list[:2]]

    gemini_payload = {
        "pair":pair,"current_price":price,
        "account_size":account_size,"risk_pct":risk["risk_pct"],
        "smc_trigger":trigger,"smc_direction":direction,
        "smc_score":score,"smc_confidence":conf,
        "smc_reasons":decision["reasons"],
        "diamond_zone": {
            "exists": bool(decision.get("diamond")),
            "type":   decision["diamond"].get("type") if decision.get("diamond") else None,
            "zone":   [decision["diamond"].get("low"), decision["diamond"].get("high")]
                      if decision.get("diamond") else None,
        },
        "funding_rate":funding_rate,
        "pre_calculated_risk":risk,
        "entry_zone":{
            "source":          entry_zone.get("source"),
            "limit_price":     entry_zone.get("entry"),      # ← the actual limit order price
            "zone_low":        entry_zone.get("zone_low"),
            "zone_high":       entry_zone.get("zone_high"),
            "entry_condition": entry_cond,                   # LIMIT_WAIT / LIMIT_NOW / MARKET
            "dist_from_cmp":   dist_from,                    # % distance from current price
            "note": (
                f"Place LIMIT {'BUY' if direction=='long' else 'SELL'} at {entry_zone.get('entry')}. "
                f"Price must {'fall' if direction=='long' else 'rise'} "
                f"{dist_from:.2f}% to fill this order."
            ) if entry_cond == "LIMIT_WAIT" else
            f"Price already inside zone. Enter limit now at {entry_zone.get('entry')}.",
        },
        "stop_loss_level":sl_level,
        "sweep_level":decision.get("sweep_level"),
        "timeframes":{
            "1d":{"trend":a1d["structure"]["trend"],
                  "bos":a1d["structure"]["bos"],
                  "ema_stack":"bull" if a1d["indicators"]["bull_ema_stack"] else
                              "bear" if a1d["indicators"]["bear_ema_stack"] else "mixed",
                  "rsi":a1d["indicators"]["rsi"],
                  "obs":trim_obs(a1d["obs"]),"fvgs":trim_fvgs(a1d["fvgs"]),
                  "sd_zones":trim_sd(a1d["sd_zones"])},
            "4h":{"trend":a4h["structure"]["trend"],
                  "bos":a4h["structure"]["bos"],"choch":a4h["structure"]["choch"],
                  "sweep":a4h["sweep"]["swept"],"sweep_dir":a4h["sweep"].get("direction"),
                  "obs":trim_obs(a4h["obs"]),"fvgs":trim_fvgs(a4h["fvgs"]),
                  "sd_zones":trim_sd(a4h["sd_zones"]),
                  "rsi":a4h["indicators"]["rsi"],"vol_spike":a4h["indicators"]["vol_spike"]},
            "1h":{"trend":a1h["structure"]["trend"],
                  "bos":a1h["structure"]["bos"],"choch":a1h["structure"]["choch"],
                  "sweep":a1h["sweep"]["swept"],"sweep_level":a1h["sweep"].get("sweep_level"),
                  "obs":trim_obs(a1h["obs"]),"fvgs":trim_fvgs(a1h["fvgs"]),
                  "sd_zones":trim_sd(a1h["sd_zones"]),
                  "in_zone":a1h["in_zone"],
                  "rsi":a1h["indicators"]["rsi"],"rsi_div":a1h["indicators"]["rsi_divergence"],
                  "vol_spike":a1h["indicators"]["vol_spike"],
                  "ema20":a1h["indicators"]["ema20"],"atr":atr_1h,
                  "candle":a1h["indicators"]["candle_pattern"]},
            "15m":{"trend":a15["structure"]["trend"],
                   "sweep":a15["sweep"]["swept"],
                   "obs":trim_obs(a15["obs"]),"fvgs":trim_fvgs(a15["fvgs"]),
                   "rsi":a15["indicators"]["rsi"],"candle":a15["indicators"]["candle_pattern"]},
            "5m":{"sweep":a5m["sweep"]["swept"],
                  "rsi":a5m["indicators"]["rsi"],
                  "candle":a5m["indicators"]["candle_pattern"],
                  "vol_spike":a5m["indicators"]["vol_spike"],
                  "ema20_dir":"up" if a5m["indicators"]["bull_ema_stack"] else "down"},
        },
        "liquidity":{
            "equal_highs":a1h["sweep"].get("equal_highs",[]),
            "equal_lows":a1h["sweep"].get("equal_lows",[]),
        },
    }

    await log("[ENGINE] ✅ Payload ready for Gemini.")
    return {
        "pair":pair,"price":price,"direction":direction,
        "decision":decision,"analyses":analyses,
        "risk":risk,"funding_rate":funding_rate,
        "gemini_payload":gemini_payload,
        "no_signal":False,"atr_1h":atr_1h,"score":score,
    }
