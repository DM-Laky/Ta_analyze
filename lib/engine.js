/**
 * lib/engine.js
 * SMC/ICT Analysis Engine
 * Primary: Liquidity Sweep → BOS/CHoCH → FVG/OB/S&D entry
 * Secondary: RSI, EMA, Volume (confirmation only)
 * Scoring: 85pts SMC + 15pts indicators = 100pts
 */

// ─────────────────────────────────────────
// Math Helpers
// ─────────────────────────────────────────

function ema(values, period) {
  const k = 2 / (period + 1);
  let result = new Array(values.length).fill(null);
  // Find first valid starting point
  let start = period - 1;
  result[start] = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = start + 1; i < values.length; i++) {
    result[i] = values[i] * k + result[i - 1] * (1 - k);
  }
  return result;
}

function rsi(closes, period = 14) {
  if (closes.length < period + 1) return new Array(closes.length).fill(50);
  const result = new Array(closes.length).fill(null);
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  result[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    result[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return result;
}

function atr(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1].close;
    return Math.max(c.high - c.low, Math.abs(c.high - prev), Math.abs(c.low - prev));
  });
  const result = new Array(candles.length).fill(null);
  result[period - 1] = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < tr.length; i++) {
    result[i] = (result[i - 1] * (period - 1) + tr[i]) / period;
  }
  return result;
}

function avgVolume(candles, period = 20) {
  if (candles.length < period) return candles[candles.length - 1]?.volume || 0;
  const slice = candles.slice(-period - 1, -1);
  return slice.reduce((s, c) => s + c.volume, 0) / slice.length;
}

// ─────────────────────────────────────────
// Swing Point Detection
// ─────────────────────────────────────────

function detectSwings(candles, lb = 5) {
  const highs = [], lows = [];
  for (let i = lb; i < candles.length - lb; i++) {
    const windowH = candles.slice(i - lb, i + lb + 1).map((c) => c.high);
    const windowL = candles.slice(i - lb, i + lb + 1).map((c) => c.low);
    if (candles[i].high === Math.max(...windowH))
      highs.push({ i, price: candles[i].high });
    if (candles[i].low === Math.min(...windowL))
      lows.push({ i, price: candles[i].low });
  }
  return { highs, lows };
}

// ─────────────────────────────────────────
// Market Structure (BOS / CHoCH)
// ─────────────────────────────────────────

function detectStructure(candles, highs, lows) {
  const res = {
    trend: "ranging", bos: false, bosLevel: null, bosDir: null,
    choch: false, chochLevel: null, hhHl: false, llLh: false,
    swingHighs: [], swingLows: [],
  };
  if (highs.length < 2 || lows.length < 2) return res;

  const rh = highs.slice(-6);
  const rl = lows.slice(-6);
  res.swingHighs = rh.map((h) => h.price);
  res.swingLows  = rl.map((l) => l.price);

  const price = candles[candles.length - 1].close;
  res.hhHl = rh.at(-1).price > rh.at(-2).price && rl.at(-1).price > rl.at(-2).price;
  res.llLh = rh.at(-1).price < rh.at(-2).price && rl.at(-1).price < rl.at(-2).price;

  if (price > rh.at(-1).price) {
    res.bos = true; res.bosLevel = rh.at(-1).price;
    res.bosDir = "bullish"; res.trend = "bullish";
  } else if (price < rl.at(-1).price) {
    res.bos = true; res.bosLevel = rl.at(-1).price;
    res.bosDir = "bearish"; res.trend = "bearish";
  } else if (res.hhHl) res.trend = "bullish";
  else if (res.llLh)   res.trend = "bearish";

  if (res.trend === "bullish" && rh.at(-1).price < rh.at(-2).price) {
    res.choch = true; res.chochLevel = rh.at(-1).price;
  } else if (res.trend === "bearish" && rl.at(-1).price > rl.at(-2).price) {
    res.choch = true; res.chochLevel = rl.at(-1).price;
  }
  return res;
}

// ─────────────────────────────────────────
// Liquidity Sweep
// ─────────────────────────────────────────

function detectSweep(candles, highs, lows, tol = 0.0025) {
  const res = {
    swept: false, direction: null,
    sweepLevel: null, sweepLow: null, sweepHigh: null,
    candles_ago: null, equalHighs: [], equalLows: [],
  };
  if (candles.length < 10) return res;

  const shPrices = highs.map((h) => h.price);
  const slPrices = lows.map((l) => l.price);

  const eqH = [], eqL = [];
  for (let i = 0; i < shPrices.length; i++)
    for (let j = i + 1; j < shPrices.length; j++)
      if (Math.abs(shPrices[i] - shPrices[j]) / shPrices[i] < tol) {
        const lvl = (shPrices[i] + shPrices[j]) / 2;
        if (!eqH.includes(lvl)) eqH.push(lvl);
      }
  for (let i = 0; i < slPrices.length; i++)
    for (let j = i + 1; j < slPrices.length; j++)
      if (Math.abs(slPrices[i] - slPrices[j]) / slPrices[i] < tol) {
        const lvl = (slPrices[i] + slPrices[j]) / 2;
        if (!eqL.includes(lvl)) eqL.push(lvl);
      }

  res.equalHighs = eqH.slice(-5);
  res.equalLows  = eqL.slice(-5);

  const scan = Math.min(40, candles.length - 1);
  for (let i = candles.length - 1; i > candles.length - scan; i--) {
    const { low: li, high: hi, close: ci, open: oi } = candles[i];
    const rng = hi - li;

    for (const eq of eqL)
      if (li < eq * (1 - tol * 0.3) && ci > eq) {
        res.swept = true; res.direction = "bullish";
        res.sweepLevel = eq; res.sweepLow = li;
        res.candles_ago = candles.length - 1 - i;
        return res;
      }
    for (const eq of eqH)
      if (hi > eq * (1 + tol * 0.3) && ci < eq) {
        res.swept = true; res.direction = "bearish";
        res.sweepLevel = eq; res.sweepHigh = hi;
        res.candles_ago = candles.length - 1 - i;
        return res;
      }

    if (i >= 5 && rng > 0) {
      const recLow  = Math.min(...candles.slice(Math.max(0, i - 10), i).map((c) => c.low));
      const recHigh = Math.max(...candles.slice(Math.max(0, i - 10), i).map((c) => c.high));
      const wDn = Math.min(oi, ci) - li;
      const wUp = hi - Math.max(oi, ci);
      if (li < recLow && ci > recLow && wDn > rng * 0.55) {
        res.swept = true; res.direction = "bullish";
        res.sweepLevel = recLow; res.sweepLow = li;
        res.candles_ago = candles.length - 1 - i;
        return res;
      }
      if (hi > recHigh && ci < recHigh && wUp > rng * 0.55) {
        res.swept = true; res.direction = "bearish";
        res.sweepLevel = recHigh; res.sweepHigh = hi;
        res.candles_ago = candles.length - 1 - i;
        return res;
      }
    }
  }
  return res;
}

// ─────────────────────────────────────────
// Order Blocks
// ─────────────────────────────────────────

function detectOrderBlocks(candles, direction) {
  const obs = [];
  const price = candles.at(-1).close;
  const ranges = candles.map((c) => c.high - c.low);
  const avgRng = ranges.reduce((a, b) => a + b, 0) / ranges.length;
  const isLong = direction === "bullish" || direction === "long";

  for (let i = 2; i < candles.length - 3; i++) {
    const c = candles[i], n = candles[i + 1];
    const up   = n.close - n.open;
    const down = n.open  - n.close;

    if (isLong && c.open > c.close && up > avgRng * 1.2) {
      const tested = candles.slice(i + 2, Math.min(i + 20, candles.length))
                            .some((x) => x.low <= c.close);
      const dist = Math.abs(price - (c.open + c.low) / 2) / price;
      obs.push({
        type: "bullish", high: c.open, low: c.low,
        mid: (c.open + c.low) / 2,
        strength: Math.min(up / (avgRng * 1.2), 1.5),
        fresh: !tested, distPct: dist * 100, index: i,
      });
    }
    if (!isLong && c.close > c.open && down > avgRng * 1.2) {
      const tested = candles.slice(i + 2, Math.min(i + 20, candles.length))
                            .some((x) => x.high >= c.open);
      const dist = Math.abs(price - (c.close + c.high) / 2) / price;
      obs.push({
        type: "bearish", high: c.high, low: c.close,
        mid: (c.close + c.high) / 2,
        strength: Math.min(down / (avgRng * 1.2), 1.5),
        fresh: !tested, distPct: dist * 100, index: i,
      });
    }
  }
  return obs
    .sort((a, b) => (b.fresh ? 1 : 0) - (a.fresh ? 1 : 0) || a.distPct - b.distPct)
    .slice(0, 5);
}

// ─────────────────────────────────────────
// Fair Value Gaps
// ─────────────────────────────────────────

function detectFVGs(candles, direction) {
  const fvgs  = [];
  const price = candles.at(-1).close;
  const isLong = direction === "bullish" || direction === "long";

  for (let i = 1; i < candles.length - 1; i++) {
    const prev = candles[i - 1], next = candles[i + 1];
    if (isLong && prev.high < next.low) {
      const top = next.low, bot = prev.high, mid = (top + bot) / 2;
      const sz  = (top - bot) / price * 100;
      if (sz > 0) fvgs.push({
        type: "bullish", top, bottom: bot, mid,
        sizePct: sz, filled: price <= bot,
        distPct: Math.abs(price - mid) / price * 100,
      });
    }
    if (!isLong && prev.low > next.high) {
      const top = prev.low, bot = next.high, mid = (top + bot) / 2;
      const sz  = (top - bot) / price * 100;
      if (sz > 0) fvgs.push({
        type: "bearish", top, bottom: bot, mid,
        sizePct: sz, filled: price >= top,
        distPct: Math.abs(price - mid) / price * 100,
      });
    }
  }
  return fvgs
    .filter((f) => !f.filled)
    .sort((a, b) => a.distPct - b.distPct)
    .slice(0, 4);
}

// ─────────────────────────────────────────
// Supply & Demand Zones
// ─────────────────────────────────────────

function detectSupplyDemand(candles, direction) {
  const zones = [];
  const price = candles.at(-1).close;
  const isLong = direction === "bullish" || direction === "long";
  const ranges = candles.map((c) => c.high - c.low);
  const avgRng = ranges.reduce((a, b) => a + b, 0) / ranges.length;

  for (let i = 3; i < candles.length - 4; i++) {
    const body = Math.abs(candles[i].close - candles[i].open);
    if (body <= avgRng * 1.8) continue;

    const isBull = candles[i].close > candles[i].open;
    if (isLong && !isBull) continue;
    if (!isLong && isBull) continue;

    const base    = candles.slice(Math.max(0, i - 5), i);
    const zoneH   = Math.max(...base.map((c) => c.high));
    const zoneL   = Math.min(...base.map((c) => c.low));
    const zoneMid = (zoneH + zoneL) / 2;
    const zoneSize = (zoneH - zoneL) / price * 100;
    if (zoneSize < 0.05) continue;

    let touches = 0, broken = false;
    for (let j = i + 1; j < candles.length; j++) {
      const c = candles[j];
      if (isBull && c.low <= zoneH && c.high >= zoneL) {
        touches++;
        if (c.close < zoneL) { broken = true; break; }
      }
      if (!isBull && c.high >= zoneL && c.low <= zoneH) {
        touches++;
        if (c.close > zoneH) { broken = true; break; }
      }
    }
    if (broken) continue;

    const freshness = touches === 0 ? "fresh"
                    : touches === 1 ? "respected"
                    : touches === 2 ? "tested" : "weak";
    const freshScore = touches === 0 ? 10 : touches === 1 ? 6 : touches === 2 ? 3 : 1;

    zones.push({
      type:       isBull ? "demand" : "supply",
      high:       zoneH, low: zoneL, mid: zoneMid,
      sizePct:    zoneSize, freshness, freshScore,
      touchCount: touches,
      inZone:     zoneL <= price && price <= zoneH,
      distPct:    Math.abs(price - zoneMid) / price * 100,
    });
  }

  return zones
    .sort((a, b) => (b.inZone ? 1 : 0) - (a.inZone ? 1 : 0)
      || b.freshScore - a.freshScore || a.distPct - b.distPct)
    .slice(0, 4);
}

// ─────────────────────────────────────────
// Diamond Zone
// ─────────────────────────────────────────

function detectDiamond(obs, fvgs, sdZones) {
  if (!obs.length || !fvgs.length) return null;
  const ob  = obs[0];
  const fvg = fvgs[0];
  const sd  = sdZones[0];

  // Full diamond: OB + FVG + S/D overlap
  if (sd) {
    const lo = Math.max(ob.low, fvg.bottom, sd.low);
    const hi = Math.min(ob.high, fvg.top, sd.high);
    if (lo < hi) return {
      type: "diamond", high: hi, low: lo, mid: (hi + lo) / 2,
      components: ["OB", "FVG", "S/D"], bonusPts: 5,
    };
  }
  // Partial: OB + FVG overlap
  const lo2 = Math.max(ob.low, fvg.bottom);
  const hi2 = Math.min(ob.high, fvg.top);
  if (lo2 < hi2) return {
    type: "partial_diamond", high: hi2, low: lo2, mid: (hi2 + lo2) / 2,
    components: ["OB", "FVG"], bonusPts: 3,
  };
  return null;
}

// ─────────────────────────────────────────
// ICT Limit Order Entry Zone
// ─────────────────────────────────────────

function findEntryZone(obs, fvgs, sdZones, diamond, price, direction) {
  const isLong = direction === "bullish" || direction === "long";
  const candidates = [];

  const entryCondition = (ep, zLo, zHi) =>
    (zLo <= price && price <= zHi) ? "LIMIT_NOW" : "LIMIT_WAIT";

  // Diamond (highest priority)
  if (diamond) {
    const ep   = isLong ? diamond.low : diamond.high;
    const cond = entryCondition(ep, diamond.low, diamond.high);
    candidates.push({
      source: diamond.type.toUpperCase(), entry: ep,
      zoneLow: diamond.low, zoneHigh: diamond.high,
      entryCondition: cond,
      distFromEntry: Math.abs(price - ep) / price * 100,
      score: 150 - Math.abs(price - ep) / price * 300,
      bonusPts: diamond.bonusPts,
    });
  }

  // FVG
  for (const fvg of fvgs.slice(0, 3)) {
    const ep   = isLong ? fvg.bottom : fvg.top;
    const cond = entryCondition(ep, fvg.bottom, fvg.top);
    const dist = Math.abs(price - ep) / price * 100;
    candidates.push({
      source: "FVG", entry: ep,
      zoneLow: fvg.bottom, zoneHigh: fvg.top,
      entryCondition: cond, distFromEntry: dist,
      score: 100 - dist * 5 + (fvg.sizePct || 0) * 8 + (cond === "LIMIT_NOW" ? 20 : 0),
      bonusPts: 0,
    });
  }

  // OB
  for (const ob of obs.filter((o) => o.fresh).slice(0, 3)) {
    const ep   = isLong ? ob.low : ob.high;
    const cond = entryCondition(ep, ob.low, ob.high);
    const dist = Math.abs(price - ep) / price * 100;
    candidates.push({
      source: "OB", entry: ep,
      zoneLow: ob.low, zoneHigh: ob.high,
      entryCondition: cond, distFromEntry: dist,
      score: 100 - dist * 4 + (ob.strength || 1) * 12 + (cond === "LIMIT_NOW" ? 20 : 0),
      bonusPts: 0,
    });
  }

  // S/D
  for (const sd of sdZones.slice(0, 2)) {
    const ep   = isLong ? sd.low : sd.high;
    const cond = entryCondition(ep, sd.low, sd.high);
    const dist = Math.abs(price - ep) / price * 100;
    candidates.push({
      source: "S/D", entry: ep,
      zoneLow: sd.low, zoneHigh: sd.high,
      entryCondition: cond, distFromEntry: dist,
      score: 90 - dist * 5 + (sd.freshScore || 0) * 6 + (cond === "LIMIT_NOW" ? 15 : 0),
      bonusPts: 0,
    });
  }

  if (!candidates.length) return {
    source: "MARKET", entry: price,
    zoneLow: price * 0.999, zoneHigh: price * 1.001,
    entryCondition: "MARKET", distFromEntry: 0, score: 30, bonusPts: 0,
  };

  candidates.sort((a, b) => b.score - a.score);
  return candidates[0];
}

// ─────────────────────────────────────────
// Candlestick Pattern
// ─────────────────────────────────────────

function candlestickPattern(candles) {
  if (candles.length < 3) return "none";
  const c = candles.at(-1), p = candles.at(-2), pp = candles.at(-3);
  const body = Math.abs(c.close - c.open);
  const uw   = c.high - Math.max(c.close, c.open);
  const lw   = Math.min(c.close, c.open) - c.low;
  const rng  = c.high - c.low;
  if (rng < 1e-10) return "doji";
  if (body / rng < 0.08)          return "doji";
  if (lw > body * 2.2 && uw < body * 0.4) return "hammer";
  if (uw > body * 2.2 && lw < body * 0.4) return "shooting_star";
  if (lw > rng * 0.62)            return "bullish_pinbar";
  if (uw > rng * 0.62)            return "bearish_pinbar";
  if (c.close > c.open && p.close < p.open &&
      c.open <= p.close && c.close >= p.open) return "bullish_engulfing";
  if (c.close < c.open && p.close > p.open &&
      c.open >= p.close && c.close <= p.open) return "bearish_engulfing";
  const pb = Math.abs(p.close - p.open), ppb = Math.abs(pp.close - pp.open);
  if (pp.close < pp.open && pb < ppb * 0.3 && c.close > c.open &&
      c.close > (pp.open + pp.close) / 2) return "morning_star";
  if (pp.close > pp.open && pb < ppb * 0.3 && c.close < c.open &&
      c.close < (pp.open + pp.close) / 2) return "evening_star";
  if (body / rng > 0.88 && c.close > c.open) return "bullish_marubozu";
  if (body / rng > 0.88 && c.close < c.open) return "bearish_marubozu";
  return "none";
}

// ─────────────────────────────────────────
// Secondary Indicators (15 pts max)
// ─────────────────────────────────────────

function computeIndicators(candles, direction) {
  const closes  = candles.map((c) => c.close);
  const rsiVals = rsi(closes, 14);
  const rsiVal  = rsiVals.at(-1) || 50;
  const e20     = ema(closes, 20).at(-1);
  const e50     = ema(closes, 50).at(-1);
  const e200    = ema(closes, 200).at(-1);
  const price   = closes.at(-1);
  const bullEma = price > e20 && e20 > e50;
  const bearEma = price < e20 && e20 < e50;
  const avgVol  = avgVolume(candles, 20);
  const volSpike = candles.at(-1).volume > avgVol * 1.8;
  const atrVals = atr(candles, 14);
  const atrVal  = atrVals.at(-1) || 0;
  const candle  = candlestickPattern(candles);
  const isLong  = direction === "bullish" || direction === "long";

  const BULL_CANDLES = new Set(["hammer","bullish_pinbar","bullish_engulfing","morning_star","bullish_marubozu"]);
  const BEAR_CANDLES = new Set(["shooting_star","bearish_pinbar","bearish_engulfing","evening_star","bearish_marubozu"]);

  let pts = 0;
  if (isLong) {
    if (rsiVal < 45)        pts += 3;
    if (bullEma)            pts += 4;
    if (volSpike)           pts += 5;
    if (BULL_CANDLES.has(candle)) pts += 3;
  } else {
    if (rsiVal > 55)        pts += 3;
    if (bearEma)            pts += 4;
    if (volSpike)           pts += 5;
    if (BEAR_CANDLES.has(candle)) pts += 3;
  }

  return {
    rsi: Math.round(rsiVal * 100) / 100,
    ema20: e20, ema50: e50, ema200: e200,
    bullEmaStack: bullEma, bearEmaStack: bearEma,
    volSpike, atr: atrVal,
    candlePattern: candle, pts: Math.min(pts, 15),
  };
}

// ─────────────────────────────────────────
// 85 / 15 Scoring System
// ─────────────────────────────────────────

function computeScore(struct, sweep, obs, fvgs, sdZones, diamond, indics, direction) {
  const bd = {};
  let smcTotal = 0;

  // Structure (25pts)
  let s = 0;
  if (struct.bos)    s += 10;
  if (struct.choch)  s += 8;
  const isLong = direction === "bullish" || direction === "long";
  if (isLong  && struct.hhHl) s += 7;
  if (!isLong && struct.llLh) s += 7;
  bd.structure = Math.min(s, 25); smcTotal += bd.structure;

  // Sweep (20pts)
  let sw = 0;
  if (sweep.swept) {
    const ago = sweep.candles_ago || 99;
    sw = ago <= 5 ? 20 : ago <= 15 ? 16 : ago <= 30 ? 10 : 6;
  }
  bd.sweep = sw; smcTotal += sw;

  // Order Blocks (15pts)
  let ob = 0;
  const freshOBs = obs.filter((o) => o.fresh);
  if (freshOBs.length) {
    ob += 8;
    if (freshOBs[0].distPct < 1.0) ob += 4;
    if (freshOBs[0].strength > 1.2) ob += 3;
  }
  bd.orderBlocks = Math.min(ob, 15); smcTotal += bd.orderBlocks;

  // FVG (10pts)
  let fvgPts = 0;
  if (fvgs.length) {
    fvgPts += 6;
    if (fvgs[0].distPct < 1.5) fvgPts += 2;
    if (fvgs[0].sizePct > 0.3)  fvgPts += 2;
  }
  bd.fvg = Math.min(fvgPts, 10); smcTotal += bd.fvg;

  // Supply/Demand (15pts)
  let sdPts = 0;
  if (sdZones.length) {
    const sd = sdZones[0];
    sdPts += sd.inZone ? 8 : sd.distPct < 2 ? 5 : sd.distPct < 5 ? 3 : 0;
    sdPts += Math.min(sd.freshScore, 4);
    if (sd.freshness === "respected") sdPts += 3;
  }
  bd.supplyDemand = Math.min(sdPts, 15); smcTotal += bd.supplyDemand;

  // Diamond bonus
  bd.diamondBonus = diamond ? diamond.bonusPts : 0;
  smcTotal = Math.min(smcTotal + bd.diamondBonus, 85);
  bd.smcTotal = smcTotal;

  // Indicators (15pts)
  bd.indicators = indics.pts;

  const total = Math.min(smcTotal + indics.pts, 100);
  bd.total = total;
  bd.grade = total >= 95 ? "ELITE" : total >= 85 ? "STRONG"
           : total >= 75 ? "STANDARD" : total >= 60 ? "WEAK" : "NO_SIGNAL";
  return bd;
}

// ─────────────────────────────────────────
// Risk Calculator (score-scaled, $6 cap)
// ─────────────────────────────────────────

function calculateRisk(entry, stopLoss, accountSize = 100, score = 75) {
  const MAX_MARGIN = parseFloat(process.env.MAX_MARGIN_PER_TRADE || 6);

  // Risk % and leverage cap scale with score
  let riskPct, levCap, label;
  if (score >= 95)      { riskPct = 0.020; levCap = 15; label = "ELITE";    }
  else if (score >= 85) { riskPct = 0.015; levCap = 12; label = "STRONG";   }
  else if (score >= 75) { riskPct = 0.010; levCap = 8;  label = "STANDARD"; }
  else                  { riskPct = 0.005; levCap = 5;  label = "WEAK";     }

  const riskAmount = accountSize * riskPct;
  const slDistPct  = Math.abs(entry - stopLoss) / entry || 0.01;
  let notional     = riskAmount / slDistPct;
  const rawLev     = notional / accountSize;

  const leverage   = Math.max(1, Math.min(Math.ceil(rawLev), levCap, 20));
  let margin       = Math.min(notional / leverage, accountSize * 0.4, MAX_MARGIN);
  notional         = margin * leverage;

  const isLong     = stopLoss < entry;
  const d1 = slDistPct * 1.5, d2 = slDistPct * 2.5, d3 = slDistPct * 4.0;
  const tp1 = isLong ? entry * (1 + d1) : entry * (1 - d1);
  const tp2 = isLong ? entry * (1 + d2) : entry * (1 - d2);
  const tp3 = isLong ? entry * (1 + d3) : entry * (1 - d3);

  return {
    riskAmount: Math.round(riskAmount * 100) / 100,
    notional:   Math.round(notional * 100) / 100,
    margin:     Math.round(margin * 100) / 100,
    leverage, levCap, label, riskPct,
    slDistPct:  Math.round(slDistPct * 10000) / 100,
    rr:         Math.round(d2 / slDistPct * 100) / 100,
    tp1, tp2, tp3,
    hardInvalidation: isLong ? stopLoss * 0.998 : stopLoss * 1.002,
    softWarning:      isLong ? stopLoss * 1.004 : stopLoss * 0.996,
  };
}

// ─────────────────────────────────────────
// Signal Decision Engine
// ─────────────────────────────────────────

function makeSignalDecision(analyses) {
  const { a1d, a4h, a1h, a15m, a5m } = analyses;
  let direction, conf, trigger, entryZone, slLevel, sweepLevel, reasons = [];

  // T1: 4H sweep + 1H BOS
  if (a4h.sweep.swept && (a1h.structure.bos || a1h.structure.choch)) {
    direction   = a4h.sweep.direction; conf = 82; trigger = "4H_SWEEP_1H_BOS";
    sweepLevel  = a4h.sweep.sweepLevel;
    entryZone   = a1h.entryZone;
    slLevel     = direction === "bullish"
      ? (a4h.sweep.sweepLow  || sweepLevel) * 0.999
      : (a4h.sweep.sweepHigh || sweepLevel) * 1.001;
    reasons.push(`4H liquidity sweep @ ${sweepLevel?.toFixed(6)} + 1H BOS confirmed`);
  }
  // T2: 1H sweep + BOS/CHoCH
  else if (a1h.sweep.swept && (a1h.structure.bos || a1h.structure.choch)) {
    direction   = a1h.sweep.direction; conf = 78; trigger = "1H_SWEEP_BOS";
    sweepLevel  = a1h.sweep.sweepLevel;
    entryZone   = a1h.entryZone;
    slLevel     = direction === "bullish"
      ? (a1h.sweep.sweepLow  || sweepLevel) * 0.999
      : (a1h.sweep.sweepHigh || sweepLevel) * 1.001;
    reasons.push(`1H sweep @ ${sweepLevel?.toFixed(6)} + BOS/CHoCH`);
  }
  // T3: 15m sweep + 1H aligned
  else if (a15m.sweep.swept && a1h.structure.trend === a15m.sweep.direction
           && a1h.structure.trend !== "ranging") {
    direction   = a15m.sweep.direction; conf = 74; trigger = "15m_SWEEP_1H_ALIGNED";
    sweepLevel  = a15m.sweep.sweepLevel;
    entryZone   = a15m.entryZone;
    slLevel     = direction === "bullish"
      ? (a15m.sweep.sweepLow  || sweepLevel) * 0.999
      : (a15m.sweep.sweepHigh || sweepLevel) * 1.001;
    reasons.push(`15m sweep + 1H ${direction} structure`);
  }
  // T4: BOS + fresh OB + FVG on 1H
  else if (a1h.structure.bos && a1h.obs.some((o) => o.fresh) && a1h.fvgs.length) {
    direction   = a1h.structure.bosDir || "bullish"; conf = 72; trigger = "1H_BOS_OB_FVG";
    entryZone   = a1h.entryZone;
    const fOB   = a1h.obs.find((o) => o.fresh);
    slLevel     = direction === "bullish" ? fOB.low * 0.999 : fOB.high * 1.001;
    reasons.push(`1H BOS (${direction}) + fresh OB + unfilled FVG`);
  }
  // T5: 4H BOS + 1H CHoCH
  else if (a4h.structure.bos && a1h.structure.choch) {
    direction   = a4h.structure.bosDir || "bullish"; conf = 70; trigger = "4H_BOS_1H_CHOCH";
    entryZone   = a1h.entryZone;
    const fOB   = a1h.obs.find((o) => o.fresh);
    slLevel     = fOB
      ? (direction === "bullish" ? fOB.low * 0.999 : fOB.high * 1.001)
      : (direction === "bullish" ? a1h.price * 0.985 : a1h.price * 1.015);
    reasons.push(`4H BOS → 1H CHoCH cascade (${direction})`);
  }

  if (!direction) return {
    signal: false,
    reasons: ["No SMC trigger: no sweep, no BOS+OB+FVG combination found"],
  };

  const normDir = direction === "bullish" ? "long" : direction === "bearish" ? "short" : direction;

  // Score from 1H analysis
  let score = a1h.score.total || conf;

  // Diamond bonus
  if (a1h.diamond) {
    score = Math.min(score + a1h.diamond.bonusPts, 100);
    reasons.push(`⬡ Diamond Zone (${a1h.diamond.components.join("+")}) +${a1h.diamond.bonusPts}pts`);
  }

  // HTF alignment bonus
  if (normDir === "long") {
    if (a1d.structure.trend === "bullish") { score = Math.min(score + 3, 100); reasons.push("1D bullish aligned"); }
    if (a4h.structure.trend === "bullish") score = Math.min(score + 2, 100);
  } else {
    if (a1d.structure.trend === "bearish") { score = Math.min(score + 3, 100); reasons.push("1D bearish aligned"); }
    if (a4h.structure.trend === "bearish") score = Math.min(score + 2, 100);
  }

  // Structure reasons
  if (a1h.structure.bos)   reasons.push(`1H BOS ${a1h.structure.bosDir || ""}`);
  if (a1h.structure.choch) reasons.push("1H CHoCH");
  if (a4h.structure.bos)   reasons.push(`4H BOS ${a4h.structure.bosDir || ""}`);
  if (a1h.fvgs.length)     reasons.push(`${a1h.fvgs.length} unfilled FVG(s) on 1H`);
  const freshCount = a1h.obs.filter((o) => o.fresh).length;
  if (freshCount)           reasons.push(`${freshCount} fresh OB(s) on 1H`);
  if (a1h.indicators.volSpike) reasons.push("Volume spike on 1H");
  if (!slLevel) {
    const ep = entryZone?.entry || a1h.price;
    slLevel = normDir === "long" ? ep * 0.985 : ep * 1.015;
  }
  if (!entryZone?.entry) {
    entryZone = { source: "MARKET", entry: a1h.price, zoneLow: a1h.price * 0.999, zoneHigh: a1h.price * 1.001, entryCondition: "MARKET", distFromEntry: 0 };
  }

  return {
    signal: true, direction: normDir, conf, trigger,
    entryZone, slLevel: Math.round(slLevel * 1e8) / 1e8,
    sweepLevel, score: Math.min(score, 98), reasons,
    diamond: a1h.diamond || null,
  };
}

// ─────────────────────────────────────────
// Per-Timeframe Analysis
// ─────────────────────────────────────────

function analyzeTimeframe(candles, tf, dirHint = "any") {
  const lb = tf === "5m" || tf === "15m" ? 3 : 5;
  const { highs, lows } = detectSwings(candles, lb);
  const structure = detectStructure(candles, highs, lows);
  const sweep     = detectSweep(candles, highs, lows);
  const price     = candles.at(-1).close;

  let dir = dirHint;
  if (dir === "any") {
    dir = sweep.swept ? sweep.direction
        : structure.trend !== "ranging" ? structure.trend : "bullish";
  }

  const obs      = detectOrderBlocks(candles, dir);
  const fvgs     = detectFVGs(candles, dir);
  const sdZones  = detectSupplyDemand(candles, dir);
  const diamond  = detectDiamond(obs, fvgs, sdZones);
  const indics   = computeIndicators(candles, dir);
  const scoreBd  = computeScore(structure, sweep, obs, fvgs, sdZones, diamond, indics, dir);
  const entryZone = findEntryZone(obs, fvgs, sdZones, diamond, price, dir);

  return {
    tf, price, structure, sweep, obs, fvgs, sdZones, diamond,
    entryZone, indicators: indics, score: scoreBd, direction: dir,
  };
}

// ─────────────────────────────────────────
// Main Full Analysis (5 TFs)
// ─────────────────────────────────────────

function runFullAnalysis(candles1d, candles4h, candles1h, candles15m, candles5m) {
  const a1d  = analyzeTimeframe(candles1d,  "1d");
  const a4h  = analyzeTimeframe(candles4h,  "4h");
  const a1h  = analyzeTimeframe(candles1h,  "1h");
  const a15m = analyzeTimeframe(candles15m, "15m");
  const a5m  = analyzeTimeframe(candles5m,  "5m");

  const decision = makeSignalDecision({ a1d, a4h, a1h, a15m, a5m });

  return {
    analyses: { a1d, a4h, a1h, a15m, a5m },
    decision,
    price: a1h.price,
  };
}

module.exports = {
  runFullAnalysis,
  analyzeTimeframe,
  calculateRisk,
  detectSwings,
  detectStructure,
  detectSweep,
  detectOrderBlocks,
  detectFVGs,
  detectSupplyDemand,
  detectDiamond,
  findEntryZone,
  computeScore,
};
