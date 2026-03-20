/**
 * api/analyze.js
 * POST /api/analyze
 * Body: { pair: "BTCUSDT" }
 * Fetches 1D+4H+1H+15m+5m candles, runs full SMC analysis, saves to KV
 * Does NOT call Gemini (separate step to stay under 10s limit)
 */

const { fetchOHLCV, fetchTicker, fetchFundingRate } = require("../lib/binance");
const { runFullAnalysis, calculateRisk }            = require("../lib/engine");
const { setLastAnalysis }                           = require("../lib/kv");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const logs = [];
  const log  = (msg) => { logs.push({ ts: new Date().toISOString().split("T")[1].split(".")[0], msg }); console.log(msg); };

  try {
    let { pair } = req.body;
    if (!pair) return res.status(400).json({ error: "pair required" });

    // Normalise: BTC/USDT → BTCUSDT
    const symbol = pair.replace("/", "").toUpperCase();
    const display = symbol.replace("USDT", "/USDT");

    log(`[ANALYZE] ═══ Starting analysis: ${display} ═══`);

    // Fetch all timeframes in parallel (3 groups to avoid rate limits)
    log("[ANALYZE] Fetching 1D · 4H · 1H · 15m · 5m candles from Binance...");

    const [c1d, c4h, c1h] = await Promise.all([
      fetchOHLCV(symbol, "1d",  150),
      fetchOHLCV(symbol, "4h",  150),
      fetchOHLCV(symbol, "1h",  150),
    ]);
    const [c15m, c5m, ticker, fundingRate] = await Promise.all([
      fetchOHLCV(symbol, "15m", 150),
      fetchOHLCV(symbol, "5m",  150),
      fetchTicker(symbol),
      fetchFundingRate(symbol),
    ]);

    const price = ticker.price;
    log(`[ANALYZE] Data fetched. Price: ${price} | Funding: ${(fundingRate * 100).toFixed(4)}%`);

    // Run SMC engine
    log("[SMC] Running full 5-TF SMC/ICT + Supply/Demand analysis...");
    const { analyses, decision } = runFullAnalysis(c1d, c4h, c1h, c15m, c5m);

    // Log per-timeframe summary
    for (const [label, a] of [["1D", analyses.a1d], ["4H", analyses.a4h], ["1H", analyses.a1h], ["15m", analyses.a15m], ["5m", analyses.a5m]]) {
      log(`[SMC] ${label} | Trend:${a.structure.trend.toUpperCase()} | BOS:${a.structure.bos ? "✓" : "✗"} | CHoCH:${a.structure.choch ? "✓" : "✗"} | Sweep:${a.sweep.swept ? "✓ " + a.sweep.direction.toUpperCase() : "✗"} | FVGs:${a.fvgs.length} | OBs:${a.obs.filter((o) => o.fresh).length}f | RSI:${a.indicators.rsi} | Score:${a.score.total}`);
    }

    if (!decision.signal) {
      log(`[SMC] ⚠️ No signal: ${decision.reasons.join(" | ")}`);
      await setLastAnalysis({ pair: display, symbol, price, fundingRate, analyses: summarizeAnalyses(analyses), decision, noSignal: true });
      return res.json({ success: true, noSignal: true, reason: decision.reasons.join(" | "), price, logs });
    }

    const { direction, trigger, entryZone, slLevel, score, reasons, diamond } = decision;
    log(`[SMC] ✅ SIGNAL: ${direction.toUpperCase()} | Trigger: ${trigger} | Score: ${score}/100`);
    log(`[SMC] Entry zone: ${entryZone.source} @ ${entryZone.entry?.toFixed(8)} [${entryZone.zoneLow?.toFixed(8)}–${entryZone.zoneHigh?.toFixed(8)}]`);
    log(`[SMC] Condition: ${entryZone.entryCondition} | Distance from CMP: ${entryZone.distFromEntry?.toFixed(3)}%`);
    if (entryZone.entryCondition === "LIMIT_WAIT")
      log(`[SMC] ⏳ Place LIMIT ${direction === "long" ? "BUY" : "SELL"} at ${entryZone.entry?.toFixed(8)} — wait for retrace`);
    else
      log(`[SMC] ✅ Price inside zone — enter immediately at ${entryZone.entry?.toFixed(8)}`);
    log(`[SMC] SL: ${slLevel?.toFixed(8)} | Sweep level: ${decision.sweepLevel}`);
    for (const r of reasons) log(`[SMC] ✓ ${r}`);
    if (diamond) log(`[SMC] ⬡ Diamond Zone: ${diamond.components.join("+")} | ${diamond.low?.toFixed(8)}–${diamond.high?.toFixed(8)}`);

    // Risk calculation
    const accountSize = parseFloat(process.env.ACCOUNT_SIZE || 100);
    const risk = calculateRisk(entryZone.entry, slLevel, accountSize, score);
    log(`[ORDER] Risk: ${risk.label} | Margin: $${risk.margin} | Lev: ${risk.leverage}x | RR: 1:${risk.rr}`);

    // Build Gemini payload (saved to KV, sent in /api/signal)
    const geminiPayload = buildGeminiPayload({
      pair: display, symbol, price, fundingRate,
      direction, trigger, score, reasons,
      entryZone, slLevel, diamond,
      analyses, risk,
    });

    await setLastAnalysis({
      pair: display, symbol, price, fundingRate,
      analyses: summarizeAnalyses(analyses),
      decision, risk, geminiPayload,
      noSignal: false,
    });

    log("[ANALYZE] ✅ Analysis complete. Click GET SIGNAL to send to Gemini.");

    return res.json({
      success:   true,
      noSignal:  false,
      pair:      display,
      price,
      direction,
      trigger,
      score,
      grade:     decision.score?.grade || gradeScore(score),
      entryZone,
      slLevel,
      risk,
      diamond,
      reasons,
      fundingRate,
      logs,
    });

  } catch (err) {
    log(`[ANALYZE] ❌ Error: ${err.message}`);
    console.error(err);
    return res.status(500).json({ error: err.message, logs });
  }
};

function gradeScore(s) {
  return s >= 95 ? "ELITE" : s >= 85 ? "STRONG" : s >= 75 ? "STANDARD" : s >= 60 ? "WEAK" : "NO_SIGNAL";
}

function summarizeAnalyses(analyses) {
  const out = {};
  for (const [tf, a] of Object.entries(analyses)) {
    out[tf] = {
      trend:   a.structure.trend,
      bos:     a.structure.bos,
      choch:   a.structure.choch,
      sweep:   a.sweep.swept,
      fvgs:    a.fvgs.length,
      freshOBs: a.obs.filter((o) => o.fresh).length,
      rsi:     a.indicators.rsi,
      score:   a.score.total,
      candle:  a.indicators.candlePattern,
    };
  }
  return out;
}

function buildGeminiPayload({ pair, price, fundingRate, direction, trigger, score, reasons, entryZone, slLevel, diamond, analyses, risk }) {
  const a1h  = analyses.a1h;
  const a4h  = analyses.a4h;
  const a1d  = analyses.a1d;
  const a15m = analyses.a15m;
  const a5m  = analyses.a5m;

  const trim = (obs)  => obs.slice(0, 3).map((o) => ({ type: o.type, high: o.high, low: o.low, mid: o.mid, fresh: o.fresh, distPct: o.distPct }));
  const trimF = (fvgs) => fvgs.slice(0, 3).map((f) => ({ type: f.type, top: f.top, bottom: f.bottom, mid: f.mid, sizePct: f.sizePct }));
  const trimSD = (sds) => sds.slice(0, 2).map((s) => ({ type: s.type, high: s.high, low: s.low, freshness: s.freshness, inZone: s.inZone }));

  return {
    pair, current_price: price, funding_rate: fundingRate,
    account_size: parseFloat(process.env.ACCOUNT_SIZE || 100),
    smc_trigger: trigger, smc_direction: direction,
    smc_score: score, smc_reasons: reasons,
    entry_zone: {
      source: entryZone.source, limit_price: entryZone.entry,
      zone_low: entryZone.zoneLow, zone_high: entryZone.zoneHigh,
      entry_condition: entryZone.entryCondition,
      dist_from_cmp: entryZone.distFromEntry,
      note: entryZone.entryCondition === "LIMIT_WAIT"
        ? `Place LIMIT ${direction === "long" ? "BUY" : "SELL"} at ${entryZone.entry?.toFixed(8)}. Price must ${direction === "long" ? "fall" : "rise"} ${entryZone.distFromEntry?.toFixed(2)}% to fill.`
        : `Price already inside zone. Enter limit at ${entryZone.entry?.toFixed(8)} now.`,
    },
    stop_loss_level: slLevel,
    diamond_zone: diamond ? { exists: true, type: diamond.type, components: diamond.components, low: diamond.low, high: diamond.high } : { exists: false },
    pre_calculated_risk: risk,
    timeframes: {
      "1d":  { trend: a1d.structure.trend,  bos: a1d.structure.bos,  rsi: a1d.indicators.rsi, obs: trim(a1d.obs),  fvgs: trimF(a1d.fvgs),  sd_zones: trimSD(a1d.sdZones) },
      "4h":  { trend: a4h.structure.trend,  bos: a4h.structure.bos,  choch: a4h.structure.choch, sweep: a4h.sweep.swept, rsi: a4h.indicators.rsi, vol_spike: a4h.indicators.volSpike, obs: trim(a4h.obs), fvgs: trimF(a4h.fvgs), sd_zones: trimSD(a4h.sdZones) },
      "1h":  { trend: a1h.structure.trend,  bos: a1h.structure.bos,  choch: a1h.structure.choch, sweep: a1h.sweep.swept, in_zone: a1h.entryZone?.entryCondition === "LIMIT_NOW", rsi: a1h.indicators.rsi, rsi_div: "none", vol_spike: a1h.indicators.volSpike, candle: a1h.indicators.candlePattern, obs: trim(a1h.obs), fvgs: trimF(a1h.fvgs), sd_zones: trimSD(a1h.sdZones) },
      "15m": { trend: a15m.structure.trend, sweep: a15m.sweep.swept, rsi: a15m.indicators.rsi, candle: a15m.indicators.candlePattern, obs: trim(a15m.obs), fvgs: trimF(a15m.fvgs) },
      "5m":  { sweep: a5m.sweep.swept, rsi: a5m.indicators.rsi, candle: a5m.indicators.candlePattern, vol_spike: a5m.indicators.volSpike },
    },
    liquidity: { equal_highs: a1h.sweep.equalHighs || [], equal_lows: a1h.sweep.equalLows || [] },
  };
}
