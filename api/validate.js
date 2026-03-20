/**
 * api/validate.js
 * POST /api/validate
 * Body: { pair, entry, sl, tp, direction }
 * Fetches live data, runs SMC check, sends to Gemini for validation
 */

const { fetchOHLCV, fetchTicker }       = require("../lib/binance");
const { analyzeTimeframe, calculateRisk } = require("../lib/engine");
const { validateExternalSignal }         = require("../lib/gemini");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const logs = [];
  const log  = (msg) => { logs.push({ ts: new Date().toISOString().split("T")[1].split(".")[0], msg }); console.log(msg); };

  try {
    const { pair, entry, sl, tp, direction } = req.body;
    if (!pair || !entry || !sl || !tp) {
      return res.status(400).json({ error: "pair, entry, sl, tp required", logs });
    }

    const symbol  = pair.replace("/", "").toUpperCase();
    const display = symbol.replace("USDT", "/USDT");
    const eFloat  = parseFloat(entry);
    const slFloat = parseFloat(sl);
    const tpFloat = parseFloat(tp);
    const dir     = (direction || (eFloat > slFloat ? "long" : "short")).toLowerCase();

    log(`[VALIDATE] ═══ Validating external signal: ${display} ═══`);
    log(`[VALIDATE] Direction: ${dir.toUpperCase()} | Entry: ${eFloat} | SL: ${slFloat} | TP: ${tpFloat}`);

    // Fetch live data (1H + 4H only — faster, stays under 10s)
    const [ticker, c1h, c4h] = await Promise.all([
      fetchTicker(symbol),
      fetchOHLCV(symbol, "1h", 100),
      fetchOHLCV(symbol, "4h", 100),
    ]);

    const price = ticker.price;
    log(`[VALIDATE] Current price: ${price} | Change 24h: ${ticker.change?.toFixed(2)}%`);

    // Quick SMC analysis
    const a1h = analyzeTimeframe(c1h, "1h", dir === "long" ? "bullish" : "bearish");
    const a4h = analyzeTimeframe(c4h, "4h", dir === "long" ? "bullish" : "bearish");

    log(`[VALIDATE] 1H — Trend:${a1h.structure.trend.toUpperCase()} BOS:${a1h.structure.bos ? "✓" : "✗"} Sweep:${a1h.sweep.swept ? "✓" : "✗"} FVGs:${a1h.fvgs.length} OBs:${a1h.obs.filter((o) => o.fresh).length}`);
    log(`[VALIDATE] 4H — Trend:${a4h.structure.trend.toUpperCase()} BOS:${a4h.structure.bos ? "✓" : "✗"} Sweep:${a4h.sweep.swept ? "✓" : "✗"}`);

    // Check if entry is near any zone
    const nearOB  = a1h.obs.some((o) => o.low <= eFloat && eFloat <= o.high);
    const nearFVG = a1h.fvgs.some((f) => f.bottom <= eFloat && eFloat <= f.top);
    const nearSD  = a1h.sdZones.some((s) => s.low <= eFloat && eFloat <= s.high);
    log(`[VALIDATE] Entry proximity — Near OB: ${nearOB ? "✓" : "✗"} | Near FVG: ${nearFVG ? "✓" : "✗"} | Near S/D: ${nearSD ? "✓" : "✗"}`);

    // RR check
    const slDist = Math.abs(eFloat - slFloat);
    const tpDist = Math.abs(tpFloat - eFloat);
    const rr     = Math.round(tpDist / slDist * 100) / 100;
    log(`[VALIDATE] RR: 1:${rr} | SL dist: ${(slDist / eFloat * 100).toFixed(3)}%`);

    // Build Gemini payload
    const payload = {
      pair:           display,
      current_price:  price,
      signal: {
        direction: dir, entry: eFloat, sl: slFloat, tp: tpFloat,
        rr, slDistPct: slDist / eFloat * 100,
      },
      market_structure: {
        "1h": {
          trend:   a1h.structure.trend,
          bos:     a1h.structure.bos,
          choch:   a1h.structure.choch,
          sweep:   a1h.sweep.swept,
          sweepLevel: a1h.sweep.sweepLevel,
          rsi:     a1h.indicators.rsi,
          candle:  a1h.indicators.candlePattern,
          obs:     a1h.obs.slice(0, 3).map((o) => ({ type: o.type, high: o.high, low: o.low, fresh: o.fresh })),
          fvgs:    a1h.fvgs.slice(0, 3).map((f) => ({ type: f.type, top: f.top, bottom: f.bottom })),
          sd:      a1h.sdZones.slice(0, 2).map((s) => ({ type: s.type, high: s.high, low: s.low, freshness: s.freshness })),
        },
        "4h": {
          trend:   a4h.structure.trend,
          bos:     a4h.structure.bos,
          sweep:   a4h.sweep.swept,
          rsi:     a4h.indicators.rsi,
        },
      },
      entry_zone_check: { nearOB, nearFVG, nearSD },
    };

    log("[VALIDATE] Sending to Gemini for validation...");
    const verdict = await validateExternalSignal(payload);

    log(`[VALIDATE] Gemini verdict: ${verdict.verdict} | Confidence: ${verdict.confidence}%`);
    for (const r of verdict.reasons || []) log(`[VALIDATE] ${r}`);
    log(`[VALIDATE] Risk: ${verdict.risk_assessment}`);
    log(`[VALIDATE] Suggestion: ${verdict.suggestion}`);

    return res.json({
      success:  true,
      pair:     display,
      price,
      signal:   { direction: dir, entry: eFloat, sl: slFloat, tp: tpFloat, rr },
      verdict,
      entryZoneCheck: { nearOB, nearFVG, nearSD },
      structure: {
        trend1h: a1h.structure.trend, bos1h: a1h.structure.bos,
        trend4h: a4h.structure.trend, sweep1h: a1h.sweep.swept,
      },
      logs,
    });

  } catch (err) {
    log(`[VALIDATE] ❌ ${err.message}`);
    console.error(err);
    return res.status(500).json({ error: err.message, logs });
  }
};
