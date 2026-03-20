/**
 * api/check.js
 * POST /api/check
 * Manual re-check of active trade against current market structure
 * Replaces the automated 5-min loop for Vercel serverless
 */

const { fetchOHLCV, fetchTicker }              = require("../lib/binance");
const { analyzeTimeframe }                     = require("../lib/engine");
const { checkTrade }                           = require("../lib/gemini");
const { getTradeState, setTradeState }         = require("../lib/kv");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const logs = [];
  const log  = (msg) => { logs.push({ ts: new Date().toISOString().split("T")[1].split(".")[0], msg }); console.log(msg); };

  try {
    log("[CHECK] Loading active trade state...");
    const state = await getTradeState();

    if (!state?.active) {
      return res.json({ success: true, status: "NO_TRADE", message: "No active signal to check.", logs });
    }

    const { pair, direction, entry, stopLoss, hardInvalidation, softWarning, tp1, tp2, tp3, tp1Hit } = state;
    const symbol = pair.replace("/", "");

    log(`[CHECK] Trade: ${pair} ${direction.toUpperCase()} @ ${entry}`);

    // Fetch latest price and 1H candles
    const [ticker, c1h, c15m] = await Promise.all([
      fetchTicker(symbol),
      fetchOHLCV(symbol, "1h", 50),
      fetchOHLCV(symbol, "15m", 50),
    ]);

    const price = ticker.price;
    log(`[CHECK] CMP: ${price}`);

    // PnL estimate
    const pnlPct  = direction === "long"
      ? (price - entry) / entry * 100
      : (entry - price) / entry * 100;
    const pnlSign = pnlPct >= 0 ? "+" : "";
    log(`[CHECK] PnL estimate: ${pnlSign}${pnlPct.toFixed(3)}%`);

    // TP hit detection
    const tpHits = [];
    if (!tp1Hit && tp1) {
      const tp1Hit_ = direction === "long" ? price >= tp1 : price <= tp1;
      if (tp1Hit_) {
        tpHits.push("TP1");
        await setTradeState({ ...state, tp1Hit: true, beMoved: true });
        log(`[CHECK] 🎯 TP1 HIT! SL moved to break-even @ ${entry}`);
      }
    }
    if (tp2) {
      const tp2Hit = direction === "long" ? price >= tp2 : price <= tp2;
      if (tp2Hit) tpHits.push("TP2");
    }
    if (tp3) {
      const tp3Hit = direction === "long" ? price >= tp3 : price <= tp3;
      if (tp3Hit) tpHits.push("TP3");
    }

    // Rules-based checks
    const a1h  = analyzeTimeframe(c1h,  "1h");
    const a15m = analyzeTimeframe(c15m, "15m");
    const rsi1h    = a1h.indicators.rsi;
    const trend1h  = a1h.structure.trend;
    const bos1h    = a1h.structure.bos;
    const bosDir1h = a1h.structure.bosDir;
    const candle15 = a15m.indicators.candlePattern;

    // Hard triggers
    let hardTrigger = null;
    if (hardInvalidation) {
      if (direction === "long"  && price < hardInvalidation) hardTrigger = { trigger: "HARD_INVALIDATION", detail: `Price ${price} < invalidation ${hardInvalidation}` };
      if (direction === "short" && price > hardInvalidation) hardTrigger = { trigger: "HARD_INVALIDATION", detail: `Price ${price} > invalidation ${hardInvalidation}` };
    }
    if (!hardTrigger && bos1h) {
      if (direction === "long"  && bosDir1h === "bearish") hardTrigger = { trigger: "OPPOSITE_BOS_1H", detail: "Bearish BOS on 1H — long invalidated" };
      if (direction === "short" && bosDir1h === "bullish") hardTrigger = { trigger: "OPPOSITE_BOS_1H", detail: "Bullish BOS on 1H — short invalidated" };
    }
    if (!hardTrigger && direction === "long"  && rsi1h > 78) hardTrigger = { trigger: "RSI_OVERBOUGHT", detail: `RSI 1H overbought (${rsi1h.toFixed(0)})` };
    if (!hardTrigger && direction === "short" && rsi1h < 22) hardTrigger = { trigger: "RSI_OVERSOLD",   detail: `RSI 1H oversold (${rsi1h.toFixed(0)})` };

    // Soft warnings
    const warnings = [];
    if (softWarning) {
      if (direction === "long"  && price < softWarning) warnings.push(`⚠️ Approaching soft warning @ ${softWarning}`);
      if (direction === "short" && price > softWarning) warnings.push(`⚠️ Approaching soft warning @ ${softWarning}`);
    }
    const BEAR_15 = new Set(["shooting_star","bearish_pinbar","bearish_engulfing","evening_star"]);
    const BULL_15 = new Set(["hammer","bullish_pinbar","bullish_engulfing","morning_star"]);
    if (direction === "long"  && BEAR_15.has(candle15)) warnings.push(`⚠️ Counter-trend 15m pattern: ${candle15}`);
    if (direction === "short" && BULL_15.has(candle15)) warnings.push(`⚠️ Counter-trend 15m pattern: ${candle15}`);
    if (direction === "long"  && rsi1h > 68 && rsi1h < 78) warnings.push(`⚠️ RSI 1H approaching overbought (${rsi1h.toFixed(0)})`);
    if (direction === "short" && rsi1h < 32 && rsi1h > 22) warnings.push(`⚠️ RSI 1H approaching oversold (${rsi1h.toFixed(0)})`);

    for (const w of warnings) log(`[CHECK] ${w}`);

    let status = "HOLD", geminiVerdict = null;

    if (hardTrigger) {
      log(`[CHECK] 🚨 HARD TRIGGER: ${hardTrigger.trigger} — ${hardTrigger.detail}`);
      log("[CHECK] Escalating to Gemini for final verdict...");
      try {
        geminiVerdict = await checkTrade(state, price, {
          trend: trend1h, rsi: rsi1h, bos: bos1h, pattern: candle15,
        });
        const action = geminiVerdict.action || "HOLD";
        log(`[CHECK] Gemini verdict: ${action} — ${geminiVerdict.reason}`);
        status = action;
        if (action === "CLOSE_NOW")      log("[CHECK] ⛔ CLOSE TRADE NOW");
        else if (action === "CLOSE_WARNING") log("[CHECK] ⚠️ CLOSE WARNING");
        else                              log("[CHECK] ✅ Gemini says HOLD");
      } catch (e) {
        log(`[CHECK] ❌ Gemini check failed: ${e.message}. Defaulting HOLD.`);
        status = "HOLD";
      }
    } else {
      const structOk = (direction === "long" && trend1h === "bullish") ||
                       (direction === "short" && trend1h === "bearish");
      status = "HOLD";
      log(`[CHECK] ✅ No hard triggers | 1H trend: ${trend1h.toUpperCase()} | RSI: ${rsi1h.toFixed(1)} | Status: HOLD`);
    }

    // Update state
    const cycles = (state.monitorCycles || 0) + 1;
    await setTradeState({
      ...state,
      monitorCycles: cycles,
      lastCheck:     Date.now(),
      lastStatus:    status,
      lastPrice:     price,
      lastPnlPct:    Math.round(pnlPct * 1000) / 1000,
    });

    log(`[CHECK] Check cycle #${cycles} complete.`);

    return res.json({
      success:  true,
      status,
      price,
      pnlPct:   Math.round(pnlPct * 1000) / 1000,
      pnlSign,
      tpHits,
      warnings,
      hardTrigger,
      geminiVerdict,
      rsi1h:    Math.round(rsi1h * 100) / 100,
      trend1h,
      cycle:    cycles,
      beMoved:  state.beMoved || tpHits.includes("TP1"),
      logs,
    });

  } catch (err) {
    log(`[CHECK] ❌ ${err.message}`);
    console.error(err);
    return res.status(500).json({ error: err.message, logs });
  }
};
