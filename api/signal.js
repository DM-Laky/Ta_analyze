/**
 * api/signal.js
 * POST /api/signal
 * Reads saved analysis from KV, sends to Gemini, saves signal to KV
 */

const { generateSignal }              = require("../lib/gemini");
const { getLastAnalysis, setTradeState } = require("../lib/kv");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const logs = [];
  const log  = (msg) => { logs.push({ ts: new Date().toISOString().split("T")[1].split(".")[0], msg }); console.log(msg); };

  try {
    log("[SIGNAL] Loading saved analysis from KV...");
    const saved = await getLastAnalysis();

    if (!saved) return res.status(400).json({ error: "No analysis found. Run Analyze first.", logs });
    if (saved.noSignal) return res.json({ success: true, noSignal: true, reason: saved.decision?.reasons?.join(" | ") || "No SMC signal found", logs });

    const { pair, price, geminiPayload, risk, decision } = saved;
    if (!geminiPayload) return res.status(400).json({ error: "No Gemini payload found. Re-run Analyze.", logs });

    log(`[SIGNAL] Sending ${pair} analysis to Gemini 1.5 Flash...`);
    log(`[SIGNAL] SMC Trigger: ${geminiPayload.smc_trigger} | Score: ${geminiPayload.smc_score}/100`);
    log(`[SIGNAL] Entry zone: ${geminiPayload.entry_zone?.source} @ ${geminiPayload.entry_zone?.limit_price?.toFixed ? geminiPayload.entry_zone.limit_price.toFixed(8) : geminiPayload.entry_zone?.limit_price}`);
    log(`[SIGNAL] Condition: ${geminiPayload.entry_zone?.entry_condition} | Note: ${geminiPayload.entry_zone?.note}`);

    const signal = await generateSignal(geminiPayload);
    if (!signal) return res.status(500).json({ error: "Gemini returned null response", logs });

    const mode       = signal.mode || "no_signal";
    const confidence = signal.confidence || 0;
    const direction  = signal.direction  || "none";
    const entryType  = signal.entry_type || "standard";

    log(`[SIGNAL] ✅ Gemini: ${mode.toUpperCase()} ${direction.toUpperCase()} | Confidence: ${confidence}% | ${entryType.toUpperCase()}`);
    log(`[SIGNAL] Entry: ${signal.entry} | SL: ${signal.stop_loss} | Cond: ${signal.entry_condition}`);
    log(`[SIGNAL] ${signal.entry_note || ""}`);
    if (signal.tp_levels?.length)
      log("[SIGNAL] " + signal.tp_levels.map((tp, i) => `TP${i + 1}: ${tp}`).join(" | "));
    log(`[SIGNAL] Lev: ${signal.leverage}x | Margin: $${signal.margin_usdt} | RR: 1:${signal.rr}`);
    log(`[SIGNAL] Reasoning: ${signal.reasoning}`);

    if (mode === "no_signal" || confidence < 65) {
      log(`[SIGNAL] ⚠️ No valid setup (conf=${confidence}%)`);
      return res.json({ success: true, noSignal: true, reason: signal.reasoning || "Low confidence", confidence, logs });
    }

    // Save trade state to KV
    const tp = signal.tp_levels || [];
    const tradeState = {
      active:           true,
      inTrade:          false,
      pair,
      direction,
      mode,
      entry:            signal.entry,
      stopLoss:         signal.stop_loss,
      tp1:              tp[0] || null,
      tp2:              tp[1] || null,
      tp3:              tp[2] || null,
      leverage:         signal.leverage,
      marginUsdt:       signal.margin_usdt,
      rr:               signal.rr,
      confidence,
      entryCondition:   signal.entry_condition,
      entryNote:        signal.entry_note || "",
      entryType,
      hardInvalidation: signal.hard_invalidation,
      softWarning:      signal.soft_warning,
      liquidityAbove:   signal.liquidity_above,
      liquidityBelow:   signal.liquidity_below,
      reasoning:        signal.reasoning,
      score:            geminiPayload.smc_score,
      trigger:          geminiPayload.smc_trigger,
      signalTs:         Date.now(),
      tp1Hit:           false,
      beMoved:          false,
      monitorCycles:    0,
    };
    await setTradeState(tradeState);
    log("[SIGNAL] ✅ Trade state saved to KV.");

    return res.json({ success: true, noSignal: false, signal: tradeState, logs });

  } catch (err) {
    log(`[SIGNAL] ❌ ${err.message}`);
    console.error(err);
    return res.status(500).json({ error: err.message, logs });
  }
};
