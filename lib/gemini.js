/**
 * lib/gemini.js
 * Gemini 1.5 Flash integration
 * Two modes: signal generation + external signal validation
 */

const { GoogleGenerativeAI } = require("@google/generative-ai");

function getClient() {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  return genAI.getGenerativeModel({
    model: "gemini-1.5-flash",
    generationConfig: {
      temperature:     0.1,
      maxOutputTokens: 1024,
      responseMimeType: "application/json",
    },
  });
}

// ─────────────────────────────────────────
// Signal Generation
// ─────────────────────────────────────────

const SIGNAL_SYSTEM = `You are an ICT (Inner Circle Trader) institutional crypto futures analyst.

The SMC engine has identified a setup. Your job: validate and return the final trade parameters.

CRITICAL — ENTRY IS ALWAYS A LIMIT ORDER:
- LONG:  entry = BOTTOM of FVG or LOW of OB (price must retrace DOWN to fill)
- SHORT: entry = TOP of FVG or HIGH of OB (price must retrace UP to fill)
- entry_condition "LIMIT_WAIT" = price not yet at zone, place limit and wait
- entry_condition "LIMIT_NOW"  = price already inside zone, enter immediately

STOP LOSS:
- Long:  SL BELOW sweep_low wick — never inside OB/FVG
- Short: SL ABOVE sweep_high wick

TP LEVELS (ICT liquidity targets):
- TP1: nearest equal high/low (1.5x SL distance)
- TP2: next major swing (2.5x SL distance)
- TP3: HTF structure target (4x SL distance)

RETURN ONLY JSON:
{
  "mode": "scalp|swing|no_signal",
  "direction": "long|short|none",
  "entry": <float — limit order price>,
  "entry_condition": "LIMIT_WAIT|LIMIT_NOW|MARKET",
  "entry_note": "<one sentence: where price must go to fill>",
  "stop_loss": <float>,
  "tp_levels": [<tp1>, <tp2>, <tp3>],
  "leverage": <int 1-20>,
  "margin_usdt": <float>,
  "rr": <float>,
  "confidence": <int 70-98>,
  "entry_type": "sniper|strong|standard",
  "hard_invalidation": <float>,
  "soft_warning": <float>,
  "liquidity_above": <float|null>,
  "liquidity_below": <float|null>,
  "reasoning": "<3-4 sentences: sweep, entry zone, SL logic, TP target>"
}`;

async function generateSignal(payload) {
  const model  = getClient();
  const prompt = `${SIGNAL_SYSTEM}\n\nAnalyze this ICT/SMC data and return the trade signal JSON:\n\n${JSON.stringify(payload, null, 2)}`;

  const result = await model.generateContent(prompt);
  const raw    = result.response.text().trim().replace(/^```json\n?/, "").replace(/```$/, "").trim();
  const signal = JSON.parse(raw);

  // Normalize types
  signal.confidence     = parseInt(signal.confidence || 70);
  signal.leverage       = parseInt(signal.leverage   || 1);
  signal.rr             = parseFloat(signal.rr       || 1.5);
  signal.margin_usdt    = parseFloat(signal.margin_usdt || 0);
  signal.entry          = parseFloat(signal.entry    || 0);
  signal.stop_loss      = parseFloat(signal.stop_loss || 0);
  signal.entry_condition = signal.entry_condition || "LIMIT_WAIT";
  if (!Array.isArray(signal.tp_levels)) signal.tp_levels = [];

  // Fallback TPs from pre-calculated risk
  if (!signal.tp_levels.length && payload.pre_calculated_risk) {
    const r = payload.pre_calculated_risk;
    signal.tp_levels = [r.tp1, r.tp2, r.tp3].filter(Boolean);
  }

  return signal;
}

// ─────────────────────────────────────────
// Trade Re-check (no Gemini — rules only)
// Returns { action, reason, invalidated }
// ─────────────────────────────────────────

const CHECK_SYSTEM = `You are monitoring an active ICT futures trade. A structural alert has triggered.

RULES:
- CLOSE_NOW: hard invalidation breached, or opposite BOS on 1H confirmed
- CLOSE_WARNING: soft warning hit, or RSI extreme against trade direction
- HOLD: structure intact, OB still holds, minor noise only

Return ONLY JSON: {"action":"HOLD|CLOSE_WARNING|CLOSE_NOW","reason":"<one sentence>","invalidated":false}`;

async function checkTrade(tradeState, currentPrice, liveData) {
  const model = getClient();
  const payload = {
    trade: {
      pair:             tradeState.pair,
      direction:        tradeState.direction,
      entry:            tradeState.entry,
      stop_loss:        tradeState.stopLoss,
      hard_invalidation: tradeState.hardInvalidation,
      soft_warning:     tradeState.softWarning,
    },
    current_price: currentPrice,
    live_1h: {
      trend:   liveData.trend,
      rsi:     liveData.rsi,
      bos:     liveData.bos,
      pattern: liveData.pattern,
    },
  };

  const prompt  = `${CHECK_SYSTEM}\n\n${JSON.stringify(payload, null, 2)}`;
  const result  = await model.generateContent(prompt);
  const raw     = result.response.text().trim().replace(/^```json\n?/, "").replace(/```$/, "").trim();
  return JSON.parse(raw);
}

// ─────────────────────────────────────────
// External Signal Validator
// ─────────────────────────────────────────

const VALIDATE_SYSTEM = `You are an ICT/SMC signal validator. A user received a trading signal from an external source.
Your job: verify if this signal is technically sound based on current market structure.

CHECK:
1. Is the entry price near a valid OB, FVG, or S/D zone?
2. Is the SL beyond the nearest sweep low/high?
3. Does the TP target a liquidity pool or swing high/low?
4. Does the current market structure support the trade direction?
5. Is the RR ratio at least 1:1.5?

Return ONLY JSON:
{
  "verdict": "VALID|CAUTION|INVALID",
  "confidence": <int 0-100>,
  "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"],
  "risk_assessment": "<one sentence on risk>",
  "suggestion": "<one sentence improvement or confirmation>"
}`;

async function validateExternalSignal(payload) {
  const model  = getClient();
  const prompt = `${VALIDATE_SYSTEM}\n\nValidate this external signal:\n\n${JSON.stringify(payload, null, 2)}`;

  const result = await model.generateContent(prompt);
  const raw    = result.response.text().trim().replace(/^```json\n?/, "").replace(/```$/, "").trim();
  const data   = JSON.parse(raw);
  data.confidence = parseInt(data.confidence || 50);
  return data;
}

module.exports = { generateSignal, checkTrade, validateExternalSignal };
