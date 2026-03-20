"""
ai.py — APEX V10 Gemini Integration (Rewrite)
Gemini receives pre-decided SMC signal with entry zone.
It validates and returns the precise limit order price.
"""

import json
import asyncio
import logging
from typing import Optional
import google.generativeai as genai

logger = logging.getLogger("ai")

SIGNAL_PROMPT = """You are an ICT (Inner Circle Trader) institutional crypto futures analyst.

The SMC engine has identified a setup using liquidity sweep, BOS/CHoCH, Order Blocks, FVGs, and Supply/Demand zones.
The engine has already calculated the LIMIT ORDER entry price.
Your job: validate it and return the final precise trade parameters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — ENTRY IS ALWAYS A LIMIT ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ICT entries are NEVER market orders (unless price is already inside the zone).
The entry price = where price must RETRACE TO before filling.

LONG  setup: price swept lows → bounced up → FVG/OB left BELOW current price
             → Entry = BOTTOM of FVG or LOW of OB (price must come BACK DOWN to fill)
             → Example: Sweep at 0.0969, bounce to 0.1002, FVG at 0.0975–0.0985
               → Limit BUY at 0.0976 (FVG bottom)
               → Wait for price to retrace to 0.0976, then it fills and runs up

SHORT setup: price swept highs → dropped → FVG/OB left ABOVE current price
             → Entry = TOP of FVG or HIGH of OB (price must rally BACK UP to fill)
             → Example: Sweep at 1.150, drop to 1.080, FVG at 1.110–1.120
               → Limit SELL at 1.119 (FVG top)
               → Wait for price to retrace to 1.119, then it fills and drops

ENTRY CONDITION RULES:
- "LIMIT_WAIT" → place limit order now, wait for retrace. DO NOT adjust entry toward current price.
- "LIMIT_NOW"  → price is inside zone, enter at zone bottom (long) or top (short) immediately.
- "MARKET"     → no clean zone, use current price only as last resort.

You MUST return the entry price within the provided entry_zone [zone_low, zone_high].
If entry_condition is LIMIT_WAIT: entry must be ≤ zone_high (long) or ≥ zone_low (short).

STOP LOSS:
- Long:  SL must be BELOW the sweep_low wick — below ALL wicks, not just the zone.
- Short: SL must be ABOVE the sweep_high wick.
- Never place SL inside the FVG or OB.

TAKE PROFIT (ICT liquidity targets):
- TP1: nearest equal high/low (1.5x SL distance from entry)
- TP2: next major swing point (2.5x SL distance)
- TP3: HTF structure target (4x SL distance)

MODE:
- SCALP if SL distance < 0.8% and trigger is 1H/15m sweep
- SWING if SL distance ≥ 1% or trigger is 4H/1D sweep

CONFIDENCE:
- 90–98: sweep + all TFs aligned + diamond zone + volume
- 80–89: sweep + 3/4 TFs aligned + OB/FVG confirmed
- 70–79: BOS+OB+FVG only, no sweep, clean structure
- Below 70: return no_signal

RETURN ONLY VALID JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

{
  "mode": "scalp|swing|no_signal",
  "direction": "long|short|none",
  "entry": <float — limit order price, NOT current price>,
  "entry_condition": "LIMIT_WAIT|LIMIT_NOW|MARKET",
  "entry_note": "<one sentence explaining where price must go to fill>",
  "stop_loss": <float>,
  "tp_levels": [<tp1>, <tp2>, <tp3>],
  "leverage": <int 1-20>,
  "margin_usdt": <float>,
  "rr": <float>,
  "confidence": <int 70-98>,
  "entry_type": "sniper|strong|standard",
  "hard_invalidation": <float>,
  "soft_warning": <float>,
  "ob_zone": [<low>, <high>],
  "liquidity_above": <float|null>,
  "liquidity_below": <float|null>,
  "reasoning": "<3-4 sentences: sweep location, entry zone type, where price must retrace, SL logic, key TP>"
}"""

MONITOR_PROMPT = """You are monitoring an active ICT futures trade. A structural alert fired.
CLOSE_NOW: hard invalidation breached or opposite 1H BOS confirmed.
CLOSE_WARNING: soft warning hit or RSI extreme against trade.
HOLD: minor noise, structure intact.
Return ONLY: {"action": "HOLD|CLOSE_WARNING|CLOSE_NOW", "reason": "<one sentence>", "invalidated": <bool>}"""


class GeminiAnalyst:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._signal_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SIGNAL_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, max_output_tokens=1024,
                response_mime_type="application/json",
            )
        )
        self._monitor_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=MONITOR_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.05, max_output_tokens=256,
                response_mime_type="application/json",
            )
        )

    def _clean(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    async def _call(self, model, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        return resp.text

    async def generate_signal(self, payload: dict) -> Optional[dict]:
        try:
            prompt = ("Validate and refine this ICT/SMC signal. "
                      "Return precise limit order entry within the entry_zone.\n\n"
                      + json.dumps(payload, indent=2, default=str))
            raw    = await self._call(self._signal_model, prompt)
            signal = json.loads(self._clean(raw))

            signal["confidence"]     = int(signal.get("confidence", 70))
            signal["leverage"]        = int(signal.get("leverage", 1))
            signal["rr"]              = float(signal.get("rr", 1.5))
            signal["margin_usdt"]     = float(signal.get("margin_usdt", 0))
            signal["entry"]           = float(signal.get("entry", 0))
            signal["stop_loss"]       = float(signal.get("stop_loss", 0))
            signal["entry_condition"] = signal.get("entry_condition", "LIMIT_WAIT")
            signal["entry_note"]      = signal.get("entry_note", "")
            if not isinstance(signal.get("tp_levels"), list):
                signal["tp_levels"] = []
            if not signal["tp_levels"]:
                r = payload.get("pre_calculated_risk", {})
                signal["tp_levels"] = [r.get("tp1"), r.get("tp2"), r.get("tp3")]
            signal["pair"] = payload.get("pair")
            return signal
        except Exception as e:
            logger.error(f"[AI] generate_signal error: {e}")
            return None

    async def evaluate_structure(self, trade_state: dict,
                                  current_price: float,
                                  live_analysis: dict) -> dict:
        try:
            data = {
                "trade": {k: trade_state.get(k) for k in
                          ["pair","direction","entry","stop_loss",
                           "hard_invalidation","soft_warning"]},
                "current_price": current_price,
                "trigger":       live_analysis.get("trigger"),
                "live_1h":       {k: live_analysis.get(k) for k in
                                  ["trend","rsi","bos","pattern"]},
            }
            raw = await self._call(self._monitor_model, json.dumps(data, default=str))
            return json.loads(self._clean(raw))
        except Exception as e:
            logger.error(f"[AI] evaluate_structure error: {e}")
            return {"action": "HOLD", "reason": "AI check failed", "invalidated": False}
