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

SIGNAL_PROMPT = """You are an ICT (Inner Circle Trader) trained institutional crypto futures analyst.

The system has already detected an SMC signal using liquidity sweep, BOS/CHoCH, Order Blocks, and Fair Value Gaps.
Your job is to VALIDATE and REFINE the entry — specifically the limit order price within the identified zone.

YOUR RULES:
1. The SMC engine has already confirmed the direction. Trust it unless something is fundamentally wrong.
2. Refine the entry price within the provided entry_zone [low, high]:
   - For LONG: entry near the BOTTOM of the OB or FVG (best value)
   - For SHORT: entry near the TOP of the OB or FVG (best value)
3. Stop loss MUST be below the sweep_low (long) or above sweep_high (short) — never inside the OB or FVG.
4. TP levels use liquidity pools as targets:
   - TP1: nearest equal high/low (1.5x SL distance)
   - TP2: next major swing (2.5x SL distance)
   - TP3: HTF OB or major structure (4x SL distance)
5. Mode: SCALP if SL < 0.8% and 1H sweep. SWING if 4H/1D sweep or SL > 1%.
6. Confidence: 90-98 for 4H/1D sweep + all TFs aligned. 80-89 for 1H sweep. 70-79 for BOS+OB+FVG only.
7. Only return no_signal if price is trapped between equal highs AND equal lows with no clear direction.

RETURN ONLY VALID JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

{
  "mode": "scalp|swing|no_signal",
  "direction": "long|short|none",
  "entry": <precise_float>,
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
  "reasoning": "<3-4 sentences: sweep location, entry zone, SL logic, TP target>"
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
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=SIGNAL_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, max_output_tokens=1024,
                response_mime_type="application/json",
            )
        )
        self._monitor_model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
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

            signal["confidence"]  = int(signal.get("confidence", 70))
            signal["leverage"]    = int(signal.get("leverage", 1))
            signal["rr"]          = float(signal.get("rr", 1.5))
            signal["margin_usdt"] = float(signal.get("margin_usdt", 0))
            signal["entry"]       = float(signal.get("entry", 0))
            signal["stop_loss"]   = float(signal.get("stop_loss", 0))
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
