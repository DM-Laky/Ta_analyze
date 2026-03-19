"""
ai.py — APEX V10 Gemini 1.5 Flash Integration
Two roles:
  1. generate_signal()    — Full analysis → structured trade signal
  2. evaluate_structure() — Called ONLY when hard monitor trigger fires
"""

import json
import asyncio
import logging
from typing import Optional
import google.generativeai as genai

logger = logging.getLogger("ai")

# ─────────────────────────────────────────────
# System Prompt — Signal Generation
# ─────────────────────────────────────────────

SIGNAL_SYSTEM_PROMPT = """You are APEX, an elite institutional crypto futures trader with 20 years of ICT/SMC experience.

You receive pre-analyzed multi-timeframe market data that has already passed through a strict confluence scoring system and pre-signal filters. Your job is to make the FINAL trading decision.

YOUR DECISION FRAMEWORK:

1. DETERMINE MODE:
   - SCALP: If 1H + 15m + 5m show tight alignment and sniper score >= 3. SL < 0.8%. Quick trade.
   - SWING: If 1D + 4H + 1H + 15m all align. SL 1-3%. Hold for days.
   - NO_SIGNAL: If you see any of these — avoid trade entirely:
     * Timeframe conflict (e.g., 1D bullish but 4H bearish)
     * Price between equal highs/equal lows with no clear direction
     * No fresh OB or FVG near price
     * RSI divergence against your direction
     * Confluence score < 75

2. ENTRY RULES (ICT-style):
   - Entry MUST be at or inside a fresh Order Block or Fair Value Gap
   - Never chase — if price is already far from OB/FVG, wait
   - Scalp entry: use 15m or 1H OB
   - Swing entry: use 4H or 1D OB

3. STOP LOSS RULES:
   - SL must be BELOW the OB low (long) or ABOVE the OB high (short)
   - SL must be BEYOND the nearest liquidity zone
   - Never place SL inside the OB — it will get swept

4. TP RULES (ICT liquidity targets):
   - TP1: First imbalance / minor swing high/low (1.5x SL distance)
   - TP2: Major swing point / equal highs/lows (2.5x SL distance)
   - TP3: Major HTF OB or previous structure (4x SL distance)

5. LEVERAGE (already calculated — validate and adjust):
   - Tighter SL = higher leverage is mathematically correct
   - Never recommend above 20x for futures
   - For $100 account: be conservative, max 15x

6. CONFIDENCE:
   - 90-100: All 5 TFs aligned, sniper entry, fresh OB, volume confirmation
   - 80-89: 4/5 TFs aligned, good OB, standard entry
   - 70-79: 3/5 aligned — only take if RR > 3
   - Below 70: Do NOT signal

RETURN ONLY VALID JSON. NO MARKDOWN. NO PREAMBLE. NO EXPLANATION OUTSIDE JSON.

{
  "mode": "scalp|swing|no_signal",
  "direction": "long|short|none",
  "entry": <float|null>,
  "stop_loss": <float|null>,
  "tp_levels": [<tp1_float>, <tp2_float>, <tp3_float>],
  "leverage": <int>,
  "margin_usdt": <float>,
  "rr": <float>,
  "confidence": <int 0-100>,
  "entry_type": "sniper|strong|standard|none",
  "hard_invalidation": <float|null>,
  "soft_warning": <float|null>,
  "ob_zone": [<low_float>, <high_float>] or null,
  "liquidity_above": <float|null>,
  "liquidity_below": <float|null>,
  "confluence_score": <int>,
  "reasoning": "<max 4 sentences: TF alignment, entry logic, risk, key invalidation>"
}"""


# ─────────────────────────────────────────────
# System Prompt — Monitor Re-evaluation
# ─────────────────────────────────────────────

MONITOR_SYSTEM_PROMPT = """You are APEX trade monitor. A trade is active and a structural break has been detected.

Analyze the situation and decide: HOLD, CLOSE_WARNING, or CLOSE_NOW.

RULES:
- CLOSE_NOW: Hard invalidation level breached, or opposite BOS on 1H confirmed
- CLOSE_WARNING: Soft warning level hit, or RSI extreme against trade
- HOLD: Minor noise, structure still intact, OB still holds

Return ONLY JSON:
{
  "action": "HOLD|CLOSE_WARNING|CLOSE_NOW",
  "reason": "<one sentence>",
  "invalidated": <bool>
}"""


# ─────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────

class GeminiAnalyst:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._signal_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SIGNAL_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.15,
                max_output_tokens=1024,
                response_mime_type="application/json",
            )
        )
        self._monitor_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=MONITOR_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
                response_mime_type="application/json",
            )
        )

    def _clean_json(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    async def _call(self, model, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        return response.text

    async def generate_signal(self, payload: dict, risk_data: dict) -> Optional[dict]:
        """
        Send full analysis payload to Gemini → receive trade signal.
        risk_data contains pre-calculated position sizing for Gemini to validate.
        """
        try:
            combined = {
                **payload,
                "pre_calculated_risk": risk_data,
                "instruction": (
                    "Analyze this complete multi-timeframe SMC/ICT data. "
                    "The risk calculations are provided as a reference — you may adjust leverage "
                    "by ±2x based on your assessment. "
                    "Return your signal as JSON only."
                )
            }
            prompt = json.dumps(combined, indent=2, default=str)
            raw    = await self._call(self._signal_model, prompt)
            signal = json.loads(self._clean_json(raw))

            # Normalize types
            signal["confidence"]  = int(signal.get("confidence", 0))
            signal["leverage"]    = int(signal.get("leverage", 1))
            signal["rr"]          = float(signal.get("rr", 0))
            signal["margin_usdt"] = float(signal.get("margin_usdt", 0))
            if "tp_levels" not in signal or not isinstance(signal["tp_levels"], list):
                signal["tp_levels"] = []
            signal["pair"]            = payload.get("pair")
            signal["confluence_score"] = payload.get("confluence_score", 0)

            return signal

        except json.JSONDecodeError as e:
            logger.error(f"[AI] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"[AI] generate_signal error: {e}")
            return None

    async def evaluate_structure(self, trade_state: dict,
                                  current_price: float,
                                  live_analysis: dict) -> dict:
        """
        Called ONLY when hard monitor trigger fires.
        Minimal Gemini call — fast verdict.
        """
        try:
            prompt_data = {
                "trade": {
                    "pair":             trade_state.get("pair"),
                    "direction":        trade_state.get("direction"),
                    "entry":            trade_state.get("entry"),
                    "stop_loss":        trade_state.get("stop_loss"),
                    "hard_invalidation": trade_state.get("hard_invalidation"),
                    "soft_warning":     trade_state.get("soft_warning"),
                    "ob_zone":          trade_state.get("ob_zone"),
                },
                "current_price": current_price,
                "trigger": live_analysis.get("trigger"),
                "live_1h": {
                    "trend":   live_analysis.get("trend"),
                    "rsi":     live_analysis.get("rsi"),
                    "bos":     live_analysis.get("bos"),
                    "pattern": live_analysis.get("pattern"),
                }
            }
            raw    = await self._call(self._monitor_model, json.dumps(prompt_data, default=str))
            result = json.loads(self._clean_json(raw))
            return result

        except Exception as e:
            logger.error(f"[AI] evaluate_structure error: {e}")
            return {"action": "HOLD", "reason": "AI evaluation failed — defaulting to hold", "invalidated": False}
