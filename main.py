"""
main.py — APEX V10 Trading Terminal Server
FastAPI + WebSocket + IP display + full route handling
"""

import asyncio
import json
import logging
import os
import socket
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from engine import ExchangeManager, run_full_analysis, calculate_risk
from ai import GeminiAnalyst
from monitor import TradeMonitor
from state import (
    load_trade_state, save_trade_state, apply_signal_to_state,
    mark_trade_entered, full_reset, get_pnl_pct, get_pnl_usdt,
    save_analysis_cache, load_analysis_cache, is_in_trade
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

load_dotenv()

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "100"))
RISK_PCT           = float(os.getenv("RISK_PCT", "0.015"))
PORT               = int(os.getenv("PORT", "8080"))

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────

class AppState:
    ws_clients:    list[WebSocket] = []
    log_buffer:    list[dict]      = []
    exchange:      Optional[ExchangeManager] = None
    ai:            Optional[GeminiAnalyst]   = None
    monitor:       Optional[TradeMonitor]    = None
    server_ip:     str = "unknown"
    analysis_busy: bool = False

app_state = AppState()


# ─────────────────────────────────────────────
# IP Detection
# ─────────────────────────────────────────────

def get_public_ip() -> str:
    try:
        import urllib.request
        return urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────────────────────────────────
# WebSocket Broadcasting
# ─────────────────────────────────────────────

async def ws_log(message: str, level: str = "INFO"):
    ts    = datetime.utcnow().strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "msg": message}
    app_state.log_buffer.append(entry)
    if len(app_state.log_buffer) > 600:
        app_state.log_buffer = app_state.log_buffer[-600:]

    logger.info(message) if level != "ERROR" else logger.error(message)

    payload = json.dumps({"type": "log", "data": entry})
    await _broadcast_raw(payload)


async def ws_broadcast(event_type: str, data: dict):
    payload = json.dumps({"type": event_type, "data": data}, default=str)
    await _broadcast_raw(payload)


async def _broadcast_raw(payload: str):
    dead = []
    for ws in app_state.ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in app_state.ws_clients:
            app_state.ws_clients.remove(ws)


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    app_state.server_ip = get_public_ip()

    print("\n" + "═" * 60)
    print("  APEX V10 — AI Crypto Trading Terminal")
    print("═" * 60)
    print(f"  Server IP  : {app_state.server_ip}")
    print(f"  Port       : {PORT}")
    print(f"  Account    : ${ACCOUNT_SIZE}")
    print(f"  Risk/Trade : {RISK_PCT*100}%")
    print("═" * 60)
    print(f"\n  ⚠️  Add this IP to Binance API whitelist:")
    print(f"  ➜  {app_state.server_ip}\n")
    print("=" * 60 + "\n")

    app_state.exchange = ExchangeManager(BINANCE_API_KEY, BINANCE_API_SECRET)
    app_state.ai       = GeminiAnalyst(GEMINI_API_KEY)
    app_state.monitor  = TradeMonitor(
        exchange=app_state.exchange,
        ai_analyst=app_state.ai,
        log_fn=ws_log,
        broadcast_fn=ws_broadcast,
        interval=300,
    )

    # Resume monitoring if trade was active on restart
    if is_in_trade():
        state = load_trade_state()
        logger.info(f"[STARTUP] Resuming monitor for {state.get('pair')}...")
        await app_state.monitor.start()

    yield

    # ── Shutdown ──
    logger.info("[SHUTDOWN] Stopping services...")
    await app_state.monitor.stop()
    await app_state.exchange.close()
    logger.info("[SHUTDOWN] Clean exit.")


app = FastAPI(title="APEX V10 Trading Terminal", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ─────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    trade_state = load_trade_state()
    cache       = load_analysis_cache()
    return {
        "server_ip":     app_state.server_ip,
        "account_size":  ACCOUNT_SIZE,
        "risk_pct":      RISK_PCT,
        "trade_state":   trade_state,
        "has_cache":     cache is not None,
        "analysis_busy": app_state.analysis_busy,
        "monitor_running": app_state.monitor._running if app_state.monitor else False,
    }


@app.get("/api/top_pairs")
async def get_top_pairs():
    """Fetch top 10 Binance Futures pairs by volume."""
    try:
        await ws_log("[SCREENER] Fetching top 10 volume pairs from Binance Futures...")
        pairs = await app_state.exchange.get_top_volume_pairs(10)
        await ws_log(f"[SCREENER] ✅ Fetched {len(pairs)} pairs.")
        return {"pairs": pairs}
    except Exception as e:
        await ws_log(f"[SCREENER] ❌ Failed: {e}", level="ERROR")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/analyze")
async def start_analysis(request: Request):
    """
    Start full 5-TF analysis + Gemini signal generation for a pair.
    """
    if app_state.analysis_busy:
        return JSONResponse(status_code=409,
                            content={"error": "Analysis already running. Please wait."})

    body = await request.json()
    pair = body.get("pair", "").strip().upper()
    if not pair:
        return JSONResponse(status_code=400, content={"error": "Pair is required."})
    if "/" not in pair:
        pair += "/USDT"

    app_state.analysis_busy = True
    await ws_broadcast("status", {"analysis_busy": True, "pair": pair})

    async def run():
        try:
            await ws_log(f"[APEX] ═══ Starting analysis for {pair} ═══")
            await ws_log(f"[APEX] Account: ${ACCOUNT_SIZE} | Risk: {RISK_PCT*100}% per trade")

            # ── Engine analysis ──
            result = await run_full_analysis(
                exchange=app_state.exchange,
                pair=pair,
                account_size=ACCOUNT_SIZE,
                risk_pct=RISK_PCT,
                log_fn=ws_log,
            )

            if result.get("error"):
                await ws_log(f"[APEX] ❌ Analysis error: {result['error']}", level="ERROR")
                await ws_broadcast("analysis_error", {"error": result["error"]})
                return

            if result.get("no_signal"):
                reason = result.get("no_signal_reason", "Insufficient confluence")
                await ws_log(f"[APEX] ⚠️ NO SIGNAL — {reason}")
                await ws_broadcast("no_signal", {
                    "pair": pair, "reason": reason,
                    "confluence": result.get("confluence", {})
                })
                save_analysis_cache(result)
                return

            # ── Pre-calculate risk for Gemini ──
            price   = result["price"]
            atr_1h  = result.get("atr_1h", price * 0.005)
            a4h     = result["analyses"].get("4h", {})
            obs_4h  = a4h.get("order_blocks", [])
            best_ob = next((o for o in obs_4h if o.get("fresh")), None)

            # Estimate SL from OB
            prelim_direction = result["gemini_payload"]["timeframes"]["1d"]["trend"]
            if prelim_direction == "bullish" and best_ob and best_ob["type"] == "bullish":
                est_entry = price
                est_sl    = best_ob["low"] * 0.999
            elif prelim_direction == "bearish" and best_ob and best_ob["type"] == "bearish":
                est_entry = price
                est_sl    = best_ob["high"] * 1.001
            else:
                est_entry = price
                est_sl    = price * (0.985 if prelim_direction == "bullish" else 1.015)

            risk_data = calculate_risk(
                entry=est_entry, stop_loss=est_sl,
                account=ACCOUNT_SIZE, risk_pct=RISK_PCT,
                atr=atr_1h
            )
            await ws_log(
                f"[ORDER] Pre-calc risk → Margin: ${risk_data.get('margin_usdt', 0)} | "
                f"Leverage: {risk_data.get('leverage', 1)}x | RR: {risk_data.get('rr', 0)}"
            )

            # ── Gemini signal ──
            await ws_log("[AI] 🤖 Sending scored analysis to Gemini 1.5 Flash...")
            signal = await app_state.ai.generate_signal(result["gemini_payload"], risk_data)

            if not signal:
                await ws_log("[AI] ❌ Gemini returned no valid signal.", level="ERROR")
                await ws_broadcast("no_signal", {"pair": pair, "reason": "Gemini returned null"})
                return

            mode       = signal.get("mode", "no_signal")
            direction  = signal.get("direction", "none")
            confidence = signal.get("confidence", 0)
            entry_type = signal.get("entry_type", "standard")

            await ws_log(
                f"[AI] ✅ Gemini response → Mode: {mode.upper()} | "
                f"Direction: {direction.upper()} | Confidence: {confidence}% | "
                f"Entry: {entry_type.upper()}"
            )
            await ws_log(f"[AI] Reasoning: {signal.get('reasoning', '')}")

            if mode == "no_signal" or confidence < 70:
                await ws_log(
                    f"[APEX] ⚠️ Gemini found no valid signal "
                    f"(confidence={confidence}%, mode={mode})"
                )
                await ws_broadcast("no_signal", {
                    "pair": pair, "reason": signal.get("reasoning", "Low confidence"),
                    "confidence": confidence
                })
                return

            # ── Log trade levels ──
            tp_levels = signal.get("tp_levels", [])
            await ws_log(f"[SIGNAL] Entry: {signal.get('entry')} | SL: {signal.get('stop_loss')}")
            if tp_levels:
                tps = " | ".join(f"TP{i+1}: {tp}" for i, tp in enumerate(tp_levels))
                await ws_log(f"[SIGNAL] {tps}")
            await ws_log(
                f"[SIGNAL] Leverage: {signal.get('leverage')}x | "
                f"Margin: ${signal.get('margin_usdt')} | RR: {signal.get('rr')}"
            )
            await ws_log(
                f"[SIGNAL] ⚡ {entry_type.upper()} {direction.upper()} on {pair} | "
                f"Confluence: {signal.get('confluence_score', 0)}/100"
            )

            # ── Save to state + cache ──
            saved_state = apply_signal_to_state(signal)
            save_analysis_cache({**result, "signal": signal})

            # ── Broadcast to dashboard ──
            await ws_broadcast("signal", {
                "pair":            pair,
                "mode":            mode,
                "direction":       direction,
                "entry":           signal.get("entry"),
                "stop_loss":       signal.get("stop_loss"),
                "tp1":             tp_levels[0] if len(tp_levels) > 0 else None,
                "tp2":             tp_levels[1] if len(tp_levels) > 1 else None,
                "tp3":             tp_levels[2] if len(tp_levels) > 2 else None,
                "leverage":        signal.get("leverage"),
                "margin_usdt":     signal.get("margin_usdt"),
                "rr":              signal.get("rr"),
                "confidence":      confidence,
                "entry_type":      entry_type,
                "confluence_score": signal.get("confluence_score", 0),
                "reasoning":       signal.get("reasoning"),
                "hard_invalidation": signal.get("hard_invalidation"),
                "soft_warning":    signal.get("soft_warning"),
                "ob_zone":         signal.get("ob_zone"),
                "liquidity_above": signal.get("liquidity_above"),
                "liquidity_below": signal.get("liquidity_below"),
                "timestamp":       datetime.utcnow().isoformat(),
            })

        except Exception as e:
            await ws_log(f"[APEX] ❌ Unexpected error: {e}", level="ERROR")
            await ws_broadcast("analysis_error", {"error": str(e)})
        finally:
            app_state.analysis_busy = False
            await ws_broadcast("status", {"analysis_busy": False})

    asyncio.create_task(run())
    return {"status": "started", "pair": pair}


@app.post("/api/enter_trade")
async def enter_trade():
    """Mark trade as entered — starts the 5-min monitoring loop."""
    state = load_trade_state()
    if not state.get("active"):
        return JSONResponse(status_code=400, content={"error": "No active signal to enter."})
    if state.get("in_trade"):
        return JSONResponse(status_code=400, content={"error": "Already in a trade."})

    mark_trade_entered()
    await ws_log(f"[TRADE] ✅ Trade entered: {state['pair']} {state['direction'].upper()} @ {state['entry']}")
    await app_state.monitor.start()
    return {"status": "ok", "message": "Trade entered. Monitoring started."}


@app.post("/api/reset")
async def reset():
    """Full reset — clear state, cache, stop monitoring."""
    await app_state.monitor.stop()
    full_reset()
    await ws_log("[RESET] 🔄 System reset. All state and cache cleared. Ready for new trade.")
    await ws_broadcast("reset", {"status": "clean"})
    return {"status": "ok"}


@app.get("/api/trade_state")
async def get_trade_state():
    return load_trade_state()


@app.get("/api/logs")
async def get_logs():
    return {"logs": app_state.log_buffer[-100:]}


@app.get("/api/wallet")
async def get_wallet():
    try:
        balance = await app_state.exchange.fetch_balance()
        return balance
    except Exception as e:
        return {"error": str(e), "total": 0, "free": 0, "used": 0}


@app.get("/api/live_price")
async def get_live_price(pair: str):
    try:
        ticker = await app_state.exchange.fetch_ticker(pair)
        price  = float(ticker.get("last", 0))
        state  = load_trade_state()
        pnl_pct  = get_pnl_pct(price) if state.get("in_trade") else 0.0
        pnl_usdt = get_pnl_usdt(price) if state.get("in_trade") else 0.0
        return {"pair": pair, "price": price, "pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.ws_clients.append(websocket)
    logger.info(f"[WS] Client connected ({len(app_state.ws_clients)} total)")

    # Send buffered logs
    for entry in app_state.log_buffer[-100:]:
        try:
            await websocket.send_text(json.dumps({"type": "log", "data": entry}))
        except Exception:
            break

    # Send current state snapshot
    try:
        state = load_trade_state()
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "server_ip":  app_state.server_ip,
                "trade_state": state,
                "account_size": ACCOUNT_SIZE,
            }
        }, default=str))
    except Exception:
        pass

    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if websocket in app_state.ws_clients:
            app_state.ws_clients.remove(websocket)
        logger.info(f"[WS] Client disconnected ({len(app_state.ws_clients)} remaining)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
