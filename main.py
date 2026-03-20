"""
main.py — APEX V10 Trading Terminal
FastAPI + WebSocket + async engine + IP display on startup
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
    load_trade_state, apply_signal_to_state,
    mark_trade_entered, full_reset,
    get_pnl_pct, get_pnl_usdt,
    save_analysis_cache, load_analysis_cache,
    is_in_trade,
)

load_dotenv()

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "100"))
RISK_PCT           = float(os.getenv("RISK_PCT", "0.015"))
PORT               = int(os.getenv("PORT", "8080"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


class AppState:
    ws_clients:    list = []
    log_buffer:    list = []
    exchange:      Optional[ExchangeManager] = None
    ai:            Optional[GeminiAnalyst]   = None
    monitor:       Optional[TradeMonitor]    = None
    server_ip:     str  = "detecting..."
    analysis_busy: bool = False

app_state = AppState()


def get_server_ip() -> str:
    try:
        import urllib.request
        return urllib.request.urlopen("https://api.ipify.org", timeout=4).read().decode().strip()
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


async def ws_log(message: str, level: str = "INFO"):
    ts    = datetime.utcnow().strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "msg": message}
    app_state.log_buffer.append(entry)
    if len(app_state.log_buffer) > 600:
        app_state.log_buffer = app_state.log_buffer[-600:]
    if level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)
    await _bcast(json.dumps({"type": "log", "data": entry}))


async def ws_broadcast(event_type: str, data: dict):
    await _bcast(json.dumps({"type": event_type, "data": data}, default=str))


async def _bcast(payload: str):
    dead = []
    for ws in app_state.ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in app_state.ws_clients:
            app_state.ws_clients.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.server_ip = get_server_ip()
    print(f"""
╔══════════════════════════════════════════════════╗
║          APEX V10 — AI Trading Terminal          ║
╠══════════════════════════════════════════════════╣
║  Server IP  : {app_state.server_ip:<34}║
║  Port       : {PORT:<34}║
║  Account    : ${ACCOUNT_SIZE:<33}║
║  Risk/Trade : {RISK_PCT*100:.1f}%{'':<32}║
╠══════════════════════════════════════════════════╣
║  ⚠  Whitelist this IP in Binance API settings   ║
║  ➜  {app_state.server_ip:<44}║
╚══════════════════════════════════════════════════╝
""")
    app_state.exchange = ExchangeManager(BINANCE_API_KEY, BINANCE_API_SECRET)
    app_state.ai       = GeminiAnalyst(GEMINI_API_KEY)
    app_state.monitor  = TradeMonitor(
        exchange=app_state.exchange,
        ai_analyst=app_state.ai,
        log_fn=ws_log,
        broadcast_fn=ws_broadcast,
        interval=300,
    )
    if is_in_trade():
        st = load_trade_state()
        logger.info(f"[STARTUP] Resuming monitor for {st.get('pair')}")
        await app_state.monitor.start()
    yield
    logger.info("[SHUTDOWN] Stopping...")
    if app_state.monitor:
        await app_state.monitor.stop()
    if app_state.exchange:
        await app_state.exchange.close()
    logger.info("[SHUTDOWN] Done.")


app = FastAPI(title="APEX V10", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    return {
        "server_ip":       app_state.server_ip,
        "account_size":    ACCOUNT_SIZE,
        "risk_pct":        RISK_PCT,
        "trade_state":     load_trade_state(),
        "analysis_busy":   app_state.analysis_busy,
        "monitor_running": app_state.monitor._running if app_state.monitor else False,
    }


@app.get("/api/top_pairs")
async def get_top_pairs():
    try:
        await ws_log("[SCREENER] Fetching top 10 Binance Futures pairs by volume...")
        pairs = await app_state.exchange.get_top_volume_pairs(10)
        if not pairs:
            await ws_log("[SCREENER] ⚠️ No pairs returned. Check API key + IP whitelist.", level="ERROR")
            return {"pairs": [], "warning": "No pairs returned — check IP whitelist"}
        await ws_log(f"[SCREENER] ✅ {len(pairs)} pairs: {', '.join(p['pair'] for p in pairs[:5])}...")
        return {"pairs": pairs}
    except Exception as e:
        await ws_log(f"[SCREENER] ❌ {e}", level="ERROR")
        return JSONResponse(status_code=500, content={"error": str(e), "pairs": []})


@app.post("/api/analyze")
async def start_analysis(request: Request):
    if app_state.analysis_busy:
        return JSONResponse(status_code=409, content={"error": "Analysis already running."})

    body = await request.json()
    pair = body.get("pair", "").strip().upper().replace(" ", "")
    if not pair:
        return JSONResponse(status_code=400, content={"error": "Pair required."})
    if "/" not in pair:
        pair = pair + "/USDT"

    app_state.analysis_busy = True
    await ws_broadcast("status", {"analysis_busy": True, "pair": pair})

    async def run():
        try:
            await ws_log(f"[APEX] ═══ Analysis started: {pair} ═══")
            await ws_log(f"[APEX] Account:${ACCOUNT_SIZE} Risk:{RISK_PCT*100}%")

            result = await run_full_analysis(
                exchange=app_state.exchange,
                pair=pair,
                account_size=ACCOUNT_SIZE,
                risk_pct=RISK_PCT,
                log_fn=ws_log,
            )

            if result.get("error"):
                await ws_log(f"[APEX] ❌ {result['error']}", level="ERROR")
                await ws_broadcast("analysis_error", {"error": result["error"]})
                return

            if result.get("no_signal"):
                reason = result.get("no_signal_reason", "SMC conditions not met")
                await ws_log(f"[APEX] ⚠️ NO SIGNAL — {reason}")
                await ws_broadcast("no_signal", {"pair": pair, "reason": reason})
                save_analysis_cache(result)
                return

            await ws_log("[AI] 🤖 Sending ICT signal to Gemini 1.5 Flash for refinement...")
            signal = await app_state.ai.generate_signal(result["gemini_payload"])

            if not signal:
                await ws_log("[AI] ❌ Gemini returned null", level="ERROR")
                await ws_broadcast("no_signal", {"pair": pair, "reason": "Gemini null response"})
                return

            mode       = signal.get("mode", "no_signal")
            direction  = signal.get("direction", "none")
            confidence = signal.get("confidence", 0)
            entry_type = signal.get("entry_type", "standard")

            await ws_log(f"[AI] {mode.upper()} {direction.upper()} conf:{confidence}% {entry_type.upper()}")
            await ws_log(f"[AI] {signal.get('reasoning','')}")

            if mode == "no_signal" or confidence < 65:
                await ws_log(f"[APEX] ⚠️ Gemini: no setup (conf={confidence}%)")
                await ws_broadcast("no_signal", {
                    "pair": pair, "reason": signal.get("reasoning","Low confidence"),
                    "confidence": confidence,
                })
                return

            tp_levels = signal.get("tp_levels", [])
            await ws_log(f"[SIGNAL] Entry:{signal.get('entry')} SL:{signal.get('stop_loss')}")
            if tp_levels:
                await ws_log("[SIGNAL] " + " | ".join(f"TP{i+1}:{v}" for i,v in enumerate(tp_levels)))
            await ws_log(f"[SIGNAL] {signal.get('leverage')}x | Margin:${signal.get('margin_usdt')} | RR:1:{signal.get('rr')}")
            await ws_log(f"[SIGNAL] ⚡ {entry_type.upper()} {direction.upper()} {pair} | Confidence:{confidence}%")

            apply_signal_to_state(signal)
            save_analysis_cache({**result, "signal": signal})

            await ws_broadcast("signal", {
                "pair":             pair,
                "mode":             mode,
                "direction":        direction,
                "entry":            signal.get("entry"),
                "stop_loss":        signal.get("stop_loss"),
                "tp1":              tp_levels[0] if len(tp_levels) > 0 else None,
                "tp2":              tp_levels[1] if len(tp_levels) > 1 else None,
                "tp3":              tp_levels[2] if len(tp_levels) > 2 else None,
                "leverage":         signal.get("leverage"),
                "margin_usdt":      signal.get("margin_usdt"),
                "rr":               signal.get("rr"),
                "confidence":       confidence,
                "entry_type":       entry_type,
                "confluence_score": result.get("decision", {}).get("final_confidence", confidence),
                "reasoning":        signal.get("reasoning"),
                "hard_invalidation":signal.get("hard_invalidation"),
                "soft_warning":     signal.get("soft_warning"),
                "ob_zone":          signal.get("ob_zone"),
                "liquidity_above":  signal.get("liquidity_above"),
                "liquidity_below":  signal.get("liquidity_below"),
                "timestamp":        datetime.utcnow().isoformat(),
            })

        except Exception as e:
            await ws_log(f"[APEX] ❌ {e}", level="ERROR")
            await ws_broadcast("analysis_error", {"error": str(e)})
        finally:
            app_state.analysis_busy = False
            await ws_broadcast("status", {"analysis_busy": False})

    asyncio.create_task(run())
    return {"status": "started", "pair": pair}


@app.post("/api/enter_trade")
async def enter_trade():
    state = load_trade_state()
    if not state.get("active"):
        return JSONResponse(status_code=400, content={"error": "No active signal."})
    if state.get("in_trade"):
        return JSONResponse(status_code=400, content={"error": "Already in trade."})
    mark_trade_entered()
    await ws_log(f"[TRADE] ✅ Entered: {state['pair']} {(state.get('direction')or'').upper()} @ {state.get('entry')}")
    await app_state.monitor.start()
    return {"status": "ok"}


@app.post("/api/reset")
async def reset():
    await app_state.monitor.stop()
    full_reset()
    await ws_log("[RESET] 🔄 System reset. Ready for new trade.")
    await ws_broadcast("reset", {"status": "clean"})
    return {"status": "ok"}


@app.get("/api/trade_state")
async def get_trade_state():
    return load_trade_state()


@app.get("/api/wallet")
async def get_wallet():
    try:
        return await app_state.exchange.fetch_balance()
    except Exception as e:
        return {"error": str(e), "total": 0.0, "free": 0.0, "used": 0.0}


@app.get("/api/live_price")
async def live_price(pair: str):
    try:
        ticker   = await app_state.exchange.fetch_ticker(pair)
        price    = float(ticker.get("last", 0))
        state    = load_trade_state()
        pnl_pct  = get_pnl_pct(price) if state.get("in_trade") else 0.0
        pnl_usdt = get_pnl_usdt(price) if state.get("in_trade") else 0.0
        return {"pair": pair, "price": price,
                "pnl_pct": round(pnl_pct,3), "pnl_usdt": round(pnl_usdt,4)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.ws_clients.append(websocket)
    logger.info(f"[WS] Connected ({len(app_state.ws_clients)} total)")

    for entry in app_state.log_buffer[-100:]:
        try:
            await websocket.send_text(json.dumps({"type": "log", "data": entry}))
        except Exception:
            break

    try:
        state = load_trade_state()
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "server_ip":    app_state.server_ip,
                "trade_state":  state,
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
        logger.info(f"[WS] Disconnected ({len(app_state.ws_clients)} remaining)")


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
