"""
main.py — SMC Alert Bot  (Auto-Execution Edition)

Pipeline
────────
  hourly_scanner_loop   — HTF scan → WATCHLIST_1 alerts + charts  (60 min)
  sniper_loop           — POI watch → sweep → CHoCH → ORDER PLACED  (5 s)
  execution_loop        — Fill detection, cancel-if-missed, close detect  (15 s)
  pnl_tracking_loop     — Live unrealised PnL updates to Telegram  (120 s)

4-Stage Execution Alerts
────────────────────────
  🔵 Stage 1 — Order Placed     (LIMIT at 30% retrace, SL + TP3@90% attached)
  🟢 Stage 2 — Trade Opened     (limit order filled, position LIVE)
  📊 Stage 3 — Live PnL Update  (every 2 min while position open)
  🏆/🔴 Stage 4 — Trade Closed  (SL or TP hit, final result)
"""

from __future__ import annotations

import asyncio
import gc
import glob
import json
import logging
import os
import time as _time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from dotenv import load_dotenv

from scan import run_scan, fetch_ohlcv
from watching import run_watcher
from sniper import run_sniper, run_choch_monitor
from smc import SniperSignal
from charting import generate_watchlist_chart, generate_sniper_chart
from execution import (
    place_trade, monitor_trades, get_all_live_pnl,
    emergency_cancel_pending, emergency_close_profitable,
    get_btc_5min_change, register_error_callback,
    set_crash_active, is_crash_active,
    get_public_exchange, get_active_trade_count,
    TradeRecord, RISK_PER_TRADE,
)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")    # needed for execution

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID not set in .env")

# Auto-execution is silently disabled when API keys are missing
AUTO_EXECUTE    = bool(BINANCE_API_KEY and os.getenv("BINANCE_API_SECRET", ""))
MAX_LIVE_TRADES = 2    # hard cap: block new entries when 2 trades are open

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("app")
if not AUTO_EXECUTE:
    logger.warning(
        "BINANCE_API_KEY / BINANCE_API_SECRET not set — running in ALERT-ONLY mode."
    )

# ─────────────────────────────────────────────────────────────────────
# Bot & Dispatcher
# ─────────────────────────────────────────────────────────────────────

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp  = Dispatcher()

# symbol → Telegram message_id of the "Trade Opened" message (edited live for PnL)
_pnl_msg_ids: dict[str, int] = {}


# ─────────────────────────────────────────────────────────────────────
# ── Telegram formatters ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

def format_watchlist_alert(entry: dict) -> str:
    arrow = "🟢" if entry.get("direction") == "bullish" else "🔴"
    poi   = entry.get("htf_poi", {})
    bd    = entry.get("breakdown", {})
    lines = [
        f"{arrow} <b>WATCHLIST ALERT</b> {arrow}",
        "",
        f"📊 <b>Coin:</b>  <code>{entry.get('symbol', '?')}</code>",
        f"📈 <b>Direction:</b>  <code>{entry.get('direction', '?').upper()}</code>",
        f"⭐ <b>Score:</b>  <code>{entry.get('score', '?')} / 100</code>",
        "",
        "🎯 <b>HTF POI Zone:</b>",
        f"   High: <code>{poi.get('high', 0):.6f}</code>",
        f"   Low:  <code>{poi.get('low', 0):.6f}</code>",
        f"   Type: <code>{poi.get('type', '?').upper()}</code>",
        "",
        "<b>Score Breakdown:</b>",
    ]
    for k, v in bd.items():
        lines.append(f"  • {k}: <code>{v}</code>")
    lines += ["", "⏳ <i>Monitoring for price approach…</i>",
              "<i>— DM_LKY SMC Bot —</i>"]
    return "\n".join(lines)


def format_poi_touched_alert(entry: dict) -> str:
    arrow = "🟢" if entry.get("direction") == "bullish" else "🔴"
    poi   = entry.get("htf_poi", {})
    sym   = entry.get("symbol", "?")
    dir_  = entry.get("direction", "?").upper()
    return "\n".join([
        f"🟡 <b>POI TOUCHED — {sym}</b>",
        "",
        f"📊 <b>Coin:</b>  <code>{sym}</code>",
        f"📈 <b>Direction:</b>  {arrow} <code>{dir_}</code>",
        "",
        "🎯 <b>HTF POI Zone:</b>",
        f"   High: <code>{poi.get('high', 0):.6f}</code>",
        f"   Low:  <code>{poi.get('low', 0):.6f}</code>",
        "",
        "⏳ <i>Price entered POI. Watching for Liquidity Sweep…</i>",
        "<b>Do NOT enter yet.</b>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_sweep_alert(sweep_entry: dict) -> str:
    sym   = sweep_entry.get("symbol", "?")
    dir_  = sweep_entry.get("direction", "?")
    arrow = "🟢" if dir_ == "bullish" else "🔴"
    poi   = sweep_entry.get("htf_poi", {})
    sw    = sweep_entry.get("sweep", {})
    return "\n".join([
        f"🟠 <b>LIQUIDITY SWEEP — {sym}</b>",
        "",
        f"📊 <b>Coin:</b>  <code>{sym}</code>",
        f"📈 <b>Direction:</b>  {arrow} <code>{dir_.upper()}</code>",
        "",
        f"⚡ <b>Sweep Level:</b>  <code>{sw.get('swept_price', 0):.6f}</code>",
        f"📍 <b>Wick Extreme:</b>  <code>{sw.get('wick_extreme', 0):.6f}</code>",
        f"🎯 <b>POI:</b>  "
        f"<code>{poi.get('low', 0):.6f} – {poi.get('high', 0):.6f}</code>",
        "",
        "⏳ <i>Sweep confirmed! Watching for CHoCH to fire sniper entry…</i>",
        "<b>Get ready. Entry incoming.</b>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_sniper_alert(sig: SniperSignal) -> str:
    arrow = "🟢" if sig.direction == "LONG" else "🔴"
    lines = [
        f"{arrow} <b>SMC SNIPER SIGNAL</b> {arrow}",
        "",
        f"📊 <b>Coin:</b>  <code>{sig.symbol}</code>",
        f"📈 <b>Direction:</b>  <code>{sig.direction}</code>",
        f"📋 <b>Order:</b>  <code>{sig.order_type}</code>",
        "",
        f"🔵 <b>Entry Zone:</b>  <code>{sig.entry_low:.6f} – {sig.entry_high:.6f}</code>",
        f"🔴 <b>Stop Loss:</b>  <code>{sig.stop_loss:.6f}</code>",
        "",
        "━━━ <b>6-TARGET MOONSHOT</b> ━━━",
        f"🎯 <b>TP1 (1:1):</b>      <code>{sig.tp1:.6f}</code>",
        f"🎯 <b>TP2 (1:1.5):</b>    <code>{sig.tp2:.6f}</code>",
        f"🎯 <b>TP3 (1:2) ★:</b>    <code>{sig.tp3:.6f}</code>  ← $3.00 / 90% close",
        f"🎯 <b>TP4 (1:4):</b>      <code>{sig.tp4:.6f}</code>",
        f"🚀 <b>TP5 (1:6):</b>     <code>{sig.tp5:.6f}</code>",
        f"💎 <b>TP6 (Runner):</b>  <code>{sig.tp6:.6f}</code>",
        "",
        "━━━ <b>RISK MANAGEMENT</b> ━━━",
        f"⚖️ <b>Fixed Risk:</b>  <code>${RISK_PER_TRADE:.2f}</code>",
        f"📦 <b>Position Size:</b>  <code>{sig.position_size:.4f}</code>",
        f"📐 <b>R:R:</b>  <code>1 : {sig.risk_reward}</code>",
        "",
        "<b>Confluences:</b>",
    ]
    for c in sig.confluences:
        lines.append(f"  • {c}")
    lines += ["",
              "⚠️ <i>Auto-execution active. LIMIT at 30% retrace. SL + TP3(90%) placed on fill.</i>",
              "<i>— DM_LKY SMC Bot —</i>"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# ── 4-Stage Execution Alert Formatters ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────

def format_order_placed(rec: TradeRecord) -> str:
    arrow = "🟢" if rec.direction == "LONG" else "🔴"
    return "\n".join([
        f"🔵 <b>ORDER PLACED  [Binance Futures]</b>  {arrow}",
        "",
        f"📊 <b>Symbol:</b>    <code>{rec.symbol}</code>",
        f"📈 <b>Direction:</b> <code>{rec.direction}</code>",
        "",
        f"📐 <b>Entry (30% Retrace):</b>  <code>{rec.entry_price:.6f}</code>",
        f"🔴 <b>Stop Loss:</b>            <code>{rec.stop_loss:.6f}</code>",
        f"🎯 <b>Primary TP  (TP3):</b>    <code>{rec.tp3:.6f}</code>",
        f"🚀 <b>Runner  TP  (TP6):</b>    <code>{rec.tp6:.6f}</code>",
        "",
        "━━━ <b>ORDER DETAILS</b> ━━━",
        f"📦 <b>Quantity:</b>     <code>{rec.qty:.6f}</code>",
        f"⚡ <b>Leverage:</b>    <code>{rec.leverage}×  ISOLATED</code>",
        f"💰 <b>Margin Cost:</b>  <code>${rec.margin_cost:.4f}</code>",
        f"⚖️ <b>Fixed Risk:</b>   <code>${RISK_PER_TRADE:.2f}</code>",
        "",
        "⏳ <i>LIMIT at 30% retrace. SL + TP3(90%) placed on fill.</i>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_trade_opened(rec: TradeRecord) -> str:
    arrow = "🟢" if rec.direction == "LONG" else "🔴"
    return "\n".join([
        f"🟢 <b>TRADE OPENED — {rec.symbol}</b>  {arrow}",
        "",
        f"✅ <b>LIMIT filled at:</b>  <code>{rec.entry_price:.6f}</code>",
        f"📈 <b>Direction:</b>  <code>{rec.direction}</code>",
        f"🔴 <b>SL:</b>   <code>{rec.stop_loss:.6f}</code>  ← STOP_MARKET active",
        f"🎯 <b>TP3:</b>  <code>{rec.tp3:.6f}</code>  ← 90% close (TAKE_PROFIT_MARKET)",
        f"🚀 <b>TP6:</b>  <code>{rec.tp6:.6f}</code>  ← runner target",
        "",
        f"📦 <b>Size:</b>    <code>{rec.qty:.6f}</code>  ×  "
        f"<code>{rec.leverage}× isolated</code>",
        f"💰 <b>Margin:</b>  <code>${rec.margin_cost:.4f}</code>",
        "",
        "📊 <i>Position LIVE. PnL updates every 2 min.</i>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_pnl_update(rec: TradeRecord, pnl_usd: float, pnl_pct: float) -> str:
    sign   = "+" if pnl_usd >= 0 else ""
    emoji  = "📈" if pnl_usd >= 0 else "📉"
    return "\n".join([
        f"📊 <b>LIVE PnL UPDATE — {rec.symbol}</b>",
        "",
        f"{emoji} <b>Unrealised PnL:</b>  "
        f"<code>{sign}${pnl_usd:.4f}  ({sign}{pnl_pct:.2f}%)</code>",
        "",
        f"📐 <b>Entry:</b>  <code>{rec.entry_price:.6f}</code>",
        f"🔴 <b>SL:</b>  <code>{rec.stop_loss:.6f}</code>",
        f"🎯 <b>TP3:</b>  <code>{rec.tp3:.6f}</code>  ← primary target",
        f"🚀 <b>TP6:</b>  <code>{rec.tp6:.6f}</code>  ← runner",
        f"📦 <b>Qty:</b>  <code>{rec.qty:.6f}</code>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_live_dashboard(rec: TradeRecord, pnl_usd: float, pnl_pct: float) -> str:
    """Live dashboard message edited in-place in Telegram (anti-spam PnL update)."""
    arrow  = "🟢" if rec.direction == "LONG" else "🔴"
    sign   = "+" if pnl_usd >= 0 else ""
    bar_emoji = "📈" if pnl_usd >= 0 else "📉"
    status = rec.status.replace("_", " ")
    return "\n".join([
        f"📊 <b>LIVE DASHBOARD — {rec.symbol}</b>  {arrow}",
        "",
        f"{bar_emoji} <b>PnL:</b>  <code>{sign}${pnl_usd:.4f}  ({sign}{pnl_pct:.2f}%)</code>",
        f"🟡 <b>Status:</b>  <code>{status}</code>",
        "",
        f"📐 <b>Entry:</b>     <code>{rec.entry_price:.6f}</code>",
        f"🔴 <b>SL:</b>        <code>{rec.stop_loss:.6f}</code>",
        f"🎯 <b>TP3 (90%):</b> <code>{rec.tp3:.6f}</code>",
        f"🚀 <b>TP6 Runner:</b> <code>{rec.tp6:.6f}</code>",
        "",
        f"📦 <b>Qty:</b>  <code>{rec.qty:.6f}</code>   "
        f"⚡ <b>Leverage:</b>  <code>{rec.leverage}× ISO</code>",
        "<i>— DM_LKY SMC Bot  [updates every 30 s] —</i>",
    ])


def format_trade_closed(rec: TradeRecord) -> str:
    pnl_usd = rec.pnl_usd
    win     = pnl_usd >= 0
    emoji   = "🏆" if win else "🔴"
    sign    = "+" if win else ""
    result  = "PROFIT" if win else "LOSS"
    reason  = rec.close_reason or "SL/TP"
    tick    = "\u2705" if win else "\u274c"
    return "\n".join([
        f"{emoji} <b>TRADE CLOSED — {rec.symbol}</b>",
        "",
        f"{tick} <b>Result:</b>  "
        f"<code>{sign}${pnl_usd:.4f}  [{result}]</code>",
        f"❓ <b>Reason:</b>  <code>{reason}</code>",
        "",
        f"📐 <b>Entry:</b>  <code>{rec.entry_price:.6f}</code>",
        f"🔴 <b>SL:</b>     <code>{rec.stop_loss:.6f}</code>",
        f"🎯 <b>TP3:</b>    <code>{rec.tp3:.6f}</code>",
        f"🚀 <b>TP6:</b>    <code>{rec.tp6:.6f}</code>",
        f"📦 <b>Qty:</b>    <code>{rec.qty:.6f}</code>",
        f"⚡ <b>Leverage:</b> <code>{rec.leverage}× isolated</code>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_runner_active(rec: TradeRecord) -> str:
    """TP3 hit (90% closed) — runner 10% now protected at break-even."""
    return "\n".join([
        f"🚀 <b>TP3 HIT — RUNNER ACTIVE — {rec.symbol}</b>",
        "",
        f"✅ <b>90% closed at TP3:</b>  <code>{rec.tp3:.6f}</code>",
        f"🔐 <b>SL moved to break-even:</b>  <code>{rec.entry_price:.6f}</code>",
        f"🚀 <b>Runner target (TP6):</b>  <code>{rec.tp6:.6f}</code>",
        "",
        f"📦 <b>Runner qty (≈10%):</b>  <code>{rec.qty * 0.10:.6f}</code>",
        "",
        "<i>Risk-free runner. Worst case = break-even.</i>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


def format_order_cancelled(rec: TradeRecord, reason: str) -> str:
    return "\n".join([
        f"❌ <b>ORDER CANCELLED — {rec.symbol}</b>",
        "",
        f"📊 <b>Symbol:</b>  <code>{rec.symbol}</code>",
        f"📈 <b>Direction:</b>  <code>{rec.direction}</code>",
        f"❓ <b>Reason:</b>  <code>{reason}</code>",
        "",
        f"📐 <b>Entry was:</b>  <code>{rec.entry_price:.6f}</code>",
        f"🎯 <b>TP1 was:</b>    <code>{rec.tp1:.6f}</code>",
        "",
        "<i>Entry missed or timed out. Waiting for next setup.</i>",
        "<i>— DM_LKY SMC Bot —</i>",
    ])


# ─────────────────────────────────────────────────────────────────────
# ── Resource Management Helpers ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

def _is_system_busy() -> bool:
    """
    Returns True if the sniper room OR execution engine is active.
    The scanner must NOT run while busy.
    """
    if DATA_DIR.exists():
        for fpath in DATA_DIR.glob("*.json"):
            try:
                with open(fpath) as f:
                    d = json.load(f)
                if d.get("status") in ("WATCHLIST_2_POI_TOUCHED", "WATCHLIST_2_SWEEP"):
                    return True
            except Exception:
                pass
    return get_active_trade_count() > 0


def _cleanup_temp_files() -> None:
    """Delete all orphaned .png chart files from the system temp directory."""
    import tempfile
    tmp_dir = tempfile.gettempdir()
    for pattern in ("wl1_*.png", "wl2_*.png", "snp_*.png", "tmp*.png"):
        for fpath in glob.glob(os.path.join(tmp_dir, pattern)):
            try:
                os.unlink(fpath)
                logger.debug("Cleaned temp: %s", fpath)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────
# ── Alert senders ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

async def _send(text: str) -> None:
    """Safe Telegram message send with HTML parse mode."""
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text,
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)


async def _send_photo(chart_path: str, caption: str) -> None:
    try:
        photo = FSInputFile(chart_path)
        await bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID, photo=photo,
            caption=caption, parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        logger.error("Telegram photo send failed: %s", exc)
        await _send(caption)   # fallback to text
    finally:
        try:
            os.unlink(chart_path)
        except Exception:
            pass
        plt.close("all")


async def send_watchlist_alert(entry: dict) -> None:
    text       = format_watchlist_alert(entry)
    chart_path = None
    symbol     = entry.get("symbol")
    poi        = entry.get("htf_poi", {})

    if symbol and poi.get("high") and poi.get("low"):
        exc_obj = await get_public_exchange()
        try:
            df_chart = await fetch_ohlcv(exc_obj, symbol, "15m", 3)
            if df_chart is not None and len(df_chart) > 20:
                chart_path = generate_watchlist_chart(
                    df_chart, symbol, entry.get("direction", "bullish"),
                    poi["high"], poi["low"], entry.get("score", 0),
                    timeframe="15m",
                )
        except Exception as exc:
            logger.warning("chart gen failed %s: %s", symbol, exc)

    if chart_path and os.path.exists(chart_path):
        await _send_photo(chart_path, text)
    else:
        await _send(text)
    logger.info("Watchlist alert sent: %s", symbol)


async def send_poi_touched_alert(entry: dict) -> None:
    await _send(format_poi_touched_alert(entry))
    logger.info("POI touched alert sent: %s", entry.get("symbol"))


async def send_sweep_alert(sweep_entry: dict) -> None:
    await _send(format_sweep_alert(sweep_entry))
    logger.info("Sweep alert sent: %s", sweep_entry.get("symbol"))


async def send_sniper_alert(signal: SniperSignal) -> None:
    """Send signal text + 5m chart."""
    text       = format_sniper_alert(signal)
    chart_path = None

    exc_obj = await get_public_exchange()
    try:
        since_ms = int((_time.time() - 3 * 3600) * 1000)
        raw = await exc_obj.fetch_ohlcv(signal.symbol, "5m", since=since_ms, limit=200)
        if raw:
            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(str)
            chart_path = generate_sniper_chart(
                df, signal.symbol, signal.direction,
                signal.entry_high, signal.entry_low, signal.stop_loss,
                signal.tp1, signal.tp2, signal.tp3,
                signal.tp4, signal.tp5, signal.tp6,
                timeframe="5m",
            )
    except Exception as exc:
        logger.warning("sniper chart failed %s: %s", signal.symbol, exc)

    if chart_path and os.path.exists(chart_path):
        await _send_photo(chart_path, text)
    else:
        await _send(text)
    logger.info("Sniper alert sent: %s", signal.symbol)


# ── Execution alerts (Stage 1-4) ──────────────────────────────────────

async def send_order_placed(rec: TradeRecord) -> None:
    await _send(format_order_placed(rec))
    logger.info("Stage 1 — Order Placed alert sent: %s", rec.symbol)


async def send_trade_opened(rec: TradeRecord) -> None:
    try:
        msg = await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=format_trade_opened(rec),
            parse_mode=ParseMode.HTML,
        )
        _pnl_msg_ids[rec.symbol] = msg.message_id
    except Exception as exc:
        logger.error("send_trade_opened failed: %s", exc)
    logger.info("Stage 2 — Trade Opened alert sent: %s", rec.symbol)


async def send_pnl_update(rec: TradeRecord, pnl_usd: float, pnl_pct: float) -> None:
    await _send(format_pnl_update(rec, pnl_usd, pnl_pct))
    logger.info("Stage 3 — PnL update sent: %s  $%.4f", rec.symbol, pnl_usd)


async def send_trade_closed(rec: TradeRecord) -> None:
    _pnl_msg_ids.pop(rec.symbol, None)
    await _send(format_trade_closed(rec))
    logger.info("Stage 4 — Trade Closed alert sent: %s  PnL=$%.4f", rec.symbol, rec.pnl_usd)


async def send_order_cancelled(rec: TradeRecord, reason: str) -> None:
    await _send(format_order_cancelled(rec, reason))
    logger.info("Order Cancelled alert sent: %s  reason=%s", rec.symbol, reason)


async def send_runner_active(rec: TradeRecord) -> None:
    await _send(format_runner_active(rec))
    logger.info("Stage 3b — Runner Active alert sent: %s", rec.symbol)


# ─────────────────────────────────────────────────────────────────────
# ── Background loops ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

async def hourly_scanner_loop() -> None:
    """HTF screener — runs every 3600 s. Sends start/finish Telegram notifications."""
    while True:
        try:
            await _send(
                "🔍 <b>Hourly Scan STARTED</b>\n"
                "Scanning 400 USDT-M Futures symbols concurrently (sem=30)…"
            )
            logger.info("── Hourly scan starting ──")
            results = await run_scan()
            logger.info("── Scan done: %d on WATCHLIST_1 ──", len(results))

            if results:
                lines = [f"✅ <b>Hourly Scan COMPLETE — {len(results)} coin(s) added to Watchlist 1</b>"]
                for r in results:
                    a = "🟢" if r.get("direction") == "bullish" else "🔴"
                    lines.append(
                        f"  {a} <code>{r['symbol']}</code>  "
                        f"score=<code>{r.get('score', '?')}</code>  "
                        f"dir=<code>{r.get('direction', '?')}</code>"
                    )
                await _send("\n".join(lines))
            else:
                await _send("✅ <b>Hourly Scan COMPLETE</b>\nNo new coins qualified this scan.")

            for entry in results:
                await send_watchlist_alert(entry)
        except Exception as exc:
            logger.error("Hourly scan error: %s", exc, exc_info=True)
            await _send(f"⚠️ <b>Scan Error:</b>\n<code>{type(exc).__name__}: {exc}</code>")
        await asyncio.sleep(3600)


async def sniper_loop() -> None:
    """
    5-second loop: watcher promotion + sweep detection.
    CHoCH confirmation is handled by choch_monitor_loop() at 2 s cadence.
    """
    while True:
        try:
            # ── Watcher: WATCHLIST_1 → WATCHLIST_2_POI_TOUCHED ───────
            promoted = await run_watcher()
            for entry in promoted:
                await send_poi_touched_alert(entry)

            # ── Sniper: sweep detection (POI_TOUCHED entries only) ────
            sweep_alerts, _ = await run_sniper()

            for sw in sweep_alerts:
                await send_sweep_alert(sw)

        except Exception as exc:
            logger.error("Sniper loop error: %s", exc, exc_info=True)

        await asyncio.sleep(5)


async def execution_loop() -> None:
    """
    15-second loop: fill detection, cancel-if-missed, partial close, close detection.
    Sends Stage 2 (filled), Stage 3b (runner), Stage 4 (closed) Telegram alerts.
    """
    if not AUTO_EXECUTE:
        return

    while True:
        try:
            result = await monitor_trades()

            # Stage 2 — newly filled orders (SL + TP3 placed)
            for rec in result.filled:
                await send_trade_opened(rec)

            # Stage 3b — TP3 hit, runner 10% now at break-even
            for rec in result.runners:
                await send_runner_active(rec)

            # Cancelled orders
            for rec in result.cancelled:
                await send_order_cancelled(rec, rec.close_reason or "Entry missed / timeout")

            # Stage 4 — position fully closed
            for rec in result.closed:
                await send_trade_closed(rec)
            if result.closed:
                gc.collect()

        except Exception as exc:
            logger.error("Execution loop error: %s", exc, exc_info=True)

        await asyncio.sleep(15)


async def choch_monitor_loop() -> None:
    """
    2-second ultra-fast CHoCH confirmation loop.
    Processes only WATCHLIST_2_SWEEP coins — zero entry misses.
    """
    while True:
        try:
            signals = await run_choch_monitor()
            for sig in signals:
                await send_sniper_alert(sig)
                if AUTO_EXECUTE:
                    if get_active_trade_count() >= MAX_LIVE_TRADES:
                        logger.info("Trade cap reached — skip %s", sig.symbol)
                        continue
                    try:
                        rec = await place_trade(sig)
                        if rec is not None:
                            await send_order_placed(rec)
                    except RuntimeError as exc:
                        logger.warning("Auto-execute blocked: %s", exc)
                    except Exception as exc:
                        logger.error("place_trade failed: %s", exc, exc_info=True)
        except Exception as exc:
            logger.error("CHoCH monitor error: %s", exc, exc_info=True)
        await asyncio.sleep(2)


async def pnl_tracking_loop() -> None:
    """
    Stage 3 — every 30 s, EDIT the Trade Opened message in-place with live PnL.
    Anti-spam: no new notifications, single "live dashboard" message per trade.
    """
    if not AUTO_EXECUTE:
        return

    while True:
        await asyncio.sleep(30)
        try:
            pnl_data = await get_all_live_pnl()
            for rec, pnl_usd, pnl_pct in pnl_data:
                msg_id = _pnl_msg_ids.get(rec.symbol)
                if msg_id:
                    try:
                        await bot.edit_message_text(
                            chat_id=TELEGRAM_CHAT_ID,
                            message_id=msg_id,
                            text=format_live_dashboard(rec, pnl_usd, pnl_pct),
                            parse_mode=ParseMode.HTML,
                        )
                        logger.info("PnL dashboard edited: %s  $%.4f", rec.symbol, pnl_usd)
                    except Exception as edit_exc:
                        logger.debug("edit_message failed %s: %s — sending new", rec.symbol, edit_exc)
                        try:
                            new_msg = await bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=format_live_dashboard(rec, pnl_usd, pnl_pct),
                                parse_mode=ParseMode.HTML,
                            )
                            _pnl_msg_ids[rec.symbol] = new_msg.message_id
                        except Exception as send_exc:
                            logger.error("PnL fallback send failed %s: %s", rec.symbol, send_exc)
                else:
                    try:
                        new_msg = await bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=format_live_dashboard(rec, pnl_usd, pnl_pct),
                            parse_mode=ParseMode.HTML,
                        )
                        _pnl_msg_ids[rec.symbol] = new_msg.message_id
                    except Exception as send_exc:
                        logger.error("PnL send failed %s: %s", rec.symbol, send_exc)
        except Exception as exc:
            logger.error("PnL tracking loop error: %s", exc, exc_info=True)


async def _idle_cleanup_loop() -> None:
    """Every 60 s: remove orphaned chart files and force GC when the system is idle."""
    while True:
        await asyncio.sleep(60)
        try:
            if not _is_system_busy():
                _cleanup_temp_files()
                gc.collect()
        except Exception as exc:
            logger.debug("Idle cleanup error: %s", exc)


async def btc_crash_monitor_loop() -> None:
    """
    Emergency: checks BTC 5-min price change every 30 s.
    If BTC drops > 2% in 5 minutes:
      • Set crash flag (blocks new entries)
      • Cancel all PENDING orders
      • Close all LIVE positions with PnL >= +$0.20
      • Send 🚨 emergency Telegram alert
    Crash flag auto-clears if BTC recovers above the threshold.
    """
    if not AUTO_EXECUTE:
        return

    BTC_CRASH_THRESHOLD = -2.0   # % drop in 5 min
    already_crashed     = False

    while True:
        await asyncio.sleep(30)
        try:
            change = await get_btc_5min_change()
            if change is None:
                continue

            if change <= BTC_CRASH_THRESHOLD and not already_crashed:
                already_crashed = True
                set_crash_active(True)

                alert = (
                    f"🚨 <b>BTC CRASH DETECTED!</b>\n"
                    f"BTC dropped <code>{change:.2f}%</code> in 5 minutes.\n"
                    "Emergency protocols activated:\n"
                    "• All new trade entries BLOCKED\n"
                    "• Cancelling all pending orders…\n"
                    "• Closing profitable positions…\n"
                    "<i>— DM_LKY SMC Bot —</i>"
                )
                await _send(alert)
                logger.warning("BTC CRASH DETECTED: %.2f%% drop", change)

                # Cancel all pending limit orders
                cancelled = await emergency_cancel_pending()
                if cancelled:
                    syms = ", ".join(r.symbol for r in cancelled)
                    await _send(
                        f"❌ <b>Emergency: {len(cancelled)} order(s) cancelled</b>\n"
                        f"<code>{syms}</code>"
                    )

                # Close profitable live positions
                closed = await emergency_close_profitable(min_profit_usd=0.20)
                if closed:
                    lines = [f"🔴 <b>Emergency Closed {len(closed)} position(s):</b>"]
                    for r in closed:
                        lines.append(
                            f"  <code>{r.symbol}</code>  "
                            f"PnL: <code>${r.pnl_usd:+.4f}</code>"
                        )
                    await _send("\n".join(lines))

            elif change > BTC_CRASH_THRESHOLD and already_crashed:
                # BTC recovered
                already_crashed = False
                set_crash_active(False)
                await _send(
                    f"✅ <b>BTC stabilised</b>  ({change:+.2f}% over 5 min).\n"
                    "New trade entries RE-ENABLED."
                )

        except Exception as exc:
            logger.error("BTC crash monitor error: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────
# ── Telegram command handlers ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

@dp.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    mode    = "🤖 AUTO-EXECUTION ACTIVE" if AUTO_EXECUTE else "📡 ALERT-ONLY MODE"
    crash   = "  [🚨 CRASH PROTOCOL ON]" if is_crash_active() else ""
    exch    = "Binance USDT-M Futures" if AUTO_EXECUTE else "N/A"
    await message.answer(
        f"🤖 <b>SMC Sniper Bot</b>\n\n"
        f"<b>Mode:</b>   {mode}{crash}\n"
        f"<b>Exchange:</b> <code>{exch}</code>\n\n"
        "<b>Commands:</b>\n"
        "📡 /scan      — Force HTF scan\n"
        "📊 /status    — Watchlists + open trades\n"
        "💰 /pnl       — Live PnL snapshot\n"
        "🛑 /trades    — All active trades\n"
        "₿ /btcstatus — BTC crash status\n",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("scan"))
async def cmd_scan(message: types.Message) -> None:
    await message.answer("⏳ Running forced HTF scan…", parse_mode=ParseMode.HTML)
    try:
        results = await run_scan()
        if results:
            lines = ["✅ <b>Scan complete — WATCHLIST_1:</b>\n"]
            for r in results:
                arrow = "🟢" if r.get("direction") == "bullish" else "🔴"
                lines.append(
                    f"{arrow} <code>{r['symbol']}</code>  "
                    f"dir=<code>{r['direction']}</code>  "
                    f"score=<code>{r.get('score','?')}</code>"
                )
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
            for entry in results:
                await send_watchlist_alert(entry)
        else:
            await message.answer("✅ Scan complete — no coins qualified.",
                                  parse_mode=ParseMode.HTML)
    except Exception as exc:
        logger.error("/scan error: %s", exc, exc_info=True)
        await message.answer(f"❌ Scan error: {exc}", parse_mode=ParseMode.HTML)


@dp.message(Command("status"))
async def cmd_status(message: types.Message) -> None:
    wl1, wl2_poi, wl2_sweep = [], [], []
    if DATA_DIR.exists():
        for fpath in DATA_DIR.glob("*.json"):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                status = data.get("status", "")
                sym    = data.get("symbol", fpath.stem)
                dir_   = data.get("direction", "?")
                arrow  = "🟢" if dir_ == "bullish" else "🔴"
                item   = f"{arrow} <code>{sym}</code> ({dir_})"
                if status == "WATCHLIST_1":
                    wl1.append(item)
                elif status == "WATCHLIST_2_POI_TOUCHED":
                    wl2_poi.append(item)
                elif status == "WATCHLIST_2_SWEEP":
                    wl2_sweep.append(item + " ⚡")
            except Exception:
                pass

    lines = [f"📊 <b>Bot Status</b>  [{'AUTO-EXEC' if AUTO_EXECUTE else 'ALERT-ONLY'}]\n"]
    lines.append(f"🔍 <b>WATCHLIST 1</b> ({len(wl1)}):")
    lines.extend([f"  {e}" for e in wl1] or ["  — empty"])
    lines.append(f"\n🟡 <b>POI TOUCHED</b> ({len(wl2_poi)}):")
    lines.extend([f"  {e}" for e in wl2_poi] or ["  — none"])
    lines.append(f"\n🟠 <b>SWEEP DETECTED</b> ({len(wl2_sweep)}):")
    lines.extend([f"  {e}" for e in wl2_sweep] or ["  — none"])

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@dp.message(Command("trades"))
async def cmd_trades(message: types.Message) -> None:
    from execution import _load_all_trades
    trades = _load_all_trades()
    if not trades:
        await message.answer("📭 No tracked trades.", parse_mode=ParseMode.HTML)
        return

    lines = ["💼 <b>Active Trades</b>\n"]
    for t in trades:
        arrow = "🟢" if t.direction == "LONG" else "🔴"
        pnl_str = f"  PnL: <code>${t.pnl_usd:+.4f}</code>" if t.status == "LIVE" else ""
        lines.append(
            f"{arrow} <code>{t.symbol}</code>  [{t.status}]"
            f"  entry=<code>{t.entry_price:.5f}</code>"
            f"  qty=<code>{t.qty:.4f}</code>"
            f"  SL=<code>{t.stop_loss:.5f}</code>"
            f"{pnl_str}"
        )
    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@dp.message(Command("pnl"))
async def cmd_pnl(message: types.Message) -> None:
    if not AUTO_EXECUTE:
        await message.answer("ℹ️ Auto-execution disabled — no live positions tracked.",
                              parse_mode=ParseMode.HTML)
        return
    await message.answer("⏳ Fetching live PnL…", parse_mode=ParseMode.HTML)
    try:
        pnl_data = await get_all_live_pnl()
        if not pnl_data:
            await message.answer("📭 No live positions open.", parse_mode=ParseMode.HTML)
            return
        for rec, pnl_usd, pnl_pct in pnl_data:
            await send_pnl_update(rec, pnl_usd, pnl_pct)
    except Exception as exc:
        logger.error("/pnl error: %s", exc, exc_info=True)
        await message.answer(f"❌ PnL fetch error: {exc}", parse_mode=ParseMode.HTML)


@dp.message(Command("btcstatus"))
async def cmd_btcstatus(message: types.Message) -> None:
    try:
        change = await get_btc_5min_change()
        crash  = is_crash_active()
        emoji  = "🚨" if crash else ("📉" if (change or 0) < -1.0 else "📊")
        chg_str = f"{change:+.2f}%" if change is not None else "N/A"
        status  = "🚨 CRASH PROTOCOL ACTIVE" if crash else "✅ Normal"
        await message.answer(
            f"{emoji} <b>BTC Market Status</b>\n\n"
            f"₿ <b>5-min change:</b>  <code>{chg_str}</code>\n"
            f"🛡 <b>Crash protocol:</b>  {status}\n"
            "<i>Crash triggers at −2% drop in 5 min.</i>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        await message.answer(f"❌ BTC status error: {exc}", parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────────────────────────
# ── Main entry ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("Starting SMC Sniper Bot  [mode=%s]  [exchange=Binance Futures]",
                "AUTO-EXECUTE" if AUTO_EXECUTE else "ALERT-ONLY")

    # Register Telegram as the error sink for all CCXT exceptions
    if AUTO_EXECUTE:
        register_error_callback(
            lambda msg: bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=msg,
                parse_mode=ParseMode.HTML,
            )
        )

    asyncio.create_task(hourly_scanner_loop())       # 60-min HTF scan
    asyncio.create_task(sniper_loop())               # 5-s sweep detection
    asyncio.create_task(choch_monitor_loop())        # 2-s CHoCH confirmation
    asyncio.create_task(execution_loop())            # 15-s fill / cancel / close
    asyncio.create_task(pnl_tracking_loop())         # 30-s PnL edit-in-place
    asyncio.create_task(btc_crash_monitor_loop())    # 30-s BTC crash guard
    asyncio.create_task(_idle_cleanup_loop())        # 60-s idle GC + file cleanup

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
