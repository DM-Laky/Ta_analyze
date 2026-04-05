"""
main.py — GOLD HUNTER PRO
===========================
Main 24/7 orchestrator.

Loop architecture:
  • Every 60 min (London/NY session): Deep SMC scan → watchlist adds
  • Every 30 sec: Watchlist price-check → POI trigger
  • Every 10 sec: Triggered entries → 1M confirmation → entry signal
  • Cleanup:  expired watchlist entries removed every 30 min

Run:
    python main.py
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from analysis.smc_engine import smc_engine, SetupCandidate
from config import config
from core.data_fetcher import fetcher
from core.session_manager import session_manager
from signals.entry_engine import entry_engine, EntrySignal
from signals.watchlist import watchlist, WatchlistEntry
from telegram.bot import telegram_bot
from telegram.chart_generator import chart_generator
from telegram.message_formatter import (
    format_entry_signal,
    format_poi_triggered_alert,
    format_startup_message,
    format_system_status,
    format_watchlist_alert,
)
from utils.logger import log, setup_logger

# ── Setup ─────────────────────────────────────────────────────────────────────

setup_logger(level=config.LOG_LEVEL, log_file=config.LOG_FILE)

START_TIME = datetime.now(timezone.utc)


# ── Job: Deep SMC Analysis ────────────────────────────────────────────────────

def job_deep_analysis():
    """
    Runs every DEEP_ANALYSIS_INTERVAL_MIN minutes.
    Skips if not in an active session.
    """
    session = session_manager.get_current_session()
    session_manager.log_session()

    if not session.is_active:
        log.info("⏸️  Off-session — skipping deep analysis")
        return

    log.info("=" * 60)
    log.info("🔬 DEEP SMC ANALYSIS — %s", session)
    log.info("=" * 60)

    for symbol in config.SYMBOLS:
        try:
            candidates = smc_engine.analyze(symbol)
            log.info(
                "   %s → %d candidates found",
                symbol, len(candidates),
            )

            for candidate in candidates:
                entry = watchlist.add(candidate)
                # Callback fires automatically if on_watchlist_add is set

        except Exception as exc:
            log.exception("Deep analysis error for %s: %s", symbol, exc)


# ── Job: Price Check (Watchlist Monitor) ─────────────────────────────────────

def job_price_check():
    """Every 30s — check if price entered any POI."""
    try:
        watchlist.check_prices()
    except Exception as exc:
        log.exception("Price check error: %s", exc)


# ── Job: Entry Confirmation ───────────────────────────────────────────────────

def job_entry_confirmation():
    """
    Every 10s — for all triggered watchlist entries, check 1M for confirmation.
    """
    triggered = watchlist.get_triggered_entries()
    if not triggered:
        return

    log.debug("Checking %d triggered entries for confirmation", len(triggered))

    for entry in triggered:
        try:
            price_info = fetcher.get_current_price(entry.symbol)
            if price_info is None:
                continue

            current_price = price_info["mid"]

            # Pre-fetch M15 once — reused by entry_engine for chart
            df_m15 = fetcher.get_candles(entry.symbol, "M15", config.CHART_CANDLES_SHOWN)
            signal, confirmation = entry_engine.check(entry, current_price, df_m15=df_m15)

            if signal:
                log.info("🎯 ENTRY SIGNAL CONFIRMED: %s", signal)
                watchlist.mark_confirmed(entry.uid)
                _send_entry_signal(signal)

        except Exception as exc:
            log.exception(
                "Entry confirmation error for %s: %s", entry.uid, exc
            )


# ── Job: Cleanup ──────────────────────────────────────────────────────────────

def job_cleanup():
    """Every 30 min — remove expired watchlist entries."""
    watchlist.cleanup_expired()
    log.info("🧹 Cleanup done. %s", watchlist.summary())


# ── Job: Status Heartbeat ─────────────────────────────────────────────────────

def job_status():
    """Every 6 hours — send a status ping to Telegram."""
    uptime = datetime.now(timezone.utc) - START_TIME
    hours, rem = divmod(int(uptime.total_seconds()), 3600)
    minutes = rem // 60
    uptime_str = f"{hours}h {minutes}m"

    session = session_manager.get_current_session()
    text = format_system_status(
        watchlist_summary=watchlist.summary(),
        session_str=str(session),
        uptime_str=uptime_str,
    )
    telegram_bot.send_text(text)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _on_watchlist_add(entry: WatchlistEntry):
    """Send watchlist alert to Telegram when new POI is added."""
    log.info("📢 Sending watchlist alert: %s", entry)
    try:
        c = entry.candidate

        # Fetch M15 candles for chart
        df = fetcher.get_candles(c.symbol, "M15", config.CHART_CANDLES_SHOWN)

        text = format_watchlist_alert(entry)

        if df is not None:
            chart = chart_generator.watchlist_chart(df, c, title=None)
            # Send chart + text separately
            telegram_bot.send_photo(
                chart,
                caption=f"📋 WATCHLIST | {c.symbol} {c.direction} | {c.poi_type}",
            )

        telegram_bot.send_text(text)

    except Exception as exc:
        log.exception("Error sending watchlist alert: %s", exc)


def _on_poi_triggered(entry: WatchlistEntry, price: float):
    """Send a quick triggered alert."""
    log.info("⚡ Sending POI triggered alert: %s @ %.5f", entry, price)
    try:
        text = format_poi_triggered_alert(entry, price)
        telegram_bot.send_text(text)
    except Exception as exc:
        log.exception("Error sending triggered alert: %s", exc)


def _on_expired(entry: WatchlistEntry):
    log.info("⏱️ Watchlist entry expired: %s", entry.uid)
    # Optionally notify Telegram
    # telegram_bot.send_text(f"⏱️ POI expired: {entry.symbol} {entry.direction} `{entry.uid}`")


def _send_entry_signal(signal: EntrySignal):
    """Send the full entry signal with chart."""
    try:
        text = format_entry_signal(signal)

        if signal.df_m15 is not None:
            chart = chart_generator.entry_chart(signal.df_m15, signal)
            telegram_bot.send_photo(
                chart,
                caption=(
                    f"🎯 ENTRY | {signal.symbol} {signal.direction} "
                    f"| SL: {signal.stop_loss:.5f} | TP1: {signal.tp1:.5f}"
                ),
            )

        telegram_bot.send_text(text)
        log.info("📤 Entry signal sent: %s", signal.summary_line())

    except Exception as exc:
        log.exception("Error sending entry signal: %s", exc)


# ── Startup ───────────────────────────────────────────────────────────────────

def startup():
    log.info("╔══════════════════════════════════════╗")
    log.info("║     GOLD HUNTER PRO — STARTING       ║")
    log.info("╚══════════════════════════════════════╝")

    # Connect MT5
    mt5_ok = fetcher.connect()
    if not mt5_ok:
        log.warning("MT5 not connected — running in demo/simulation mode")

    # Connect Telegram callbacks
    watchlist.on_watchlist_add = _on_watchlist_add
    watchlist.on_poi_triggered  = _on_poi_triggered
    watchlist.on_expired        = _on_expired

    # Start Telegram bot
    telegram_bot.start()

    if telegram_bot.is_configured:
        telegram_bot.send_text(format_startup_message())

    log.info("✅ All systems online")


def shutdown(signum=None, frame=None):
    log.info("🛑 Shutting down Gold Hunter Pro...")
    telegram_bot.stop()
    fetcher.disconnect()
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    startup()

    # Graceful shutdown on SIGINT / SIGTERM
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler = BackgroundScheduler(timezone="UTC")

    # Deep analysis — top of every hour
    scheduler.add_job(
        job_deep_analysis,
        trigger=IntervalTrigger(minutes=config.DEEP_ANALYSIS_INTERVAL_MIN),
        id="deep_analysis",
        next_run_time=datetime.now(timezone.utc),  # Run immediately on start
    )

    # Price check — every 30s
    scheduler.add_job(
        job_price_check,
        trigger=IntervalTrigger(seconds=config.WATCHLIST_CHECK_INTERVAL_SEC),
        id="price_check",
    )

    # Entry confirmation — every 10s
    scheduler.add_job(
        job_entry_confirmation,
        trigger=IntervalTrigger(seconds=config.LTF_ENTRY_CHECK_INTERVAL_SEC),
        id="entry_confirmation",
    )

    # Cleanup — every 30 min
    scheduler.add_job(
        job_cleanup,
        trigger=IntervalTrigger(minutes=30),
        id="cleanup",
    )

    # Status heartbeat — every 6 hours
    scheduler.add_job(
        job_status,
        trigger=IntervalTrigger(hours=6),
        id="status_heartbeat",
    )

    scheduler.start()
    log.info(
        "⏰ Scheduler started | Jobs: %d",
        len(scheduler.get_jobs()),
    )

    log.info("")
    log.info("🟢 GOLD HUNTER PRO IS LIVE — 24/7 MODE")
    log.info("   Press Ctrl+C to stop.")
    log.info("")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
