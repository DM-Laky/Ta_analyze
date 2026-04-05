"""
telegram/message_formatter.py
===============================
Formats all Telegram alert messages.
Uses MarkdownV2 for rich formatting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from analysis.smc_engine import SetupCandidate
from signals.entry_engine import EntrySignal
from signals.watchlist import WatchlistEntry


def _esc(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in str(text))


def _pct(score: float) -> str:
    return f"{score * 100:.0f}%"


def _bars(score: float, n: int = 10) -> str:
    filled = round(score * n)
    return "█" * filled + "░" * (n - filled)


def format_watchlist_alert(entry: WatchlistEntry) -> str:
    """
    Alert sent when a new POI is added to watchlist.
    'Watching for price to enter this zone.'
    """
    c = entry.candidate
    sym_display = c.symbol
    dir_emoji = "🟢" if c.direction == "BUY" else "🔴"
    trend_emoji = {
        "BULLISH": "📈",
        "BEARISH": "📉",
        "RANGING": "↔️",
    }.get(c.h1_trend.value, "—")

    score_bar = _bars(c.setup_score)
    score_pct = _pct(c.setup_score)

    tp1_str = f"{c.tp1:.5f}" if c.tp1 else "—"
    tp2_str = f"{c.tp2:.5f}" if c.tp2 else "—"
    tp3_str = f"{c.tp3:.5f}" if c.tp3 else "—"

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    poi_type_emoji = {
        "OB+FVG": "🔥",
        "OB": "📦",
        "FVG": "⚡",
    }.get(c.poi_type, "📍")

    rr = c.rr_estimate
    session_str = f"{c.session.emoji} {c.session.name}"

    text = (
        f"📋 *WATCHLIST ALERT* — {_esc(sym_display)}\n"
        f"{'─' * 32}\n"
        f"{dir_emoji} *Direction:* {_esc(c.direction)}\n"
        f"{trend_emoji} *H1 Trend:* {_esc(c.h1_trend.value)}\n"
        f"{poi_type_emoji} *POI Type:* {_esc(c.poi_type)}\n"
        f"\n"
        f"📍 *POI Zone:*\n"
        f"   Top    ➜ `{_esc(f'{c.poi_top:.5f}')}`\n"
        f"   Mid    ➜ `{_esc(f'{c.poi_mid:.5f}')}`\n"
        f"   Bottom ➜ `{_esc(f'{c.poi_bottom:.5f}')}`\n"
        f"\n"
        f"🎯 *Liquidity Targets:*\n"
        f"   TP1 ➜ `{_esc(tp1_str)}`\n"
        f"   TP2 ➜ `{_esc(tp2_str)}`\n"
        f"   TP3 ➜ `{_esc(tp3_str)}`\n"
        f"\n"
        f"📊 *Est\\. R:R:* `{_esc(f'{rr:.1f}R')}`\n"
        f"⭐ *Setup Score:* `{_esc(score_pct)}` {_esc(score_bar)}\n"
        f"\n"
        f"💡 *Reason:* _{_esc(c.reason[:200])}_\n"
        f"\n"
        f"🕐 *Session:* {_esc(session_str)}\n"
        f"⏰ *Time:* {_esc(now)}\n"
        f"🆔 ID: `{_esc(entry.uid)}`\n"
        f"{'─' * 32}\n"
        f"⏳ _Waiting for price to enter POI\\.\\.\\._"
    )
    return text


def format_entry_signal(signal: EntrySignal) -> str:
    """
    The main sniper entry alert — the one you trade from.
    """
    dir_emoji = "🟢" if signal.direction == "BUY" else "🔴"
    dir_word   = "BUY  📈" if signal.direction == "BUY" else "SELL 📉"

    conf_method = {
        "CHOCH": "🔄 CHoCH (Change of Character)",
        "VSHAPE": "⚡ V\\-Shape Momentum",
        "BOTH": "🔥 CHoCH \\+ V\\-Shape",
        "NONE": "—",
    }.get(signal.confirmation.method, signal.confirmation.method)

    conf_bar = _bars(signal.confirmation.confidence)
    conf_pct = _pct(signal.confirmation.confidence)

    entries_text = "\n".join(
        f"   E{i+1} ➜ `{_esc(f'{ep:.5f}')}`"
        for i, ep in enumerate(signal.entries)
    )

    tp2_line = (
        f"   TP2 ➜ `{_esc(f'{signal.tp2:.5f}')}` "
        f"_{_esc(f'({signal.rr2:.1f}R)') if signal.rr2 else ''}_\n"
        if signal.tp2 is not None else ""
    )
    tp3_line = (
        f"   TP3 ➜ `{_esc(f'{signal.tp3:.5f}')}`\n"
        if signal.tp3 is not None else ""
    )

    now = signal.signal_time.strftime("%H:%M:%S UTC")
    score_bar = _bars(signal.setup_score)

    risk_pips = signal.risk_pips
    poi_type_emoji = {"OB+FVG": "🔥", "OB": "📦", "FVG": "⚡"}.get(signal.poi_type, "📍")

    text = (
        f"{'═' * 34}\n"
        f"🎯 *SNIPER ENTRY SIGNAL*\n"
        f"{'═' * 34}\n"
        f"\n"
        f"{dir_emoji} *{_esc(signal.symbol)}  {_esc(dir_word)}*\n"
        f"\n"
        f"{'─' * 34}\n"
        f"📌 *ENTRY ZONE* \\({_esc(str(len(signal.entries)))} orders\\):\n"
        f"{entries_text}\n"
        f"\n"
        f"🛑 *STOP LOSS:*\n"
        f"   SL  ➜ `{_esc(f'{signal.stop_loss:.5f}')}` _{_esc(f'({risk_pips:.0f} pips)')}_\n"
        f"\n"
        f"✅ *TAKE PROFITS:*\n"
        f"   TP1 ➜ `{_esc(f'{signal.tp1:.5f}')}` _\\({_esc(f'{signal.rr1:.1f}R')}\\)_\n"
        f"{tp2_line}"
        f"{tp3_line}"
        f"{'─' * 34}\n"
        f"\n"
        f"🔍 *CONFIRMATION:* {conf_method}\n"
        f"   Confidence: `{_esc(conf_pct)}` {_esc(conf_bar)}\n"
        f"   _{_esc(signal.confirmation.details[:150])}_\n"
        f"\n"
        f"{poi_type_emoji} *POI Type:* {_esc(signal.poi_type)}\n"
        f"📊 *H1 Trend:* {_esc(signal.h1_trend.value)}\n"
        f"🌐 *Session:* {_esc(signal.session_name)}\n"
        f"\n"
        f"⭐ *Signal Score:* {_esc(_bars(signal.setup_score))} `{_esc(_pct(signal.setup_score))}`\n"
        f"\n"
        f"{'═' * 34}\n"
        f"⏰ {_esc(now)}  🆔 `{_esc(signal.uid)}`\n"
        f"{'─' * 34}\n"
        f"⚠️ _This is a manual entry signal\\.\nAlways manage your risk\\!_"
    )
    return text


def format_poi_triggered_alert(entry: WatchlistEntry, price: float) -> str:
    """Short alert when price enters POI — before confirmation."""
    dir_emoji = "🟢" if entry.direction == "BUY" else "🔴"
    return (
        f"⚡ *POI TRIGGERED* — {_esc(entry.symbol)}\n"
        f"{'─' * 28}\n"
        f"{dir_emoji} *{_esc(entry.direction)}* | Price `{_esc(f'{price:.5f}')}`\n"
        f"Zone: `{_esc(f'{entry.poi_bottom:.5f}')}` → `{_esc(f'{entry.poi_top:.5f}')}`\n"
        f"🆔 `{_esc(entry.uid)}`\n"
        f"{'─' * 28}\n"
        f"🔍 _Checking 1M for CHoCH \\/ V\\-Shape\\.\\.\\._"
    )


def format_startup_message() -> str:
    now = datetime.now(timezone.utc).strftime("%Y\\-%m\\-%d %H:%M UTC")
    return (
        f"🚀 *GOLD HUNTER PRO — ONLINE*\n"
        f"{'═' * 32}\n"
        f"✅ MT5 Connected\n"
        f"✅ SMC Engine Ready\n"
        f"✅ Telegram Bot Active\n"
        f"✅ Watchlist Initialized\n"
        f"\n"
        f"📊 *Symbols:* XAUUSD \\| EURUSD \\| GBPUSD\n"
        f"⏰ *Started:* {now}\n"
        f"{'═' * 32}\n"
        f"🎯 _Hunting precision entries\\.\\.\\._"
    )


def format_system_status(
    watchlist_summary: str,
    session_str: str,
    uptime_str: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return (
        f"📡 *SYSTEM STATUS*\n"
        f"{'─' * 28}\n"
        f"🌐 Session: {_esc(session_str)}\n"
        f"📋 {_esc(watchlist_summary)}\n"
        f"⏱️ Uptime: {_esc(uptime_str)}\n"
        f"⏰ {_esc(now)}"
    )
