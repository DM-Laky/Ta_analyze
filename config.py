"""
GOLD HUNTER PRO — Configuration
================================
All settings in one place. Edit before running.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ─── MT5 Connection ────────────────────────────────────────────────────────
    MT5_LOGIN: int = 159971596                         # Your Exness MT5 account number
    MT5_PASSWORD: str = "Laky@816"                     # MT5 password
    MT5_SERVER: str = "Exness-MT5Real20"        # Exness server name

    # ─── Telegram ──────────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = "8155409741:AAFJJpARiuRnH2Yqpyq0r1T_eI6jGd82CAQ"               # From @BotFather
    TELEGRAM_CHAT_ID: str = "5530953384"                 # Your chat/channel ID

    # ─── Symbols ───────────────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(
        default_factory=lambda: ["XAUUSD", "EURUSD", "GBPUSD"]
    )
    SYMBOL_DISPLAY: dict = field(
        default_factory=lambda: {
            "XAUUSD": "GOLD (XAU/USD) 🥇",
            "EURUSD": "EUR/USD 🇪🇺",
            "GBPUSD": "GBP/USD 🇬🇧",
        }
    )
    SYMBOL_PIP: dict = field(
        default_factory=lambda: {
            "XAUUSD": 0.01,    # 1 pip = $0.01 for gold
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
        }
    )

    # ─── Timeframes ────────────────────────────────────────────────────────────
    HTF_CANDLES: int = 100      # 1H candles for market structure
    MTF_CANDLES: int = 100      # 15M candles for OB / FVG
    LTF_CANDLES: int = 20       # 1M candles for entry confirmation

    # ─── Sessions (UTC) ────────────────────────────────────────────────────────
    LONDON_START_UTC: int = 7   # 07:00 UTC = 08:00 London (BST)
    LONDON_END_UTC: int = 16    # 16:00 UTC
    NY_START_UTC: int = 13      # 13:00 UTC = 09:00 NY (EDT)
    NY_END_UTC: int = 21        # 21:00 UTC
    # Analysis also runs at overlap (London+NY = 13:00–16:00 UTC) — premium zone

    # ─── SMC Detection Parameters ──────────────────────────────────────────────
    SWING_LEFT_BARS: int = 3    # Bars left of swing to confirm
    SWING_RIGHT_BARS: int = 3   # Bars right of swing to confirm

    FVG_MIN_PIPS: float = 5.0   # Minimum FVG size in pips (XAUUSD: points)
    OB_MIN_PIPS: float = 3.0    # Minimum OB size
    LIQ_CLUSTER_TOLERANCE: float = 0.0020   # % price tolerance for equal H/L cluster

    # ─── POI Watchlist ─────────────────────────────────────────────────────────
    POI_HIT_TOLERANCE: float = 0.0015       # 0.15% — price considered "at POI"
    WATCHLIST_EXPIRY_HOURS: int = 8         # Auto-expire POI after N hours
    MAX_WATCHLIST_PER_SYMBOL: int = 5       # Keep only top-N POIs per symbol

    # ─── Entry Confirmation (1M) ───────────────────────────────────────────────
    CONFIRMATION_LOOKBACK: int = 20         # 1M candles to scan for CHoCH/V-shape
    VSHAPE_RETRACE_MIN: float = 0.50        # V-shape must retrace ≥50% of the wick
    CHOCH_SWING_BARS: int = 2              # Swing bars for 1M CHoCH detection

    # ─── Entry & Risk ──────────────────────────────────────────────────────────
    ENTRY_SPLITS: int = 5                  # Divide entry zone into 5 laddered orders
    SL_BUFFER_PIPS: float = 8.0            # Extra buffer beyond OB/FVG for SL
    MIN_RR: float = 1.5                    # Skip signal if R:R < this
    TP_LEVELS: int = 3                     # TP1, TP2, TP3

    # ─── Scheduling ────────────────────────────────────────────────────────────
    DEEP_ANALYSIS_INTERVAL_MIN: int = 60       # Deep scan every 60 min
    WATCHLIST_CHECK_INTERVAL_SEC: int = 30     # Watchlist price-check every 30s
    LTF_ENTRY_CHECK_INTERVAL_SEC: int = 10     # 1M confirmation every 10s

    # ─── Chart ─────────────────────────────────────────────────────────────────
    CHART_CANDLES_SHOWN: int = 60          # Candles to display in chart
    CHART_WIDTH_IN: float = 14.0
    CHART_HEIGHT_IN: float = 8.0
    CHART_DPI: int = 150
    CHART_STYLE: str = "dark"              # "dark" | "light"

    # ─── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/gold_hunter.log"
    LOG_MAX_BYTES: int = 10_000_000        # 10 MB
    LOG_BACKUP_COUNT: int = 5


    def __post_init__(self):
        """Validate critical settings on startup."""
        if self.FVG_MIN_PIPS <= 0:
            raise ValueError("FVG_MIN_PIPS must be > 0")
        if self.OB_MIN_PIPS <= 0:
            raise ValueError("OB_MIN_PIPS must be > 0")
        if self.MIN_RR <= 0:
            raise ValueError("MIN_RR must be > 0")
        if self.ENTRY_SPLITS < 2:
            raise ValueError("ENTRY_SPLITS must be >= 2")
        if self.WATCHLIST_EXPIRY_HOURS <= 0:
            raise ValueError("WATCHLIST_EXPIRY_HOURS must be > 0")


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val and val.strip().lstrip("-").isdigit() else default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default) or default


def _env_list(key: str, default: list) -> list:
    val = os.getenv(key)
    if val:
        return [s.strip() for s in val.split(",") if s.strip()]
    return default


# ── Singleton ─────────────────────────────────────────────────────────────────
config = Config(
    # MT5
    MT5_LOGIN=_env_int("MT5_LOGIN", 0),
    MT5_PASSWORD=_env_str("MT5_PASSWORD", ""),
    MT5_SERVER=_env_str("MT5_SERVER", "Exness-MT5Trial"),
    # Telegram
    TELEGRAM_BOT_TOKEN=_env_str("TELEGRAM_BOT_TOKEN", ""),
    TELEGRAM_CHAT_ID=_env_str("TELEGRAM_CHAT_ID", ""),
    # Symbols
    SYMBOLS=_env_list("SYMBOLS", ["XAUUSD", "EURUSD", "GBPUSD"]),
    # Candles
    HTF_CANDLES=_env_int("HTF_CANDLES", 100),
    MTF_CANDLES=_env_int("MTF_CANDLES", 100),
    LTF_CANDLES=_env_int("LTF_CANDLES", 20),
    # Sessions
    LONDON_START_UTC=_env_int("LONDON_START_UTC", 7),
    LONDON_END_UTC=_env_int("LONDON_END_UTC", 16),
    NY_START_UTC=_env_int("NY_START_UTC", 13),
    NY_END_UTC=_env_int("NY_END_UTC", 21),
    # SMC parameters
    SWING_LEFT_BARS=_env_int("SWING_LEFT_BARS", 3),
    SWING_RIGHT_BARS=_env_int("SWING_RIGHT_BARS", 3),
    FVG_MIN_PIPS=_env_float("FVG_MIN_PIPS", 5.0),
    OB_MIN_PIPS=_env_float("OB_MIN_PIPS", 3.0),
    LIQ_CLUSTER_TOLERANCE=_env_float("LIQ_CLUSTER_TOLERANCE", 0.0020),
    # Watchlist
    POI_HIT_TOLERANCE=_env_float("POI_HIT_TOLERANCE", 0.0015),
    WATCHLIST_EXPIRY_HOURS=_env_int("WATCHLIST_EXPIRY_HOURS", 8),
    MAX_WATCHLIST_PER_SYMBOL=_env_int("MAX_WATCHLIST_PER_SYMBOL", 5),
    # Entry confirmation
    CONFIRMATION_LOOKBACK=_env_int("CONFIRMATION_LOOKBACK", 20),
    VSHAPE_RETRACE_MIN=_env_float("VSHAPE_RETRACE_MIN", 0.50),
    CHOCH_SWING_BARS=_env_int("CHOCH_SWING_BARS", 2),
    # Risk
    ENTRY_SPLITS=_env_int("ENTRY_SPLITS", 5),
    SL_BUFFER_PIPS=_env_float("SL_BUFFER_PIPS", 8.0),
    MIN_RR=_env_float("MIN_RR", 1.5),
    TP_LEVELS=_env_int("TP_LEVELS", 3),
    # Scheduling
    DEEP_ANALYSIS_INTERVAL_MIN=_env_int("DEEP_ANALYSIS_INTERVAL_MIN", 60),
    WATCHLIST_CHECK_INTERVAL_SEC=_env_int("WATCHLIST_CHECK_INTERVAL_SEC", 30),
    LTF_ENTRY_CHECK_INTERVAL_SEC=_env_int("LTF_ENTRY_CHECK_INTERVAL_SEC", 10),
    # Chart
    CHART_CANDLES_SHOWN=_env_int("CHART_CANDLES_SHOWN", 60),
    CHART_WIDTH_IN=_env_float("CHART_WIDTH_IN", 14.0),
    CHART_HEIGHT_IN=_env_float("CHART_HEIGHT_IN", 8.0),
    CHART_DPI=_env_int("CHART_DPI", 150),
    CHART_STYLE=_env_str("CHART_STYLE", "dark"),
    # Logging
    LOG_LEVEL=_env_str("LOG_LEVEL", "INFO"),
    LOG_FILE=_env_str("LOG_FILE", "logs/gold_hunter.log"),
    LOG_MAX_BYTES=_env_int("LOG_MAX_BYTES", 10_000_000),
    LOG_BACKUP_COUNT=_env_int("LOG_BACKUP_COUNT", 5),
)
