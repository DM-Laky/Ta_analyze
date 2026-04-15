from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    db_path: str = os.getenv("DB_PATH", "trading_system.db")
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.04"))
    sl_buffer_ratio: float = float(os.getenv("SL_BUFFER_RATIO", "0.0008"))
    max_active_trades: int = int(os.getenv("MAX_ACTIVE_TRADES", "2"))
    symbol_cooldown_min: int = int(os.getenv("SYMBOL_COOLDOWN_MIN", "30"))
    watchlist_expiry_hours: int = int(os.getenv("WATCHLIST_EXPIRY_HOURS", "8"))
    price_monitor_seconds: int = int(os.getenv("PRICE_MONITOR_SECONDS", "2"))
    setup_reaction_ratio: float = float(os.getenv("SETUP_REACTION_RATIO", "0.25"))
    use_limit_retest: bool = os.getenv("USE_LIMIT_RETEST", "false").lower() == "true"


settings = Settings()
