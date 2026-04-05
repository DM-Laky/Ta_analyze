"""
core/data_fetcher.py
=====================
Connects to MetaTrader 5 via the MetaTrader5 Python package.
Provides clean pandas DataFrames of OHLCV candles for any symbol/timeframe.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import config
from utils.logger import log


# ── Timeframe mapping ─────────────────────────────────────────────────────────
TF_MAP: dict[str, int] = {}
if MT5_AVAILABLE:
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }


class MT5DataFetcher:
    """Thread-safe MT5 data fetcher with auto-reconnect."""

    def __init__(self):
        self._connected = False
        self._last_reconnect = 0.0
        self._lock = threading.Lock()  # MT5 C++ API is not thread-safe

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            log.warning("MetaTrader5 package not installed — running in DEMO mode")
            return False

        if not mt5.initialize():
            log.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        if config.MT5_LOGIN and config.MT5_PASSWORD:
            ok = mt5.login(
                login=config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER,
            )
            if not ok:
                log.error("MT5 login failed: %s", mt5.last_error())
                return False

        info = mt5.account_info()
        if info:
            log.info(
                "✅ MT5 Connected | Account: %s | Server: %s | Balance: %.2f %s",
                info.login, info.server, info.balance, info.currency
            )
        self._connected = True
        return True

    def disconnect(self):
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        log.info("MT5 disconnected.")

    def _ensure_connected(self) -> bool:
        if self._connected:
            return True
        now = time.time()
        if now - self._last_reconnect < 30:
            return False
        self._last_reconnect = now
        return self.connect()

    # ── Data Fetching ─────────────────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch `count` most-recent candles for `symbol` on `timeframe`.

        Returns a DataFrame with columns:
            time, open, high, low, close, tick_volume, spread, real_volume
        Index is reset integer.
        Returns None on failure.
        """
        if not self._ensure_connected():
            log.warning("MT5 not connected — cannot fetch %s %s", symbol, timeframe)
            return self._demo_candles(symbol, timeframe, count)

        tf = TF_MAP.get(timeframe)
        if tf is None:
            log.error("Unknown timeframe: %s", timeframe)
            return None

        with self._lock:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                log.error(
                    "No data returned for %s %s — error: %s",
                    symbol, timeframe, mt5.last_error()
                )
                return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_current_price(self, symbol: str) -> Optional[dict]:
        """Return latest bid/ask for a symbol."""
        if not self._ensure_connected():
            return None
        with self._lock:
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": (tick.bid + tick.ask) / 2,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        if not self._ensure_connected():
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "point": info.point,
            "digits": info.digits,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_step": info.volume_step,
            "spread": info.spread,
        }

    # ── Demo / Fallback ───────────────────────────────────────────────────────

    def _demo_candles(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """Generate synthetic candles for testing without MT5."""
        import numpy as np

        log.debug("Generating DEMO candles for %s %s", symbol, timeframe)
        np.random.seed(42)

        base_prices = {
            "XAUUSD": 2300.0,
            "EURUSD": 1.0850,
            "GBPUSD": 1.2700,
        }
        base = base_prices.get(symbol, 1.0)
        tick_size = config.SYMBOL_PIP.get(symbol, 0.0001)

        now = pd.Timestamp.utcnow().floor("T")
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        interval = pd.Timedelta(minutes=tf_minutes.get(timeframe, 60))

        times = [now - interval * (count - 1 - i) for i in range(count)]
        closes = [base]
        for _ in range(count - 1):
            closes.append(closes[-1] + np.random.normal(0, tick_size * 5))

        rows = []
        for i, (t, c) in enumerate(zip(times, closes)):
            o = closes[i - 1] if i > 0 else c
            hi = max(o, c) + abs(np.random.normal(0, tick_size * 3))
            lo = min(o, c) - abs(np.random.normal(0, tick_size * 3))
            rows.append({
                "time": t,
                "open": round(o, 5),
                "high": round(hi, 5),
                "low": round(lo, 5),
                "close": round(c, 5),
                "volume": int(np.random.randint(100, 10000)),
                "spread": 2,
                "real_volume": 0,
            })

        return pd.DataFrame(rows)


# ── Module-level singleton ────────────────────────────────────────────────────
fetcher = MT5DataFetcher()
