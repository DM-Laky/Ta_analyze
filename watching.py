"""
watching.py — The Approach Monitor (Watchlist 1 → Watchlist 2)
Runs every minute (triggered by app.py).

1. Scan data/ for JSON files with status == "WATCHLIST_1".
2. Fetch the last few 1m candles (not just ticker CMP).
3. If any candle's HIGH/LOW wick intersected the htf_poi zone
   → promote to "WATCHLIST_2".

This catches fast wicks during news events that touch the POI and
retrace within one minute — something a single CMP check would miss.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from execution import get_public_exchange

logger = logging.getLogger("watching")

DATA_DIR = Path(__file__).parent / "data"

WICK_CHECK_CANDLES = 5  # check the last 5 one-minute candles


def _load_watchlist(status: str) -> List[dict]:
    """Load all JSON entries from data/ with the given status."""
    entries: List[dict] = []
    if not DATA_DIR.exists():
        return entries
    for fpath in DATA_DIR.glob("*.json"):
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            if data.get("status") == status:
                data["_path"] = str(fpath)
                entries.append(data)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fpath, exc)
    return entries


def _save_json(path: str, data: dict) -> None:
    """Atomic-ish write of JSON data."""
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    clean["updated_ts"] = time.time()
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


async def _fetch_recent_1m(
    exchange: ccxt.Exchange,
    symbol: str,
) -> Optional[pd.DataFrame]:
    """Fetch the last WICK_CHECK_CANDLES 1-minute candles."""
    since_ms = int((time.time() - WICK_CHECK_CANDLES * 60 - 60) * 1000)
    try:
        raw = await exchange.fetch_ohlcv(
            symbol, "1m", since=since_ms, limit=WICK_CHECK_CANDLES,
        )
    except Exception as exc:
        logger.warning("1m OHLCV fetch failed for %s: %s", symbol, exc)
        return None
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def _wick_touched_zone(
    df: pd.DataFrame,
    poi_high: float,
    poi_low: float,
) -> bool:
    """
    Check if ANY candle's wick (high/low range) intersected the POI zone.
    Two ranges [candle_low, candle_high] and [poi_low, poi_high] overlap
    when candle_low <= poi_high AND candle_high >= poi_low.
    """
    for i in range(len(df)):
        candle_low = df["low"].iat[i]
        candle_high = df["high"].iat[i]
        if candle_low <= poi_high and candle_high >= poi_low:
            return True
    return False


async def run_watcher() -> List[dict]:
    """
    Check WATCHLIST_1 coins.  If recent 1m candle wicks intersected the
    HTF POI zone, promote to WATCHLIST_2.
    """
    entries = _load_watchlist("WATCHLIST_1")
    if not entries:
        return []

    exchange = await get_public_exchange()
    promoted: List[dict] = []

    try:
        for entry in entries:
            symbol = entry.get("symbol")
            poi = entry.get("htf_poi")
            if not symbol or not poi:
                continue

            poi_high = poi.get("high")
            poi_low = poi.get("low")
            if poi_high is None or poi_low is None:
                continue

            df_1m = await _fetch_recent_1m(exchange, symbol)
            if df_1m is None or len(df_1m) == 0:
                continue

            if _wick_touched_zone(df_1m, poi_high, poi_low):
                cmp = float(df_1m["close"].iat[-1])
                logger.info(
                    "POI TOUCHED %s → WATCHLIST_2_POI_TOUCHED  "
                    "(1m wick touched POI [%.6f, %.6f], CMP=%.6f)",
                    symbol, poi_low, poi_high, cmp,
                )
                entry["status"] = "WATCHLIST_2_POI_TOUCHED"
                entry["entered_poi_ts"] = time.time()
                entry["entered_poi_price"] = cmp
                _save_json(entry["_path"], entry)
                promoted.append(entry)
    except Exception as exc:
        logger.warning("run_watcher exchange error: %s", exc)

    # Enforce WATCHLIST_2 capacity: keep at most 20 coins across all stage-2 statuses
    STAGE2_STATUSES = {"WATCHLIST_2_POI_TOUCHED", "WATCHLIST_2_SWEEP"}
    MAX_WATCHLIST_2 = 20
    wl2_files: list = []
    for fpath in DATA_DIR.glob("*.json"):
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            if data.get("status") in STAGE2_STATUSES:
                ts = data.get("entered_poi_ts", data.get("created_ts", fpath.stat().st_mtime))
                wl2_files.append((ts, fpath))
        except Exception:
            pass

    if len(wl2_files) > MAX_WATCHLIST_2:
        wl2_files.sort(key=lambda x: x[0])   # oldest first
        to_evict = wl2_files[:len(wl2_files) - MAX_WATCHLIST_2]
        for _, old_path in to_evict:
            try:
                old_path.unlink()
                logger.info("Evicted oldest WATCHLIST_2 (cap %d): %s",
                            MAX_WATCHLIST_2, old_path.name)
            except Exception:
                pass

    return promoted
