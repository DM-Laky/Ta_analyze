"""
sniper.py — 3-Stage Sniper Room (Alert-Only, HFT 10s cadence)
=============================================================
State machine per coin:

  WATCHLIST_2_POI_TOUCHED  (set by watching.py)
    ↓  sweep detected, no CHoCH yet
  WATCHLIST_2_SWEEP         → Alert 2 sent (🟠 Liq Sweep detected)
    ↓  CHoCH confirmed after sweep
  ALERT_SENT                → Alert 3 sent (sniper signal + chart)
    ↓  JSON deleted immediately

Speed: fetch only last 100 candles for 5m/1m. Called every 10 seconds.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from execution import get_public_exchange
from smc import (
    SniperSignal, build_sniper_signal,
    detect_swing_points, detect_liquidity_sweeps,
)

logger = logging.getLogger("sniper")

DATA_DIR            = Path(__file__).parent / "data"
WL2_TIMEOUT_SECONDS = 7200   # 2 h max in sniper room
WL2_DRIFT_PCT       = 8.0    # invalidate if CMP drifts >8% from POI mid


# ── JSON helpers ──────────────────────────────────────────────────────

def _load_stage2() -> List[dict]:
    entries: List[dict] = []
    if not DATA_DIR.exists():
        return entries
    for fpath in DATA_DIR.glob("*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data.get("status") in ("WATCHLIST_2_POI_TOUCHED", "WATCHLIST_2_SWEEP"):
                data["_path"] = str(fpath)
                entries.append(data)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fpath, exc)
    return entries


def _save_json(path: str, data: dict) -> None:
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    clean["updated_ts"] = time.time()
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def _delete_json(path: str) -> None:
    try:
        os.unlink(path)
        logger.info("DELETED %s", Path(path).name)
    except Exception as exc:
        logger.warning("Failed to delete %s: %s", path, exc)


def _purge_alert_sent() -> None:
    if not DATA_DIR.exists():
        return
    for fpath in DATA_DIR.glob("*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data.get("status") == "ALERT_SENT":
                fpath.unlink()
                logger.info("Purged ALERT_SENT: %s", fpath.name)
        except Exception:
            pass


# ── Fast 100-candle LTF fetch ─────────────────────────────────────────

async def _fetch_ltf(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 100,
) -> Optional[pd.DataFrame]:
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as exc:
        logger.warning("LTF %s %s: %s", symbol, timeframe, exc)
        return None
    if not raw:
        return None
    df = pd.DataFrame(
        raw, columns=["timestamp","open","high","low","close","volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(str)
    return df


# ── Sweep-only check (no CHoCH required) ─────────────────────────────

def _has_recent_sweep(
    df_ltf: pd.DataFrame,
    sweep_dir: str,
    poi_high: float,
    poi_low: float,
    lb: int = 2,
) -> Optional[dict]:
    swings = detect_swing_points(df_ltf, lb)
    if not swings:
        return None
    sweeps = detect_liquidity_sweeps(
        df_ltf, swings, sweep_dir,
        poi_high=poi_high, poi_low=poi_low,
    )
    if not sweeps:
        return None
    sw = sweeps[0]
    wick_extreme = (
        float(df_ltf["low"].iat[sw.sweep_index])
        if sweep_dir == "bullish"
        else float(df_ltf["high"].iat[sw.sweep_index])
    )
    return {
        "sweep_index":    sw.sweep_index,
        "swept_price":    sw.swept_price,
        "recovery_index": sw.recovery_index,
        "direction":      sw.direction,
        "wick_extreme":   wick_extreme,
    }


# ── Main sniper loop ──────────────────────────────────────────────────

async def run_sniper() -> Tuple[List[dict], List[SniperSignal]]:
    """
    Returns (sweep_alerts, sniper_signals).
    Called every 10 seconds from main.py.
    """
    _purge_alert_sent()

    entries = _load_stage2()
    if not entries:
        return [], []

    exchange       = await get_public_exchange()
    sweep_alerts:   List[dict]         = []
    sniper_signals: List[SniperSignal] = []

    try:
        for entry in entries:
            fpath     = entry.get("_path", "")
            symbol    = entry.get("symbol")
            direction = entry.get("direction")
            poi       = entry.get("htf_poi", {})
            status    = entry.get("status", "")

            if not symbol or not direction or not poi:
                _delete_json(fpath); continue

            poi_high = poi.get("high")
            poi_low  = poi.get("low")
            if poi_high is None or poi_low is None:
                _delete_json(fpath); continue

            # Timeout check
            entered_ts = entry.get("entered_poi_ts", entry.get("created_ts", 0))
            if time.time() - entered_ts > WL2_TIMEOUT_SECONDS:
                logger.info("TIMEOUT %s → deleting", symbol)
                _delete_json(fpath); continue

            # CMP drift check
            try:
                ticker  = await exchange.fetch_ticker(symbol)
                cmp_now = ticker.get("last")
                if cmp_now is not None:
                    poi_mid = (poi_high + poi_low) / 2
                    drift   = abs(cmp_now - poi_mid) / poi_mid * 100 if poi_mid else 0
                    if drift > WL2_DRIFT_PCT:
                        logger.info(
                            "DRIFT %s %.1f%% → deleting", symbol, drift
                        )
                        _delete_json(fpath); continue
            except Exception:
                pass

            # Fetch 100-candle LTF
            try:
                df_5m = await _fetch_ltf(exchange, symbol, "5m", 100)
                df_1m = await _fetch_ltf(exchange, symbol, "1m", 100)
                df_1h = await _fetch_ltf(exchange, symbol, "1h", 100)
            except Exception as exc:
                logger.warning("Fetch %s: %s", symbol, exc); continue

            if any(x is None or len(x) < 20 for x in [df_5m, df_1m, df_1h]):
                continue

            sweep_dir = direction  # "bullish" or "bearish"

            # ── STAGE 1 → 2: POI_TOUCHED → check for sweep ───────────
            if status == "WATCHLIST_2_POI_TOUCHED":
                sweep_info = None
                for df_ltf, lb in [(df_5m, 2), (df_1m, 1)]:
                    found = _has_recent_sweep(
                        df_ltf, sweep_dir, poi_high, poi_low, lb
                    )
                    if found:
                        sweep_info = found
                        break

                if sweep_info:
                    logger.info(
                        "SWEEP %s extreme=%.6f swept=%.6f",
                        symbol, sweep_info["wick_extreme"], sweep_info["swept_price"],
                    )
                    entry["status"]     = "WATCHLIST_2_SWEEP"
                    entry["sweep_data"] = sweep_info
                    entry["sweep_ts"]   = time.time()
                    _save_json(fpath, entry)
                    sweep_alerts.append({
                        "symbol":    symbol,
                        "direction": direction,
                        "htf_poi":   poi,
                        "sweep":     sweep_info,
                    })
                else:
                    logger.debug("%s — no sweep yet, waiting…", symbol)
                continue

            # ── STAGE 2 → 3: SWEEP → check for CHoCH → signal ────────
            if status == "WATCHLIST_2_SWEEP":
                signal = build_sniper_signal(
                    symbol=symbol,
                    direction=direction,
                    df_5m=df_5m,
                    df_1m=df_1m,
                    htf_poi_high=poi_high,
                    htf_poi_low=poi_low,
                    df_1h=df_1h,
                )
                if signal is not None:
                    logger.info(
                        "🎯 SIGNAL %s %s entry=[%.6f,%.6f] SL=%.6f TP1=%.6f RR=%.2f",
                        symbol, signal.direction,
                        signal.entry_low, signal.entry_high,
                        signal.stop_loss, signal.tp1, signal.risk_reward,
                    )
                    _delete_json(fpath)
                    sniper_signals.append(signal)
                else:
                    logger.debug("%s — CHoCH not confirmed yet, waiting…", symbol)

    except Exception as exc:
        logger.warning("run_sniper exchange error: %s", exc)

    return sweep_alerts, sniper_signals
