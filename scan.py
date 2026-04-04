"""
scan.py — The LTF Scalp Screener
Runs every hour (triggered by main.py).

1. Fetch Top-400 USDT-M Futures perpetuals (Binance) by 24h quote volume.
2. Pull 1H (10 days) and 15m (3 days) OHLCV for each — 30 concurrent requests.
3. Score every coin via smc.score_coin(df_1h, df_15m, cmp).
4. Pick top-5 (score ≥ MIN_SCORE) and persist as WATCHLIST_1 JSON.
On a 16 GB server the full scan completes in seconds.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from execution import get_public_exchange
from smc import score_coin, Zone, MIN_SCORE

logger = logging.getLogger("scan")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Exchange helpers
# ─────────────────────────────────────────────────────────────────────

async def fetch_top_symbols(exchange: Any, n: int = 400) -> List[str]:
    """Return top *n* USDT-M linear perpetual symbols by 24 h quote volume."""
    markets = await exchange.load_markets()
    valid_syms = {
        sym for sym, m in markets.items()
        if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
        and m.get("active", True)
    }
    tickers = await exchange.fetch_tickers()
    usdt_perps = {
        k: v for k, v in tickers.items()
        if k in valid_syms and v.get("quoteVolume") is not None
    }
    sorted_syms = sorted(
        usdt_perps.items(),
        key=lambda x: x[1]["quoteVolume"],
        reverse=True,
    )
    return [s[0] for s in sorted_syms[:n]]


async def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since_days: int,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for *symbol* going back *since_days*."""
    since_ms = int((time.time() - since_days * 86400) * 1000)
    try:
        raw = await exchange.fetch_ohlcv(
            symbol, timeframe, since=since_ms, limit=1000,
        )
    except Exception as exc:
        logger.warning("OHLCV fetch failed for %s %s: %s", symbol, timeframe, exc)
        return None

    if not raw:
        return None

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────
# Core scan logic
# ─────────────────────────────────────────────────────────────────────

def _build_json(symbol: str, direction: str, poi: Zone) -> dict:
    return {
        "symbol": symbol,
        "status": "WATCHLIST_1",
        "direction": direction,
        "htf_poi": {
            "high": poi.high,
            "low": poi.low,
            "type": poi.kind,
        },
        "created_ts": time.time(),
        "updated_ts": time.time(),
    }


def _safe_symbol_filename(symbol: str) -> str:
    """Convert 'SOL/USDT' or 'SOL/USDT:USDT' → 'SOLUSDT'."""
    return symbol.replace("/", "").replace(":", "").replace("USDT", "", 1) + "USDT"


async def run_scan() -> List[dict]:
    """
    Execute the full hourly scan pipeline.
    Returns list of dicts that were promoted to WATCHLIST_1.
    """
    exchange = await get_public_exchange()
    results: List[dict] = []
    try:
        symbols = await fetch_top_symbols(exchange, 400)
        logger.info("Fetched %d symbols for screening.", len(symbols))
        # ── දැනට වොච්ලිස්ට් වල ඉන්න කොයින්ස් ටික හොයාගමු ──
        tracked_symbols = set()
        if DATA_DIR.exists():
            for fpath in DATA_DIR.glob("*.json"):
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    sym = data.get("symbol")
                    # WATCHLIST_1 හෝ WATCHLIST_2 වල ඉන්නවා නම් විතරක් ට්‍රැක් කරගමු
                    if sym and data.get("status") in ["WATCHLIST_1", "WATCHLIST_2"]:
                        tracked_symbols.add(sym)
                except Exception:
                    pass

        # ── ස්කෑන් ලිස්ට් එකෙන් ඒ ටික අයින් කරමු ──
        original_count = len(symbols)
        symbols = [s for s in symbols if s not in tracked_symbols]
        
        logger.info(f"Skipping {len(tracked_symbols)} coins already in watchlists. "
                    f"Scanning {len(symbols)} new coins out of {original_count}.")

        sem = asyncio.Semaphore(30)

        async def _score_symbol(sym: str) -> Optional[dict]:
            async with sem:
                df_1h = df_15m = None
                try:
                    df_1h  = await fetch_ohlcv(exchange, sym, "1h",  10)
                    df_15m = await fetch_ohlcv(exchange, sym, "15m",  3)
                    if df_1h is None or df_15m is None:
                        return None
                    if len(df_1h) < 30 or len(df_15m) < 30:
                        return None
                    cmp = df_15m["close"].iat[-1]
                    res = score_coin(df_1h, df_15m, cmp)
                    if res["score"] >= MIN_SCORE and res["direction"] and res["poi"]:
                        return {
                            "symbol":         sym,
                            "score":          res["score"],
                            "direction":      res["direction"],
                            "poi":            res["poi"],
                            "breakdown":      res["breakdown"],
                            "ob_fvg_overlap": res.get("ob_fvg_overlap", False),
                        }
                    return None
                except Exception as exc:
                    logger.warning("Error scoring %s: %s", sym, exc)
                    return None
                finally:
                    del df_1h, df_15m

        logger.info("Launching concurrent scan (%d symbols, sem=30)…", len(symbols))
        tasks = [_score_symbol(sym) for sym in symbols]
        raw_results = await asyncio.gather(*tasks)
        scored: list = [r for r in raw_results if r is not None]

        # Sort and pick top 5
        scored.sort(key=lambda x: x["score"], reverse=True)
        top5 = scored[:5]
        logger.info("Top 5 scored coins: %s",
                     [(t["symbol"], t["score"]) for t in top5])

        # Write new top-5 files (accumulate — do NOT wipe existing WATCHLIST_1)
        for entry in top5:
            fname = _safe_symbol_filename(entry["symbol"])
            fpath = DATA_DIR / f"{fname}.json"
            payload = _build_json(entry["symbol"], entry["direction"], entry["poi"])
            payload["score"] = entry["score"]
            payload["breakdown"] = entry["breakdown"]

            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)

            results.append(payload)
            logger.info("WATCHLIST_1: %s  score=%d  dir=%s",
                         entry["symbol"], entry["score"], entry["direction"])

        # Enforce WATCHLIST_1 capacity: keep at most 50 coins (evict oldest)
        MAX_WATCHLIST_1 = 50
        wl1_files: list = []
        for fpath in DATA_DIR.glob("*.json"):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                if data.get("status") == "WATCHLIST_1":
                    ts = data.get("created_ts", fpath.stat().st_mtime)
                    wl1_files.append((ts, fpath))
            except Exception:
                pass

        if len(wl1_files) > MAX_WATCHLIST_1:
            wl1_files.sort(key=lambda x: x[0])   # oldest first
            to_evict = wl1_files[:len(wl1_files) - MAX_WATCHLIST_1]
            for _, old_path in to_evict:
                try:
                    old_path.unlink()
                    logger.info("Evicted oldest WATCHLIST_1 (cap %d): %s",
                                MAX_WATCHLIST_1, old_path.name)
                except Exception:
                    pass

    except Exception as exc:
        logger.error("run_scan top-level error: %s", exc, exc_info=True)
    finally:
        gc.collect()

    return results