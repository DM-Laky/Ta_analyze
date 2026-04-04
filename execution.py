"""
execution.py — Auto-Execution Engine  (Binance USDT-M Futures, CCXT async)

Risk model
----------
  Fixed risk   = $1.50 per trade
  Qty          = $1.50 / abs(entry_40pct - buffered_SL)
  Leverage     = 25× ISOLATED margin
  Entry        = 40% retrace of displacement leg (from smc.py)

Order lifecycle
---------------
  PENDING     → LIMIT entry placed, waiting for fill
  LIVE        → position open (FULL — SL + TP3@90% orders active)
  LIVE_RUNNER → TP3 hit, 90% closed, SL moved to break-even, 10% running for TP6
  CLOSED      → all position closed (SL / TP3 full / TP6 / Emergency)
  CANCELLED   → entry missed (price hit TP1 before fill) OR timed out

Partial close logic
-------------------
  When TP3 fires (90% close order hits):
    1. Detect position qty drop to ~10% of original
    2. Cancel old SL order
    3. Place new SL at entry_price (break-even)
    4. Set status = LIVE_RUNNER  →  holds for TP6

BTC crash emergency
-------------------
  set_crash_active(True)  → blocks new place_trade() calls
  emergency_cancel_pending() → cancel all PENDING limit orders on Binance
  emergency_close_profitable(min_profit) → market-close LIVE positions with
                                           unrealised PnL >= min_profit USD

Global Telegram error callback
------------------------------
  register_error_callback(async_fn)   ← called by main.py on bot start
  All CCXT exceptions are forwarded to Telegram via this callback.

Public API
----------
  place_trade(signal)              → Optional[TradeRecord]
  monitor_trades()                 → MonitorResult
  get_all_live_pnl()               → List[Tuple[TradeRecord, float, float]]
  has_active_trade(symbol)         → bool
  get_active_trade_count()          → int
  get_public_exchange()             → ccxt.binance  (shared public instance)
  emergency_cancel_pending()       → List[TradeRecord]
  emergency_close_profitable(usd)  → List[TradeRecord]
  set_crash_active(active)
  is_crash_active()                → bool
  register_error_callback(cb)
  get_btc_5min_change()            → Optional[float]   (% change, negative = drop)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Coroutine, List, Optional, Tuple

import ccxt.async_support as ccxt

from smc import SniperSignal

logger = logging.getLogger("execution")

# ── Constants ────────────────────────────────────────────────────────
RISK_PER_TRADE          = 1.50    # fixed USD risk per trade
TARGET_LEVERAGE         = 25      # 25× isolated margin
ORDER_FILL_TIMEOUT      = 3600    # cancel unfilled orders after 1 h
TP3_PARTIAL_CLOSE_PCT   = 0.90    # close 90% at TP3
RUNNER_THRESHOLD_PCT    = 0.15    # detect TP3 hit when qty < 15% of original
MIN_PROFIT_EMERGENCY    = 0.20    # USD: emergency-close only if PnL >= this

TRADES_DIR = Path(__file__).parent / "data" / "trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# ── Module-level state ───────────────────────────────────────────────
_btc_crash_active: bool = False
_error_callback: Optional[Callable[..., Coroutine]] = None
_pub_exchange:   Optional[ccxt.binance] = None
_auth_exchange:  Optional[ccxt.binance] = None


# ─────────────────────────────────────────────────────────────────────
# TradeRecord
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:             str
    direction:          str      # "LONG" | "SHORT"
    order_id:           str      # entry limit order ID
    status:             str      # PENDING | LIVE | LIVE_RUNNER | CLOSED | CANCELLED
    entry_price:        float    # limit price placed / actual fill price
    stop_loss:          float    # buffered SL (0.175% beyond sweep wick)
    tp1:                float
    tp2:                float
    tp3:                float    # primary target — 90% close here
    tp6:                float    # runner target — 10% close here
    qty:                float    # full position size in base coin
    leverage:           int
    margin_cost:        float    # estimated initial margin USD
    placed_ts:          float
    filled_ts:          float = 0.0
    close_price:        float = 0.0
    pnl_usd:            float = 0.0
    pnl_pct:            float = 0.0
    sl_order_id:        str   = ""   # Binance STOP_MARKET order ID
    tp3_order_id:       str   = ""   # Binance TAKE_PROFIT_MARKET order ID (90%)
    runner_sl_order_id: str   = ""   # break-even SL placed after TP3 hit
    partial_closed:     bool  = False
    close_reason:       str   = ""   # SL | TP3 | TP6 | Emergency
    confluences:        List[str] = field(default_factory=list)

    @property
    def file_key(self) -> str:
        safe = self.symbol.replace("/", "").replace(":", "")
        oid  = self.order_id[:12] if len(self.order_id) >= 12 else self.order_id
        return f"{safe}_{oid}"


@dataclass
class MonitorResult:
    filled:    List[TradeRecord] = field(default_factory=list)
    cancelled: List[TradeRecord] = field(default_factory=list)
    closed:    List[TradeRecord] = field(default_factory=list)
    runners:   List[TradeRecord] = field(default_factory=list)   # TP3 hit, runner active


# ─────────────────────────────────────────────────────────────────────
# Global state helpers
# ─────────────────────────────────────────────────────────────────────

def register_error_callback(cb: Callable[..., Coroutine]) -> None:
    global _error_callback
    _error_callback = cb


def set_crash_active(active: bool) -> None:
    global _btc_crash_active
    _btc_crash_active = active
    logger.warning("BTC crash flag → %s", "ACTIVE" if active else "CLEARED")


def is_crash_active() -> bool:
    return _btc_crash_active


async def _report_error(context: str, exc: Exception) -> None:
    """Forward exception to Telegram and log it."""
    msg = (
        f"⚠️ <b>API Error [{context}]</b>\n"
        f"<code>{type(exc).__name__}: {exc}</code>"
    )
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    if _error_callback:
        try:
            await _error_callback(msg)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Exchange factories
# ─────────────────────────────────────────────────────────────────────

def _create_exchange() -> ccxt.binance:
    """Authenticated Binance Futures exchange."""
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set in .env — "
            "auto-execution is disabled."
        )
    return ccxt.binance({
        "apiKey":    api_key,
        "secret":    api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType":             "future",
            "adjustForTimeDifference": True,
        },
    })


def _create_public_exchange() -> ccxt.binance:
    """Unauthenticated Binance exchange for public market data."""
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


async def get_public_exchange() -> ccxt.binance:
    """Shared unauthenticated Binance Futures instance (lazy init, never closed)."""
    global _pub_exchange
    if _pub_exchange is None:
        inst = _create_public_exchange()
        await inst.load_markets()
        _pub_exchange = inst
    return _pub_exchange


async def _get_auth_exchange() -> ccxt.binance:
    """Shared authenticated Binance Futures instance (lazy init, never closed)."""
    global _auth_exchange
    if _auth_exchange is None:
        inst = _create_exchange()
        await inst.load_markets()
        _auth_exchange = inst
    return _auth_exchange


def get_active_trade_count() -> int:
    """Number of trades currently in PENDING, LIVE, or LIVE_RUNNER status."""
    return sum(
        1 for t in _load_all_trades()
        if t.status in ("PENDING", "LIVE", "LIVE_RUNNER")
    )


# ─────────────────────────────────────────────────────────────────────
# JSON persistence
# ─────────────────────────────────────────────────────────────────────

def _trade_path(file_key: str) -> Path:
    return TRADES_DIR / f"{file_key}.json"


def _save_trade(rec: TradeRecord) -> None:
    d = asdict(rec)
    d["updated_ts"] = time.time()
    with open(_trade_path(rec.file_key), "w") as f:
        json.dump(d, f, indent=2)
    logger.debug("Trade saved: %s  status=%s", rec.symbol, rec.status)


def _load_all_trades() -> List[TradeRecord]:
    trades: List[TradeRecord] = []
    for p in TRADES_DIR.glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            valid = {k: v for k, v in d.items() if k != "updated_ts"}
            trades.append(TradeRecord(**valid))
        except Exception as exc:
            logger.warning("Could not load trade %s: %s", p.name, exc)
    return trades


def _load_by_status(*statuses: str) -> List[TradeRecord]:
    return [t for t in _load_all_trades() if t.status in statuses]


def has_active_trade(symbol: str) -> bool:
    return any(
        t.symbol == symbol and t.status in ("PENDING", "LIVE", "LIVE_RUNNER")
        for t in _load_all_trades()
    )


# ─────────────────────────────────────────────────────────────────────
# Exchange helpers
# ─────────────────────────────────────────────────────────────────────

async def _setup_symbol(exchange: ccxt.binance, symbol: str) -> None:
    """Set ISOLATED margin + 25× leverage on Binance Futures."""
    try:
        await exchange.set_margin_mode("isolated", symbol)
        logger.info("Margin mode → ISOLATED  %s", symbol)
    except Exception as exc:
        logger.debug("set_margin_mode %s (may already be set): %s", symbol, exc)

    try:
        await exchange.set_leverage(TARGET_LEVERAGE, symbol)
        logger.info("Leverage → %d×  %s", TARGET_LEVERAGE, symbol)
    except Exception as exc:
        await _report_error(f"set_leverage {symbol}", exc)


async def _get_open_position(
    exchange: ccxt.binance,
    symbol: str,
    direction: str,
) -> Optional[dict]:
    try:
        positions = await exchange.fetch_positions([symbol])
        want_side = "long" if direction == "LONG" else "short"
        for p in positions:
            if (p.get("symbol") == symbol
                    and str(p.get("side", "")).lower() == want_side
                    and abs(float(p.get("contracts") or 0)) > 0):
                return p
    except Exception as exc:
        await _report_error(f"fetch_positions {symbol}", exc)
    return None


async def _fetch_closed_pnl(
    exchange: ccxt.binance,
    rec: TradeRecord,
) -> float:
    """Pull realised PnL from Binance income history after close."""
    try:
        since = int((rec.filled_ts or rec.placed_ts) * 1000)
        trades = await exchange.fetch_my_trades(rec.symbol, since=since, limit=50)
        total = 0.0
        for t in trades:
            info = t.get("info", {})
            pnl = float(
                info.get("realizedPnl") or info.get("closedPnl") or 0
            )
            total += pnl
        return round(total, 4)
    except Exception as exc:
        logger.warning("fetch_closed_pnl %s: %s", rec.symbol, exc)
        if rec.close_price > 0 and rec.qty > 0:
            sign = 1 if rec.direction == "LONG" else -1
            return round(sign * rec.qty * (rec.close_price - rec.entry_price), 4)
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────

def _calc_qty(
    entry_price: float,
    stop_loss:   float,
    exchange:    ccxt.binance,
    symbol:      str,
) -> Optional[float]:
    """Qty = $1.50 / |entry - SL|, rounded to Binance lot size."""
    risk_dist = abs(entry_price - stop_loss)
    if risk_dist <= 0:
        return None

    raw_qty = RISK_PER_TRADE / risk_dist
    qty_str = exchange.amount_to_precision(symbol, raw_qty)
    qty     = float(qty_str)

    try:
        mkt     = exchange.market(symbol)
        min_qty = float(mkt["limits"]["amount"].get("min") or 0.0)
        if qty < min_qty:
            logger.warning("qty %.6f < min %.6f for %s", qty, min_qty, symbol)
            return None
    except Exception:
        pass

    return qty


async def _place_sl_order(
    exchange:    ccxt.binance,
    symbol:      str,
    direction:   str,
    sl_price:    float,
    qty:         float,
) -> str:
    """
    Place a STOP_MARKET reduce-only order for the SL.
    Returns order ID or "" on failure.
    """
    close_side = "sell" if direction == "LONG" else "buy"
    qty = float(exchange.amount_to_precision(symbol, qty))
    try:
        order = await exchange.create_order(
            symbol, "stop_market", close_side, qty, None,
            params={
                "stopPrice":   exchange.price_to_precision(symbol, sl_price),
                "reduceOnly":  True,
                "workingType": "MARK_PRICE",
            },
        )
        oid = str(order["id"])
        logger.info("SL order placed  %s  price=%.6f  id=%s", symbol, sl_price, oid)
        return oid
    except Exception as exc:
        await _report_error(f"place_sl_order {symbol}", exc)
        return ""


async def _place_tp_order(
    exchange:  ccxt.binance,
    symbol:    str,
    direction: str,
    tp_price:  float,
    qty:       float,
) -> str:
    """
    Place a TAKE_PROFIT_MARKET reduce-only order for a TP level.
    Returns order ID or "" on failure.
    """
    close_side = "sell" if direction == "LONG" else "buy"
    qty = float(exchange.amount_to_precision(symbol, qty))
    try:
        order = await exchange.create_order(
            symbol, "take_profit_market", close_side, qty, None,
            params={
                "stopPrice":   exchange.price_to_precision(symbol, tp_price),
                "reduceOnly":  True,
                "workingType": "MARK_PRICE",
            },
        )
        oid = str(order["id"])
        logger.info("TP order placed  %s  price=%.6f  qty=%.6f  id=%s",
                    symbol, tp_price, qty, oid)
        return oid
    except Exception as exc:
        await _report_error(f"place_tp_order {symbol}", exc)
        return ""


async def _cancel_order_safe(
    exchange: ccxt.binance,
    order_id: str,
    symbol:   str,
) -> None:
    if not order_id:
        return
    try:
        await exchange.cancel_order(order_id, symbol)
        logger.info("Order cancelled  %s  id=%s", symbol, order_id)
    except Exception as exc:
        logger.debug("cancel_order %s %s: %s", symbol, order_id, exc)


async def _place_protection_orders(
    exchange:    ccxt.binance,
    rec:         "TradeRecord",
    fill_px:     float,
) -> None:
    """
    Place SL + TP3 (90% qty) immediately after a fill is detected.
    TP3 is RECALCULATED from the actual fill price to guarantee exact $3.00
    profit ($1.50 risk × 2:1 R:R), regardless of the signal's TP estimate.
    Writes updated order IDs to the TradeRecord and saves it.
    """
    sign = 1 if rec.direction == "LONG" else -1

    # ── SL (full qty) ─────────────────────────────────────────────────────
    sl_oid = await _place_sl_order(
        exchange, rec.symbol, rec.direction, rec.stop_loss, rec.qty
    )
    rec.sl_order_id = sl_oid

    # ── TP3: recalculated from fill_px for guaranteed 1:2 R:R (→ $3.00) ──
    risk_dist = abs(fill_px - rec.stop_loss)
    tp3_exact = fill_px + sign * 2.0 * risk_dist
    tp3_price = float(exchange.price_to_precision(rec.symbol, tp3_exact))
    rec.tp3   = tp3_price   # keep record accurate

    tp3_qty = float(
        exchange.amount_to_precision(rec.symbol, rec.qty * TP3_PARTIAL_CLOSE_PCT)
    )
    tp3_oid = await _place_tp_order(
        exchange, rec.symbol, rec.direction, tp3_price, tp3_qty
    )
    rec.tp3_order_id = tp3_oid

    logger.info(
        "PROTECTION PLACED  %s  fill=%.6f  SL_id=%s  TP3=%.6f  TP3_id=%s",
        rec.symbol, fill_px, sl_oid, tp3_price, tp3_oid,
    )


async def _fast_fill_monitor(
    symbol:   str,
    order_id: str,
) -> None:
    """
    Background task: polls for fill every 2 s for up to 30 s.
    If the limit order fills before monitor_trades() next runs,
    SL + TP3 are placed immediately — position is NEVER unprotected.
    After 30 s this task exits; the regular 15-s execution_loop takes over.
    """
    try:
        exchange = await _get_auth_exchange()
        for _ in range(15):   # 15 × 2 s = 30 s max
            await asyncio.sleep(2)
            # Re-read from disk so we see any status change by execution_loop
            all_trades = _load_all_trades()
            rec = next((r for r in all_trades if r.order_id == order_id), None)
            if rec is None or rec.status != "PENDING":
                return   # already handled (LIVE/CANCELLED) or deleted
            try:
                order = await exchange.fetch_order(order_id, symbol)
                if str(order.get("status", "")).lower() == "closed":
                    fill_px = float(order.get("average") or order.get("price")
                                    or rec.entry_price)
                    rec.entry_price = fill_px
                    rec.filled_ts   = time.time()
                    rec.status      = "LIVE"      # mark LIVE
                    _save_trade(rec)             # ← save BEFORE placing orders
                    #   monitor_trades() will now see status=LIVE and skip
                    await _place_protection_orders(exchange, rec, fill_px)
                    _save_trade(rec)             # re-save with order IDs
                    logger.info("FAST FILL  %s @ %.6f  orders protected",
                                symbol, fill_px)
                    return
            except Exception as exc:
                logger.debug("fast_fill_monitor %s: %s", symbol, exc)
    except Exception as exc:
        logger.debug("_fast_fill_monitor outer %s: %s", symbol, exc)


# ─────────────────────────────────────────────────────────────────────
# BTC crash helper (public data, no auth required)
# ─────────────────────────────────────────────────────────────────────

async def get_btc_5min_change() -> Optional[float]:
    """
    Returns BTC % price change over the last 5 minutes.
    Negative value means a price drop.
    """
    try:
        exchange = await get_public_exchange()
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1m", limit=6)
        if len(ohlcv) < 6:
            return None
        price_5min_ago = float(ohlcv[0][4])   # close of 5 bars ago
        price_now      = float(ohlcv[-1][4])   # latest close
        if price_5min_ago <= 0:
            return None
        return (price_now - price_5min_ago) / price_5min_ago * 100.0
    except Exception as exc:
        logger.warning("get_btc_5min_change: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════
# Public API — Core execution
# ═══════════════════════════════════════════════════════════════════════

async def place_trade(signal: SniperSignal) -> Optional[TradeRecord]:
    """
    Full Binance Futures entry flow:
      1. Crash-guard + double-entry guard
      2. 25× ISOLATED margin setup
      3. Qty = $1.50 / |40%-retrace-entry - buffered-SL|
      4. Place LIMIT entry order
      — SL + TP3(90%) are placed in monitor_trades() after fill —
    """
    if _btc_crash_active:
        logger.warning("place_trade blocked — BTC crash protocol active")
        return None

    if has_active_trade(signal.symbol):
        logger.info("Double-entry blocked (local): %s", signal.symbol)
        return None

    try:
        exchange = await _get_auth_exchange()

        # Exchange-level position guard
        live_pos = await _get_open_position(exchange, signal.symbol, signal.direction)
        if live_pos is not None:
            logger.info("Double-entry blocked (exchange position exists): %s",
                        signal.symbol)
            return None

        await _setup_symbol(exchange, signal.symbol)

        # Outer-edge entry for immediate fill (market-side of the 40% retrace zone)
        # LONG  → price is falling INTO the zone, hits the TOP  edge first
        # SHORT → price is rising  INTO the zone, hits the BOTTOM edge first
        entry_raw = (
            signal.entry_high
            if signal.direction == "LONG"
            else signal.entry_low
        )
        entry_price = float(exchange.price_to_precision(signal.symbol, entry_raw))
        sl_price    = float(exchange.price_to_precision(signal.symbol, signal.stop_loss))
        tp1_price   = float(exchange.price_to_precision(signal.symbol, signal.tp1))
        tp2_price   = float(exchange.price_to_precision(signal.symbol, signal.tp2))
        tp6_price   = float(exchange.price_to_precision(signal.symbol, signal.tp6))

        qty = _calc_qty(entry_price, sl_price, exchange, signal.symbol)
        if qty is None:
            await _report_error(
                f"place_trade {signal.symbol}",
                ValueError("qty calc failed — risk distance too small?"),
            )
            return None

        side = "buy" if signal.direction == "LONG" else "sell"

        order = await exchange.create_order(
            signal.symbol, "limit", side, qty, entry_price,
            params={"timeInForce": "GTC"},
        )

        margin_cost = round((qty * entry_price) / TARGET_LEVERAGE, 4)
        order_id    = str(order["id"])

        # Recalculate TP3 from entry price for guaranteed 1:2 R:R ($3.00)
        sign_dir    = 1 if signal.direction == "LONG" else -1
        risk_dist   = abs(entry_price - sl_price)
        tp3_exact   = float(exchange.price_to_precision(
            signal.symbol, entry_price + sign_dir * 2.0 * risk_dist
        ))

        rec = TradeRecord(
            symbol      = signal.symbol,
            direction   = signal.direction,
            order_id    = order_id,
            status      = "PENDING",
            entry_price = entry_price,
            stop_loss   = sl_price,
            tp1         = tp1_price,
            tp2         = tp2_price,
            tp3         = tp3_exact,
            tp6         = tp6_price,
            qty         = qty,
            leverage    = TARGET_LEVERAGE,
            margin_cost = margin_cost,
            placed_ts   = time.time(),
            confluences = list(signal.confluences),
        )
        _save_trade(rec)

        logger.info(
            "ORDER PLACED  %s %s  qty=%.6f  entry=%.6f  SL=%.6f  "
            "TP3=%.6f (1:2 R:R)  TP6=%.6f  margin=$%.4f",
            signal.symbol, signal.direction,
            qty, entry_price, sl_price, tp3_exact, tp6_price, margin_cost,
        )

        # Launch fast fill monitor — protects position within 2 s of fill
        asyncio.create_task(_fast_fill_monitor(signal.symbol, order_id))

        return rec

    except Exception as exc:
        await _report_error(f"place_trade {signal.symbol}", exc)
        return None


async def monitor_trades() -> MonitorResult:
    """
    Single-pass monitor for PENDING + LIVE + LIVE_RUNNER trades.

    PENDING:
      • Fill detection → place SL + TP3(90%) → LIVE
      • Cancel-if-missed (price hit TP1 before fill) → CANCELLED
      • Timeout (>1 h) → CANCELLED

    LIVE:
      • Partial close detection (qty drops to < 15%) → LIVE_RUNNER
        – Cancel old SL, place break-even SL
      • Full close detection (position gone) → CLOSED

    LIVE_RUNNER:
      • TP6 or BE-SL hit detection → CLOSED
    """
    result  = MonitorResult()
    pending = _load_by_status("PENDING")
    live    = _load_by_status("LIVE", "LIVE_RUNNER")

    if not pending and not live:
        return result

    try:
        exchange = await _get_auth_exchange()

        # ── PENDING — fill + cancel checks ───────────────────────────
        for rec in pending:
            try:
                order = await exchange.fetch_order(rec.order_id, rec.symbol)
                os_   = str(order.get("status", "")).lower()

                if os_ == "closed":
                    # Re-read on-disk status — fast_fill_monitor may have
                    # already promoted this record to LIVE and placed orders.
                    fresh = next(
                        (t for t in _load_all_trades()
                         if t.order_id == rec.order_id),
                        None,
                    )
                    if fresh and fresh.status != "PENDING":
                        result.filled.append(fresh)
                        continue

                    fill_px = float(order.get("average") or order.get("price")
                                    or rec.entry_price)
                    rec.entry_price = fill_px
                    rec.filled_ts   = time.time()
                    rec.status      = "LIVE"

                    await _place_protection_orders(exchange, rec, fill_px)
                    _save_trade(rec)
                    result.filled.append(rec)
                    logger.info(
                        "ORDER FILLED  %s @ %.6f  SL_id=%s  TP3=%.6f  TP3_id=%s",
                        rec.symbol, fill_px, rec.sl_order_id,
                        rec.tp3, rec.tp3_order_id,
                    )
                    continue

                if os_ in ("canceled", "cancelled", "rejected", "expired"):
                    rec.status       = "CANCELLED"
                    rec.close_reason = "External cancel"
                    _save_trade(rec)
                    result.cancelled.append(rec)
                    continue

                if time.time() - rec.placed_ts > ORDER_FILL_TIMEOUT:
                    await _cancel_order_safe(exchange, rec.order_id, rec.symbol)
                    rec.status       = "CANCELLED"
                    rec.close_reason = "Timeout (1 h)"
                    _save_trade(rec)
                    result.cancelled.append(rec)
                    logger.info("ORDER TIMEOUT cancelled  %s", rec.symbol)
                    continue

                # Cancel-if-missed
                try:
                    ticker = await exchange.fetch_ticker(rec.symbol)
                    cmp    = float(ticker.get("last") or 0)
                    missed = (
                        (rec.direction == "LONG"  and cmp >= rec.tp1) or
                        (rec.direction == "SHORT" and cmp <= rec.tp1)
                    )
                    if missed:
                        await _cancel_order_safe(exchange, rec.order_id, rec.symbol)
                        rec.status       = "CANCELLED"
                        rec.close_reason = "Entry missed — price hit TP1"
                        _save_trade(rec)
                        result.cancelled.append(rec)
                        logger.info("ENTRY MISSED  %s  CMP=%.6f TP1=%.6f",
                                    rec.symbol, cmp, rec.tp1)
                except Exception as exc:
                    logger.debug("ticker check %s: %s", rec.symbol, exc)

            except Exception as exc:
                await _report_error(f"monitor PENDING {rec.symbol}", exc)

        # ── LIVE / LIVE_RUNNER — position tracking ────────────────────
        for rec in live:
            try:
                pos = await _get_open_position(exchange, rec.symbol, rec.direction)

                if pos is None:
                    # Position fully closed — cancel any stale open orders first
                    await _cancel_order_safe(exchange, rec.sl_order_id,         rec.symbol)
                    await _cancel_order_safe(exchange, rec.tp3_order_id,        rec.symbol)
                    await _cancel_order_safe(exchange, rec.runner_sl_order_id,  rec.symbol)
                    pnl = await _fetch_closed_pnl(exchange, rec)
                    rec.status  = "CLOSED"
                    rec.pnl_usd = pnl
                    if not rec.close_reason:
                        rec.close_reason = "TP6" if rec.partial_closed else "SL/TP3"
                    _save_trade(rec)
                    result.closed.append(rec)
                    logger.info("POSITION CLOSED  %s  PnL=$%.4f  reason=%s",
                                rec.symbol, pnl, rec.close_reason)
                    continue

                current_qty = abs(float(pos.get("contracts") or 0))

                # Cache mark price for fallback PnL calc
                mark_px = (pos.get("markPrice")
                           or pos.get("info", {}).get("markPrice"))
                if mark_px:
                    rec.close_price = float(mark_px)

                # TP3 partial-close detection (LIVE → LIVE_RUNNER)
                if (rec.status == "LIVE"
                        and not rec.partial_closed
                        and 0 < current_qty < rec.qty * RUNNER_THRESHOLD_PCT):
                    logger.info(
                        "TP3 HIT  %s  original_qty=%.6f  remaining=%.6f",
                        rec.symbol, rec.qty, current_qty,
                    )
                    rec.partial_closed = True
                    rec.close_reason   = "TP3"

                    # Cancel old SL (was for full position)
                    await _cancel_order_safe(
                        exchange, rec.sl_order_id, rec.symbol
                    )

                    # Place break-even SL for runner qty
                    be_sl_oid = await _place_sl_order(
                        exchange, rec.symbol, rec.direction,
                        rec.entry_price, current_qty,
                    )
                    rec.runner_sl_order_id = be_sl_oid
                    rec.status             = "LIVE_RUNNER"
                    _save_trade(rec)
                    result.runners.append(rec)
                    continue

                _save_trade(rec)

            except Exception as exc:
                await _report_error(f"monitor LIVE {rec.symbol}", exc)

    except Exception as exc:
        await _report_error("monitor_trades top-level", exc)

    if result.closed:
        gc.collect()

    return result


async def get_all_live_pnl() -> List[Tuple[TradeRecord, float, float]]:
    """
    Returns [(rec, pnl_usd, pnl_pct)] for all LIVE + LIVE_RUNNER positions.
    pnl_pct is relative to initial margin.
    """
    live = _load_by_status("LIVE", "LIVE_RUNNER")
    if not live:
        return []

    results: List[Tuple[TradeRecord, float, float]] = []

    try:
        exchange = await _get_auth_exchange()
        for rec in live:
            try:
                pos = await _get_open_position(exchange, rec.symbol, rec.direction)
                if pos is None:
                    continue

                pnl_usd = float(pos.get("unrealizedPnl") or 0)
                init_margin = (rec.qty * rec.entry_price) / rec.leverage
                pnl_pct     = (pnl_usd / init_margin * 100) if init_margin > 0 else 0.0

                rec.pnl_usd = round(pnl_usd, 4)
                rec.pnl_pct = round(pnl_pct, 2)
                _save_trade(rec)

                results.append((rec, pnl_usd, round(pnl_pct, 2)))
            except Exception as exc:
                await _report_error(f"live_pnl {rec.symbol}", exc)

    except Exception as exc:
        await _report_error("get_all_live_pnl", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Public API — BTC Crash Emergency
# ═══════════════════════════════════════════════════════════════════════

async def emergency_cancel_pending() -> List[TradeRecord]:
    """Cancel ALL pending limit orders immediately (BTC crash protocol)."""
    pending = _load_by_status("PENDING")
    if not pending:
        return []

    cancelled: List[TradeRecord] = []
    try:
        exchange = await _get_auth_exchange()
        for rec in pending:
            try:
                await exchange.cancel_order(rec.order_id, rec.symbol)
                rec.status       = "CANCELLED"
                rec.close_reason = "BTC crash emergency"
                _save_trade(rec)
                cancelled.append(rec)
                logger.info("EMERGENCY CANCEL  %s  id=%s", rec.symbol, rec.order_id)
            except Exception as exc:
                await _report_error(f"emergency_cancel {rec.symbol}", exc)
    except Exception as exc:
        await _report_error("emergency_cancel_pending", exc)

    return cancelled


async def emergency_close_profitable(
    min_profit_usd: float = MIN_PROFIT_EMERGENCY,
) -> List[TradeRecord]:
    """
    Market-close all LIVE positions with unrealised PnL >= min_profit_usd.
    Positions in loss keep their original SL (risk management intact).
    """
    live = _load_by_status("LIVE", "LIVE_RUNNER")
    if not live:
        return []

    closed: List[TradeRecord] = []

    try:
        exchange = await _get_auth_exchange()
        for rec in live:
            try:
                pos = await _get_open_position(exchange, rec.symbol, rec.direction)
                if pos is None:
                    continue

                pnl_usd = float(pos.get("unrealizedPnl") or 0)
                if pnl_usd < min_profit_usd:
                    logger.info(
                        "Emergency skip %s — PnL $%.4f < threshold $%.2f",
                        rec.symbol, pnl_usd, min_profit_usd,
                    )
                    continue

                # Cancel SL orders first (prevent double-close)
                await _cancel_order_safe(exchange, rec.sl_order_id, rec.symbol)
                await _cancel_order_safe(exchange, rec.runner_sl_order_id, rec.symbol)
                await _cancel_order_safe(exchange, rec.tp3_order_id, rec.symbol)

                # Market close
                close_side = "sell" if rec.direction == "LONG" else "buy"
                qty        = abs(float(pos.get("contracts") or rec.qty))
                await exchange.create_order(
                    rec.symbol, "market", close_side, qty,
                    params={"reduceOnly": True},
                )
                rec.status       = "CLOSED"
                rec.pnl_usd      = round(pnl_usd, 4)
                rec.close_reason = "BTC crash emergency"
                _save_trade(rec)
                closed.append(rec)
                logger.info(
                    "EMERGENCY CLOSE  %s  PnL=$%.4f", rec.symbol, pnl_usd
                )
            except Exception as exc:
                await _report_error(f"emergency_close {rec.symbol}", exc)

    except Exception as exc:
        await _report_error("emergency_close_profitable", exc)

    return closed
