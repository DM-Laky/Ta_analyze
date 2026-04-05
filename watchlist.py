"""
signals/watchlist.py
=====================
Manages the POI Watchlist.

When a SetupCandidate is added → alert is sent to Telegram.
Every 30s the manager checks if price has entered any POI zone.
When price enters POI → triggers entry confirmation engine.
Expired or invalidated entries are cleaned up automatically.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from analysis.smc_engine import SetupCandidate
from config import config
from core.data_fetcher import fetcher
from utils.logger import log


class WatchlistState:
    WAITING   = "WAITING"     # Price not yet at POI
    TRIGGERED = "TRIGGERED"   # Price entered POI zone
    CONFIRMED = "CONFIRMED"   # Entry signal generated
    EXPIRED   = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass
class WatchlistEntry:
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    candidate: SetupCandidate = None
    state: str = WatchlistState.WAITING
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None

    @property
    def symbol(self) -> str:
        return self.candidate.symbol

    @property
    def direction(self) -> str:
        return self.candidate.direction

    @property
    def poi_top(self) -> float:
        return self.candidate.poi_top

    @property
    def poi_bottom(self) -> float:
        return self.candidate.poi_bottom

    @property
    def poi_mid(self) -> float:
        return self.candidate.poi_mid

    @property
    def is_active(self) -> bool:
        return self.state in (WatchlistState.WAITING, WatchlistState.TRIGGERED)

    @property
    def is_expired(self) -> bool:
        return self.candidate.is_expired

    def price_in_poi(self, price: float) -> bool:
        tol = self.poi_mid * config.POI_HIT_TOLERANCE
        return (self.poi_bottom - tol) <= price <= (self.poi_top + tol)

    def __repr__(self) -> str:
        return (
            f"WatchlistEntry(uid={self.uid} | {self.symbol} {self.direction} "
            f"| POI={self.poi_bottom:.5f}–{self.poi_top:.5f} | state={self.state})"
        )


class WatchlistManager:
    """
    Thread-safe watchlist.
    Callbacks:
      on_watchlist_add(entry)        — fires when new entry added
      on_poi_triggered(entry, price) — fires when price enters POI
      on_expired(entry)              — fires when entry expires
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[str, WatchlistEntry] = {}

        # Callbacks (set externally by main orchestrator)
        self.on_watchlist_add:  Optional[Callable] = None
        self.on_poi_triggered:  Optional[Callable] = None
        self.on_expired:        Optional[Callable] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, candidate: SetupCandidate) -> Optional[WatchlistEntry]:
        """Add a new POI to the watchlist. Returns None if duplicate."""
        with self._lock:
            # Duplicate check (same symbol, direction, similar POI)
            for e in self._entries.values():
                if (e.is_active and
                        e.symbol == candidate.symbol and
                        e.direction == candidate.direction and
                        abs(e.poi_mid - candidate.poi_mid) / candidate.poi_mid < 0.002):
                    log.debug("Duplicate watchlist entry skipped: %s", candidate)
                    return None

            # Max watchlist per symbol
            symbol_entries = [
                e for e in self._entries.values()
                if e.symbol == candidate.symbol and e.is_active
            ]
            if len(symbol_entries) >= config.MAX_WATCHLIST_PER_SYMBOL:
                # Remove the lowest-score entry
                worst = min(
                    symbol_entries, key=lambda e: e.candidate.setup_score
                )
                if worst.candidate.setup_score < candidate.setup_score:
                    log.info("Replacing low-score watchlist entry: %s", worst.uid)
                    worst.state = WatchlistState.CANCELLED
                else:
                    log.debug("Watchlist full, new entry skipped")
                    return None

            entry = WatchlistEntry(candidate=candidate)
            self._entries[entry.uid] = entry
            log.info("📋 Added to watchlist: %s", entry)

            if self.on_watchlist_add:
                self.on_watchlist_add(entry)

            return entry

    def check_prices(self):
        """
        Called by the scheduler every N seconds.
        Checks current price for all active watchlist entries.
        """
        active = self.get_active_entries()
        if not active:
            return

        # Group by symbol to minimise broker requests
        symbols = list({e.symbol for e in active})

        for symbol in symbols:
            price_info = fetcher.get_current_price(symbol)
            if price_info is None:
                continue

            mid = price_info["mid"]
            symbol_entries = [e for e in active if e.symbol == symbol]

            for entry in symbol_entries:
                if entry.price_in_poi(mid):
                    # Atomic state check-and-update under lock
                    with self._lock:
                        if entry.state != WatchlistState.WAITING:
                            continue  # already triggered by another thread
                        entry.state = WatchlistState.TRIGGERED
                        entry.triggered_at = datetime.now(timezone.utc)

                    log.info(
                        "🎯 POI TRIGGERED! %s %s | Price=%.5f in zone %.5f–%.5f",
                        entry.symbol, entry.direction, mid,
                        entry.poi_bottom, entry.poi_top,
                    )
                    if self.on_poi_triggered:
                        self.on_poi_triggered(entry, mid)

    def cleanup_expired(self):
        """Remove expired entries and fire callbacks."""
        with self._lock:
            for entry in list(self._entries.values()):
                if entry.is_active and entry.is_expired:
                    entry.state = WatchlistState.EXPIRED
                    log.info("⏱️ Watchlist entry expired: %s", entry.uid)
                    if self.on_expired:
                        self.on_expired(entry)

    def mark_confirmed(self, uid: str):
        with self._lock:
            if uid in self._entries:
                self._entries[uid].state = WatchlistState.CONFIRMED
                self._entries[uid].confirmed_at = datetime.now(timezone.utc)

    def mark_cancelled(self, uid: str):
        with self._lock:
            if uid in self._entries:
                self._entries[uid].state = WatchlistState.CANCELLED

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active_entries(self) -> List[WatchlistEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.is_active]

    def get_triggered_entries(self) -> List[WatchlistEntry]:
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.state == WatchlistState.TRIGGERED
            ]

    def get_all(self) -> List[WatchlistEntry]:
        with self._lock:
            return list(self._entries.values())

    def summary(self) -> str:
        entries = self.get_all()
        active  = sum(1 for e in entries if e.is_active)
        total   = len(entries)
        return f"Watchlist: {active} active / {total} total"


watchlist = WatchlistManager()
