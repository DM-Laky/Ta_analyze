from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.models import WatchlistItem
from storage.database import Database


class WatchlistManager:
    def __init__(self, db: Database, expiry_hours: int = 8) -> None:
        self.db = db
        self.expiry = timedelta(hours=expiry_hours)
        self.items: dict[str, WatchlistItem] = {}

    def upsert(self, item: WatchlistItem) -> bool:
        existing = self.items.get(item.symbol)
        if existing and existing.score >= item.score:
            return False
        self.items[item.symbol] = item
        self.db.insert_watchlist(item)
        return True

    def expire(self) -> None:
        now = datetime.now(timezone.utc)
        for symbol, item in list(self.items.items()):
            if now - item.created_at > self.expiry:
                self.db.update_watchlist_status(symbol, "EXPIRED")
                del self.items[symbol]

    def invalidate(self, symbol: str, reason: str = "INVALIDATED") -> None:
        if symbol in self.items:
            self.db.update_watchlist_status(symbol, reason)
            del self.items[symbol]

    def ranked(self) -> list[WatchlistItem]:
        return sorted(self.items.values(), key=lambda i: i.score, reverse=True)
