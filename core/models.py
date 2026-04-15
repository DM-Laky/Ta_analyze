from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Bias = Literal["BUY", "SELL", "NEUTRAL"]
Side = Literal["BUY", "SELL"]


@dataclass(slots=True)
class WatchlistItem:
    symbol: str
    bias: Bias
    range_high: float
    range_low: float
    midpoint: float
    sweep_side: str
    score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "WATCHLIST"


@dataclass(slots=True)
class SetupSignal:
    symbol: str
    bias: Side
    entry_zone_low: float
    entry_zone_high: float
    reason: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "SETUP_VALID_5M"


@dataclass(slots=True)
class EntrySignal:
    symbol: str
    side: Side
    entry_price: float
    sl_price: float
    trigger_type: str = "TBS"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class OpenTrade:
    id: int
    symbol: str
    side: Side
    qty: float
    entry_price: float
    sl_price: float
    tp_usd: float
    opened_at: datetime
    status: str = "OPEN"
    exchange_order_id: str | None = None
