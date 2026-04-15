from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from core.models import EntrySignal, OpenTrade, SetupSignal, WatchlistItem


class Database:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    active INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bias TEXT NOT NULL,
                    range_high REAL NOT NULL,
                    range_low REAL NOT NULL,
                    midpoint REAL NOT NULL,
                    sweep_side TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_watchlist_status ON watchlist(status, created_at);
                CREATE TABLE IF NOT EXISTS setups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bias TEXT NOT NULL,
                    entry_zone_low REAL NOT NULL,
                    entry_zone_high REAL NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_usd REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    close_price REAL,
                    pnl_usd REAL,
                    status TEXT NOT NULL,
                    exchange_order_id TEXT
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_symbols(self, symbols: Iterable[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            for sym in symbols:
                conn.execute(
                    """
                    INSERT INTO symbols(symbol, active, updated_at) VALUES(?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET active=excluded.active, updated_at=excluded.updated_at
                    """,
                    (sym, 1, now),
                )

    def insert_watchlist(self, item: WatchlistItem) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO watchlist(symbol,bias,range_high,range_low,midpoint,sweep_side,score,created_at,status)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    item.symbol,
                    item.bias,
                    item.range_high,
                    item.range_low,
                    item.midpoint,
                    item.sweep_side,
                    item.score,
                    item.created_at.isoformat(),
                    item.status,
                ),
            )
            return int(cur.lastrowid)

    def update_watchlist_status(self, symbol: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE watchlist SET status=? WHERE symbol=? AND status!='EXPIRED'", (status, symbol))

    def insert_setup(self, setup: SetupSignal) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO setups(symbol,bias,entry_zone_low,entry_zone_high,reason,created_at,status)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    setup.symbol,
                    setup.bias,
                    setup.entry_zone_low,
                    setup.entry_zone_high,
                    setup.reason,
                    setup.created_at.isoformat(),
                    setup.status,
                ),
            )
            return int(cur.lastrowid)

    def insert_trade(self, signal: EntrySignal, qty: float, order_id: str | None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades(symbol,side,qty,entry_price,sl_price,tp_usd,opened_at,status,exchange_order_id)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    signal.symbol,
                    signal.side,
                    qty,
                    signal.entry_price,
                    signal.sl_price,
                    0.40,
                    signal.created_at.isoformat(),
                    "OPEN",
                    order_id,
                ),
            )
            return int(cur.lastrowid)

    def close_trade(self, trade_id: int, close_price: float, pnl_usd: float, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET closed_at=?, close_price=?, pnl_usd=?, status=?
                WHERE id=?
                """,
                (datetime.now(timezone.utc).isoformat(), close_price, pnl_usd, status, trade_id),
            )

    def get_open_trades(self) -> list[OpenTrade]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()
        return [
            OpenTrade(
                id=row["id"],
                symbol=row["symbol"],
                side=row["side"],
                qty=row["qty"],
                entry_price=row["entry_price"],
                sl_price=row["sl_price"],
                tp_usd=row["tp_usd"],
                opened_at=datetime.fromisoformat(row["opened_at"]),
                exchange_order_id=row["exchange_order_id"],
            )
            for row in rows
        ]

    def log(self, level: str, message: str, **context: object) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO logs(level,message,context,created_at) VALUES(?,?,?,?)",
                (
                    level,
                    message,
                    json.dumps(context) if context else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
