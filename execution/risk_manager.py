from __future__ import annotations

from dataclasses import dataclass

from data.exchange_client import ExchangeClient


@dataclass(slots=True)
class SizingResult:
    qty: float
    leverage: int
    risk_amount: float
    actual_risk: float


class RiskManager:
    def __init__(self, exchange: ExchangeClient, risk_ratio: float = 0.04) -> None:
        self.exchange = exchange
        self.risk_ratio = risk_ratio

    def wallet_balance(self) -> float:
        bal = self.exchange.fetch_balance_sync()
        total = bal.get("USDT", {}).get("total")
        if total is None:
            total = bal.get("total", {}).get("USDT", 0)
        return float(total or 0)

    def size_position(self, symbol: str, entry_price: float, sl_price: float) -> SizingResult | None:
        distance = abs(entry_price - sl_price)
        if distance <= 0:
            return None
        wallet = self.wallet_balance()
        risk_amount = wallet * self.risk_ratio
        raw_qty = risk_amount / distance
        qty = self.exchange.amount_to_precision(symbol, raw_qty)
        market = self.exchange.market(symbol)
        min_qty = float(market.get("limits", {}).get("amount", {}).get("min") or 0)
        min_notional = float(market.get("limits", {}).get("cost", {}).get("min") or 0)
        if qty < min_qty:
            qty = min_qty
        if qty * entry_price < min_notional and entry_price > 0:
            qty = max(qty, min_notional / entry_price)
            qty = self.exchange.amount_to_precision(symbol, qty)
        actual_risk = abs(entry_price - sl_price) * qty
        if wallet <= 0 or actual_risk > risk_amount:
            return None
        notional = qty * entry_price
        leverage = max(1, min(50, int(notional / max(wallet, 1e-8)) + 1))
        return SizingResult(qty=qty, leverage=leverage, risk_amount=risk_amount, actual_risk=actual_risk)
