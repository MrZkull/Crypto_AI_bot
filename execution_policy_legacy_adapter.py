"""Backward-compatible adapter to the canonical execution_policy module.

This file intentionally delegates to the canonical policy instead of keeping a
second copy of P&L mathematics, preventing accounting drift between legacy and
current callers.
"""
from execution_policy import get_config_hash


def execute_tp1_partial(trade: dict, fill_price: float, fee_rate: float, partial_pct: float = 0.5):
    if partial_pct <= 0 or partial_pct > 1:
        raise ValueError("partial_pct must be in (0, 1]")
    side = trade["side"]
    entry = float(trade.get("entry", trade.get("simulated_entry")))
    qty = float(trade["qty"])
    exit_qty = qty * partial_pct
    gross = (fill_price - entry) * exit_qty if side == "BUY" else (entry - fill_price) * exit_qty
    fee_usd = exit_qty * fill_price * fee_rate
    trade["qty"] = qty - exit_qty
    trade["realized_pnl"] = trade.get("realized_pnl", 0.0) + gross - fee_usd
    trade["fees_paid"] = trade.get("fees_paid", trade.get("entry_fee_usd", 0.0)) + fee_usd
    return trade
