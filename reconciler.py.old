"""
reconciler.py
Task 2: Untracked Dust & Orphan Self-Healing Reconciler.

Runs once per scan cycle from smart_scheduler.py. Cross-checks live
Deribit positions against your local trades.json ledger. Any position
that exists on the exchange but is NOT accounted for locally (or has no
live SL/TP order attached) is immediately flattened with a reduce_only
market order -- this is your circuit breaker against orphaned dust.

Import and call `reconcile_positions(deribit_client, trades_store)` from
your existing smart_scheduler loop.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("reconciler")

TRADES_FILE = Path("trades.json")


class TradesStore:
    """Thin wrapper around trades.json so the reconciler and dashboard
    read/write in one place."""

    def __init__(self, path: Path = TRADES_FILE):
        self.path = path
        if not self.path.exists():
            self.path.write_text(json.dumps({"open": {}, "closed": []}, indent=2))

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    def save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def open_symbols(self) -> set:
        return set(self.load().get("open", {}).keys())

    def mark_flattened(self, symbol: str, reason: str) -> None:
        data = self.load()
        data.setdefault("closed", []).append({
            "symbol": symbol,
            "closed_at": time.time(),
            "reason": reason,
            "source": "reconciler_auto_flatten",
        })
        data.get("open", {}).pop(symbol, None)
        self.save(data)


def _has_active_protection(deribit_client, symbol: str) -> bool:
    """
    True if the position on `symbol` has at least one live stop/take-profit
    trigger order registered on the exchange. Adjust the field names below
    to match your deribit client's get_open_orders() response shape.
    """
    open_orders = deribit_client.get_open_orders(instrument_name=symbol)
    for order in open_orders:
        order_type = order.get("order_type", "")
        if order.get("reduce_only") and order_type in (
            "stop_market", "stop_limit", "take_market", "take_limit", "trigger"
        ):
            return True
    return False


def reconcile_positions(deribit_client, trades_store: TradesStore) -> List[Dict]:
    """
    Executes one reconciliation pass. Returns a list of actions taken, so
    smart_scheduler can log/alert on them.
    """
    actions = []
    live_positions = deribit_client.get_positions(kind="future")  # linear USDC perps
    tracked_symbols = trades_store.open_symbols()

    for pos in live_positions:
        symbol = pos.get("instrument_name")
        size = pos.get("size", 0)

        if not symbol or size == 0:
            continue  # flat, nothing to do

        is_tracked = symbol in tracked_symbols
        is_protected = _has_active_protection(deribit_client, symbol)

        if is_tracked and is_protected:
            continue  # healthy, known position with live SL/TP -- leave it alone

        # Orphan condition: untracked OR tracked-but-unprotected.
        reason = []
        if not is_tracked:
            reason.append("no matching record in trades.json")
        if not is_protected:
            reason.append("no active SL/TP trigger order on exchange")
        reason_str = "; ".join(reason)

        logger.error(
            f"[ORPHAN DETECTED] {symbol} size={size} reason={reason_str} "
            f"-- flattening immediately with reduce_only market order"
        )

        flatten_side = "sell" if size > 0 else "buy"
        try:
            deribit_client.place_order(
                instrument_name=symbol,
                side=flatten_side,
                amount=abs(size),
                type="market",
                reduce_only=True,
            )
            trades_store.mark_flattened(symbol, reason_str)
            actions.append({"symbol": symbol, "action": "flattened", "reason": reason_str})
        except Exception as exc:
            logger.critical(f"[RECONCILER FAILURE] Could not flatten {symbol}: {exc}")
            actions.append({"symbol": symbol, "action": "flatten_failed", "error": str(exc)})

    return actions


def get_open_positions_for_dashboard(deribit_client) -> List[Dict]:
    """
    Backing function for GET /api/trades/open.
    Renders EVERY open position regardless of notional size -- do not
    filter out sub-$10 positions. That filtering is exactly what let
    dust risk hide on the dashboard while remaining live on the exchange.
    """
    positions = deribit_client.get_positions(kind="future")
    return [
        {
            "symbol": p.get("instrument_name"),
            "size": p.get("size"),
            "average_price": p.get("average_price"),
            "mark_price": p.get("mark_price"),
            "unrealized_pnl": p.get("floating_profit_loss"),
            "notional_usd": abs(p.get("size", 0)) * p.get("mark_price", 0),
        }
        for p in positions
        if p.get("size", 0) != 0
    ]
