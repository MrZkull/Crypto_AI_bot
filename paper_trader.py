# paper_trader.py — Paper Trading Engine
#
# WHY THIS EXISTS:
#   Binance testnet geo-blocks GitHub Actions IPs (451 error).
#   Solution: Use real Binance PUBLIC market data for signals (never blocked),
#   simulate trade execution locally with realistic fills based on live prices.
#   No API keys needed for any market operations.
#   Balance, orders, PnL all tracked in local JSON files.
#
# This gives you IDENTICAL functionality to testnet with zero geo-block issues.

import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

BALANCE_FILE  = "balance.json"
TRADES_FILE   = "trades.json"


# ── Public price fetch (never geo-blocked) ────────────────

def get_live_price(symbol: str) -> float:
    """Get current market price from public Binance API. No auth required."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=8
        )
        if r.ok:
            return float(r.json()["price"])
    except Exception as e:
        log.warning(f"Price fetch {symbol}: {e}")
    return 0.0


def get_live_prices_bulk(symbols: list) -> dict:
    """Get prices for multiple symbols at once."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=8)
        if r.ok:
            return {item["symbol"]: float(item["price"])
                    for item in r.json()
                    if item["symbol"] in symbols}
    except Exception as e:
        log.warning(f"Bulk price fetch: {e}")
    return {}


# ── Virtual balance ───────────────────────────────────────

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception: pass
    return default


def save_json_file(path, data):
    try:
        import os
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"save_json {path}: {e}")


class PaperTrader:
    """
    Simulates trade execution using real live market prices.
    No exchange connection, no API keys, no geo-block possible.
    All state stored in JSON files.
    """

    STARTING_BALANCE = 10_000.0  # USDT

    def __init__(self):
        self._ensure_balance()

    def _ensure_balance(self):
        """Create balance.json with starting capital if it doesn't exist."""
        bal = load_json(BALANCE_FILE, {})
        if not bal or bal.get("usdt") in (None, 0):
            self._save_balance(self.STARTING_BALANCE)
            log.info(f"  ✅ Paper trading: initialized with {self.STARTING_BALANCE} USDT")

    def _save_balance(self, usdt: float, assets: list = None):
        save_json_file(BALANCE_FILE, {
            "usdt":       round(usdt, 4),
            "assets":     assets or [{"asset":"USDT","free":str(round(usdt,4)),"total":str(round(usdt,4))}],
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":       "paper_trading",
        })

    def get_usdt_balance(self) -> float:
        """Returns current free USDT."""
        bal = load_json(BALANCE_FILE, {})
        # Subtract capital locked in open trades
        trades    = load_json(TRADES_FILE, {})
        locked    = sum(t.get("qty", 0) * t.get("entry", 0)
                        for t in trades.values()
                        if not t.get("closed") and t.get("signal") != "RECOVERED")
        free_usdt = float(bal.get("usdt", self.STARTING_BALANCE))
        return max(0.0, free_usdt - locked)

    def get_balance(self) -> dict:
        bal = load_json(BALANCE_FILE, {})
        return {"USDT": float(bal.get("usdt", self.STARTING_BALANCE))}

    def save_balance_snapshot(self):
        """
        Update balance.json with current equity.
        Calculates unrealised PnL from open trades using live prices.
        """
        trades    = load_json(TRADES_FILE, {})
        bal       = load_json(BALANCE_FILE, {})
        base_usdt = float(bal.get("usdt", self.STARTING_BALANCE))

        symbols   = [s for s in trades if not trades[s].get("closed")]
        prices    = get_live_prices_bulk(symbols) if symbols else {}
        upnl      = 0.0

        for sym, t in trades.items():
            if t.get("closed"): continue
            live = prices.get(sym)
            if live and t.get("entry") and t.get("qty"):
                if t["signal"] in ("BUY", "RECOVERED"):
                    upnl += (live - t["entry"]) * t["qty"]
                else:
                    upnl += (t["entry"] - live) * t["qty"]

        equity = base_usdt + upnl
        assets = [
            {"asset":"USDT","free":str(round(base_usdt,4)),"total":str(round(equity,4))},
        ]
        save_json_file(BALANCE_FILE, {
            "usdt":        round(base_usdt, 4),
            "equity":      round(equity, 4),
            "unrealised":  round(upnl, 4),
            "assets":      assets,
            "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":        "paper_trading",
            "open_trades": len(symbols),
        })
        log.info(f"  ✅ balance.json: {base_usdt:.2f} USDT | "
                 f"unrealised: {upnl:+.2f} | equity: {equity:.2f}")
        return base_usdt

    # ── Order simulation ───────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Simulate market fill at current live price.
        Realistic: uses actual market price, not the signal price.
        """
        live_price = get_live_price(symbol)
        if live_price <= 0:
            raise Exception(f"Cannot get live price for {symbol}")

        # Simulate small slippage (0.05%)
        slippage = 1.0005 if side.upper() == "BUY" else 0.9995
        fill_price = live_price * slippage

        # Simulate order ID
        order_id = f"paper_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"

        log.info(f"  📝 PAPER {side} {quantity:.6f} {symbol} @ {fill_price:.4f} "
                 f"(live: {live_price:.4f})")

        return {
            "orderId":    order_id,
            "symbol":     symbol,
            "side":       side.upper(),
            "type":       "MARKET",
            "origQty":    str(quantity),
            "executedQty": str(quantity),
            "status":     "FILLED",
            "fills":      [{"price": str(fill_price), "qty": str(quantity)}],
            "price":      str(fill_price),
            "paper_fill": fill_price,
        }

    def place_limit_order(self, symbol: str, side: str, quantity: float,
                          price: float, stop_price: float = None) -> dict:
        """Record limit order — will be checked against live prices each scan."""
        order_id = f"paper_{symbol}_{side}_{int(datetime.now(timezone.utc).timestamp())}"
        order_type = "STOP_LOSS_LIMIT" if stop_price else "LIMIT"
        log.info(f"  📝 PAPER {order_type} {side} {quantity:.6f} {symbol} @ {price:.4f}")
        return {
            "orderId":    order_id,
            "symbol":     symbol,
            "side":       side.upper(),
            "type":       order_type,
            "origQty":    str(quantity),
            "price":      str(price),
            "stopPrice":  str(stop_price or price),
            "status":     "NEW",
        }

    def get_order(self, symbol: str, order_id) -> dict:
        """
        Check if a paper order (SL/TP) has been triggered by live price.
        Compares current market price against order price.
        """
        if not str(order_id).startswith("paper_"):
            return {"status": "NEW"}

        trades = load_json(TRADES_FILE, {})
        trade  = trades.get(symbol, {})
        if not trade:
            return {"status": "CANCELED"}

        live_price = get_live_price(symbol)
        if live_price <= 0:
            return {"status": "NEW"}

        signal    = trade.get("signal", "BUY")
        is_buy    = signal in ("BUY", "RECOVERED")
        tp1_price = float(trade.get("tp1", 0))
        tp2_price = float(trade.get("tp2", 0))
        sl_price  = float(trade.get("stop", 0))

        # Determine which order this is
        oids = trade.get("order_ids", {})
        if str(oids.get("tp1")) == str(order_id):
            # TP1 hit if price crossed target
            triggered = (live_price >= tp1_price) if is_buy else (live_price <= tp1_price)
            if triggered and tp1_price > 0:
                return {"status": "FILLED", "price": str(tp1_price),
                        "orderId": order_id}

        elif str(oids.get("tp2")) == str(order_id):
            triggered = (live_price >= tp2_price) if is_buy else (live_price <= tp2_price)
            if triggered and tp2_price > 0:
                return {"status": "FILLED", "price": str(tp2_price),
                        "orderId": order_id}

        elif str(oids.get("stop_loss")) == str(order_id):
            # SL triggered if price went against us
            triggered = (live_price <= sl_price) if is_buy else (live_price >= sl_price)
            if triggered and sl_price > 0:
                return {"status": "FILLED", "price": str(live_price),
                        "orderId": order_id}

        return {"status": "NEW"}

    def cancel_order(self, symbol: str, order_id) -> dict:
        """No-op for paper orders."""
        return {"status": "CANCELED", "orderId": order_id}

    def get_open_orders(self, symbol: str = None) -> list:
        """Return virtual open orders from trades.json."""
        trades = load_json(TRADES_FILE, {})
        orders = []
        for sym, t in trades.items():
            if t.get("closed"): continue
            if symbol and sym != symbol: continue
            for key, oid in t.get("order_ids", {}).items():
                if key != "entry":
                    orders.append({
                        "symbol":   sym,
                        "orderId":  oid,
                        "type":     "STOP_LOSS_LIMIT" if key=="stop_loss" else "LIMIT",
                        "side":     "SELL" if t.get("signal")=="BUY" else "BUY",
                        "origQty":  str(t.get("qty", 0)),
                        "price":    str(t.get("stop",0) if key=="stop_loss" else
                                        t.get("tp1",0) if key=="tp1" else t.get("tp2",0)),
                        "status":   "NEW",
                    })
        return orders

    def get_closed_orders(self, symbol: str, limit: int = 10) -> list:
        """Return closed orders from trade history."""
        from pathlib import Path
        try:
            if Path("trade_history.json").exists():
                with open("trade_history.json") as f:
                    hist = json.load(f)
                return [h for h in hist if h.get("symbol")==symbol][-limit:]
        except Exception: pass
        return []

    def update_balance_after_close(self, pnl: float):
        """
        Called when a trade closes — update USDT balance with realised PnL.
        """
        bal  = load_json(BALANCE_FILE, {})
        usdt = float(bal.get("usdt", self.STARTING_BALANCE))
        new_usdt = usdt + pnl
        self._save_balance(new_usdt)
        log.info(f"  💰 Balance updated: {usdt:.2f} → {new_usdt:.2f} USDT (PnL: {pnl:+.4f})")

    def test_connection(self) -> bool:
        """Always succeeds — no connection needed."""
        log.info("  ✅ Paper trading mode — no exchange connection needed")
        return True
