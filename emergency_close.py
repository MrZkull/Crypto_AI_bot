#!/usr/bin/env python3
"""
EMERGENCY CLOSE SCRIPT
Run this ONCE to:
1. Close all open positions on Deribit testnet at market price
2. Cancel all open orders
3. Clear trades.json so bot starts fresh

Usage:
  python emergency_close.py

Add DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET to .env before running.
"""

import os, json, time, requests, hmac, hashlib, logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

def authenticate(session, client_id, client_secret):
    r = session.get(f"{TESTNET_BASE}/public/auth", params={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=15)
    res = r.json().get("result", {})
    session.headers["Authorization"] = f"Bearer {res['access_token']}"
    log.info("✅ Authenticated with Deribit testnet")
    return session

def get(session, path, params=None):
    r = session.get(f"{TESTNET_BASE}{path}", params=params or {}, timeout=15)
    d = r.json()
    if "error" in d: raise Exception(d["error"])
    return d.get("result", d)

def post(session, path, body):
    r = session.post(f"{TESTNET_BASE}{path}", json=body, timeout=15)
    d = r.json()
    if "error" in d: raise Exception(d["error"])
    return d.get("result", d)

def main():
    client_id     = os.getenv("DERIBIT_CLIENT_ID", "")
    client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise ValueError("Set DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET in .env")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    authenticate(session, client_id, client_secret)

    closed_positions = 0
    cancelled_orders = 0

    # ── Step 1: Close all open positions ──────────────────────────────
    log.info("\n[1] Fetching all open positions...")
    for currency in ["BTC", "ETH", "USDC"]:
        try:
            positions = get(session, "/private/get_positions",
                           {"currency": currency, "kind": "future"})
            if not isinstance(positions, list):
                continue
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                if size == 0:
                    continue
                instrument = pos["instrument_name"]
                # To close: sell if long (size>0), buy if short (size<0)
                close_side = "sell" if size > 0 else "buy"
                close_size = abs(size)
                try:
                    result = post(session, f"/private/{close_side}", {
                        "instrument_name": instrument,
                        "amount":          close_size,
                        "type":            "market",
                        "reduce_only":     True,
                        "label":           "emergency_close",
                    })
                    order = result.get("order", result)
                    log.info(f"  ✅ CLOSED {instrument} size={close_size} "
                             f"→ order_id={order.get('order_id','?')}")
                    closed_positions += 1
                    time.sleep(0.5)
                except Exception as e:
                    log.error(f"  ❌ Failed to close {instrument}: {e}")
        except Exception as e:
            log.warning(f"  Positions {currency}: {e}")

    # ── Step 2: Cancel ALL open orders ────────────────────────────────
    log.info("\n[2] Cancelling all open orders...")
    for currency in ["BTC", "ETH", "USDC"]:
        try:
            result = post(session, "/private/cancel_all_by_currency", {
                "currency": currency,
                "kind":     "future",
            })
            n = result if isinstance(result, int) else 0
            if n > 0:
                log.info(f"  ✅ Cancelled {n} orders for {currency}")
                cancelled_orders += n
        except Exception as e:
            log.warning(f"  Cancel {currency}: {e}")

    # ── Step 3: Clear trades.json ──────────────────────────────────────
    log.info("\n[3] Clearing trades.json...")
    for path in [Path("trades.json"), Path("data/trades.json")]:
        try:
            with open(path, "w") as f:
                json.dump({}, f)
            log.info(f"  ✅ Cleared {path}")
        except Exception as e:
            log.warning(f"  Could not clear {path}: {e}")

    # ── Step 4: Get final balance ──────────────────────────────────────
    log.info("\n[4] Final balance check...")
    total_usd = 0
    for currency in ["BTC", "ETH", "USDC"]:
        try:
            s = get(session, "/private/get_account_summary",
                   {"currency": currency, "extended": "true"})
            eq_usd = float(s.get("equity_usd", 0) or s.get("equity", 0) or 0)
            if eq_usd > 0:
                log.info(f"  {currency}: ${eq_usd:.2f} USD")
                total_usd += eq_usd
        except Exception:
            pass

    log.info(f"\n{'='*50}")
    log.info(f"EMERGENCY CLOSE COMPLETE")
    log.info(f"  Positions closed: {closed_positions}")
    log.info(f"  Orders cancelled: {cancelled_orders}")
    log.info(f"  Portfolio value:  ${total_usd:.2f} USD")
    log.info(f"  trades.json:      CLEARED")
    log.info(f"\nBot will start fresh on next GitHub Actions scan.")
    log.info(f"{'='*50}")

if __name__ == "__main__":
    main()
