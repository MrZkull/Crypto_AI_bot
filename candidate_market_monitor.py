#!/usr/bin/env python3
# candidate_market_monitor.py — Prospective Shadow Position Lifecycle & Outcome Logger

import os
import sys
import json
import time
import math
import logging
import urllib.request
from pathlib import Path
from filelock import FileLock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STATE_FILE = Path("candidate_state.json")
EVENTS_LEDGER = Path("candidate_events.jsonl")
MANIFEST_FILE = Path("candidate_manifest.json")

FEE_RATE = 0.0006  # 0.06% taker fee assumption


def log_event(event_dict: dict):
    with FileLock("candidate_events.jsonl.lock"):
        with open(EVENTS_LEDGER, "a") as f:
            f.write(json.dumps(event_dict) + "\n")


def fetch_latest_candle(symbol: str) -> dict:
    """Fetches the latest closed 15m candle via resilient Binance Spot endpoints."""
    endpoints = [
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=15m&limit=2",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=2",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    raw = json.loads(resp.read().decode())
                    if len(raw) >= 1:
                        # Inspect the last fully closed candle (index -2) or current candle (index -1)
                        k = raw[-1]
                        return {
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "close_time": int(k[6])
                        }
        except Exception:
            continue
    return {}


def monitor_active_positions():
    if not STATE_FILE.exists() or not MANIFEST_FILE.exists():
        return

    with open(MANIFEST_FILE, "r") as mf:
        manifest = json.load(mf)
    candidate_id = manifest.get("candidate_id")

    with FileLock("candidate_state.json.lock"):
        try:
            with open(STATE_FILE, "r") as sf:
                state = json.load(sf)
        except Exception:
            return

        if not state:
            return

        active_keys = [k for k, v in state.items() if v.get("status") in ("ACTIVE", "COUNTERFACTUAL_ACTIVE")]
        if not active_keys:
            return

        log.info(f"Monitoring {len(active_keys)} active candidate positions...")
        modified = False

        for pid in active_keys:
            trade = state[pid]
            sym = trade["symbol"]
            side = trade["side"].upper()
            entry = float(trade["simulated_entry"])
            stop = float(trade["stop"])
            tp1 = float(trade["tp1"])
            qty = float(trade["qty"])
            risk_usd = float(trade["initial_risk_usd"])
            current_state = trade.get("state", "OPEN")

            candle = fetch_latest_candle(sym)
            if not candle:
                continue

            h, l, c = candle["high"], candle["low"], candle["close"]
            closed = False
            exit_price = 0.0
            reason = ""

            # Check Long Lifecycles
            if side == "BUY":
                # 1. Partial TP1
                if current_state in ("OPEN", "COUNTERFACTUAL_OPEN") and h >= tp1:
                    new_state = "PARTIAL_TP1"
                    trade["state"] = new_state
                    tp1_fee = (qty * 0.5) * tp1 * FEE_RATE
                    trade["tp1_fee_usd"] = float(trade.get("tp1_fee_usd", 0.0)) + tp1_fee
                    log_event({
                        "event": "PARTIAL_TP1_FILLED",
                        "candidate_id": candidate_id,
                        "pred_id": pid,
                        "tp1_price": tp1,
                        "tp1_fee": tp1_fee,
                        "timestamp_ms": int(time.time() * 1000)
                    })
                    modified = True

                # 2. Stop Loss Hit
                if l <= stop:
                    closed = True
                    exit_price = stop
                    reason = "STOP_LOSS"
                # 3. Full Take Profit Hit
                elif h >= float(trade.get("tp2", tp1)):
                    closed = True
                    exit_price = float(trade.get("tp2", tp1))
                    reason = "TAKE_PROFIT"

            # Check Short Lifecycles
            elif side == "SELL":
                # 1. Partial TP1
                if current_state in ("OPEN", "COUNTERFACTUAL_OPEN") and l <= tp1:
                    new_state = "PARTIAL_TP1"
                    trade["state"] = new_state
                    tp1_fee = (qty * 0.5) * tp1 * FEE_RATE
                    trade["tp1_fee_usd"] = float(trade.get("tp1_fee_usd", 0.0)) + tp1_fee
                    log_event({
                        "event": "PARTIAL_TP1_FILLED",
                        "candidate_id": candidate_id,
                        "pred_id": pid,
                        "tp1_price": tp1,
                        "tp1_fee": tp1_fee,
                        "timestamp_ms": int(time.time() * 1000)
                    })
                    modified = True

                # 2. Stop Loss Hit
                if h >= stop:
                    closed = True
                    exit_price = stop
                    reason = "STOP_LOSS"
                # 3. Full Take Profit Hit
                elif l <= float(trade.get("tp2", tp1)):
                    closed = True
                    exit_price = float(trade.get("tp2", tp1))
                    reason = "TAKE_PROFIT"

            if closed:
                exit_fee = qty * exit_price * FEE_RATE
                if side == "BUY":
                    gross_pnl = (exit_price - entry) * qty
                else:
                    gross_pnl = (entry - exit_price) * qty

                net_pnl = gross_pnl - float(trade.get("entry_fee_usd", 0.0)) - float(trade.get("tp1_fee_usd", 0.0)) - exit_fee
                net_r = (net_pnl / risk_usd) if risk_usd > 0 else 0.0

                trade["status"] = "CLOSED"
                trade["state"] = "CLOSED"
                trade["exit_price"] = exit_price
                trade["realized_gross_pnl"] = gross_pnl
                trade["final_exit_fee_usd"] = exit_fee
                trade["net_r"] = net_r
                trade["exit_reason"] = reason

                outcome_event = {
                    "event": "OUTCOME_OBSERVED",
                    "candidate_id": candidate_id,
                    "pred_id": pid,
                    "reason": reason,
                    "exit_price": exit_price,
                    "realized_gross_pnl": gross_pnl,
                    "final_exit_fee_usd": exit_fee,
                    "funding_usd": 0.0,
                    "net_r": net_r,
                    "timestamp_ms": int(time.time() * 1000)
                }
                log_event(outcome_event)
                log.info(f"✅ Closed Position {pid} ({side} on {sym}): Reason={reason} | Net-R={net_r:+.2f}R")
                modified = True

        if modified:
            tmp = str(STATE_FILE) + ".tmp"
            with open(tmp, "w") as sf:
                json.dump(state, sf, indent=2)
            os.replace(tmp, STATE_FILE)


if __name__ == "__main__":
    monitor_active_positions()
