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
    # Fetch the most recent fully closed 15m Binance candle.
    endpoints = [
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=15m&limit=3",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=3",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    now_ms = int(time.time() * 1000)

    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status != 200:
                    continue

                raw = json.loads(resp.read().decode())
                closed = [
                    k for k in raw
                    if len(k) >= 7 and int(k[6]) <= now_ms
                ]

                if not closed:
                    continue

                k = max(closed, key=lambda item: int(item[6]))
                return {
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "close_time": int(k[6]),
                }
        except Exception:
            continue

    return {}

def process_trade_candle(trade: dict, candle: dict) -> dict:
    # Apply one closed 15m candle to one candidate shadow trade.
    # C1: TP1 realizes 50% quantity and stores the remaining quantity.
    # C2: unordered OHLC barrier hits are fail-closed as AMBIGUOUS_BARRIER.
    side = str(trade["side"]).upper()
    entry = float(trade["simulated_entry"])
    stop = float(trade["stop"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade.get("tp2", tp1))
    original_qty = float(trade["qty"])
    risk_usd = float(trade["initial_risk_usd"])

    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported trade side: {side}")
    if original_qty <= 0 or risk_usd <= 0:
        raise ValueError("Trade quantity and initial risk must be positive.")

    high = float(candle["high"])
    low = float(candle["low"])
    current_state = trade.get("state", "OPEN")
    events = []

    if side == "BUY":
        hit_tp1 = high >= tp1
        hit_tp2 = high >= tp2
        hit_stop = low <= stop
    else:
        hit_tp1 = low <= tp1
        hit_tp2 = low <= tp2
        hit_stop = high >= stop

    if current_state in ("OPEN", "COUNTERFACTUAL_OPEN"):
        # Before TP1, the OHLC bar cannot establish TP1/TP2/STOP ordering.
        if (hit_stop and (hit_tp1 or hit_tp2)) or hit_tp2:
            trade.update({
                "status": "CLOSED",
                "state": "CLOSED",
                "exit_price": None,
                "realized_gross_pnl": None,
                "final_exit_fee_usd": 0.0,
                "net_r": None,
                "exit_reason": "AMBIGUOUS_BARRIER",
                "ambiguous_barrier": True,
            })
            events.append({
                "event": "OUTCOME_OBSERVED",
                "reason": "AMBIGUOUS_BARRIER",
                "exit_price": None,
                "realized_gross_pnl": None,
                "final_exit_fee_usd": 0.0,
                "funding_usd": 0.0,
                "net_r": None,
            })
            return {"modified": True, "closed": True, "events": events}

        if hit_stop:
            exit_price = stop
            gross = (
                (exit_price - entry) * original_qty
                if side == "BUY"
                else (entry - exit_price) * original_qty
            )
            exit_fee = original_qty * exit_price * FEE_RATE
            net = gross - float(trade.get("entry_fee_usd", 0.0)) - exit_fee

            trade.update({
                "status": "CLOSED",
                "state": "CLOSED",
                "exit_price": exit_price,
                "realized_gross_pnl": gross,
                "final_exit_fee_usd": exit_fee,
                "net_r": net / risk_usd,
                "exit_reason": "STOP_LOSS",
                "remaining_qty": 0.0,
            })
            events.append({
                "event": "OUTCOME_OBSERVED",
                "reason": "STOP_LOSS",
                "exit_price": exit_price,
                "realized_gross_pnl": gross,
                "final_exit_fee_usd": exit_fee,
                "funding_usd": 0.0,
                "net_r": trade["net_r"],
            })
            return {"modified": True, "closed": True, "events": events}

        if hit_tp1:
            partial_qty = original_qty * 0.5
            remaining_qty = original_qty - partial_qty
            tp1_gross = (
                (tp1 - entry) * partial_qty
                if side == "BUY"
                else (entry - tp1) * partial_qty
            )
            tp1_fee = partial_qty * tp1 * FEE_RATE

            trade["state"] = "PARTIAL_TP1"
            trade["remaining_qty"] = remaining_qty
            trade["tp1_realized_gross_pnl"] = (
                float(trade.get("tp1_realized_gross_pnl", 0.0)) + tp1_gross
            )
            trade["tp1_fee_usd"] = (
                float(trade.get("tp1_fee_usd", 0.0)) + tp1_fee
            )

            events.append({
                "event": "PARTIAL_TP1_FILLED",
                "tp1_price": tp1,
                "partial_qty": partial_qty,
                "remaining_qty": remaining_qty,
                "tp1_realized_gross_pnl": tp1_gross,
                "tp1_fee": tp1_fee,
            })
            return {"modified": True, "closed": False, "events": events}

        return {"modified": False, "closed": False, "events": []}

    if current_state != "PARTIAL_TP1":
        return {"modified": False, "closed": False, "events": []}

    remaining_qty = float(trade.get("remaining_qty", original_qty * 0.5))
    if remaining_qty <= 0 or remaining_qty > original_qty:
        raise ValueError("Invalid remaining_qty.")

    # After TP1, TP2 and STOP on the same OHLC candle remain unordered.
    if hit_stop and hit_tp2:
        trade.update({
            "status": "CLOSED",
            "state": "CLOSED",
            "exit_price": None,
            "realized_gross_pnl": None,
            "final_exit_fee_usd": 0.0,
            "net_r": None,
            "exit_reason": "AMBIGUOUS_BARRIER",
            "ambiguous_barrier": True,
        })
        events.append({
            "event": "OUTCOME_OBSERVED",
            "reason": "AMBIGUOUS_BARRIER",
            "exit_price": None,
            "realized_gross_pnl": None,
            "final_exit_fee_usd": 0.0,
            "funding_usd": 0.0,
            "net_r": None,
        })
        return {"modified": True, "closed": True, "events": events}

    if not hit_stop and not hit_tp2:
        return {"modified": False, "closed": False, "events": []}

    exit_price = tp2 if hit_tp2 else stop
    reason = "TAKE_PROFIT" if hit_tp2 else "STOP_LOSS"
    final_gross = (
        (exit_price - entry) * remaining_qty
        if side == "BUY"
        else (entry - exit_price) * remaining_qty
    )
    exit_fee = remaining_qty * exit_price * FEE_RATE

    total_gross = (
        float(trade.get("tp1_realized_gross_pnl", 0.0)) + final_gross
    )
    net = (
        total_gross
        - float(trade.get("entry_fee_usd", 0.0))
        - float(trade.get("tp1_fee_usd", 0.0))
        - exit_fee
    )

    trade.update({
        "status": "CLOSED",
        "state": "CLOSED",
        "exit_price": exit_price,
        "realized_gross_pnl": total_gross,
        "final_exit_fee_usd": exit_fee,
        "net_r": net / risk_usd,
        "exit_reason": reason,
        "remaining_qty": 0.0,
        "final_exit_qty": remaining_qty,
    })

    events.append({
        "event": "OUTCOME_OBSERVED",
        "reason": reason,
        "exit_price": exit_price,
        "realized_gross_pnl": total_gross,
        "final_exit_fee_usd": exit_fee,
        "funding_usd": 0.0,
        "net_r": trade["net_r"],
    })
    return {"modified": True, "closed": True, "events": events}


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

            outcome = process_trade_candle(
                trade,
                candle,
            )

            if outcome["modified"]:
                modified = True

            for event in outcome["events"]:
                event = dict(event)
                event["candidate_id"] = candidate_id
                event["pred_id"] = pid
                event["timestamp_ms"] = int(time.time() * 1000)
                log_event(event)

        if modified:
            tmp = str(STATE_FILE) + ".tmp"
            with open(tmp, "w") as sf:
                json.dump(state, sf, indent=2)
            os.replace(tmp, STATE_FILE)


if __name__ == "__main__":
    monitor_active_positions()
