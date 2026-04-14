# trade_executor.py — Production Logic with Precision Logic Integration
import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    LIVE_LIMIT, MODEL_FILE, LOG_FILE
)
from deribit_client import DeribitClient
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk

TRADES_FILE, MAX_OPEN_TRADES = "trades.json", 3
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

def load_json(path, default):
    p = Path(path)
    return json.load(open(p)) if p.exists() else default

def save_json(path, data):
    json.dump(data, open(path, "w"), indent=2, default=str)

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult, balance):
    trades = load_json(TRADES_FILE, {})
    if symbol in trades or len(trades) >= MAX_OPEN_TRADES: return False
    
    side, sl_side = ("BUY", "SELL") if signal == "BUY" else ("SELL", "BUY")
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)

    try:
        # 1. Market Entry
        res = deribit.place_market_order(symbol, side, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        if filled <= 0: return False

        actual_entry = float(order.get("average_price", entry))
        stop = deribit.round_price(symbol, actual_entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
        tp1 = deribit.round_price(symbol, actual_entry + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
        
        # 2. SL & TP
        sl_res = deribit.place_limit_order(symbol, sl_side, filled, stop, stop_price=stop)
        q1, q2 = deribit.split_amount(symbol, filled)
        tp1_res = deribit.place_limit_order(symbol, sl_side, q1, tp1)

        trades[symbol] = {"symbol": symbol, "signal": signal, "entry": actual_entry, "stop": stop, "tp1": tp1, "qty": filled, "qty_tp1": q1, "qty_tp2": q2, "order_ids": {"stop_loss": str(sl_res.get("order", sl_res).get("order_id")), "tp1": str(tp1_res.get("order", tp1_res).get("order_id"))}, "tp1_hit": False}
        save_json(TRADES_FILE, trades)
        log.info(f"✅ Live: {symbol} @ {actual_entry}")
        return True
    except Exception as e: log.error(f"❌ Failed {symbol}: {e}"); return False

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    run, mode, vol, reason = should_scan()
    if not run:
        log.info(f"  SKIPPED: {reason}")
        return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    deribit.test_connection()
    
    # Load AI Model
    pipeline = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    eff_risk = get_effective_risk(mode, vol)
    
    balance = deribit.get_total_equity_usd()
    
    # Maintenance
    clean_invalid_trades(deribit)
    # Note: ensure check_open_trades is defined in your file or imported
    try:
        check_open_trades(deribit)
    except NameError:
        log.warning("check_open_trades not implemented, skipping monitoring...")

    log.info(f"Scanning {len(SYMBOLS)} coins | Mode: {mode['label']} | Risk: {eff_risk}")

    found = 0
    for symbol in SYMBOLS:
        # Check if we have room for more trades
        current_trades = load_json(TRADES_FILE, {})
        if len(current_trades) >= MAX_OPEN_TRADES:
            log.info("  Max trades reached (3/3). Stopping scan.")
            break

        log.info(f"  ── Checking {symbol} ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        
        if sig:
            found += 1
            log.info(f"  🚀 SIGNAL FOUND: {sig['signal']} {symbol} (Conf: {sig['confidence']}%)")
            success = execute_trade(
                deribit, 
                sig["symbol"], sig["signal"], sig["entry"], sig["atr"], 
                sig["confidence"], sig["score"], sig["reasons"], 
                eff_risk, balance
            )
            if success:
                time.sleep(2) # Prevent rate limits
        else:
            time.sleep(0.3) # Save CPU

    log.info(f"\n{'═'*56}\nSCAN COMPLETE — Found {found} signal(s)\n{'═'*56}")
