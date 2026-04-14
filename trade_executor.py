import os, json, time, logging, requests, joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT, 
    MODEL_FILE, LOG_FILE, get_tier, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM
)
from deribit_client import DeribitClient
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, get_effective_risk

TRADES_FILE, MAX_OPEN_TRADES = "trades.json", 3
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", 
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

# ════════════ DATA FETCHING ═══════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    urls = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]
    for url in urls:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 100}, timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except: continue
    return pd.DataFrame()

# ════════════ SIGNAL LOGIC (FIXED FOR NaN ERRORS) ═════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        raw_entry = get_data(symbol, TIMEFRAME_ENTRY)
        if raw_entry.empty or len(raw_entry) < 30:
            log.info(f"      ML: WAITING (Insufficient Market Data)")
            return None
        
        # Calculate indicators
        df15 = add_indicators(raw_entry).fillna(0) # 🟢 FIX: Fill NaNs to prevent AI crash
        raw_1h = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(raw_1h).fillna(0) if not raw_1h.empty else pd.DataFrame()

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(0, index=df15.columns)
        
        # Bind 1h features specifically
        row["rsi_1h"] = float(r1h.get("rsi", 50))
        row["adx_1h"] = float(r1h.get("adx", 0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        # 🟢 PILLAR: DETAILED LOGS (ML, ADX, SCORE)
        af = pipeline["all_features"]
        
        # Final NaN check before AI
        X_raw = pd.DataFrame([row[af].values], columns=af).replace([np.inf, -np.inf], 0).fillna(0)
        
        Xs = pipeline["selector"].transform(X_raw)
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pipeline["ensemble"].predict(Xs)[0]]
        conf = round(float(max(prob)) * 100, 1)

        log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: 
            return None

        adx = float(row.get("adx", 0))
        log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: 
            return None

        # Score Logic
        score = 0
        if conf >= 65: score += 1
        if adx > 20: score += 1
        rsi = float(row.get("rsi", 50))
        if (sig == "BUY" and rsi < 55) or (sig == "SELL" and rsi > 45): score += 1
        
        log.info(f"      Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: 
            return None

        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score,
                "entry": float(row["close"]), "atr": float(row["atr"])}
    except Exception as e:
        log.error(f"      Error: {e}") # 🟢 Will now catch and log instead of crashing
        return None

# ════════════ EXECUTION & MAIN ════════════════════════════════════════

def execute_trade(deribit, symbol, signal, entry, atr, risk_mult, balance):
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)
    try:
        res = deribit.place_market_order(symbol, signal, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        if filled > 0:
            actual = float(order.get("average_price", entry))
            stop = deribit.round_price(symbol, actual - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
            tp1 = deribit.round_price(symbol, actual + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
            
            deribit.place_limit_order(symbol, "SELL" if signal=="BUY" else "BUY", filled, stop, stop_price=stop)
            q1, _ = deribit.split_amount(symbol, filled)
            deribit.place_limit_order(symbol, "SELL" if signal=="BUY" else "BUY", q1, tp1)
            
            trades = json.load(open("trades.json")) if Path("trades.json").exists() else {}
            trades[symbol] = {"symbol": symbol, "qty": filled, "entry": actual}
            json.dump(trades, open("trades.json", "w"))
            log.info(f"      ✅ LIVE: {symbol} @ {actual}")
            return True
    except Exception as e: log.error(f"      ❌ FAILED: {e}")
    return False

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, _ = should_scan()
    if not run: return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    deribit.test_connection()
    pipeline = joblib.load(MODEL_FILE)
    thresholds, risk_mult, balance = get_mode_thresholds(mode), get_effective_risk(mode, vol), deribit.get_total_equity_usd()

    log.info(f"Scanning {len(SYMBOLS)} coins | Mode: {mode['label']}")

    for symbol in SYMBOLS:
        current_trades = json.load(open("trades.json")) if Path("trades.json").exists() else {}
        if len(current_trades) >= MAX_OPEN_TRADES: break
        
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], risk_mult, balance)
            time.sleep(1.5)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    run_execution_scan()
