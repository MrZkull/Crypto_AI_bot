import os, json, time, logging, requests, joblib
import pandas as pd
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

# ════════════ HELPERS ════════════════════════════════════════════════

def load_json(path, default):
    return json.load(open(path)) if Path(path).exists() else default

def save_json(path, data):
    json.dump(data, open(path, "w"), indent=2, default=str)

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Robust data fetcher to fix 'Insufficient Market Data' error"""
    url = "https://api.binance.com/api/v3/klines"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 100}, timeout=15)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                if not df["close"].isnull().all():
                    return df
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf >= 65: score += 1; reasons.append("High Conf")
    adx = float(row.get("adx", 0))
    if adx > 20: score += 1; reasons.append("Strong ADX")
    rsi = float(row.get("rsi", 50))
    if (signal == "BUY" and rsi < 55) or (signal == "SELL" and rsi > 45):
        score += 1; reasons.append("RSI Align")
    return score, reasons

# ════════════ LOGIC ══════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        # Fetching data for decision
        raw_entry = get_data(symbol, TIMEFRAME_ENTRY)
        if raw_entry.empty or len(raw_entry) < 30:
            log.info(f"      ML: WAITING (Insufficent Market Data)")
            return None
        
        df15 = add_indicators(raw_entry)
        raw_confirm = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(raw_confirm) if not raw_confirm.empty else pd.DataFrame()

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        
        af = pipeline["all_features"]
        for f in ["rsi_1h", "adx_1h", "trend_1h"]:
            row[f] = float(r1h.get(f.replace("_1h", ""), 0))
        
        X = pd.DataFrame([row[af].values], columns=af)
        Xs = pipeline["selector"].transform(X)
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pipeline["ensemble"].predict(Xs)[0]]
        conf = round(float(max(prob)) * 100, 1)

        # 🟢 THE DETAILS YOU WANT (BUY/SELL/NO_TRADE + Conf + Score)
        log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
        
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: 
            return None

        adx = float(row.get("adx", 0))
        log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"      Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: return None

        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score, 
                "entry": float(row["close"]), "atr": float(row["atr"]), "reasons": reasons}
    except Exception as e:
        return None

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult, balance):
    trades = load_json(TRADES_FILE, {})
    side, sl_side, tp_side = ("BUY", "SELL", "SELL") if signal == "BUY" else ("SELL", "BUY", "BUY")
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)

    try:
        res = deribit.place_market_order(symbol, signal, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        
        if filled > 0:
            actual = float(order.get("average_price", entry))
            stop = deribit.round_price(symbol, actual - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
            tp1 = deribit.round_price(symbol, actual + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
            
            deribit.place_limit_order(symbol, sl_side, filled, stop, stop_price=stop)
            q1, q2 = deribit.split_amount(symbol, filled)
            deribit.place_limit_order(symbol, tp_side, q1, tp1)

            trades[symbol] = {"symbol": symbol, "signal": signal, "entry": actual, "qty": filled, "tp1_hit": False, "score": score, "confidence": confidence}
            save_json(TRADES_FILE, trades)
            log.info(f"      ✅ LIVE: {symbol} @ {actual}")
            return True
    except Exception as e: log.error(f"      ❌ FAILED: {e}")
    return False

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, _ = should_scan()
    if not run: return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    pipeline = joblib.load(MODEL_FILE)
    thresholds, risk_mult, balance = get_mode_thresholds(mode), get_effective_risk(mode, vol), deribit.get_total_equity_usd()

    log.info(f"Scanning {len(SYMBOLS)} coins | Open Slots: {3 - len(load_json(TRADES_FILE, {}))}/3")

    for symbol in SYMBOLS:
        if len(load_json(TRADES_FILE, {})) >= MAX_OPEN_TRADES: break
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], risk_mult, balance)
            time.sleep(1)
        else:
            time.sleep(0.5)

    log.info(f"\n{'═'*56}\nSCAN COMPLETE\n{'═'*56}")

if __name__ == "__main__":
    run_execution_scan()
