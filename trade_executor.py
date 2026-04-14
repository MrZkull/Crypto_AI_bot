import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT, MODEL_FILE, LOG_FILE, get_tier, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM
from deribit_client import DeribitClient
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, get_effective_risk

TRADES_FILE = "trades.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

def load_json(path, default):
    return json.load(open(path)) if Path(path).exists() else default

def save_json(path, data):
    json.dump(data, open(path, "w"), indent=2, default=str)

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Robust data fetcher to stop the 'Fast Scan' issue"""
    for _ in range(3): # Try 3 times
        try:
            r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": 100}, timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in df.columns: df[c] = pd.to_numeric(df[c])
                return df
        except: time.sleep(1)
    return pd.DataFrame()

def generate_signal(symbol, pipeline, thresholds):
    df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
    df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
    if df15.empty or len(df15) < 30: return None

    row = df15.iloc[-1].copy()
    r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
    af = pipeline["all_features"]
    
    # Prediction logic
    X = pd.DataFrame([row[af].values], columns=af)
    Xs = pipeline["selector"].transform(X)
    prob = pipeline["ensemble"].predict_proba(Xs)[0]
    sig = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pipeline["ensemble"].predict(Xs)[0]]
    conf = round(float(max(prob)) * 100, 1)

    # 🟢 DETAILS LOGS (Now actually triggered)
    log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
    if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

    adx = float(row.get("adx", 0))
    log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
    if adx < thresholds["min_adx"]: return None

    return {"symbol": symbol, "signal": sig, "confidence": conf, "entry": float(row["close"]), "atr": float(row["atr"])}

def execute_trade(deribit, symbol, signal, entry, atr, balance, risk_mult):
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)
    try:
        res = deribit.place_market_order(symbol, signal, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        if filled > 0:
            actual = float(order.get("average_price", entry))
            stop = deribit.round_price(symbol, actual - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
            tp1 = deribit.round_price(symbol, actual + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
            
            sl_res = deribit.place_limit_order(symbol, "SELL" if signal=="BUY" else "BUY", filled, stop, stop_price=stop)
            q1, q2 = deribit.split_amount(symbol, filled)
            tp1_res = deribit.place_limit_order(symbol, "SELL" if signal=="BUY" else "BUY", q1, tp1)
            
            trades = load_json(TRADES_FILE, {})
            trades[symbol] = {"symbol": symbol, "signal": signal, "entry": actual, "qty": filled}
            save_json(TRADES_FILE, trades)
            log.info(f"      ✅ Trade Executed: {symbol}")
            return True
    except Exception as e: log.error(f"      ❌ Failed: {e}")
    return False

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, reason = should_scan()
    if not run: return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    pipeline = joblib.load(MODEL_FILE)
    thresholds, risk_mult, balance = get_mode_thresholds(mode), get_effective_risk(mode, vol), deribit.get_total_equity_usd()

    for symbol in SYMBOLS:
        if len(load_json(TRADES_FILE, {})) >= 3: break
        log.info(f"\n  ── Checking {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], balance, risk_mult)
            time.sleep(2)
        else: time.sleep(0.5) # Slower scan to ensure data arrives

if __name__ == "__main__":
    run_execution_scan()
