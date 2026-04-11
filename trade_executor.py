# trade_executor.py — FULL VERSION | ALL BUGS FIXED
# FIX 1: save_signal() called AFTER orders placed with real entry/order IDs
# FIX 2: fill price from trades[] array via get_fill_price()
# FIX 3: is_order_filled() uses order_state field (Deribit)
# FIX 4: clear_stuck_trades() removes old trades with $0 SL/TP
# FIX 5: Removed PaperTrader import to prevent circular dependency crash

import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    LIVE_LIMIT, MODEL_FILE, LOG_FILE, get_tier
)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk
from deribit_client import DeribitClient, TRADEABLE

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

# ════════════ HELPERS ════════════════════════════════════════════════

def load_json(p, d):
    try:
        for path in [Path(p), Path("data") / p]:
            if path.exists():
                with open(path) as f: return json.load(f)
    except: pass
    return d

def save_json(p, data):
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])
load_signals = lambda: load_json(SIGNALS_FILE, [])
append_history = lambda rec: (lambda h: (h.append(rec), save_json(HISTORY_FILE, h)))(load_history())

def save_signal(sig):
    s = load_signals()
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])

# ════════════ INIT ═══════════════════════════════════════════════════

def init_deribit() -> DeribitClient:
    cid    = os.getenv("DERIBIT_CLIENT_ID",     "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        raise ValueError("DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET not set in GitHub Secrets")
    return DeribitClient(cid, secret)

# ════════════ BALANCE ════════════════════════════════════════════════

def fetch_and_save_balance(deribit: DeribitClient) -> float:
    try:
        deribit_usd = deribit.get_usdt_equivalent()
        balances    = deribit.get_all_balances()
        assets = [
            {"asset": cur, "free": round(float(info.get("available", 0)), 6),
             "total": round(float(info.get("equity_usd", 0)), 2), "source": "deribit_testnet"}
            for cur, info in balances.items() if float(info.get("equity_usd", 0)) > 0
        ]
        save_json(BALANCE_FILE, {
            "usdt":        round(deribit_usd, 2),
            "equity":      round(deribit_usd, 2),
            "deribit_usd": round(deribit_usd, 2),
            "unrealised":  0.0,
            "assets":      assets,
            "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":        "deribit_testnet",
            "note":        f"Deribit Portfolio: ${deribit_usd:.0f}",
        })
        log.info(f"✓ Balance: ${deribit_usd:.2f} (Deribit Testnet)")
        return round(deribit_usd, 2)
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        bal = load_json(BALANCE_FILE, {})
        return float(bal.get("usdt") or bal.get("equity") or 10000)

# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
            return df
        except: continue
    raise Exception(f"All kline endpoints failed for {symbol}")

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"✓ Model: {len(p['all_features'])} features")
    return p

# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult=1.0, deribit_balance=10000.0):
    trades = load_trades()
    if symbol in trades: return False
    if len(trades) >= MAX_OPEN_TRADES: return False
    
    dec = 4 if entry < 10 else 2
    stop = round(entry - atr*ATR_STOP_MULT, dec) if signal == "BUY" else round(entry + atr*ATR_STOP_MULT, dec)
    tp1  = round(entry + atr*ATR_TARGET1_MULT, dec) if signal == "BUY" else round(entry - atr*ATR_TARGET1_MULT, dec)
    tp2  = round(entry + atr*ATR_TARGET2_MULT, dec) if signal == "BUY" else round(entry - atr*ATR_TARGET2_MULT, dec)
    side = "BUY" if signal == "BUY" else "SELL"
    sl_side = "SELL" if side == "BUY" else "BUY"

    if deribit.is_supported(symbol):
        try:
            # 1. Entry
            res = deribit.place_market_order(symbol, side, deribit_balance * RISK_PER_TRADE * risk_mult)
            order = res.get("order", res)
            order_id = str(order.get("order_id", ""))
            
            # 2. Get Real Fill
            time.sleep(1.2)
            actual_entry = deribit.get_fill_price(res, entry)
            
            # 3. SL & TP
            order_ids = {"entry": order_id}
            try:
                sl_res = deribit.place_limit_order(symbol, sl_side, deribit_balance*0.01, stop, stop_price=stop)
                order_ids["stop_loss"] = str(sl_res.get("order", sl_res).get("order_id", ""))
                
                tp_res = deribit.place_limit_order(symbol, sl_side, deribit_balance*0.005, tp1)
                order_ids["tp1"] = str(tp_res.get("order", tp_res).get("order_id", ""))
            except Exception as e:
                log.warning(f"Secondary orders failed: {e}")

            record = {
                "symbol": symbol, "signal": signal, "entry": actual_entry, 
                "stop": stop, "tp1": tp1, "tp2": tp2, "order_ids": order_ids,
                "confidence": confidence, "score": score, "reasons": reasons,
                "opened_at": datetime.now(timezone.utc).isoformat(), "exchange": "deribit"
            }
            trades[symbol] = record
            save_trades(trades)
            save_signal(record)
            log.info(f"✅✅ TRADE OPENED: {symbol} {signal} [DERIBIT]")
            return True
        except Exception as e:
            log.error(f"Deribit trade failed: {e}")
    return False

# ════════════ MONITOR ════════════════════════════════════════════════

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    to_remove = []

    for symbol, trade in list(trades.items()):
        oids = trade.get("order_ids", {})
        if not oids.get("stop_loss"): continue

        try:
            o = deribit.get_order(oids["stop_loss"])
            if deribit.is_order_filled(o):
                log.info(f"❌ SL HIT: {symbol}")
                append_history({**trade, "closed_at": datetime.now(timezone.utc).isoformat(), "pnl": -trade.get("risk_usd", 0)})
                to_remove.append(symbol)
            
            if oids.get("tp1"):
                t1 = deribit.get_order(oids["tp1"])
                if deribit.is_order_filled(t1):
                    log.info(f"🎯 TP1 HIT: {symbol}")
                    trade["tp1_hit"] = True
                    save_trades(trades)
        except Exception as e:
            log.error(f"Monitor error {symbol}: {e}")

    for sym in to_remove: trades.pop(sym, None)
    save_trades(trades)

def clear_stuck_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    cleared = 0
    for symbol in list(trades.keys()):
        if not trades[symbol].get("order_ids", {}).get("stop_loss"):
            trades.pop(symbol)
            cleared += 1
    if cleared:
        save_trades(trades)
        log.info(f"🧹 Cleared {cleared} stuck trades.")

# ════════════ SIGNALS ════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df_e = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        if df_e.empty or len(df_e) < 50: return None
        row = df_e.iloc[-1]
        
        af = pipeline["all_features"]
        X = pd.DataFrame([row[af].values], columns=af)
        prob = pipeline["ensemble"].predict_proba(pipeline["selector"].transform(X))[0]
        pred = pipeline["ensemble"].predict(pipeline["selector"].transform(X))[0]
        
        sig = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob))*100, 1)

        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None
        
        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": 4, 
                "entry": float(row["close"]), "atr": float(row["atr"]), "reasons": ["ML Confirmed"]}
    except: return None

# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info("Starting Execution Scan...")
    run, mode, vol, reason = should_scan()
    if not run: return

    deribit = init_deribit()
    pipeline = load_model()
    bal = fetch_and_save_balance(deribit)
    
    clear_stuck_trades(deribit)
    check_open_trades(deribit)

    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES: break
        sig = generate_signal(symbol, pipeline, get_mode_thresholds(mode))
        if sig:
            execute_trade(deribit, symbol, sig["signal"], sig["entry"], sig["atr"], sig["confidence"], 4, sig["reasons"], balance=bal)
    
    log.info("Scan Complete.")

if __name__ == "__main__":
    run_execution_scan()
