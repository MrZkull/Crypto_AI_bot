# trade_executor.py — Merged Stable Version (Precision + Trailing Stop)
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
from smart_scheduler import (
    should_scan, get_mode_thresholds, check_correlation, get_effective_risk
)
from deribit_client import DeribitClient, TRADEABLE

TRADES_FILE      = "trades.json"
HISTORY_FILE     = "trade_history.json"
SIGNALS_FILE     = "signals.json"
MODE_FILE        = "scan_mode.json"
BALANCE_FILE     = "balance.json"
MAX_OPEN_TRADES  = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ════════════ FILE I/O ════════════════════════════════════════════════

def load_json(path, default):
    try:
        for p in [Path(path), Path("data") / Path(path).name]:
            if p.exists():
                with open(p) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp,"w") as f: json.dump(data,f,indent=2,default=str)
        os.replace(tmp, path)
    except Exception as e: log.error(f"save_json {path}: {e}")
    try:
        d = Path("data"); d.mkdir(exist_ok=True)
        tmp = str(d / Path(path).name) + ".tmp"
        with open(tmp,"w") as f: json.dump(data,f,indent=2,default=str)
        os.replace(tmp, str(d / Path(path).name))
    except Exception: pass

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])
load_signals = lambda: load_json(SIGNALS_FILE, [])
append_history = lambda rec: (lambda h: (h.append(rec), save_json(HISTORY_FILE, h)))(load_history())

def save_signal(sig):
    s = load_signals()
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"✓ Model: {len(p['all_features'])} features | 73.1% accuracy")
    return p

# ════════════ EXCHANGE INIT ═══════════════════════════════════════════

def init_deribit() -> DeribitClient:
    cid    = os.getenv("DERIBIT_CLIENT_ID",     "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret: raise ValueError("DERIBIT_CLIENT keys not set!")
    return DeribitClient(cid, secret)

def fetch_and_save_balance(deribit: DeribitClient) -> float:
    try:
        total_usd = deribit.get_usdt_equivalent()
        balances  = deribit.get_all_balances()
        positions = deribit.get_positions()
        
        upnl = sum(float(p.get("floating_profit_loss_usd", 0) or 0) for p in positions)
        assets = [{"asset": cur, "free": round(info.get("available",0), 6), 
                   "total": round(info.get("equity_usd",0), 2)} for cur, info in balances.items()]
        
        save_json(BALANCE_FILE, {
            "usdt": round(total_usd, 2), 
            "equity": round(total_usd + upnl, 2),
            "unrealised": round(upnl, 4), 
            "assets": assets,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "deribit_testnet"
        })
        log.info(f"✓ Balance: ${total_usd:.2f} USD | Unrealised: {upnl:+.2f}")
        return total_usd
    except Exception as e:
        log.error(f"Balance save failed: {e}")
        return 0.0

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
    raise Exception(f"Kline endpoints failed for {symbol}")

# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult=1.0, balance=10000.0):
    trades = load_trades()
    if symbol in trades: return False
    if len(trades) >= MAX_OPEN_TRADES: return False
    if not check_correlation(trades, signal): return False
    if not deribit.is_supported(symbol): return False

    dec = 4 if entry < 10 else 2
    side = "BUY" if signal == "BUY" else "SELL"
    sl_side = "SELL" if side == "BUY" else "BUY"

    # Contract calculation logic from yesterday's working file
    amount = max(10, round((balance * 0.01 * risk_mult) / (atr * 1.5) * entry / 10) * 10)
    amount_tp1 = max(10, round(amount * 0.5 / 10) * 10)
    amount_tp2 = max(10, round(amount * 0.5 / 10) * 10)
    risk_usd   = round(balance * RISK_PER_TRADE * risk_mult, 2)

    try:
        # 1. PLACE MARKET ENTRY
        res = deribit.place_market_order(symbol, side, amount)
        if not res: return False
        
        order_ids = {"entry": str(res.get("order", res).get("order_id", ""))}
        
        # 2. GET REAL FILL PRICE & RECALC TARGETS
        time.sleep(1.2)
        actual_entry = deribit.get_fill_price(res, entry)
        
        if signal == "BUY":
            stop = round(actual_entry - atr * ATR_STOP_MULT, dec)
            tp1  = round(actual_entry + atr * ATR_TARGET1_MULT, dec)
            tp2  = round(actual_entry + atr * ATR_TARGET2_MULT, dec)
        else:
            stop = round(actual_entry + atr * ATR_STOP_MULT, dec)
            tp1  = round(actual_entry - atr * ATR_TARGET1_MULT, dec)
            tp2  = round(actual_entry - atr * ATR_TARGET2_MULT, dec)

        # 3. PLACE SL & TP
        try:
            sl_res = deribit.place_limit_order(symbol, sl_side, amount, stop, stop_price=stop)
            order_ids["stop_loss"] = str(sl_res.get("order", sl_res).get("order_id", ""))
        except Exception as e: log.warning(f"SL failed: {e}")

        try:
            t1_res = deribit.place_limit_order(symbol, sl_side, amount_tp1, tp1)
            order_ids["tp1"] = str(t1_res.get("order", t1_res).get("order_id", ""))
        except Exception as e: log.warning(f"TP1 failed: {e}")

        try:
            t2_res = deribit.place_limit_order(symbol, sl_side, amount_tp2, tp2)
            order_ids["tp2"] = str(t2_res.get("order", t2_res).get("order_id", ""))
        except Exception as e: log.warning(f"TP2 failed: {e}")

        record = {
            "symbol": symbol, "signal": signal, "entry": actual_entry, 
            "stop": stop, "tp1": tp1, "tp2": tp2, "qty": amount,
            "qty_tp1": amount_tp1, "qty_tp2": amount_tp2,
            "risk_usd": risk_usd, "order_ids": order_ids,
            "confidence": confidence, "score": score, "reasons": reasons,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "tp1_hit": False, "tp2_hit": False, "closed": False
        }
        trades[symbol] = record
        save_trades(trades)
        save_signal(record)
        log.info(f"✅✅ TRADE OPENED: {symbol} {signal} [DERIBIT]")
        return True
    except Exception as e:
        log.error(f"Live trade failed: {e}")
        return False

# ════════════ MONITOR ════════════════════════════════════════════════

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    to_remove = []

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue
        
        oids = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec = 4 if entry < 10 else 2

        try:
            # ── TP1 Logic ──────────────────────────────────────────
            if not trade["tp1_hit"] and "tp1" in oids:
                o = deribit.get_order(oids["tp1"])
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    log.info(f"🎯 TP1 HIT: {symbol}")
                    # Move SL to Breakeven
                    if "stop_loss" in oids:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"] == "BUY" else "BUY"
                            be_res = deribit.place_limit_order(symbol, sl_side, trade["qty_tp2"], entry, stop_price=entry)
                            trade["order_ids"]["stop_loss"] = str(be_res.get("order", be_res).get("order_id", ""))
                            trade["stop"] = entry
                            _send(f"🛡️ *{symbol}* SL moved to breakeven @ {entry}")
                        except: pass

            # ── TRAILING STOP (Yesterday's merged fix) ─────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and "stop_loss" in oids:
                live = deribit.get_live_price(symbol)
                halfway = (entry + trade["tp2"]) / 2
                if ((trade["signal"] == "BUY" and live >= halfway) or (trade["signal"] == "SELL" and live <= halfway)) and trade["stop"] == entry:
                    try:
                        deribit.cancel_order(oids["stop_loss"])
                        sl_side = "SELL" if trade["signal"] == "BUY" else "BUY"
                        trail_res = deribit.place_limit_order(symbol, sl_side, trade["qty_tp2"], trade["tp1"], stop_price=trade["tp1"])
                        trade["order_ids"]["stop_loss"] = str(trail_res.get("order", trail_res).get("order_id", ""))
                        trade["stop"] = trade["tp1"]
                        _send(f"🚀 *{symbol}* Trailing Stop moved to TP1 @ {trade['tp1']}")
                    except: pass

            # ── SL/TP2 Final Check ─────────────────────────────────
            for key in ["stop_loss", "tp2"]:
                if key in oids:
                    o = deribit.get_order(oids[key])
                    if deribit.is_order_filled(o):
                        trade["closed"] = True
                        log.info(f"🏁 CLOSED ({key}): {symbol}")
                        _record_close(trade, deribit.get_fill_price(o, trade[key.replace("_loss","")]), 0.0, f"{key} hit")
                        to_remove.append(symbol)
                        break

        except Exception as e: log.error(f"Monitor error {symbol}: {e}")

    for sym in to_remove: trades.pop(sym, None)
    save_trades(trades)

def clear_stuck_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    cleared = 0
    for symbol in list(trades.keys()):
        oids = trades[symbol].get("order_ids", {})
        if not oids.get("stop_loss") or oids.get("stop_loss") == "None":
            trades.pop(symbol); cleared += 1
    if cleared: save_trades(trades)

def _record_close(trade, px, pnl, reason):
    h = load_history()
    h.append({**trade, "close_price": px, "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})
    save_json(HISTORY_FILE, h)

# ════════════ SIGNALS ════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        if df.empty or len(df) < 50: return None
        row = df.iloc[-1]
        
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
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    
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
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], 
                          sig["confidence"], 4, sig["reasons"], risk_mult=get_effective_risk(mode, vol), balance=bal)
    
    log.info("Scan Complete.")

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if tok and cid: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", 
                                  data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=10)

if __name__ == "__main__":
    run_execution_scan()
