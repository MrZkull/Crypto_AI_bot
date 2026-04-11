# trade_executor.py — FULL INTEGRATED VERSION (Detailed Logging Restored)
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
append_history = lambda rec: (lambda h: (h.append(rec), save_json(HISTORY_FILE, h)))(load_history())

def save_signal(sig):
    s = load_json(SIGNALS_FILE, [])
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])

# ════════════ BALANCE & MONITOR ══════════════════════════════════════

def fetch_and_save_balance(deribit: DeribitClient) -> float:
    try:
        total_usd = deribit.get_usdt_equivalent()
        balances  = deribit.get_all_balances()
        positions = deribit.get_positions()
        upnl = sum(float(p.get("floating_profit_loss_usd", 0) or 0) for p in positions)
        assets = [{"asset": cur, "free": round(info.get("available",0), 6), 
                   "total": round(info.get("equity_usd",0), 2)} for cur, info in balances.items()]
        save_json(BALANCE_FILE, {
            "usdt": round(total_usd, 2), "equity": round(total_usd + upnl, 2),
            "unrealised": round(upnl, 4), "assets": assets,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        })
        log.info(f"✓ Balance: ${total_usd:.2f} USD | Unrealised: {upnl:+.2f}")
        return total_usd
    except Exception as e:
        log.error(f"Balance save failed: {e}")
        return 0.0

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    to_remove = []
    for symbol, trade in list(trades.items()):
        oids = trade.get("order_ids", {})
        try:
            if not trade.get("tp1_hit") and oids.get("tp1"):
                o = deribit.get_order(oids["tp1"])
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    deribit.cancel_order(oids["stop_loss"])
                    new_sl = deribit.place_limit_order(symbol, "SELL" if trade["signal"]=="BUY" else "BUY", 
                                                     trade.get("qty_tp2", trade["qty"]), trade["entry"], stop_price=trade["entry"])
                    trade["order_ids"]["stop_loss"] = str(new_sl.get("order_id", ""))
                    trade["stop"] = trade["entry"]
                    log.info(f"🎯 TP1 HIT: {symbol}")

            live = deribit.get_live_price(symbol)
            if trade.get("tp1_hit") and trade["stop"] == trade["entry"]:
                halfway = (float(trade["entry"]) + float(trade["tp2"])) / 2
                if (trade["signal"] == "BUY" and live >= halfway) or (trade["signal"] == "SELL" and live <= halfway):
                    deribit.cancel_order(oids["stop_loss"])
                    new_sl = deribit.place_limit_order(symbol, "SELL" if trade["signal"]=="BUY" else "BUY", 
                                                     trade.get("qty_tp2", trade["qty"]), trade["tp1"], stop_price=trade["tp1"])
                    trade["order_ids"]["stop_loss"] = str(new_sl.get("order_id", ""))
                    trade["stop"] = trade["tp1"]
                    log.info(f"🚀 {symbol} Trailing Stop moved to TP1")

            for key in ["stop_loss", "tp2"]:
                if oids.get(key):
                    o = deribit.get_order(oids[key])
                    if deribit.is_order_filled(o):
                        append_history({**trade, "closed_at": datetime.now(timezone.utc).isoformat()})
                        to_remove.append(symbol)
                        break
        except Exception as e: log.error(f"Monitor error {symbol}: {e}")
    for sym in to_remove: trades.pop(sym, None)
    save_trades(trades)

# ════════════ SIGNAL GENERATION (RESTORED LOGGING) ═══════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}, timeout=15)
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
            return df
        except: continue
    raise Exception(f"Kline endpoints failed for {symbol}")

def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf >= 70:   score += 1; reasons.append(f"High conf ({conf:.0f}%)")
    elif conf >= 55: score += 1; reasons.append(f"Conf ({conf:.0f}%)")
    
    adx = float(row.get("adx",0))
    if adx > 20:   score += 1; reasons.append(f"Strong ADX {adx:.0f}")
    elif adx > 15: score += 1; reasons.append(f"ADX {adx:.0f}")

    rsi = float(row.get("rsi",50))
    if signal == "BUY" and rsi < 50:   score += 1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal == "SELL" and rsi > 50: score += 1; reasons.append(f"RSI bearish ({rsi:.0f})")

    e20, e50 = float(row.get("ema20",0)), float(row.get("ema50",0))
    if signal == "BUY" and e20 > e50:   score += 1; reasons.append("EMA bullish")
    elif signal == "SELL" and e20 < e50: score += 1; reasons.append("EMA bearish")

    c20, c50 = float(r1h.get("ema20",0)), float(r1h.get("ema50",0))
    if signal == "BUY" and c20 > c50:   score += 1; reasons.append("1h confirms")
    elif signal == "SELL" and c20 < c50: score += 1; reasons.append("1h confirms")

    if not reasons: reasons.append(f"ML {conf:.0f}%")
    return score, reasons

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df15.empty or len(df15) < 50: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        af = pipeline["all_features"]
        X = pd.DataFrame([row[af].values], columns=af)
        prob = pipeline["ensemble"].predict_proba(pipeline["selector"].transform(X))[0]
        pred = pipeline["ensemble"].predict(pipeline["selector"].transform(X))[0]
        
        sig = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx",0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"    Score: {score} (need ≥{thresholds['min_score']})")

        entry = float(row["close"])
        atr = float(row["atr"])

        if score < thresholds["min_score"]:
            log.info(f"    ❌ Rejected: Score too low")
            return None

        log.info(f"    ✅ SIGNAL: {sig} {conf:.1f}%")
        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score, 
                "entry": entry, "atr": atr, "reasons": reasons}
    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}")
        return None

# ════════════ TRADE EXECUTION ════════════════════════════════════════

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult, balance):
    trades = load_trades()
    dec = 4 if entry < 10 else 2
    side = "BUY" if signal == "BUY" else "SELL"
    amount = max(1, round((balance * 0.01 * risk_mult) / (atr * 1.5) * entry / 10) * 10)
    try:
        res = deribit.place_market_order(symbol, side, amount)
        order_id = str(res.get("order_id", res.get("order", {}).get("order_id", "")))
        time.sleep(1.2)
        actual_entry = deribit.get_fill_price(res, entry)
        
        stop = round(actual_entry - atr*ATR_STOP_MULT, dec) if signal == "BUY" else round(actual_entry + atr*ATR_STOP_MULT, dec)
        tp1  = round(actual_entry + atr*ATR_TARGET1_MULT, dec) if signal == "BUY" else round(actual_entry - atr*ATR_TARGET1_MULT, dec)
        tp2  = round(actual_entry + atr*ATR_TARGET2_MULT, dec) if signal == "BUY" else round(actual_entry - atr*ATR_TARGET2_MULT, dec)
        
        sl_res = deribit.place_limit_order(symbol, "SELL" if side == "BUY" else "BUY", amount, stop, stop_price=stop)
        tp_res = deribit.place_limit_order(symbol, "SELL" if side == "BUY" else "BUY", amount*0.5, tp1)
        
        record = {"symbol": symbol, "signal": signal, "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2, 
                  "qty": amount, "order_ids": {"entry": order_id, "stop_loss": str(sl_res.get("order_id", ""))}, 
                  "confidence": confidence, "score": score, "reasons": reasons, "opened_at": datetime.now(timezone.utc).isoformat()}
        trades[symbol] = record
        save_trades(trades)
        save_signal(record)
        log.info(f"✅✅ TRADE OPENED: {symbol}")
    except Exception as e: log.error(f"Trade failed: {e}")

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    pipeline = joblib.load(MODEL_FILE)
    bal = fetch_and_save_balance(deribit)
    run, mode, vol, reason = should_scan()
    check_open_trades(deribit)
    
    if not run: log.info(f"  SKIPPING TRADES: {reason}")
    for symbol in SYMBOLS:
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, get_mode_thresholds(mode))
        if run and sig and len(load_trades()) < MAX_OPEN_TRADES:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], get_effective_risk(mode, vol), bal)
    log.info("Scan Complete.")

if __name__ == "__main__":
    run_execution_scan()
