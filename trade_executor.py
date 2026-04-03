# trade_executor.py — FIXED for 451 geo-block
# Root cause: CCXT calls /api/v3/exchangeInfo before every request → 451 from GitHub Actions
# Fix: Replace CCXT entirely with direct HMAC-signed REST calls (binance_client.py)
# This skips exchangeInfo completely — balance, trading, monitoring all work again.

import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    LIVE_LIMIT, MODEL_FILE, LOG_FILE, get_tier, MAX_SAME_DIRECTION
)
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds,
    check_correlation, get_effective_risk
)
from binance_client import BinanceTestnet

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3
TP1_CLOSE_PCT   = 0.5
TP2_CLOSE_PCT   = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# FILE I/O
# ══════════════════════════════════════════════════════════

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path, data):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"save_json {path}: {e}")


def load_trades():  return load_json(TRADES_FILE,  {})
def save_trades(d): save_json(TRADES_FILE, d)
def load_history(): return load_json(HISTORY_FILE, [])
def load_signals(): return load_json(SIGNALS_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, sigs[-500:])


# ══════════════════════════════════════════════════════════
# EXCHANGE — Direct REST, no CCXT exchangeInfo
# ══════════════════════════════════════════════════════════

def init_exchange() -> BinanceTestnet:
    key    = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_SECRET",  "")
    if not key or not secret:
        raise ValueError("BINANCE_API_KEY or BINANCE_SECRET missing in .env / GitHub Secrets")
    ex = BinanceTestnet(api_key=key, secret=secret)
    log.info("✓ Exchange: Binance TESTNET (direct REST — no exchangeInfo)")
    return ex


def load_model():
    pipeline = joblib.load(MODEL_FILE)
    required = ["ensemble", "selector", "all_features", "best_features", "label_map"]
    missing  = [k for k in required if k not in pipeline]
    if missing:
        raise ValueError(f"Model missing keys: {missing}")
    log.info(f"✓ Model loaded — {len(pipeline['all_features'])} features")
    return pipeline


def save_balance_json(ex: BinanceTestnet):
    """
    Writes balance.json so Render dashboard can read it.
    GitHub Actions = US IP = no geo-block.
    Render = India IP = 451 on Binance. We bypass by reading the file instead.
    """
    try:
        balances = ex.get_balance()
        usdt     = balances.get("USDT", 0.0)

        assets = [
            {"asset": asset, "free": str(round(amt, 6)), "total": str(round(amt, 6))}
            for asset, amt in balances.items()
        ]

        save_json(BALANCE_FILE, {
            "usdt":       round(usdt, 4),
            "assets":     assets,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })
        log.info(f"  ✅ balance.json saved: {usdt:.2f} USDT ({len(assets)} assets)")
        return usdt
    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════
# MARKET DATA (public endpoint — never geo-blocked)
# ══════════════════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    url    = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def calc_pos_size(balance: float, entry: float, stop: float,
                  risk_mult: float = 1.0):
    risk_usd = balance * RISK_PER_TRADE * risk_mult
    dist     = abs(entry - stop)
    if dist <= 0:
        return 0.0, 0.0
    qty = risk_usd / dist
    # Safety cap: never more than 20% of balance per trade
    max_usd = balance * 0.20
    if qty * entry > max_usd:
        qty = max_usd / entry
    return round(qty, 6), round(risk_usd, 2)


# ══════════════════════════════════════════════════════════
# TRADE EXECUTION
# ══════════════════════════════════════════════════════════

def execute_trade(ex: BinanceTestnet, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already open — skip")
        return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info(f"  Max trades ({MAX_OPEN_TRADES}) reached — skip")
        return False
    if not check_correlation(trades, signal):
        return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop    = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry + atr * ATR_TARGET2_MULT, dec)
        side    = "BUY";  sl_side = "SELL"; tp_side = "SELL"
    else:
        stop    = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry - atr * ATR_TARGET2_MULT, dec)
        side    = "SELL"; sl_side = "BUY";  tp_side = "BUY"

    try:
        balance = ex.get_usdt_balance()
    except Exception as e:
        log.error(f"  Balance fetch failed: {e}")
        return False

    if balance < 10:
        _warn(f"⚠️ Balance too low ({balance:.2f} USDT)")
        return False

    qty, risk_usd = calc_pos_size(balance, entry, stop, risk_mult)
    if qty <= 0:
        log.warning(f"  Zero position size for {symbol}")
        return False

    qty_tp1   = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2   = round(qty * TP2_CLOSE_PCT, 6)
    order_ids = {}

    log.info(f"  Placing {signal} {symbol} qty={qty} risk={risk_usd:.2f} USDT")

    try:
        # Market entry
        entry_order  = ex.place_market_order(symbol, side, qty)
        order_ids["entry"] = str(entry_order.get("orderId", ""))
        actual_entry = float(entry_order.get("fills", [{}])[0].get("price", entry) or entry)
        if actual_entry == 0:
            actual_entry = float(entry_order.get("price", entry) or entry)
        log.info(f"  ✅ Entry filled @ {actual_entry:.{dec}f}")
        time.sleep(1.5)

        # Stop loss
        try:
            sl = ex.place_limit_order(symbol, sl_side, qty, stop, stop_price=stop)
            order_ids["stop_loss"] = str(sl.get("orderId", ""))
            log.info(f"  ✅ SL placed @ {stop}")
        except Exception as e:
            log.warning(f"  SL failed: {e}")
            # Fallback to plain limit
            try:
                sl = ex.place_limit_order(symbol, sl_side, qty, stop)
                order_ids["stop_loss"] = str(sl.get("orderId", ""))
                log.info(f"  ✅ SL (limit) @ {stop}")
            except Exception as e2:
                log.warning(f"  SL limit fallback also failed: {e2}")

        # TP1
        try:
            t1 = ex.place_limit_order(symbol, tp_side, qty_tp1, tp1)
            order_ids["tp1"] = str(t1.get("orderId", ""))
            log.info(f"  ✅ TP1 placed @ {tp1}")
        except Exception as e:
            log.warning(f"  TP1 failed: {e}")

        # TP2
        try:
            t2 = ex.place_limit_order(symbol, tp_side, qty_tp2, tp2)
            order_ids["tp2"] = str(t2.get("orderId", ""))
            log.info(f"  ✅ TP2 placed @ {tp2}")
        except Exception as e:
            log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Order error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance,
        "risk_mult": risk_mult, "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol),
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}")
    return True


# ══════════════════════════════════════════════════════════
# MONITORING & RECOVERY
# ══════════════════════════════════════════════════════════

def clean_ghost_orders(ex: BinanceTestnet):
    """Cancel orphaned limit orders to free locked balance."""
    trades    = load_trades()
    valid_ids = set()
    for t in trades.values():
        valid_ids.update(str(v) for v in t.get("order_ids", {}).values())
    try:
        open_orders = ex.get_open_orders()
        cancelled   = 0
        for o in open_orders:
            if str(o.get("orderId")) not in valid_ids:
                try:
                    ex.cancel_order(o["symbol"], o["orderId"])
                    cancelled += 1
                except Exception: pass
        if cancelled:
            log.info(f"  ✅ Cancelled {cancelled} ghost orders — balance freed")
    except Exception as e:
        log.warning(f"  Ghost sweep failed (safe): {e}")


def sync_trade_history(ex: BinanceTestnet):
    """Rebuild history from Binance only when local file is empty."""
    if len(load_history()) > 0:
        return
    log.info("  🔄 Rebuilding history from Binance...")
    rebuilt = []
    try:
        for sym in SYMBOLS:
            try:
                orders = ex.get_closed_orders(sym, limit=10)
            except Exception: continue
            filled   = [o for o in orders if o.get("status") == "FILLED"]
            entries  = [o for o in filled if o.get("type") == "MARKET"]
            exits    = [o for o in filled if o.get("type") != "MARKET"]
            for entry in entries:
                xo = next((x for x in exits if x["time"] > entry["time"]), None)
                if not xo: continue
                qty = float(entry.get("executedQty") or 0)
                ep  = float(entry.get("price") or 0)
                xp  = float(xo.get("price") or 0)
                if qty == 0 or ep == 0: continue
                is_buy = entry.get("side") == "BUY"
                pnl    = (xp-ep)*qty if is_buy else (ep-xp)*qty
                rebuilt.append({
                    "symbol": sym, "signal": "BUY" if is_buy else "SELL",
                    "entry": ep, "close_price": xp, "pnl": round(pnl, 4),
                    "opened_at": datetime.fromtimestamp(entry["time"]/1000, tz=timezone.utc).isoformat(),
                    "closed_at": datetime.fromtimestamp(xo["time"]/1000, tz=timezone.utc).isoformat(),
                    "close_reason": "Binance Sync",
                })
        if rebuilt:
            rebuilt.sort(key=lambda x: x["closed_at"])
            save_json(HISTORY_FILE, rebuilt)
            log.info(f"  ✅ Rebuilt {len(rebuilt)} trades")
    except Exception as e:
        log.warning(f"  History sync failed: {e}")


def auto_recover_trades(ex: BinanceTestnet):
    """Reconstruct open trades from Binance open orders."""
    trades = load_trades()
    try:
        open_orders    = ex.get_open_orders()
        active_symbols = list(set(o["symbol"] for o in open_orders))
        recovered      = 0
        for sym in active_symbols:
            if sym not in trades:
                sym_orders = [o for o in open_orders if o["symbol"] == sym]
                total_qty  = sum(float(o.get("origQty", 0)) for o in sym_orders)
                oids       = {}
                for o in sym_orders:
                    ot = o.get("type","")
                    if ot == "STOP_LOSS_LIMIT" or "STOP" in ot:
                        oids["stop_loss"] = str(o["orderId"])
                    elif "tp1" not in oids:
                        oids["tp1"] = str(o["orderId"])
                    else:
                        oids["tp2"] = str(o["orderId"])
                trades[sym] = {
                    "symbol": sym, "signal": "RECOVERED",
                    "entry": float(sym_orders[0].get("price", 0)),
                    "stop": 0, "tp1": 0, "tp2": 0,
                    "qty": total_qty, "qty_tp1": total_qty/2, "qty_tp2": total_qty/2,
                    "risk_usd": 0, "balance_at_open": 0, "risk_mult": 1.0,
                    "order_ids": oids, "tp1_hit": False, "tp2_hit": False, "closed": False,
                    "confidence": 0, "score": 0, "reasons": ["🔄 Auto-recovered"],
                    "tier": get_tier(sym),
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                }
                recovered += 1
        if recovered:
            save_trades(trades)
            log.info(f"  ✅ Recovered {recovered} orphaned trades")
    except Exception as e:
        log.warning(f"  Auto-recover failed (safe): {e}")


def check_open_trades(ex: BinanceTestnet):
    trades = load_trades()
    if not trades:
        log.info("  No open trades")
        return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue
        try:
            oids  = trade.get("order_ids", {})
            entry = float(trade["entry"])

            # Check TP1
            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.get_order(symbol, oids["tp1"])
                    if o.get("status") == "FILLED":
                        trade["tp1_hit"] = True
                        avg  = float(o.get("price") or trade["tp1"])
                        pnl  = _calc_pnl(trade, avg, "tp1")
                        log.info(f"  🎯 TP1 HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, avg, trade["opened_at"])
                        # Move SL to breakeven
                        if "stop_loss" in oids:
                            try:
                                ex.cancel_order(symbol, oids["stop_loss"])
                                sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                                new_sl  = ex.place_limit_order(
                                    symbol, sl_side, trade["qty_tp2"], entry, stop_price=entry)
                                trade["order_ids"]["stop_loss"] = str(new_sl.get("orderId",""))
                                trade["stop"] = entry
                                _send(f"🛡️ *{symbol}* SL moved to entry — risk-free now!")
                            except Exception as be:
                                log.warning(f"  Breakeven SL failed: {be}")
                except Exception as e:
                    log.warning(f"  TP1 check {symbol}: {e}")

            # Check TP2
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o = ex.get_order(symbol, oids["tp2"])
                    if o.get("status") == "FILLED":
                        trade["tp2_hit"] = True; trade["closed"] = True
                        avg = float(o.get("price") or trade["tp2"])
                        pnl = _calc_pnl(trade, avg, "tp2")
                        log.info(f"  ✅ TP2 HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, avg, trade["opened_at"])
                        _record_close(trade, avg, pnl, "TP2 hit")
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  TP2 check {symbol}: {e}")

            # Check SL
            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o = ex.get_order(symbol, oids["stop_loss"])
                    if o.get("status") == "FILLED":
                        trade["closed"] = True
                        avg = float(o.get("price") or trade["stop"])
                        pnl = _calc_pnl(trade, avg, "sl")
                        log.info(f"  ❌ SL HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, avg, trade["opened_at"])
                        _record_close(trade, avg, pnl, "SL hit")
                        _cancel_remaining(ex, symbol, oids, trade)
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  SL check {symbol}: {e}")

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove):
        trades.pop(sym, None)
    save_trades(trades)


def _calc_pnl(trade, close_price, close_type):
    qty = (trade["qty_tp1"] if close_type=="tp1" else
           trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    if trade["signal"] == "BUY":
        return round((close_price - trade["entry"]) * qty, 4)
    return round((trade["entry"] - close_price) * qty, 4)


def _cancel_remaining(ex: BinanceTestnet, symbol, oids, trade):
    for key in ("tp1", "tp2"):
        if key in oids and not trade.get(f"{key}_hit"):
            try: ex.cancel_order(symbol, oids[key])
            except Exception: pass


def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade, "close_price": close_price, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason,
    })


# ══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df15.empty or len(df15) < 50:
            return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"]   = float(r1h.get("rsi",   50))
        row["adx_1h"]   = float(r1h.get("adx",    0))
        row["trend_1h"] = float(r1h.get("trend",  0))

        all_feat = pipeline["all_features"]
        missing  = [f for f in all_feat if f not in row.index]
        if missing:
            log.warning(f"    Missing features: {missing[:5]}")
            return None

        X_raw  = pd.DataFrame([row[all_feat].values], columns=all_feat)
        X_sel  = pipeline["selector"].transform(X_raw)
        pred   = pipeline["ensemble"].predict(X_sel)[0]
        prob   = pipeline["ensemble"].predict_proba(X_sel)[0]
        signal = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf   = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {signal} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if signal == "NO_TRADE" or conf < thresholds["min_confidence"]:
            return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]:
            return None

        score, reasons = _quality_score(row, r1h, signal, conf)
        log.info(f"    Score: {score}/6 (need ≥{thresholds['min_score']})")

        entry = float(row["close"])
        atr   = float(row["atr"])
        dec   = 4 if entry < 10 else 2

        if signal == "BUY":
            stop = round(entry - atr*ATR_STOP_MULT, dec)
            tp1  = round(entry + atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry + atr*ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr*ATR_STOP_MULT, dec)
            tp1  = round(entry - atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry - atr*ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                         "entry":entry,"atr":atr,"reasons":reasons,"rejected":True,
                         "reject_reason":f"score {score}<{thresholds['min_score']}",
                         "stop":stop,"tp1":tp1,"tp2":tp2})
            return None

        return {"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except requests.exceptions.HTTPError as e:
        log.warning(f"    HTTP error {symbol}: {e}")
        return None
    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}")
        return None


def _quality_score(row, row_1h, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75:   score+=1; reasons.append(f"High AI conf ({confidence:.0f}%)")
    elif confidence >= 65: score+=1; reasons.append(f"Good AI conf ({confidence:.0f}%)")
    elif confidence >= 60: reasons.append(f"AI conf ({confidence:.0f}%)")

    adx = float(row.get("adx", 0))
    if adx > 25:   score+=1; reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 20: score+=1; reasons.append(f"Moderate ADX {adx:.0f}")

    rsi = float(row.get("rsi", 50))
    if signal=="BUY"  and rsi < 45: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi > 55: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

    e20 = float(row.get("ema20",0)); e50 = float(row.get("ema50",0))
    if signal=="BUY"  and e20 > e50: score+=1; reasons.append("EMA20>EMA50 uptrend")
    elif signal=="SELL" and e20 < e50: score+=1; reasons.append("EMA20<EMA50 downtrend")

    c20 = float(row_1h.get("ema20",0)); c50 = float(row_1h.get("ema50",0))
    if signal=="BUY"  and c20 > c50: score+=1; reasons.append("1h uptrend confirmed")
    elif signal=="SELL" and c20 < c50: score+=1; reasons.append("1h downtrend confirmed")

    return score, reasons


# ══════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════

def _send(text):
    token   = os.getenv("TELEGRAM_TOKEN",   "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id":chat_id,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {
            "active":  "📈 *Active hours* — conf≥65% | score≥3 | full risk",
            "quiet":   "🌙 *Quiet hours* — conf≥72% | score≥4 | 50% risk",
            "weekend": "📅 *Weekend mode* — conf≥65% | score≥3 | 75% risk",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult=1.0):
    emoji   = "🟢" if signal=="BUY" else "🔴"
    stars   = "⭐"*min(score,5)
    dec     = 4 if entry<10 else 2
    sl_pct  = abs((stop-entry)/entry*100)
    t1_pct  = abs((tp1-entry)/entry*100)
    risk_n  = "" if risk_mult>=1.0 else f"\n⚡ Risk reduced to {int(risk_mult*100)}%"
    rlines  = "\n".join([f"  • {r}" for r in reasons])
    _send(
        f"🤖 *TESTNET TRADE OPENED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ {get_tier(symbol)}{risk_n}\n"
        f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}/6*\n\n"
        f"⚡ *ENTRY:* `{entry:.{dec}f}`\n"
        f"🛑 *STOP:* `{stop:.{dec}f}` (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:* `{tp1:.{dec}f}` (+{t1_pct:.1f}%)\n"
        f"💰 Pos: `{qty*entry:.2f} USDT` · Risk: `{risk_usd:.2f} USDT`\n\n"
        f"📊 *Reasons:*\n{rlines}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_Binance Testnet_"
    )

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji = "✅" if pnl>0 else "❌"
    dec   = 4 if entry<10 else 2
    try:
        dur = str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur = "—"
    _send(f"🤖 *TRADE CLOSED*\n{emoji} *{result} — {symbol}*\n"
          f"📥 `{entry:.{dec}f}` → 📤 `{close_price:.{dec}f}`\n"
          f"💵 *PnL: `{pnl:+.4f} USDT`* · ⏱️ {dur}\n_Binance Testnet_")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"  Scan SKIPPED: {reason}")
        return

    effective_risk = get_effective_risk(mode, vol)
    vol_warning    = vol["message"] if vol.get("warn") else None

    log.info(f"  Mode:{mode['label']} conf≥{mode['min_confidence']}% "
             f"score≥{mode['min_score']} ADX≥{mode['min_adx']} "
             f"risk:{effective_risk:.2f}")

    exchange   = init_exchange()
    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    # 1. Save balance FIRST — dashboard reads this
    save_balance_json(exchange)

    # 2. Recover lost state
    auto_recover_trades(exchange)
    sync_trade_history(exchange)

    # 3. Free locked balance
    clean_ghost_orders(exchange)

    # 4. Check open trades for TP/SL
    log.info(f"\n[1/2] Checking open trades...")
    check_open_trades(exchange)

    # 5. Scan for new signals
    trades = load_trades()
    log.info(f"\n[2/2] Scanning {len(SYMBOLS)} symbols | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"      conf≥{thresholds['min_confidence']}% score≥{thresholds['min_score']} ADX≥{thresholds['min_adx']}")

    signals_found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  🛑 Max trades reached — scan complete")
            break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.3)
            continue

        signals_found += 1
        log.info(f"  ✅ SIGNAL: {sig['signal']} {sig['confidence']:.1f}% score={sig['score']}")

        if vol_warning:
            sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warning}"]

        execute_trade(exchange, sig["symbol"], sig["signal"], sig["entry"],
                      sig["atr"], sig["confidence"], sig["score"],
                      sig["reasons"], effective_risk)
        time.sleep(1)

    # 6. Save balance again at end
    save_balance_json(exchange)

    log.info(f"\n{'═'*56}")
    log.info(f"SCAN DONE — {signals_found} signal(s) found")
    log.info(f"{'═'*56}\n")


def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    lines = ["🔍 *Bot Diagnostic*",
             f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             "━━━━━━━━━━━━━━━━━━━━"]
    try:
        mode = get_scan_mode(); vol = check_btc_volatility()
        eff  = get_effective_risk(mode, vol)
        lines += [f"📋 Mode: *{mode['label']}*",
                  f"📊 BTC ATR: *{vol['atr_pct']:.2f}%* ({vol['status']})",
                  f"⚙️ Conf≥{mode['min_confidence']}% | Score≥{mode['min_score']}/6 | ADX≥{mode['min_adx']}",
                  f"⚡ Effective risk: *{int(eff*100)}%*"]
    except Exception as e: lines.append(f"❌ Scheduler: {e}")
    try:
        ex  = init_exchange()
        bal = ex.get_usdt_balance()
        lines += [f"💰 Balance: *{bal:.2f} USDT*", f"📂 Open: *{len(load_trades())}*",
                  f"✅ Direct REST connection works — no 451 error!"]
    except Exception as e: lines.append(f"❌ Exchange: {e}")
    try:
        p = load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")
    real = [h for h in load_history() if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]
    wr   = round(len(wins)/len(real)*100,1) if real else 0
    tpnl = sum(h.get("pnl",0) for h in real)
    lines += [f"📈 Win rate: *{wr}%* ({len(wins)}W/{len(real)-len(wins)}L)",
              f"💵 Real PnL: *{tpnl:+.4f} USDT*"]
    _send("\n".join(lines))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    else:
        run_execution_scan()
