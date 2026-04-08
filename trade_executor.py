# trade_executor.py — Delta Exchange India Testnet version
# Real balance, real orders, real SL/TP — legal in India, no geo-block
# Signals from public Binance data (never blocked anywhere)

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
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk
from delta_client import DeltaClient

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3
TP1_CLOSE_PCT   = 0.5

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
            with open(path) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e: log.error(f"save_json {path}: {e}")

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

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"✓ Model: {len(p['all_features'])} features")
    return p


# ══════════════════════════════════════════════════════════
# EXCHANGE INIT
# ══════════════════════════════════════════════════════════

def init_exchange() -> DeltaClient:
    key    = os.getenv("DELTA_API_KEY",    "")
    secret = os.getenv("DELTA_API_SECRET", "")
    if not key or not secret:
        raise ValueError(
            "DELTA_API_KEY or DELTA_API_SECRET not set!\n"
            "Add to GitHub Actions secrets AND Render environment variables."
        )
    ex = DeltaClient(api_key=key, api_secret=secret)
    if not ex.test_connection():
        raise ConnectionError("Delta Exchange testnet connection failed")
    return ex


def save_balance_json(ex: DeltaClient):
    """Writes balance.json so Render dashboard can display real balance."""
    try:
        balances = ex.get_wallet_balance()
        usdt     = balances.get("USDT", 0.0)
        positions = ex.get_positions()
        upnl     = sum(float(p.get("unrealized_pnl", 0) or 0) for p in positions)

        assets = [{"asset": k, "free": str(round(v, 4)), "total": str(round(v, 4))}
                  for k, v in balances.items() if v > 0]

        save_json(BALANCE_FILE, {
            "usdt":        round(usdt, 4),
            "equity":      round(usdt + upnl, 4),
            "unrealised":  round(upnl, 4),
            "assets":      assets,
            "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "exchange":    "Delta Exchange India Testnet",
            "open_positions": len(positions),
        })
        log.info(f"  💰 Balance: {usdt:.2f} USDT | PnL: {upnl:+.2f} | Equity: {usdt+upnl:.2f}")
        return usdt
    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════
# MARKET DATA (public Binance — never geo-blocked)
# ══════════════════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    url    = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
    return df


# ══════════════════════════════════════════════════════════
# TRADE EXECUTION — real Delta Exchange orders
# ══════════════════════════════════════════════════════════

def execute_trade(ex: DeltaClient, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already open"); return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info(f"  Max trades reached"); return False
    if not check_correlation(trades, signal): return False

    # Check Delta supports this coin
    try: ex.get_product(symbol)
    except ValueError as e:
        log.warning(f"  {symbol}: {e} — skipping"); return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop  = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry + atr * ATR_TARGET2_MULT, dec)
        side  = "buy";  sl_side = "sell"; tp_side = "sell"
    else:
        stop  = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry - atr * ATR_TARGET2_MULT, dec)
        side  = "sell"; sl_side = "buy";  tp_side = "buy"

    try: balance = ex.get_usdt_balance()
    except Exception as e:
        log.error(f"  Balance failed: {e}"); return False

    if balance < 5:
        _warn(f"⚠️ Balance very low ({balance:.2f} USDT) — click 'Reload Wallet' on Delta testnet")
        return False

    contracts = ex.calc_contracts(balance, entry, stop, risk_mult)
    half_c    = max(1, contracts // 2)
    rest_c    = contracts - half_c
    risk_usd  = round(balance * RISK_PER_TRADE * risk_mult, 2)
    order_ids = {}

    log.info(f"  {signal} {symbol} | {contracts} contracts | risk≈{risk_usd:.2f} USDT")
    log.info(f"  Entry:{entry:.{dec}f} SL:{stop:.{dec}f} TP1:{tp1:.{dec}f} TP2:{tp2:.{dec}f}")

    try:
        # Market entry
        eo = ex.place_market_order(symbol, side, contracts)
        order_ids["entry"] = str(eo.get("id",""))
        # Best actual fill price
        actual_entry = float(eo.get("average_fill_price") or
                             eo.get("limit_price") or entry)
        if actual_entry == 0: actual_entry = entry
        log.info(f"  ✅ Filled @ {actual_entry:.{dec}f}")
        time.sleep(1.0)

        # Stop Loss
        try:
            sl = ex.place_limit_order(symbol, sl_side, contracts, stop, stop_price=stop)
            order_ids["stop_loss"] = str(sl.get("id",""))
        except Exception as e: log.warning(f"  SL failed: {e}")

        # TP1 (half)
        try:
            t1 = ex.place_limit_order(symbol, tp_side, half_c, tp1)
            order_ids["tp1"] = str(t1.get("id",""))
        except Exception as e: log.warning(f"  TP1 failed: {e}")

        # TP2 (rest)
        if rest_c > 0:
            try:
                t2 = ex.place_limit_order(symbol, tp_side, rest_c, tp2)
                order_ids["tp2"] = str(t2.get("id",""))
            except Exception as e: log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Order error {symbol}: {e}"); return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": contracts, "qty_tp1": half_c, "qty_tp2": rest_c,
        "risk_usd": risk_usd, "balance_at_open": balance,
        "risk_mult": risk_mult, "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol), "exchange": "delta_testnet",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, contracts, risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} {contracts} contracts")
    return True


# ══════════════════════════════════════════════════════════
# MONITORING — check real Delta order status
# ══════════════════════════════════════════════════════════

def check_open_trades(ex: DeltaClient):
    trades = load_trades()
    if not trades:
        log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol); continue
        try:
            oids  = trade.get("order_ids", {})
            entry = float(trade["entry"])

            # ── TP1 ──────────────────────────────────────
            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.get_order(oids["tp1"])
                    st = o.get("state", o.get("status",""))
                    if st in ("closed","filled","cancelled") and o.get("size_filled", o.get("filled_size",0)):
                        trade["tp1_hit"] = True
                        fill = float(o.get("average_fill_price") or o.get("limit_price") or trade["tp1"])
                        pnl  = _pnl(trade, fill, "tp1")
                        log.info(f"  🎯 TP1 HIT {symbol} pnl≈{pnl:+.4f} USDT")
                        _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, fill, trade["opened_at"])
                        # Move SL to breakeven (risk-free)
                        if "stop_loss" in oids:
                            try:
                                ex.cancel_order(symbol, oids["stop_loss"])
                                sl_side = "sell" if trade["signal"]=="BUY" else "buy"
                                new_sl  = ex.place_limit_order(
                                    symbol, sl_side, trade["qty_tp2"], entry, stop_price=entry)
                                trade["order_ids"]["stop_loss"] = str(new_sl.get("id",""))
                                trade["stop"] = entry
                                _send(f"🛡️ *{symbol}* SL moved to breakeven — risk-free!")
                            except Exception as be: log.warning(f"  Breakeven SL: {be}")
                except Exception as e: log.warning(f"  TP1 {symbol}: {e}")

            # ── TP2 ──────────────────────────────────────
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o  = ex.get_order(oids["tp2"])
                    st = o.get("state", o.get("status",""))
                    if st in ("closed","filled") and o.get("size_filled", o.get("filled_size",0)):
                        trade["tp2_hit"] = True; trade["closed"] = True
                        fill = float(o.get("average_fill_price") or o.get("limit_price") or trade["tp2"])
                        pnl  = _pnl(trade, fill, "tp2")
                        log.info(f"  ✅ TP2 HIT {symbol} pnl≈{pnl:+.4f} USDT")
                        _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, fill, trade["opened_at"])
                        _record_close(trade, fill, pnl, "TP2 hit")
                        to_remove.append(symbol)
                except Exception as e: log.warning(f"  TP2 {symbol}: {e}")

            # ── SL ───────────────────────────────────────
            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o  = ex.get_order(oids["stop_loss"])
                    st = o.get("state", o.get("status",""))
                    if st in ("closed","filled") and o.get("size_filled", o.get("filled_size",0)):
                        trade["closed"] = True
                        fill = float(o.get("average_fill_price") or o.get("limit_price") or trade["stop"])
                        pnl  = _pnl(trade, fill, "sl")
                        log.info(f"  ❌ SL HIT {symbol} pnl≈{pnl:+.4f} USDT")
                        _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, fill, trade["opened_at"])
                        _record_close(trade, fill, pnl, "SL hit")
                        for key in ("tp1","tp2"):
                            if key in oids and not trade.get(f"{key}_hit"):
                                try: ex.cancel_order(symbol, oids[key])
                                except Exception: pass
                        to_remove.append(symbol)
                except Exception as e: log.warning(f"  SL {symbol}: {e}")

        except Exception as e: log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def _pnl(trade, close_px, close_type):
    """Estimate PnL in USDT from Delta futures position."""
    qty = trade["qty_tp1"] if close_type=="tp1" else \
          trade["qty_tp2"] if close_type=="tp2" else trade["qty"]
    diff = (close_px - trade["entry"]) if trade["signal"]=="BUY" \
           else (trade["entry"] - close_px)
    # Delta contract value varies — use $0.001 BTC/contract approximation
    # Actual PnL visible on Delta Exchange dashboard in real time
    return round(diff * qty * 0.001, 4)


def _record_close(trade, close_px, pnl, reason):
    append_history({**trade, "close_price":close_px, "pnl":pnl,
                    "closed_at":datetime.now(timezone.utc).isoformat(), "close_reason":reason})


# ══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════

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

        all_feat = pipeline["all_features"]
        missing  = [f for f in all_feat if f not in row.index]
        if missing: log.warning(f"    Missing: {missing[:3]}"); return None

        X_raw  = pd.DataFrame([row[all_feat].values], columns=all_feat)
        X_sel  = pipeline["selector"].transform(X_raw)
        pred   = pipeline["ensemble"].predict(X_sel)[0]
        prob   = pipeline["ensemble"].predict_proba(X_sel)[0]
        signal = {0:"BUY",1:"SELL",2:"NO_TRADE"}[pred]
        conf   = round(float(max(prob))*100, 1)

        log.info(f"    ML: {signal} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if signal=="NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx",0))
        log.info(f"    ADX:{adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, signal, conf)
        log.info(f"    Score:{score}/5 (need ≥{thresholds['min_score']})")

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2

        if signal=="BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec); tp1=round(entry+atr*ATR_TARGET1_MULT,dec); tp2=round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec); tp1=round(entry-atr*ATR_TARGET1_MULT,dec); tp2=round(entry-atr*ATR_TARGET2_MULT,dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                         "entry":entry,"atr":atr,"reasons":reasons,"rejected":True,
                         "reject_reason":f"score {score}<{thresholds['min_score']}",
                         "stop":stop,"tp1":tp1,"tp2":tp2})
            return None

        return {"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}"); return None


def _quality_score(row, r1h, signal, confidence):
    score, reasons = 0, []
    if confidence>=70:   score+=1; reasons.append(f"High AI conf ({confidence:.0f}%)")
    elif confidence>=60: score+=1; reasons.append(f"Good AI conf ({confidence:.0f}%)")
    elif confidence>=55: reasons.append(f"AI conf ({confidence:.0f}%)")

    adx=float(row.get("adx",0))
    if adx>25: score+=1; reasons.append(f"Strong ADX {adx:.0f}")
    elif adx>18: score+=1; reasons.append(f"Moderate ADX {adx:.0f}")

    rsi=float(row.get("rsi",50))
    if signal=="BUY"  and rsi<50: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

    e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
    if signal=="BUY"  and e20>e50: score+=1; reasons.append("EMA trend: UP")
    elif signal=="SELL" and e20<e50: score+=1; reasons.append("EMA trend: DOWN")

    c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
    if signal=="BUY"  and c20>c50: score+=1; reasons.append("1h confirms UP")
    elif signal=="SELL" and c20<c50: score+=1; reasons.append("1h confirms DOWN")
    return score, reasons


# ══════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════

def _send(text):
    t=os.getenv("TELEGRAM_TOKEN",""); c=os.getenv("TELEGRAM_CHAT_ID","")
    if not t or not c: return
    try: requests.post(f"https://api.telegram.org/bot{t}/sendMessage",
         data={"chat_id":c,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode):
    last=load_json(MODE_FILE,{})
    if last.get("mode")!=mode["mode"]:
        msgs={"active":"📈 *Active hours* — conf≥58% | lower threshold","quiet":"🌙 *Quiet hours* — conf≥65%","weekend":"📅 *Weekend*"}
        _send(msgs.get(mode["mode"],"Mode changed"))
        save_json(MODE_FILE,{"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, contracts, risk_usd, balance, reasons, risk_mult=1.0):
    emoji="🟢" if signal=="BUY" else "🔴"; stars="⭐"*min(score,5); dec=4 if entry<10 else 2
    sl_pct=abs((stop-entry)/entry*100); t1_pct=abs((tp1-entry)/entry*100); t2_pct=abs((tp2-entry)/entry*100)
    risk_n="" if risk_mult>=1.0 else f"\n⚡ Risk {int(risk_mult*100)}% (volatility)"
    rlines="\n".join([f"  • {r}" for r in reasons])
    _send(
        f"🤖 *DELTA TESTNET TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ {get_tier(symbol)}{risk_n}\n"
        f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}/5*\n\n"
        f"⚡ *ENTRY:* `{entry:.{dec}f}`\n"
        f"🛑 *STOP LOSS:* `{stop:.{dec}f}` (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:* `{tp1:.{dec}f}` (+{t1_pct:.1f}%)\n"
        f"🎯 *TP2:* `{tp2:.{dec}f}` (+{t2_pct:.1f}%)\n"
        f"📦 Contracts: `{contracts}` · Risk≈`{risk_usd:.2f} USDT`\n"
        f"💼 Balance: `{balance:.2f} USDT`\n\n"
        f"📊 *Reasons:*\n{rlines}\n━━━━━━━━━━━━━━━━━━━━\n"
        f"_Delta Exchange India Testnet_"
    )

def _send_close_alert(symbol, result, pnl, entry, fill, opened_at):
    emoji="✅" if pnl>0 else "❌"; dec=4 if entry<10 else 2
    try: dur=str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur="—"
    _send(f"🤖 *DELTA TRADE CLOSED*\n{emoji} *{result} — {symbol}*\n"
          f"📥 `{entry:.{dec}f}` → 📤 `{fill:.{dec}f}`\n"
          f"💵 *PnL≈ `{pnl:+.4f} USDT`* · ⏱️ {dur}\n_Delta Testnet_")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"Exchange: Delta Exchange India Testnet")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"  SKIPPED: {reason}"); return

    effective_risk = get_effective_risk(mode, vol)
    vol_warning    = vol["message"] if vol.get("warn") else None

    log.info(f"  Mode:{mode['label']} conf≥{mode['min_confidence']}% "
             f"score≥{mode['min_score']} ADX≥{mode['min_adx']} risk:{effective_risk:.2f}")

    exchange   = init_exchange()
    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    # Step 1: Real balance from Delta
    log.info(f"\n[0] Fetching real balance from Delta testnet...")
    bal = save_balance_json(exchange)
    if bal < 5:
        _warn("⚠️ Balance very low! Click 'Reload Wallet' on testnet.delta.exchange")

    # Step 2: Check open trades for TP/SL hits
    log.info(f"\n[1] Checking open trades...")
    check_open_trades(exchange)
    save_balance_json(exchange)   # refresh after any closes

    # Step 3: Scan for new signals
    trades = load_trades()
    log.info(f"\n[2] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    conf≥{thresholds['min_confidence']}% | score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']}")

    signals_found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  🛑 Max trades reached"); break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.3); continue

        signals_found += 1
        log.info(f"  ✅ SIGNAL: {sig['signal']} {sig['confidence']:.1f}% score={sig['score']}")

        if vol_warning:
            sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warning}"]

        execute_trade(exchange, sig["symbol"], sig["signal"], sig["entry"],
                      sig["atr"], sig["confidence"], sig["score"],
                      sig["reasons"], effective_risk)
        time.sleep(0.5)

    save_balance_json(exchange)   # final balance save

    log.info(f"\n{'═'*56}")
    log.info(f"SCAN DONE — {signals_found} signal(s) | trades:{len(load_trades())}/{MAX_OPEN_TRADES}")
    log.info(f"{'═'*56}\n")


def run_diagnostic():
    log.info("Running diagnostic...")
    try:
        ex  = init_exchange()
        bal = ex.get_usdt_balance()
        pos = ex.get_positions()
        products = list(ex._products.keys())
        msg = (f"🔍 *Delta Testnet Diagnostic*\n"
               f"💰 Balance: *{bal:.2f} USDT*\n"
               f"📂 Open positions: {len(pos)}\n"
               f"📦 Supported coins: {len(products)}\n"
               f"✅ Connection: OK — no geo-block!")
        log.info(msg)
        _send(msg)
    except Exception as e:
        msg = f"❌ Diagnostic failed: {e}"
        log.error(msg); _send(msg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    else:
        run_execution_scan()
