# trade_executor.py
import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from persistence import load_json as remote_load_json, save_json as remote_save_json

try:
    import ccxt
except ImportError as e:
    raise ImportError("Run: pip install ccxt") from e

from config import (
    SYMBOLS,
    ATR_STOP_MULT,
    ATR_TARGET1_MULT,
    ATR_TARGET2_MULT,
    RISK_PER_TRADE,
    TIMEFRAME_ENTRY,
    TIMEFRAME_CONFIRM,
    LIVE_LIMIT,
    MODEL_FILE,
    LOG_FILE,
    get_tier,
)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk

load_dotenv(dotenv_path=".env", override=True)

TRADES_FILE = "trades.json"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
MODE_FILE = "scan_mode.json"
BALANCE_FILE = "balance.json"

MAX_OPEN_TRADES = 3
TP1_CLOSE_PCT = 0.5
TP2_CLOSE_PCT = 0.5

# Paper fallback
PAPER_MODE_ON_GEOBLOCK = True
PAPER_START_USDT = 10000.0
PAPER_MODE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_json_local_or_remote(path, default):
    try:
        if Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    try:
        return remote_load_json(path, default)
    except Exception:
        return default


def save_json_local_and_remote(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    try:
        remote_save_json(path, data)
    except Exception:
        pass


def load_trades():
    return load_json_local_or_remote(TRADES_FILE, {})


def save_trades(d):
    save_json_local_and_remote(TRADES_FILE, d)


def load_history():
    return load_json_local_or_remote(HISTORY_FILE, [])


def load_signals():
    return load_json_local_or_remote(SIGNALS_FILE, [])


def append_history(rec):
    h = load_history()
    h.append(rec)
    save_json_local_and_remote(HISTORY_FILE, h)


def save_signal(sig):
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json_local_and_remote(SIGNALS_FILE, sigs[-500:])


def _send(text):
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


def _send_open_alert(symbol, signal, confidence, score, entry, stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult=1.0):
    mode_lbl = "PAPER" if PAPER_MODE else "TESTNET"
    emoji = "🟢" if signal == "BUY" else "🔴"
    stars = "⭐" * min(score, 5)
    dec = 4 if entry < 10 else 2
    fp = lambda v: f"{v:,.{dec}f}"
    reasons_txt = "\n".join([f"  • {r}" for r in reasons])
    risk_note = "" if risk_mult >= 1.0 else f"\n⚡ *Risk reduced to {int(risk_mult*100)}%*"
    _send(
        f"🤖 *{mode_lbl} TRADE OPENED*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n🏷️ Tier: _{get_tier(symbol)}_{risk_note}\n"
        f"🎯 Confidence: *{confidence:.1f}%* · Score: *{score}/6*\n\n"
        f"⚡ *ENTRY:* `{fp(entry)}`\n🛑 *STOP LOSS:* `{fp(stop)}`\n"
        f"🎯 *TARGET 1:* `{fp(tp1)}`\n🎯 *TARGET 2:* `{fp(tp2)}`\n\n"
        f"💰 *Position:* `{round(qty*entry,2)} USDT`\n⚠️ *Risk:* `{risk_usd:.2f} USDT`\n"
        f"💼 *Balance:* `{balance:.2f} USDT`\n\n📊 *Reasons:*\n{reasons_txt}"
    )


def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji = "✅" if pnl > 0 else "❌"
    dec = 4 if entry < 10 else 2
    try:
        dur = str(datetime.now(timezone.utc) - datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception:
        dur = "—"
    _send(
        f"🤖 *TRADE CLOSED*\n━━━━━━━━━━━━━━━━━━━━\n\n{emoji} *{result} — {symbol}*\n\n"
        f"📥 Entry: `{entry:.{dec}f}`\n📤 Close: `{close_price:.{dec}f}`\n"
        f"💵 *PnL:* `{pnl:+.4f} USDT`\n⏱️ Duration: {dur}"
    )


def init_exchange():
    key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_SECRET", "")
    if not key or not secret:
        raise ValueError("BINANCE_API_KEY or BINANCE_SECRET missing")

    ex = ccxt.binance({"apiKey": key, "secret": secret, "options": {"defaultType": "spot"}, "enableRateLimit": True})
    ex.set_sandbox_mode(True)
    ex.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    log.info("✓ Exchange: Binance TESTNET")
    return ex


def load_model():
    return joblib.load(MODEL_FILE)


def get_balance_usdt(ex):
    global PAPER_MODE

    if PAPER_MODE:
        bal = load_json_local_or_remote(BALANCE_FILE, {})
        usdt_total = float(bal.get("usdt_total", PAPER_START_USDT) or PAPER_START_USDT)
        usdt_free = float(bal.get("usdt_free", usdt_total) or usdt_total)
        data = {
            "usdt": usdt_total,
            "usdt_free": usdt_free,
            "usdt_used": max(0.0, usdt_total - usdt_free),
            "usdt_total": usdt_total,
            "assets": bal.get("assets") or [{"asset": "USDT", "free": usdt_free, "used": usdt_total - usdt_free, "total": usdt_total}],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "mode": "paper",
        }
        save_json_local_and_remote(BALANCE_FILE, data)
        return usdt_free

    try:
        b = ex.fetch_balance()
        usdt_node = b.get("USDT", {}) if isinstance(b.get("USDT", {}), dict) else {}
        usdt_free = float(usdt_node.get("free", 0) or 0)
        usdt_used = float(usdt_node.get("used", 0) or 0)
        usdt_total = float(usdt_node.get("total", usdt_free + usdt_used) or (usdt_free + usdt_used))

        totals = b.get("total", {}) if isinstance(b.get("total", {}), dict) else {}
        frees = b.get("free", {}) if isinstance(b.get("free", {}), dict) else {}
        useds = b.get("used", {}) if isinstance(b.get("used", {}), dict) else {}
        assets = []
        for a, t in totals.items():
            try:
                t = float(t or 0)
                if t <= 0:
                    continue
                assets.append({"asset": a, "free": float(frees.get(a, 0) or 0), "used": float(useds.get(a, 0) or 0), "total": t})
            except Exception:
                continue

        save_json_local_and_remote(BALANCE_FILE, {
            "usdt": usdt_total,
            "usdt_free": usdt_free,
            "usdt_used": usdt_used,
            "usdt_total": usdt_total,
            "assets": assets,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "mode": "live",
        })
        return usdt_free

    except Exception as e:
        err = str(e)
        log.error(f"Balance fetch failed: {err}")

        if PAPER_MODE_ON_GEOBLOCK and "451" in err:
            PAPER_MODE = True
            log.warning("Geo restriction detected (451). Switching to PAPER MODE.")
            save_json_local_and_remote(BALANCE_FILE, {
                "usdt": PAPER_START_USDT,
                "usdt_free": PAPER_START_USDT,
                "usdt_used": 0.0,
                "usdt_total": PAPER_START_USDT,
                "assets": [{"asset": "USDT", "free": PAPER_START_USDT, "used": 0.0, "total": PAPER_START_USDT}],
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
                "mode": "paper",
                "fallback_reason": err,
            })
            return PAPER_START_USDT

        save_json_local_and_remote(BALANCE_FILE, {
            "usdt": 0.0,
            "usdt_free": 0.0,
            "usdt_used": 0.0,
            "usdt_total": 0.0,
            "assets": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": err,
            "mode": "live",
        })
        return 0.0


def get_data(symbol, interval):
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def calc_pos_size(balance, entry, stop, risk_mult=1.0):
    risk_usd = balance * RISK_PER_TRADE * risk_mult
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0, 0.0
    qty = risk_usd / dist
    max_usd = balance * 0.20
    if qty * entry > max_usd:
        qty = max_usd / entry
    return round(qty, 6), round(risk_usd, 2)


def _pnl(trade, close_price, close_type):
    entry = trade["entry"]
    qty = trade["qty_tp1"] if close_type == "tp1" else trade["qty_tp2"] if close_type == "tp2" else trade["qty"]
    return round((close_price - entry) * qty if trade["signal"] == "BUY" else (entry - close_price) * qty, 4)


def _record_close(trade, close_price, pnl, reason):
    append_history({**trade, "close_price": close_price, "pnl": pnl, "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})


def _public_price(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=8)
        if r.ok:
            return float(r.json().get("price"))
    except Exception:
        pass
    return None


def execute_trade(ex, symbol, signal, entry, atr, confidence, score, reasons, risk_mult=1.0):
    trades = load_trades()
    if symbol in trades or len(trades) >= MAX_OPEN_TRADES:
        return False
    if not check_correlation(trades, signal):
        return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr * ATR_STOP_MULT, dec)
        tp1 = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2 = round(entry + atr * ATR_TARGET2_MULT, dec)
        side = "buy"
    else:
        stop = round(entry + atr * ATR_STOP_MULT, dec)
        tp1 = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2 = round(entry - atr * ATR_TARGET2_MULT, dec)
        side = "sell"

    balance = get_balance_usdt(ex)
    if balance < 10:
        return False

    qty, risk_usd = calc_pos_size(balance, entry, stop, risk_mult)
    if qty <= 0:
        return False

    qty_tp1 = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2 = round(qty * TP2_CLOSE_PCT, 6)
    log.info(f"  Placing {signal} {symbol} | qty={qty} | risk={risk_usd:.2f} USDT")

    try:
        if PAPER_MODE:
            actual_entry = entry
            ts = int(time.time() * 1000)
            order_ids = {"entry": f"PAPER-ENTRY-{ts}", "stop_loss": f"PAPER-SL-{ts}", "tp1": f"PAPER-TP1-{ts}", "tp2": f"PAPER-TP2-{ts}"}
        else:
            sl_side = "sell" if side == "buy" else "buy"
            entry_order = ex.create_order(symbol, "market", side, qty)
            actual_entry = float(entry_order.get("average", entry) or entry)
            order_ids = {"entry": entry_order.get("id")}
            time.sleep(1.2)

            try:
                sl = ex.create_order(symbol, "stop_loss_limit", sl_side, qty, stop, params={"stopPrice": stop, "timeInForce": "GTC"})
            except Exception:
                sl = ex.create_order(symbol, "limit", sl_side, qty, stop, params={"stopPrice": stop, "timeInForce": "GTC"})
            order_ids["stop_loss"] = sl.get("id")

            for k, px, q in [("tp1", tp1, qty_tp1), ("tp2", tp2, qty_tp2)]:
                try:
                    tpo = ex.create_order(symbol, "take_profit_limit", sl_side, q, px, params={"stopPrice": px, "timeInForce": "GTC"})
                except Exception:
                    tpo = ex.create_order(symbol, "limit", sl_side, q, px, params={"stopPrice": px, "timeInForce": "GTC"})
                order_ids[k] = tpo.get("id")

    except Exception as e:
        log.error(f"  Trade error: {e}")
        return False

    record = {
        "symbol": symbol,
        "signal": signal,
        "entry": actual_entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "qty": qty,
        "qty_tp1": qty_tp1,
        "qty_tp2": qty_tp2,
        "risk_usd": risk_usd,
        "balance_at_open": balance,
        "risk_mult": risk_mult,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False,
        "tp2_hit": False,
        "closed": False,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "tier": get_tier(symbol),
        "mode": "paper" if PAPER_MODE else "live",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry, stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅ TRADE OPENED: {symbol} {signal} ({'PAPER' if PAPER_MODE else 'LIVE'})")
    return True


def check_open_trades(ex):
    trades = load_trades()
    if not trades:
        return
    to_remove = []

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        live = None
        if PAPER_MODE:
            live = _public_price(symbol)
        else:
            try:
                live = float((ex.fetch_ticker(symbol) or {}).get("last"))
            except Exception:
                live = None

        if not live:
            continue

        entry = trade["entry"]
        if not trade.get("tp1_hit") and ((trade["signal"] == "BUY" and live >= trade["tp1"]) or (trade["signal"] == "SELL" and live <= trade["tp1"])):
            trade["tp1_hit"] = True
            pnl = _pnl(trade, live, "tp1")
            _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, live, trade["opened_at"])

        if trade.get("tp1_hit") and not trade.get("tp2_hit") and ((trade["signal"] == "BUY" and live >= trade["tp2"]) or (trade["signal"] == "SELL" and live <= trade["tp2"])):
            trade["tp2_hit"] = True
            trade["closed"] = True
            pnl = _pnl(trade, live, "tp2")
            _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, live, trade["opened_at"])
            _record_close(trade, live, pnl, "TP2 hit" + (" (paper)" if PAPER_MODE else ""))
            to_remove.append(symbol)
            continue

        if not trade.get("closed") and ((trade["signal"] == "BUY" and live <= trade["stop"]) or (trade["signal"] == "SELL" and live >= trade["stop"])):
            trade["closed"] = True
            pnl = _pnl(trade, live, "sl")
            _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, live, trade["opened_at"])
            _record_close(trade, live, pnl, "SL hit" + (" (paper)" if PAPER_MODE else ""))
            to_remove.append(symbol)

    save_trades(trades)
    for sym in set(to_remove):
        trades.pop(sym, None)
    save_trades(trades)


def auto_recover_trades(ex):
    if PAPER_MODE or ex is None:
        return
    try:
        trades = load_trades()
        open_orders = ex.fetch_open_orders()
        active = sorted(set(o.get("symbol") for o in open_orders if o.get("symbol")))
        for sym in active:
            if sym in trades:
                continue
            sym_orders = [o for o in open_orders if o.get("symbol") == sym]
            if not sym_orders:
                continue
            qty = sum(float(o.get("amount") or 0) for o in sym_orders)
            trades[sym] = {
                "symbol": sym,
                "signal": "RECOVERED",
                "entry": float(sym_orders[0].get("average") or sym_orders[0].get("price") or 0),
                "stop": 0,
                "tp1": 0,
                "tp2": 0,
                "qty": qty,
                "qty_tp1": qty / 2,
                "qty_tp2": qty / 2,
                "risk_usd": 0,
                "balance_at_open": 0,
                "risk_mult": 1.0,
                "order_ids": {},
                "tp1_hit": False,
                "tp2_hit": False,
                "closed": False,
                "confidence": 100,
                "score": 6,
                "reasons": ["🔄 Recovered by Auto-Sync"],
                "tier": get_tier(sym),
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
        save_trades(trades)
    except Exception:
        pass


def sync_trade_history(ex):
    if PAPER_MODE or ex is None or load_history():
        return
    # keep lightweight; no rebuild when paper mode


def _quality_score(row_entry, row_confirm, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75:
        score += 1
        reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        score += 1
        reasons.append(f"Good AI confidence ({confidence:.0f}%)")
    elif confidence >= 60:
        reasons.append(f"AI confidence ({confidence:.0f}%)")

    adx = float(row_entry.get("adx", 0))
    if adx > 25:
        score += 1
        reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 20:
        score += 1
        reasons.append(f"Moderate trend ADX {adx:.0f}")

    rsi = float(row_entry.get("rsi", 50))
    if signal == "BUY" and rsi < 45:
        score += 1
        reasons.append(f"RSI bullish zone ({rsi:.0f})")
    elif signal == "SELL" and rsi > 55:
        score += 1
        reasons.append(f"RSI bearish zone ({rsi:.0f})")

    e20 = float(row_entry.get("ema20", 0))
    e50 = float(row_entry.get("ema50", 0))
    if signal == "BUY" and e20 > e50:
        score += 1
        reasons.append("EMA20 > EMA50 uptrend")
    elif signal == "SELL" and e20 < e50:
        score += 1
        reasons.append("EMA20 < EMA50 downtrend")

    c20 = float(row_confirm.get("ema20", 0))
    c50 = float(row_confirm.get("ema50", 0))
    if signal == "BUY" and c20 > c50:
        score += 1
        reasons.append("1h confirms uptrend")
    elif signal == "SELL" and c20 < c50:
        score += 1
        reasons.append("1h confirms downtrend")

    return score, reasons


def generate_signal(symbol, pipeline, thresholds):
    try:
        df_entry = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df_entry.empty or len(df_entry) < 50:
            return None

        row_entry = df_entry.iloc[-1].copy()
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)
        row_entry["rsi_1h"] = float(row_confirm.get("rsi", 50))
        row_entry["adx_1h"] = float(row_confirm.get("adx", 0))
        row_entry["trend_1h"] = float(row_confirm.get("trend", 0))

        all_feat = pipeline["all_features"]
        selector = pipeline["selector"]
        ensemble = pipeline["ensemble"]

        X_raw = pd.DataFrame([row_entry[all_feat].values], columns=all_feat)
        pred = ensemble.predict(selector.transform(X_raw))[0]
        prob = ensemble.predict_proba(selector.transform(X_raw))[0]
        signal = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pred]
        confidence = round(float(max(prob)) * 100, 1)

        if signal == "NO_TRADE" or confidence < thresholds["min_confidence"]:
            return None
        if float(row_entry.get("adx", 0)) < thresholds["min_adx"]:
            return None

        score, reasons = _quality_score(row_entry, row_confirm, signal, confidence)
        entry = float(row_entry["close"])
        atr = float(row_entry["atr"])
        dec = 4 if entry < 10 else 2
        stop = round(entry - atr * ATR_STOP_MULT, dec) if signal == "BUY" else round(entry + atr * ATR_STOP_MULT, dec)
        tp1 = round(entry + atr * ATR_TARGET1_MULT, dec) if signal == "BUY" else round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2 = round(entry + atr * ATR_TARGET2_MULT, dec) if signal == "BUY" else round(entry - atr * ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol": symbol, "signal": signal, "confidence": confidence, "score": score, "entry": entry, "atr": atr, "reasons": reasons, "rejected": True, "reject_reason": f"score {score} < {thresholds['min_score']}", "stop": stop, "tp1": tp1, "tp2": tp2})
            return None

        return {"symbol": symbol, "signal": signal, "confidence": confidence, "score": score, "entry": entry, "atr": atr, "stop": stop, "tp1": tp1, "tp2": tp2, "reasons": reasons}

    except Exception:
        return None


def check_mode_switch(mode):
    last = load_json_local_or_remote(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {
            "active": "📈 *Active hours* — conf ≥60% | score ≥2 | full risk",
            "quiet": "🌙 *Quiet hours* — conf ≥68% | score ≥3 | 75% risk",
            "weekend": "📅 *Weekend mode* — conf ≥65% | score ≥2 | 75% risk",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json_local_and_remote(MODE_FILE, {"mode": mode["mode"], "since": datetime.now(timezone.utc).isoformat()})


def run_execution_scan():
    global PAPER_MODE

    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    exchange = None
    try:
        exchange = init_exchange()
    except Exception as e:
        if PAPER_MODE_ON_GEOBLOCK:
            PAPER_MODE = True
            log.warning(f"Exchange init failed. Switching to paper mode: {e}")
        else:
            log.error(f"Exchange init failed: {e}")
            return

    bal = get_balance_usdt(exchange)
    bal_state = load_json_local_or_remote(BALANCE_FILE, {})
    if bal_state.get("error") and not PAPER_MODE:
        log.error(f"Balance fetch error: {bal_state.get('error')}")
        return

    if PAPER_MODE:
        log.warning("Running in PAPER MODE.")
    if bal <= 0:
        log.warning("Balance is zero — no new trades can be opened.")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"Scan SKIPPED: {reason}")
        return

    effective_risk = get_effective_risk(mode, vol)
    thresholds = get_mode_thresholds(mode)
    pipeline = load_model()

    auto_recover_trades(exchange)
    sync_trade_history(exchange)
    check_open_trades(exchange)

    signals_found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            break
        sig = generate_signal(symbol, pipeline, thresholds)
        if not sig:
            time.sleep(0.3)
            continue
        signals_found += 1
        if vol.get("warn"):
            sig["reasons"] = list(sig.get("reasons", [])) + [f"⚠️ {vol.get('message', '')}"]
        execute_trade(exchange, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], effective_risk)
        time.sleep(0.6)

    log.info(f"\n{'═'*56}\nSCAN DONE — {signals_found} signal(s) found\n{'═'*56}\n")


if __name__ == "__main__":
    run_execution_scan()
