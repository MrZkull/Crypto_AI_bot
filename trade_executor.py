# trade_executor.py — All critical bugs fixed
#
# BUG FIXES:
# 1. MAX_OPEN_TRADES enforced BEFORE signal scan — hard stop at 3
# 2. Full record saved (stop/tp1/tp2/order_ids/confidence/score) — dashboard shows values
# 3. check_open_trades() called every scan — TP1/SL detection works
# 4. After TP1 hit → SL moves to entry (breakeven) — trade becomes risk-free
# 5. Ghost trade cleaner checks live Deribit positions
# 6. TP2 placed as plain limit (not stop-limit) — correct order type

import os, json, time, logging, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, MODEL_FILE, LOG_FILE, get_tier,
    TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, LIVE_LIMIT
)
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, get_effective_risk

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3   # HARD LIMIT — never open more than 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)


# ════════ FILE I/O ════════════════════════════════════════════════════

def load_json(path, default):
    try:
        for p in [Path(path), Path("data") / Path(path).name]:
            if p.exists():
                with open(p) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    for dest in [Path(path), Path("data") / Path(path).name]:
        try:
            dest.parent.mkdir(exist_ok=True)
            tmp = str(dest) + ".tmp"
            with open(tmp,"w") as f: json.dump(data, f, indent=2, default=str)
            os.replace(tmp, str(dest))
        except Exception as e: log.error(f"save_json {dest}: {e}")

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    s = load_json(SIGNALS_FILE, [])
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])


# ════════ BALANCE ════════════════════════════════════════════════════

def save_balance(deribit: DeribitClient) -> float:
    try:
        bals      = deribit.get_all_balances()
        total_usd = deribit.get_total_equity_usd()
        positions = deribit.get_positions()
        upnl      = sum(float(p.get("floating_profit_loss_usd") or
                              p.get("floating_profit_loss") or 0) for p in positions)
        assets    = [{"asset": c, "free": str(round(i["available"],6)),
                      "total": str(round(i["equity_usd"],2))} for c,i in bals.items()]
        save_json(BALANCE_FILE, {
            "usdt": round(total_usd,2), "equity": round(total_usd+upnl,2),
            "unrealised": round(upnl,4), "assets": assets,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "deribit_testnet", "exchange": "Deribit(by Coinbase) Testnet",
            "open_positions": len(positions),
        })
        log.info(f"  Balance: ${total_usd:.2f} | upnl:{upnl:+.2f} | positions:{len(positions)}")
        return total_usd
    except Exception as e:
        log.error(f"  save_balance: {e}"); return 0.0


# ════════ MARKET DATA ════════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":LIVE_LIMIT}, timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:,:6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception: continue
    return pd.DataFrame()


# ════════ SIGNAL GENERATION ══════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY)).fillna(0)
        df1h_raw = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(df1h_raw).fillna(0) if not df1h_raw.empty else pd.DataFrame()
        if df15.empty or len(df15) < 30: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(0, index=df15.columns)
        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        af    = pipeline["all_features"]
        X     = pd.DataFrame([row[af].values], columns=af).replace([np.inf,-np.inf],0).fillna(0)
        Xs    = pipeline["selector"].transform(X)
        pred  = pipeline["ensemble"].predict(Xs)[0]
        prob  = pipeline["ensemble"].predict_proba(Xs)[0]
        sig   = {0:"BUY",1:"SELL",2:"NO_TRADE"}[pred]
        conf  = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        # Quality score
        score = 0
        if conf >= 65: score += 1
        if adx > 20:   score += 1
        rsi = float(row.get("rsi", 50))
        if (sig=="BUY" and rsi < 55) or (sig=="SELL" and rsi > 45): score += 1

        log.info(f"    Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: return None

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec); tp1=round(entry+atr*ATR_TARGET1_MULT,dec); tp2=round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec); tp1=round(entry-atr*ATR_TARGET1_MULT,dec); tp2=round(entry-atr*ATR_TARGET2_MULT,dec)

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2}
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}"); return None


# ════════ EXECUTE TRADE ══════════════════════════════════════════════

def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float) -> bool:
    symbol = sig["symbol"]; signal = sig["signal"]
    entry  = sig["entry"];  atr    = sig["atr"]
    stop   = sig["stop"];   tp1    = sig["tp1"]; tp2 = sig["tp2"]

    trades = load_trades()
    # HARD LIMIT — never exceed MAX_OPEN_TRADES
    if len([t for t in trades.values() if not t.get("closed",False)]) >= MAX_OPEN_TRADES:
        log.info(f"  🛑 MAX TRADES REACHED ({MAX_OPEN_TRADES}) — skipping {symbol}")
        return False
    if symbol in trades:
        log.info(f"  {symbol}: already open — skip"); return False
    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit — skip"); return False

    dec     = 4 if entry < 10 else 2
    side    = "BUY"  if signal == "BUY" else "SELL"
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"

    total_q          = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
    risk_usd         = round(balance * RISK_PER_TRADE * risk_mult, 2)

    log.info(f"  {signal} {symbol} total={total_q} tp1={qty_tp1} tp2={qty_tp2}")
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids    = {}
    actual_entry = entry

    try:
        # 1. Market entry
        entry_res   = deribit.place_market_order(symbol, side, total_q)
        entry_order = entry_res.get("order", entry_res)
        order_ids["entry"] = str(entry_order.get("order_id",""))
        actual_entry = deribit.get_fill_price(entry_res, entry) or entry

        state  = entry_order.get("order_state","")
        filled = float(entry_order.get("filled_amount",0) or 0)
        if state == "cancelled" and filled == 0:
            log.warning(f"  Market order cancelled (thin book) — skip {symbol}")
            return False

        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}")
        time.sleep(1.5)

        # Recalc levels from actual fill price
        if signal == "BUY":
            stop = deribit.round_price(symbol, actual_entry - atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET2_MULT)
        else:
            stop = deribit.round_price(symbol, actual_entry + atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET2_MULT)

        # 2. Stop Loss — stop_limit with trigger_price
        try:
            sl_res = deribit.place_limit_order(symbol, sl_side, total_q, stop, stop_price=stop)
            sl_o   = sl_res.get("order", sl_res)
            oid    = str(sl_o.get("order_id",""))
            if oid: order_ids["stop_loss"] = oid
            log.info(f"  ✅ SL @ {stop:.{dec}f} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  SL failed: {e}")

        # 3. TP1 — plain limit (reduce_only)
        try:
            if qty_tp1 > 0:
                tp1_res = deribit.place_limit_order(symbol, tp_side, qty_tp1, tp1)
                tp1_o   = tp1_res.get("order", tp1_res)
                oid     = str(tp1_o.get("order_id",""))
                if oid: order_ids["tp1"] = oid
                log.info(f"  ✅ TP1 @ {tp1:.{dec}f} × {qty_tp1} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  TP1 failed: {e}")

        # 4. TP2 — plain limit (reduce_only)
        try:
            if qty_tp2 > 0:
                tp2_res = deribit.place_limit_order(symbol, tp_side, qty_tp2, tp2)
                tp2_o   = tp2_res.get("order", tp2_res)
                oid     = str(tp2_o.get("order_id",""))
                if oid: order_ids["tp2"] = oid
                log.info(f"  ✅ TP2 @ {tp2:.{dec}f} × {qty_tp2} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        _send(f"⚠️ Trade error {symbol}: {e}")
        return False

    # FULL record — dashboard reads stop/tp1/tp2/confidence/score from here
    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": total_q, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance, "risk_mult": risk_mult,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": sig["confidence"],   # dashboard AI%
        "score":      sig["score"],         # dashboard Score
        "tier":       get_tier(symbol),
        "exchange":   "deribit_testnet",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal({**record, "type": "executed"})

    _send_open_alert(symbol, signal, sig["confidence"], sig["score"],
                     actual_entry, stop, tp1, tp2,
                     total_q, qty_tp1, qty_tp2, risk_usd, balance)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} | SL:{order_ids.get('stop_loss','!')} "
             f"TP1:{order_ids.get('tp1','!')} TP2:{order_ids.get('tp2','!')}")
    return True


# ════════ MONITOR OPEN TRADES ════════════════════════════════════════

def _fp(o: dict, fb: float) -> float:
    p = float(o.get("average_price") or o.get("price") or 0)
    return p if p > 0 else fb

def _pnl(trade, close_price, close_type) -> float:
    qty  = float(trade["qty_tp1"] if close_type=="tp1" else
                 trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    diff = (close_price-trade["entry"]) if trade["signal"]=="BUY" else (trade["entry"]-close_price)
    return round(diff * qty, 4)

def _record_close(trade, close_price, pnl, reason):
    append_history({**trade,"close_price":close_price,"pnl":pnl,
                    "closed_at":datetime.now(timezone.utc).isoformat(),"close_reason":reason})

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue
        oids  = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec   = 4 if entry < 10 else 2

        def get_o(key):
            if key not in oids or not oids[key] or str(oids[key]) in ("","None"): return {}
            return deribit.get_order(str(oids[key]))

        try:
            # ── TP1 ─────────────────────────────────────────────────
            if not trade.get("tp1_hit") and "tp1" in oids:
                o = get_o("tp1")
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    fill = _fp(o, trade["tp1"])
                    pnl  = _pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"🎯 *TP1 HIT — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")

                    # Move SL to entry (breakeven) — trade now risk-free
                    if oids.get("stop_loss") and trade.get("qty_tp2", 0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                            qty_rem = trade["qty_tp2"]
                            be_res  = deribit.place_limit_order(symbol, sl_side, qty_rem,
                                                                entry, stop_price=entry)
                            be_o    = be_res.get("order", be_res)
                            new_id  = str(be_o.get("order_id",""))
                            if new_id:
                                trade["order_ids"]["stop_loss"] = new_id
                                trade["stop"] = entry
                                log.info(f"  🛡️ SL → breakeven {entry:.{dec}f} — risk-free!")
                                _send(f"🛡️ *{symbol} RISK-FREE*\nSL moved to entry `{entry:.{dec}f}`\nBest: TP2 profit | Worst: break even")
                        except Exception as be:
                            log.warning(f"  Breakeven SL: {be}")

            # ── Trailing stop (price halfway to TP2 after TP1 hit)
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                live = deribit.get_live_price(symbol)
                if live > 0:
                    halfway  = (entry + float(trade["tp2"])) / 2
                    at_trail = ((trade["signal"]=="BUY" and live >= halfway) or
                                (trade["signal"]=="SELL" and live <= halfway))
                    sl_is_be = abs(float(trade.get("stop",0)) - entry) < entry*0.001
                    if at_trail and sl_is_be and trade.get("qty_tp2",0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                            sl_res  = deribit.place_limit_order(symbol, sl_side, trade["qty_tp2"],
                                                                trade["tp1"], stop_price=trade["tp1"])
                            sl_o = sl_res.get("order",sl_res)
                            new_id = str(sl_o.get("order_id",""))
                            if new_id:
                                trade["order_ids"]["stop_loss"] = new_id
                                trade["stop"] = trade["tp1"]
                                log.info(f"  🚀 {symbol} trailing SL → TP1 {trade['tp1']:.{dec}f}")
                                _send(f"🚀 *{symbol}* Trailing SL → TP1 `{trade['tp1']:.{dec}f}` — profit locked!")
                        except Exception as e:
                            log.warning(f"  Trailing SL: {e}")

            # ── TP2 ─────────────────────────────────────────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and "tp2" in oids:
                o = get_o("tp2")
                if deribit.is_order_filled(o):
                    trade["tp2_hit"] = True; trade["closed"] = True
                    fill = _fp(o, trade["tp2"])
                    pnl  = _pnl(trade, fill, "tp2")
                    log.info(f"  ✅ TP2 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"✅ *FULL WIN — {symbol}*\nTP2 @ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    _record_close(trade, fill, pnl, "TP2 hit")
                    to_remove.append(symbol)

            # ── SL ──────────────────────────────────────────────────
            if not trade.get("closed") and oids.get("stop_loss"):
                o = get_o("stop_loss")
                if deribit.is_order_filled(o):
                    trade["closed"] = True
                    fill = _fp(o, trade["stop"])
                    pnl  = _pnl(trade, fill, "sl")
                    result_label = "BREAK-EVEN ⚖️" if abs(fill-entry)<entry*0.001 else "STOPPED OUT ❌"
                    log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f} ({result_label})")
                    _send(f"{'⚖️' if 'BREAK' in result_label else '❌'} *{result_label} — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    _record_close(trade, fill, pnl, result_label)
                    for k in ("tp1","tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def clean_ghost_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    live_pos = {}
    for p in deribit.get_positions():
        if float(p.get("size",0)) != 0:
            inst = p.get("instrument_name","")
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            live_pos[f"{base}USDT"] = True

    to_remove = []
    for sym, trade in trades.items():
        if float(trade.get("stop",0))==0 or float(trade.get("tp1",0))==0:
            log.warning(f"  🗑️ {sym}: stop/tp1=0 — removing broken state")
            to_remove.append(sym); continue
        if sym not in live_pos:
            log.warning(f"  🗑️ {sym}: no live Deribit position — ghost trade")
            _record_close(trade, float(trade.get("entry",0)), 0.0, "Ghost — auto-removed")
            to_remove.append(sym)

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)
        log.info(f"  ✅ Removed {len(to_remove)} ghost trade(s): {to_remove}")
        _send(f"🧹 Removed {len(to_remove)} ghost trade(s): {', '.join(to_remove)}")


# ════════ TELEGRAM ════════════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _send_open_alert(symbol, signal, conf, score, entry, stop, tp1, tp2,
                     total_q, tp1_q, tp2_q, risk_usd, balance):
    emoji="🟢" if signal=="BUY" else "🔴"; dec=4 if entry<10 else 2
    sl_p=abs((stop-entry)/entry*100); t1_p=abs((tp1-entry)/entry*100); t2_p=abs((tp2-entry)/entry*100)
    _send(f"🤖 *DERIBIT TRADE OPENED*\n━━━━━━━━━━━━━━━━━━━━\n"
          f"{emoji} *{signal} — {symbol}* ⭐×{score}\n🎯 {conf:.1f}% conf\n\n"
          f"⚡ Entry: `{entry:.{dec}f}`\n"
          f"🛑 SL:    `{stop:.{dec}f}` (-{sl_p:.1f}%)\n"
          f"🎯 TP1:   `{tp1:.{dec}f}` (+{t1_p:.1f}%) × {tp1_q}\n"
          f"🎯 TP2:   `{tp2:.{dec}f}` (+{t2_p:.1f}%) × {tp2_q}\n"
          f"📦 Total: {total_q} · Risk: ${risk_usd:.2f}\n"
          f"💼 ${balance:.2f} portfolio\n━━━━━━━━━━━━━━━━━━━━")


# ════════ MAIN ════════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    run, mode, vol, _ = should_scan()
    if not run: log.info("  Scan skipped"); return

    deribit    = DeribitClient(os.getenv("DERIBIT_CLIENT_ID",""), os.getenv("DERIBIT_CLIENT_SECRET",""))
    deribit.test_connection()
    pipeline   = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    risk_mult  = get_effective_risk(mode, vol)

    log.info(f"  {mode['label']} | conf≥{thresholds['min_confidence']}% | "
             f"score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{risk_mult:.2f}")

    log.info("\n[0] Balance...")
    balance = save_balance(deribit)

    log.info("\n[1] Monitor open trades (TP/SL detection)...")
    check_open_trades(deribit)
    save_balance(deribit)

    log.info("\n[2] Clean ghost trades...")
    clean_ghost_trades(deribit)

    # HARD CHECK before scanning — abort if already at max
    current_open = len([t for t in load_trades().values() if not t.get("closed",False)])
    if current_open >= MAX_OPEN_TRADES:
        log.info(f"\n[3] MAX TRADES REACHED ({current_open}/{MAX_OPEN_TRADES}) — no new trades allowed")
        save_balance(deribit)
        return

    log.info(f"\n[3] Scanning {len(SYMBOLS)} coins | Open:{current_open}/{MAX_OPEN_TRADES}")
    log.info(f"    Tradeable: {TRADEABLE_SYMBOLS[:8]}")

    found = 0
    for symbol in SYMBOLS:
        # Reload trades each iteration — reflects any just-placed trades
        current_open = len([t for t in load_trades().values() if not t.get("closed",False)])
        if current_open >= MAX_OPEN_TRADES:
            log.info(f"  🛑 Max trades ({MAX_OPEN_TRADES}) reached — stopping scan")
            break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.2); continue

        found += 1
        if execute_trade(deribit, sig, risk_mult, balance):
            time.sleep(1.5)

    save_balance(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | ${balance:.2f}\n{'═'*56}")


if __name__ == "__main__":
    run_execution_scan()
