# trade_executor.py — V2.4: Ghost Cleansing & True SELL Unblock

import os, json, time, logging, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, MODEL_FILE, LOG_FILE, get_tier,
    TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, LIVE_LIMIT, MAX_TRADE_AGE_HOURS
)
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds, get_effective_risk, check_correlation
)

TRADES_FILE        = "trades.json"
HISTORY_FILE       = "trade_history.json"
SIGNALS_FILE       = "signals.json"
BALANCE_FILE       = "balance.json"
MAX_OPEN_TRADES    = 4

# --- V2 PRO CONSTANTS ---
COOLDOWN_FILE      = "cooldown.json"
RELIABILITY_FILE   = "reliability.json"
COOLDOWN_HOURS     = 2
GHOST_STRIKE_LIMIT = 3
MAX_DAILY_TRADES   = 8
FUNDING_WARN_PCT   = 0.05
FUNDING_SKIP_PCT   = 0.10
# ------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
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

# --- V2 PRO HELPERS ---
def load_cooldown() -> dict: return load_json(COOLDOWN_FILE, {})
def save_cooldown(d: dict): save_json(COOLDOWN_FILE, d)
def load_reliability() -> dict: return load_json(RELIABILITY_FILE, {})
def save_reliability(d: dict): save_json(RELIABILITY_FILE, d)

def _add_cooldown(symbol: str, reason: str):
    cd = load_cooldown()
    cd[symbol] = {
        "blocked_until": (datetime.now(timezone.utc).timestamp() + COOLDOWN_HOURS * 3600),
        "reason": reason,
        "blocked_at": datetime.now(timezone.utc).isoformat(),
    }
    save_cooldown(cd)
    log.info(f"  🔒 {symbol}: cooldown {COOLDOWN_HOURS}h — {reason}")

def _is_on_cooldown(symbol: str) -> bool:
    cd = load_cooldown()
    if symbol not in cd: return False
    if datetime.now(timezone.utc).timestamp() < cd[symbol]["blocked_until"]:
        remaining = (cd[symbol]["blocked_until"] - datetime.now(timezone.utc).timestamp()) / 3600
        log.info(f"  ⏳ {symbol}: cooldown {remaining:.1f}h remaining — skip")
        return True
    cd.pop(symbol); save_cooldown(cd)
    return False

def _record_ghost(symbol: str):
    rel = load_reliability()
    if symbol not in rel: rel[symbol] = {"ghosts": 0, "wins": 0, "losses": 0}
    rel[symbol]["ghosts"] = rel[symbol].get("ghosts", 0) + 1
    save_reliability(rel)
    log.info(f"  📊 {symbol}: ghost count now {rel[symbol]['ghosts']}")

def _record_outcome(symbol: str, won: bool):
    rel = load_reliability()
    if symbol not in rel: rel[symbol] = {"ghosts": 0, "wins": 0, "losses": 0}
    if won: rel[symbol]["wins"] = rel[symbol].get("wins", 0) + 1
    else: rel[symbol]["losses"] = rel[symbol].get("losses", 0) + 1
    save_reliability(rel)

def _is_unreliable(symbol: str) -> bool:
    rel = load_reliability()
    if symbol not in rel: return False
    ghosts = rel[symbol].get("ghosts", 0)
    wins   = rel[symbol].get("wins", 0)
    if ghosts >= GHOST_STRIKE_LIMIT and ghosts > wins * 2:
        log.warning(f"  🚫 {symbol}: unreliable ({ghosts} ghosts, {wins} wins) — skip")
        return True
    return False

def _get_daily_trade_count() -> int:
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist   = load_history()
    return sum(1 for h in hist if h.get("opened_at", "")[:10] == today and h.get("close_reason") != "Ghost — PnL unrecoverable")

def _check_funding_rate(deribit, symbol: str, signal: str) -> bool:
    try:
        rate = deribit.get_funding_rate(symbol)
        rate_pct = abs(rate) * 100
        if signal == "BUY" and rate > 0:
            if rate_pct >= FUNDING_SKIP_PCT:
                log.warning(f"  💸 {symbol}: funding {rate_pct:.3f}% (8h) — skip LONG (>{FUNDING_SKIP_PCT}%)")
                return False
            elif rate_pct >= FUNDING_WARN_PCT:
                log.warning(f"  ⚠️ {symbol}: funding {rate_pct:.3f}% (8h) — high for LONG")
        if signal == "SELL" and rate < 0:
            if rate_pct >= FUNDING_SKIP_PCT:
                log.warning(f"  💸 {symbol}: funding -{rate_pct:.3f}% (8h) — skip SHORT")
                return False
    except Exception as e: log.debug(f"  funding check {symbol}: {e}")
    return True

def _wait_for_position(deribit, symbol: str, side: str, entry_oid: str, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    step     = 0.5
    while time.time() < deadline:
        time.sleep(step)
        try:
            size = deribit.get_position_size(symbol)
            if abs(size) > 0:
                log.info(f"  ✅ Position confirmed: {symbol} size={size}")
                return True
        except Exception: pass
        if entry_oid:
            try:
                chk = deribit.get_order(entry_oid)
                if chk.get("order_state", "") == "filled": return True
            except Exception: pass
        step = min(step * 1.5, 2.0)
    log.warning(f"  ⚠️ {symbol}: position not confirmed after {timeout}s")
    return False

def _place_tp_with_fallback(deribit, symbol: str, side: str, qty, price: float, label: str, trade: dict, key: str, dec: int) -> str:
    try:
        res = deribit.place_limit_order(symbol, side, qty, price, use_reduce_only=True)
        o   = res.get("order", res)
        oid = str(o.get("order_id", ""))
        if oid:
            log.info(f"  🛠 Re-placed {label} {symbol} @ {price:.{dec}f}  id:{oid}")
            return oid
    except Exception as e1:
        err = str(e1).lower()
        if "11030" in err or "invalid_reduce_only" in err:
            try:
                pos_size = deribit.get_position_size(symbol)
                if abs(pos_size) > 0:
                    log.warning(f"  {label} {symbol}: 11030 but position exists — retrying without reduce_only")
                    res2 = deribit.place_limit_order(symbol, side, qty, price, use_reduce_only=False)
                    o2   = res2.get("order", res2)
                    oid2 = str(o2.get("order_id", ""))
                    if oid2:
                        log.info(f"  🛠 {label} {symbol} @ {price:.{dec}f}  id:{oid2} [no-reduce-only fallback]")
                        return oid2
                else:
                    log.info(f"  {label} re-place {symbol}: no position — ghost cleaner handles")
            except Exception as e2: log.warning(f"  {label} fallback {symbol}: {e2}")
        else:
            log.warning(f"  {label} re-place {symbol}: {e1}")
    return ""
# ------------------------


# ════════════ BALANCE ════════════════════════════════════════════════

def save_balance(deribit: DeribitClient) -> float:
    try:
        bals  = deribit.get_all_balances()
        total = deribit.get_total_equity_usd()
        pos   = deribit.get_positions()
        upnl  = sum(float(p.get("floating_profit_loss_usd") or
                          p.get("floating_profit_loss") or 0) for p in pos)
        assets = [{"asset":c,"free":str(round(i["available"],6)),
                   "total":str(round(i["equity_usd"],2))} for c,i in bals.items()]
        save_json(BALANCE_FILE, {
            "usdt":round(total,2),"equity":round(total+upnl,2),
            "unrealised":round(upnl,4),"assets":assets,
            "updated_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":"deribit_testnet","exchange":"Deribit(by Coinbase) Testnet",
            "open_positions":len(pos),
        })
        log.info(f"  Balance: ${total:.2f} | upnl:{upnl:+.2f} | positions:{len(pos)}")
        return total
    except Exception as e:
        log.error(f"  save_balance: {e}"); return 0.0


# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url,
                params={"symbol":symbol,"interval":interval,"limit":LIVE_LIMIT},
                timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:,:6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception: continue
    return pd.DataFrame()


# ════════════ SIGNAL GENERATION ══════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY)).fillna(0)
        df1h_raw = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(df1h_raw).fillna(0) if not df1h_raw.empty else pd.DataFrame()
        
        # Fetch 4H data early so ML can use it
        df4h_raw = get_data(symbol, "4h")
        df4h = add_indicators(df4h_raw).fillna(0) if not df4h_raw.empty else pd.DataFrame()

        if df15.empty or len(df15) < 30: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(0, index=df15.columns)
        r4h = df4h.iloc[-1] if not df4h.empty else pd.Series(0, index=df15.columns)

        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        # Real 4H Input for ML Pipeline
        row["rsi_4h"]   = float(r4h.get("rsi",   50))
        row["trend_4h"] = float(r4h.get("trend",  0))

        af   = pipeline["all_features"]
        X    = pd.DataFrame([row[af].values], columns=af).replace([np.inf,-np.inf],0).fillna(0)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        
        sig  = pipeline["label_map"][int(pred)]
        conf = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score = 0
        reasons = []

        # ── 4h trend bias (filters trades against dominant trend) ──────────
        e20_4h = float(r4h.get("ema20", 0)) if not df4h.empty else 0
        e50_4h = float(r4h.get("ema50", 0)) if not df4h.empty else 0
        rsi_4h = float(r4h.get("rsi", 50))  if not df4h.empty else 50
        trend_bars = 0

        if not df4h.empty:
            if sig == "BUY" and e20_4h < e50_4h:
                log.info(f"    [FILTER:4H_BIAS] 4h bearish — skip BUY {symbol}")
                return None
            if sig == "SELL" and e20_4h > e50_4h:
                BIG_THREE = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
                if symbol in BIG_THREE:
                    log.info(f"    [FILTER:4H_BIAS] 4h bullish — hard block SELL {symbol}")
                    return None
                elif rsi_4h > 65:
                    log.info(f"    [FILTER:4H_BIAS] 4h strongly bullish (RSI {rsi_4h:.0f}) — skip SELL {symbol}")
                    return None
                else:
                    score -= 1
                    reasons.append(f"4h counter-trend (-1)")
                    log.info(f"    4h mildly bullish — score -1 for SELL {symbol}")

            # How many 4h candles has the trend been active?
            for i in range(1, min(6, len(df4h))):
                r_prev = df4h.iloc[-(i+1)]
                if sig == "BUY" and float(r_prev.get("ema20", 0)) > float(r_prev.get("ema50", 0)):
                    trend_bars += 1
                elif sig == "SELL" and float(r_prev.get("ema20", 0)) < float(r_prev.get("ema50", 0)):
                    trend_bars += 1
                else:
                    break

            # Require trend to have been in place for at least 2 candles (8h)
            if trend_bars < 2:
                log.info(f"    [FILTER:4H_FRESH] trend too fresh ({trend_bars} bars) — skip {symbol}")
                return None

        # Base Scoring
        if conf >= 70:   score+=1; reasons.append(f"High conf ({conf:.0f}%)")
        elif conf >= 60: score+=1; reasons.append(f"Conf ({conf:.0f}%)")
        
        adx_val = adx
        if adx_val > 25:   score+=1; reasons.append(f"Strong ADX {adx_val:.0f}")
        elif adx_val > 18: score+=1; reasons.append(f"ADX {adx_val:.0f}")
        
        rsi = float(row.get("rsi", 50))
        if sig=="BUY"  and rsi < 50: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
        elif sig=="SELL" and rsi > 50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")
        
        e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
        if sig=="BUY"  and e20>e50: score+=1; reasons.append("EMA bullish")
        elif sig=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")
        
        c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
        if sig=="BUY"  and c20>c50: score+=1; reasons.append("1h confirms")
        elif sig=="SELL" and c20<c50: score+=1; reasons.append("1h confirms")

        # ── 4h confirmation score ──────────────────────────────────────────
        if sig == "BUY"  and e20_4h > e50_4h and rsi_4h > 45:
            score += 1; reasons.append(f"4h trend aligns (RSI {rsi_4h:.0f})")
        elif sig == "SELL" and e20_4h < e50_4h and rsi_4h < 55:
            score += 1; reasons.append(f"4h trend aligns (RSI {rsi_4h:.0f})")

        # Score bonus for established trend
        if trend_bars >= 4:
            score += 1; reasons.append(f"4h trend established ({trend_bars*4}h)")

        # ── Volume confirmation ────
        vol_prev = float(df15["volume"].iloc[-2])
        vol_ma20 = float(df15["volume"].rolling(20).mean().iloc[-2])

        if vol_ma20 <= 0 or vol_prev <= 0:
            log.info(f"    Volume data missing — skipping volume gate")
        else:
            if vol_prev < vol_ma20 * 0.5:
                log.info(f"    [FILTER:VOL] Low volume ({vol_prev:.0f} < {vol_ma20:.0f}) — skip {symbol}")
                return None  # signal not confirmed by volume

            if vol_prev > vol_ma20 * 1.5:
                score += 1; reasons.append(f"Volume surge {vol_prev/vol_ma20:.1f}×")

        # Dynamic Execution Scoring
        effective_min = thresholds["min_score"] + (1 if sig == "SELL" else 0)
        log.info(f"    Score: {score} (need ≥{effective_min})")
        if score < effective_min:
            log.info(f"    [FILTER:SCORE] Too low ({score} < {effective_min}) — skip {symbol}")
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "reasons":reasons,"rejected":True,
                "reject_reason":f"score {score}<{effective_min}"})
            return None

        if not reasons: reasons.append(f"ML {conf:.0f}%")

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec)
            tp1 =round(entry+atr*ATR_TARGET1_MULT,dec)
            tp2 =round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec)
            tp1 =round(entry-atr*ATR_TARGET1_MULT,dec)
            tp2 =round(entry-atr*ATR_TARGET2_MULT,dec)

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}"); return None


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float) -> bool:
    symbol=sig["symbol"]; signal=sig["signal"]
    entry=sig["entry"]; atr=sig["atr"]
    stop=sig["stop"]; tp1=sig["tp1"]; tp2=sig["tp2"]

    trades     = load_trades()
    open_count = len([t for t in trades.values() if not t.get("closed",False)])
    if open_count >= MAX_OPEN_TRADES:
        log.info(f"  🛑 MAX TRADES ({MAX_OPEN_TRADES}) — skip {symbol}"); return False
    if symbol in trades and not trades[symbol].get("closed",False):
        log.info(f"  {symbol}: already open — skip"); return False
    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit — skip"); return False
    if not check_correlation(trades, signal): return False

    # V2 GATE CHECKS
    daily_count = _get_daily_trade_count()
    if daily_count >= MAX_DAILY_TRADES:
        log.info(f"  📊 Daily trade limit reached ({daily_count}/{MAX_DAILY_TRADES}) — skip {symbol}")
        return False
    if _is_on_cooldown(symbol): return False
    if _is_unreliable(symbol): return False
    if not _check_funding_rate(deribit, symbol, signal): return False

    dec=4 if entry<10 else 2
    side   ="BUY"  if signal=="BUY" else "SELL"
    sl_side="SELL" if signal=="BUY" else "BUY"
    tp_side="SELL" if signal=="BUY" else "BUY"

    total_q          = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
    risk_usd         = round(balance * RISK_PER_TRADE * risk_mult, 2)

    log.info(f"  {signal} {symbol} total={total_q} risk=${risk_usd:.2f}")
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids = {}; actual_entry = entry
    try:
        er = deribit.place_market_order(symbol, side, total_q)
        eo = er.get("order", er)
        order_ids["entry"] = str(eo.get("order_id",""))
        actual_entry = deribit.get_fill_price(er, entry) or entry
        
        o_state = eo.get("order_state", "").lower()
        filled  = float(eo.get("filled_amount", 0) or 0)
        
        if o_state == "cancelled" and filled == 0:
            log.warning(f"  Market cancelled (thin book) — skip")
            return False
            
        if o_state == "open" and filled == 0:
            log.warning(f"  Market order stuck as 'open' (zero liquidity) — cancelling & skipping")
            try: deribit.cancel_order(order_ids["entry"])
            except Exception: pass
            return False

        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}")
        
        # V2 POLLING
        position_confirmed = _wait_for_position(deribit, symbol, side, order_ids.get("entry", ""))
        if not position_confirmed:
            log.warning(f"  ⚠️ {symbol}: position unconfirmed — SL/TP may fail (will retry next scan)")

        if signal=="BUY":
            stop=deribit.round_price(symbol,actual_entry-atr*ATR_STOP_MULT)
            tp1 =deribit.round_price(symbol,actual_entry+atr*ATR_TARGET1_MULT)
            tp2 =deribit.round_price(symbol,actual_entry+atr*ATR_TARGET2_MULT)
        else:
            stop=deribit.round_price(symbol,actual_entry+atr*ATR_STOP_MULT)
            tp1 =deribit.round_price(symbol,actual_entry-atr*ATR_TARGET1_MULT)
            tp2 =deribit.round_price(symbol,actual_entry-atr*ATR_TARGET2_MULT)

        tick = deribit.get_tick_size(symbol)
        sl_limit = deribit.round_price(symbol, stop - (tick * 3) if signal=="BUY" else stop + (tick * 3))

        for label, qty, price, sl_p, key in [
            ("SL",  total_q, sl_limit, stop, "stop_loss"),
            ("TP1", qty_tp1, tp1,  None, "tp1"),
            ("TP2", qty_tp2, tp2,  None, "tp2"),
        ]:
            if qty <= 0: continue
            try:
                res = deribit.place_limit_order(
                    symbol, sl_side if label=="SL" else tp_side,
                    qty, price, stop_price=sl_p)
                o   = res.get("order", res)
                oid = str(o.get("order_id",""))
                if oid: order_ids[key] = oid
                log.info(f"  ✅ {label} @ {price:.{dec}f} × {qty} id:{oid or 'MISSING'}")
            except Exception as e: log.warning(f"  {label} failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}"); _send(f"⚠️ {symbol}: {e}"); return False

    record = {
        "symbol":symbol,"signal":signal,"entry":actual_entry,
        "stop":stop,"tp1":tp1,"tp2":tp2,
        "qty":total_q,"qty_tp1":qty_tp1,"qty_tp2":qty_tp2,
        "risk_usd":risk_usd,"balance_at_open":balance,"risk_mult":risk_mult,
        "order_ids":order_ids,
        "opened_at":datetime.now(timezone.utc).isoformat(),
        "tp1_hit":False,"tp2_hit":False,"closed":False,
        "confidence":sig["confidence"],"score":sig["score"],
        "reasons":sig.get("reasons",[]),"tier":get_tier(symbol),
        "exchange":"deribit_testnet",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal({**record,"type":"executed"})
    _send_open_alert(symbol, signal, sig["confidence"], sig["score"],
                     actual_entry, stop, tp1, tp2, total_q, qty_tp1, qty_tp2, risk_usd, balance)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}")
    return True


# ════════════ MONITOR OPEN TRADES ════════════════════════════════════

def _safe_get_order(deribit, oid_str):
    if not oid_str or oid_str in ("", "None"): return {}
    try: return deribit.get_order(oid_str)
    except Exception: return {}

def fp(o, fb):
    p = float(o.get("average_price") or o.get("last_price") or o.get("price") or 0)
    return p if p > 0 else fb

def _pnl(t, cp, ct):
    qty  = float(t["qty_tp1"] if ct == "tp1" else t["qty_tp2"] if ct == "tp2" else t["qty"])
    diff = (cp - float(t["entry"])) if t["signal"] == "BUY" else (float(t["entry"]) - cp)
    return round(diff * qty, 4)

def _close_record(t, cp, pnl, reason):
    append_history({**t, "close_price": cp, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})

def _replace_missing_orders(deribit: DeribitClient, symbol: str, trade: dict) -> bool:
    oids   = trade.get("order_ids", {})
    signal = trade["signal"]
    entry  = float(trade["entry"])
    stop   = float(trade.get("stop", 0))
    tp1    = float(trade.get("tp1", 0))
    tp2    = float(trade.get("tp2", 0))
    qty    = float(trade.get("qty", 0))
    qty_t1 = float(trade.get("qty_tp1", 0))
    qty_t2 = float(trade.get("qty_tp2", 0))
    dec    = 4 if entry < 10 else 2
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"
    changed = False

    # ── Re-place SL ───────────────────────────────────────────────────────
    sl_oid  = str(oids.get("stop_loss", ""))
    need_sl = (not sl_oid or sl_oid in ("", "None")) and stop > 0 and qty > 0
    if need_sl and not trade.get("closed"):
        try:
            tick     = deribit.get_tick_size(symbol)
            sl_limit = deribit.round_price(
                symbol, stop - (tick * 3) if sl_side == "SELL" else stop + (tick * 3)
            )
            res = deribit.place_limit_order(
                symbol, sl_side, qty, sl_limit,
                stop_price=stop, use_reduce_only=True
            )
            o   = res.get("order", res)
            oid = str(o.get("order_id", ""))
            if oid:
                trade["order_ids"]["stop_loss"] = oid
                log.info(f"  🛠 Re-placed SL {symbol} @ {sl_limit:.{dec}f}"
                         f"  trigger={stop:.{dec}f}  id:{oid}")
                changed = True
        except Exception as e:
            err = str(e).lower()
            log.warning(f"  SL re-place {symbol}: {e}")
            if any(x in err for x in ("trigger_price_too_low", "trigger_price_too_high",
                                       "10035", "10036")):
                log.warning(f"  🚨 {symbol}: SL trigger already passed — MARKET CLOSE")
                try:
                    close_side = "SELL" if signal == "BUY" else "BUY"
                    close_qty  = (float(trade.get("qty_tp2", qty))
                                  if trade.get("tp1_hit") else qty)
                    if close_qty > 0:
                        deribit.place_market_order(symbol, close_side, close_qty)
                    live = deribit.get_live_price(symbol)
                    pnl  = _pnl(trade, live if live > 0 else stop, "sl")
                    _close_record(trade, live if live > 0 else stop, pnl,
                                  "SL missed — market close")
                    _send(f"🚨 *SL MISSED — {symbol}*\n"
                          f"Trigger passed. Closed @ `{live:.{dec}f}` | PnL≈`{pnl:+.4f}`")
                    if pnl < 0:
                        _add_cooldown(symbol, f"SL missed market close pnl={pnl:+.4f}")
                        _record_outcome(symbol, won=False)
                    trade["closed"] = True
                    changed = True
                except Exception as me:
                    log.error(f"  Emergency SL close {symbol}: {me}")

    # ── Re-place TP1 ──────────────────────────────────────────────────────
    tp1_oid  = str(oids.get("tp1", ""))
    need_tp1 = (not tp1_oid or tp1_oid in ("", "None")) and tp1 > 0 and qty_t1 > 0
    if need_tp1 and not trade.get("tp1_hit") and not trade.get("closed"):
        oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t1, tp1, "TP1", trade, "tp1", dec)
        if oid:
            trade["order_ids"]["tp1"] = oid
            changed = True

    # ── Re-place TP2 ──────────────────────────────────────────────────────
    tp2_oid  = str(oids.get("tp2", ""))
    need_tp2 = (not tp2_oid or tp2_oid in ("", "None")) and tp2 > 0 and qty_t2 > 0
    if need_tp2 and not trade.get("tp2_hit") and not trade.get("closed"):
        oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t2, tp2, "TP2", trade, "tp2", dec)
        if oid:
            trade["order_ids"]["tp2"] = oid
            changed = True

    return changed


def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    live_positions = set()
    try:
        for p in deribit.get_positions():
            if float(p.get("size", 0)) != 0:
                inst = p.get("instrument_name", "")
                base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                live_positions.add(f"{base}USDT")
    except Exception as e:
        log.warning(f"  Could not fetch live positions: {e}")

    for symbol, trade in list(trades.items()):

        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        entry  = float(trade["entry"])
        stop   = float(trade.get("stop", 0))
        tp1_p  = float(trade.get("tp1",  0))
        tp2_p  = float(trade.get("tp2",  0))
        dec    = 4 if entry < 10 else 2
        signal = trade["signal"]
        oids   = trade.get("order_ids", {})

        if _replace_missing_orders(deribit, symbol, trade): save_trades(trades)

        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        try:
            live = deribit.get_live_price(symbol)
        except Exception as e: log.warning(f"  {symbol}: live price error — {e}"); continue
        if live <= 0: log.warning(f"  {symbol}: live price = 0 — skip"); continue

        if live_positions and symbol not in live_positions and not trade.get("tp1_hit"):
            log.warning(f"  {symbol}: no live position found — ghost cleaner will handle")
            continue

        risk_usd = float(trade.get("risk_usd", 0))
        if risk_usd > 0:
            mae_qty  = float(trade.get("qty_tp2", 0)) if trade.get("tp1_hit") else float(trade["qty"])
            mae_diff = (live - entry) if signal == "BUY" else (entry - live)
            mae_pnl  = round(mae_diff * mae_qty, 4)

            if mae_pnl < -(risk_usd * 3):
                log.warning(f"  🚨 {symbol}: MAE ${mae_pnl:.2f} > 3× risk ${risk_usd:.2f} — FORCE CLOSE")
                try:
                    close_side = "SELL" if signal == "BUY" else "BUY"
                    if mae_qty > 0:
                        deribit.place_market_order(symbol, close_side, mae_qty)
                    for k in ("tp1", "tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    lbl = "Max adverse excursion ❌"
                    _close_record(trade, live, mae_pnl, lbl)
                    _send(f"🚨 *FORCE CLOSE — {symbol}*\nLoss `{mae_pnl:+.4f}` exceeded 3× risk\nLive @ `{live:.{dec}f}`")
                    
                    if mae_pnl < 0:
                        _add_cooldown(symbol, f"MAE close pnl={mae_pnl:+.4f}")
                        _record_outcome(symbol, won=False)
                        
                    trade["closed"] = True
                    to_remove.append(symbol)
                    continue
                except Exception as e:
                    log.error(f"  MAE force-close {symbol}: {e}")

        try:
            # ── TP1 CHECK ─────────────────────────────────────────────────
            if not trade.get("tp1_hit"):
                o      = _safe_get_order(deribit, str(oids.get("tp1", "")))
                state  = o.get("order_state", "").lower()

                tp1_order_filled = deribit.is_order_filled(o)

                filled_amt = float(o.get("filled_amount", 0) or 0)
                total_amt  = float(o.get("amount", 0) or 0)
                tp1_partial = (total_amt > 0 and filled_amt / total_amt >= 0.8 and filled_amt > 0)

                tp1_price_hit = tp1_p > 0 and ((signal == "BUY" and live >= tp1_p) or (signal == "SELL" and live <= tp1_p))
                tp1_o_gone = state in ("filled", "cancelled", "closed", "rejected", "")

                if tp1_order_filled or tp1_partial or (tp1_price_hit and tp1_o_gone):
                    trade["tp1_hit"] = True
                    method = ("order" if tp1_order_filled else "partial" if tp1_partial else "price-fallback")
                    fill   = fp(o, tp1_p) if (tp1_order_filled or tp1_partial) else live
                    pnl    = _pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f}  pnl≈{pnl:+.4f}  [{method}]")
                    _send(f"🎯 *TP1 HIT — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}` | [{method}]")
                    
                    # V2 FIX: Record TP1
                    append_history({
                        **trade,
                        "close_price": fill,
                        "pnl":         pnl,
                        "qty":         float(trade.get("qty_tp1", 0)),
                        "closed_at":   datetime.now(timezone.utc).isoformat(),
                        "close_reason": f"TP1 hit [{method}]",
                        "partial":      True,
                    })

                    if oids.get("stop_loss") and float(trade.get("qty_tp2", 0)) > 0:
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                        try:
                            sl_s = "SELL" if signal == "BUY" else "BUY"
                            tick = deribit.get_tick_size(symbol)
                            be_limit = entry + tick if sl_s == "BUY" else entry - tick

                            be   = deribit.place_limit_order(
                                symbol, sl_s, float(trade["qty_tp2"]),
                                be_limit, stop_price=entry
                            )
                            be_o = be.get("order", be)
                            nid  = str(be_o.get("order_id", ""))
                            if nid:
                                trade["order_ids"]["stop_loss"] = nid
                                trade["stop"] = entry
                            _send(f"🛡️ *{symbol} RISK-FREE* SL→entry `{entry:.{dec}f}`")
                        except Exception as be_e:
                            log.warning(f"  BE SL {symbol}: {be_e}")

            # ── TRAILING STOP (TP1 hit → halfway to TP2) ──────────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                halfway = (entry + tp2_p) / 2
                at_half = ((signal == "BUY"  and live >= halfway) or (signal == "SELL" and live <= halfway))
                sl_at_be = abs(float(trade.get("stop", 0)) - entry) < entry * 0.001
                if at_half and sl_at_be and float(trade.get("qty_tp2", 0)) > 0:
                    try:
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                        sl_s = "SELL" if signal == "BUY" else "BUY"
                        tick = deribit.get_tick_size(symbol)
                        sl_lim = tp1_p + tick if sl_s == "BUY" else tp1_p - tick

                        sl_r = deribit.place_limit_order(
                            symbol, sl_s, float(trade["qty_tp2"]),
                            sl_lim, stop_price=tp1_p
                        )
                        sl_o = sl_r.get("order", sl_r)
                        nid  = str(sl_o.get("order_id", ""))
                        if nid:
                            trade["order_ids"]["stop_loss"] = nid
                            trade["stop"] = tp1_p
                        _send(f"🚀 *{symbol}* Trail SL→TP1 `{tp1_p:.{dec}f}` locked!")
                    except Exception as e:
                        log.warning(f"  Trail SL {symbol}: {e}")

            # ── TP2 CHECK ─────────────────────────────────────────────────
            if not trade.get("tp2_hit"):
                o      = _safe_get_order(deribit, str(oids.get("tp2", "")))
                state2 = o.get("order_state", "").lower()

                tp2_order_filled = deribit.is_order_filled(o)

                filled_amt2 = float(o.get("filled_amount", 0) or 0)
                total_amt2  = float(o.get("amount", 0) or 0)
                tp2_partial = (total_amt2 > 0 and filled_amt2 / total_amt2 >= 0.8 and filled_amt2 > 0)

                tp2_price_hit = tp2_p > 0 and ((signal == "BUY" and live >= tp2_p) or (signal == "SELL" and live <= tp2_p))
                tp2_o_gone = state2 in ("filled", "cancelled", "closed", "rejected", "")

                if tp2_order_filled or tp2_partial or (tp2_price_hit and tp2_o_gone):
                    if not trade.get("tp1_hit"):
                        trade["tp1_hit"] = True
                        log.info(f"  🎯 TP1 {symbol} auto-marked (huge candle blew past both)")

                    method2 = ("order" if tp2_order_filled else "partial" if tp2_partial else "price-fallback")
                    fill2   = fp(o, tp2_p) if (tp2_order_filled or tp2_partial) else live
                    pnl2    = _pnl(trade, fill2, "tp2")

                    trade["tp2_hit"] = True
                    trade["closed"]  = True
                    log.info(f"  ✅ TP2 {symbol} @ {fill2:.{dec}f}  pnl≈{pnl2:+.4f}  [{method2}]")
                    _send(f"✅ *FULL WIN — {symbol}*\nTP2 @ `{fill2:.{dec}f}` | PnL ≈ `{pnl2:+.4f}` | [{method2}]")
                    _close_record(trade, fill2, pnl2, "TP2 hit")

                    _record_outcome(symbol, won=True)
                    cd = load_cooldown()
                    cd.pop(symbol, None); save_cooldown(cd)

                    if oids.get("stop_loss"):
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                    to_remove.append(symbol)
                    continue

            # ── SL CHECK ──────────────────────────────────────────────────
            if not trade.get("closed") and oids.get("stop_loss"):
                sl_o      = _safe_get_order(deribit, str(oids["stop_loss"]))
                sl_state  = sl_o.get("order_state", "").lower()

                sl_hit = deribit.is_sl_triggered(sl_o)

                sl_breached = stop > 0 and ((signal == "BUY" and live <= stop * 0.999) or (signal == "SELL" and live >= stop * 1.001))
                sl_not_waiting = sl_state not in ("untriggered", "open") or not sl_state

                # V2 BUG 1 FIX: Mark-price breach
                mark_price = deribit.get_mark_price(symbol)
                mark_breached = stop > 0 and mark_price > 0 and (
                    (signal == "BUY"  and mark_price <= stop * 0.998) or
                    (signal == "SELL" and mark_price >= stop * 1.002)
                )

                if mark_breached and sl_not_waiting and not sl_hit:
                    log.warning(
                        f"  ⚠️ {symbol}: MARK PRICE {mark_price:.{dec}f} past SL {stop:.{dec}f}"
                        f" (last_price lag) — MARK-PRICE BREACH close"
                    )
                    try:
                        close_side = "SELL" if signal == "BUY" else "BUY"
                        close_qty  = (float(trade.get("qty_tp2", 0))
                                      if trade.get("tp1_hit") else float(trade["qty"]))
                        if close_qty > 0:
                            deribit.place_market_order(symbol, close_side, close_qty)
                        for k in ("tp1", "tp2"):
                            if oids.get(k) and not trade.get(f"{k}_hit"):
                                try: deribit.cancel_order(oids[k])
                                except Exception: pass
                        sl_hit = True
                        log.warning(f"  Mark-price emergency close executed for {symbol}")
                    except Exception as e:
                        log.error(f"  Mark-breach close {symbol}: {e}")


                if not sl_hit and sl_breached and sl_not_waiting:
                    log.warning(f"  ⚠️ {symbol}: price {live:.{dec}f} past SL {stop:.{dec}f} (state='{sl_state}') — SCENARIO B market close")
                    try:
                        close_side = "SELL" if signal == "BUY" else "BUY"
                        close_qty  = (float(trade.get("qty_tp2", 0)) if trade.get("tp1_hit") else float(trade["qty"]))
                        if close_qty > 0:
                            deribit.place_market_order(symbol, close_side, close_qty)
                        for k in ("tp1", "tp2"):
                            if oids.get(k) and not trade.get(f"{k}_hit"):
                                try: deribit.cancel_order(oids[k])
                                except Exception: pass
                        sl_hit = True
                    except Exception as e:
                        log.error(f"  Scenario-B close {symbol}: {e}")

                if sl_hit:
                    trade["closed"] = True
                    fill = fp(sl_o, stop)
                    if fill == 0 or fill == stop or fill == float(trade.get("stop", 0)):
                        fill = live if live > 0 else stop

                    slippage_pct = abs(fill - stop) / stop * 100
                    if slippage_pct > 0.5:
                        log.warning(f"  ⚠️ {symbol}: SL slippage {slippage_pct:.2f}% (expected {stop:.{dec}f}, got {fill:.{dec}f})")

                    pnl = _pnl(trade, fill, "sl")
                    lbl = ("BREAK-EVEN ⚖️" if abs(fill - entry) < entry * 0.002 else "STOPPED OUT ❌")
                    log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f}  pnl≈{pnl:+.4f}  [state={sl_state}]")
                    _send(f"{'⚖️' if 'BREAK' in lbl else '❌'} *{lbl} — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`" + (f"\n⚠️ Slippage: {slippage_pct:.2f}%" if slippage_pct > 0.5 else ""))
                    _close_record(trade, fill, pnl, lbl)
                    
                    if pnl < 0:
                        _add_cooldown(symbol, f"SL hit pnl={pnl:+.4f}")
                        _record_outcome(symbol, won=False)
                    else:
                        _record_outcome(symbol, won=True)

                    for k in ("tp1", "tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def check_stale_trades(deribit: DeribitClient):
    trades = load_trades()
    now    = datetime.now(timezone.utc)
    to_remove = []

    for symbol, trade in trades.items():
        if trade.get("closed") or trade.get("tp2_hit"): continue
        try:
            age_h = (now - datetime.fromisoformat(trade.get("opened_at", "").replace("Z", ""))).total_seconds() / 3600
        except Exception: continue

        if age_h <= MAX_TRADE_AGE_HOURS: continue

        log.warning(f"  ⏰ {symbol}: {age_h:.0f}h — time-based exit")
        try:
            for k, oid in trade.get("order_ids", {}).items():
                if k != "entry" and oid and str(oid) not in ("", "None"):
                    try: deribit.cancel_order(oid)
                    except Exception: pass

            live      = deribit.get_live_price(symbol)
            close_qty = (float(trade.get("qty_tp2", 0)) if trade.get("tp1_hit") else float(trade["qty"]))
            if close_qty > 0:
                close_side = "SELL" if trade["signal"] == "BUY" else "BUY"
                try: deribit.place_market_order(symbol, close_side, close_qty)
                except Exception as me: log.warning(f"  Time-exit market order {symbol}: {me}")

            close_price = live if live > 0 else float(trade["entry"])
            pnl         = _pnl(trade, close_price, "sl")
            reason      = f"Time exit ({age_h:.0f}h)"
            _close_record(trade, close_price, pnl, reason)
            _send(f"⏰ *TIME EXIT — {symbol}*\n{age_h:.0f}h | PnL≈`{pnl:+.4f}` | @ `{close_price:.4f}`")
            to_remove.append(symbol)
        except Exception as e: log.error(f"  Time exit {symbol}: {e}")

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)


def clean_ghost_trades(deribit: DeribitClient):
    trades=load_trades()
    if not trades: return
    live_pos = {}
    for p in deribit.get_positions():
        if float(p.get("size",0))!=0:
            inst=p.get("instrument_name","")
            base=inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            live_pos[f"{base}USDT"]=True

    to_remove=[]
    for symbol, trade in trades.items():
        if float(trade.get("stop",0))==0 or float(trade.get("tp1",0))==0:
            log.warning(f"  🗑️ {symbol}: broken state — remove")
            to_remove.append(symbol); continue
            
        # ZOMBIE CLEANER: Purge corrupted score:0 / confidence:0 records
        if float(trade.get("score", 1)) == 0 and float(trade.get("confidence", 1)) == 0:
            log.warning(f"  🗑️ {symbol}: score=0 confidence=0 — broken record, removing")
            _close_record(trade, float(trade.get("entry", 0)), 0.0, "Ghost — broken record")
            to_remove.append(symbol)
            continue
            
        if symbol not in live_pos:
            log.warning(f"  🕵️ {symbol}: no position — recovering PnL...")
            real_pnl=None; real_close=None; reason="Closed on exchange"
            try:
                fills=deribit.get_trade_history_for_instrument(symbol,count=20)
                entry=float(trade["entry"]); entry_dir=trade["signal"]
                close_fills=[f for f in fills
                    if (entry_dir=="BUY" and f.get("direction")=="sell") or
                       (entry_dir=="SELL" and f.get("direction")=="buy")]
                if close_fills:
                    latest=close_fills[0]; real_close=float(latest.get("price",0) or 0)
                    qty=float(trade.get("qty",0))
                    if real_close>0 and qty>0:
                        diff=(real_close-entry) if entry_dir=="BUY" else (entry-real_close)
                        real_pnl=round(diff*qty,4)
                        tp1_p=float(trade.get("tp1",0)); tp2_p=float(trade.get("tp2",0))
                        if entry_dir=="BUY":
                            if real_close >= tp2_p*0.998: reason="TP2 hit"
                            elif real_close >= tp1_p*0.998: reason="TP1 hit"
                            else: reason="SL hit"
                        else:
                            if real_close <= tp2_p*1.002: reason="TP2 hit"
                            elif real_close <= tp1_p*1.002: reason="TP1 hit"
                            else: reason="SL hit"
                        log.info(f"  ✅ Recovered {symbol}: close={real_close:.4f} pnl={real_pnl:+.4f} ({reason})")
            except Exception as e: log.warning(f"  PnL recovery {symbol}: {e}")

            if real_pnl is not None and real_close is not None:
                _close_record(trade,real_close,real_pnl,reason)
                _send(f"{'✅' if real_pnl>0 else '❌'} *{reason} — {symbol}*\nPnL:`{real_pnl:+.4f}`")
            else:
                _close_record(trade,float(trade.get("entry",0)),0.0,"Ghost — PnL unrecoverable")
                _record_ghost(symbol)
                _add_cooldown(symbol, "Ghost trade — position unrecoverable")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym,None)
        save_trades(trades)
        log.info(f"  Processed {len(to_remove)} ghost/closed trade(s)")

def check_funding_rates(deribit) -> None:
    trades = load_trades()
    if not trades: return
    for symbol, trade in trades.items():
        if trade.get("closed"): continue
        try:
            rate     = deribit.get_funding_rate(symbol)
            rate_pct = rate * 100
            signal   = trade["signal"]
            is_paying = (signal == "BUY" and rate > 0) or (signal == "SELL" and rate < 0)
            if is_paying and abs(rate_pct) >= FUNDING_WARN_PCT:
                age_h = 0
                try:
                    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(
                        trade.get("opened_at","").replace("Z",""))).total_seconds() / 3600
                except Exception: pass
                total_drag = abs(rate_pct) * (age_h / 8)
                log.warning(f"  💸 {symbol} {signal}: funding {rate_pct:+.3f}%/8h | age={age_h:.0f}h | drag≈{total_drag:.3f}%")
                if abs(rate_pct) >= FUNDING_SKIP_PCT:
                    _send(f"💸 *FUNDING ALERT — {symbol}*\n{signal} paying `{abs(rate_pct):.3f}%` per 8h\n"
                          f"Position age: `{age_h:.0f}h` | Total drag ≈ `{total_drag:.3f}%`\nConsider manual close if TP unlikely soon.")
        except Exception as e: log.debug(f"  funding monitor {symbol}: {e}")

# ════════════ TELEGRAM ════════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"},timeout=10)
    except Exception: pass

def _send_open_alert(sym,sig,conf,score,entry,stop,tp1,tp2,qty,q1,q2,risk,bal):
    e="🟢" if sig=="BUY" else "🔴"; d=4 if entry<10 else 2
    sp=abs((stop-entry)/entry*100); t1=abs((tp1-entry)/entry*100); t2=abs((tp2-entry)/entry*100)
    _send(f"🤖 *DERIBIT TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
          f"{e} *{sig} — {sym}* ⭐×{score}\n🎯 {conf:.1f}% conf\n\n"
          f"⚡ Entry: `{entry:.{d}f}`\n"
          f"🛑 SL:    `{stop:.{d}f}` (-{sp:.1f}%)\n"
          f"🎯 TP1:   `{tp1:.{d}f}` (+{t1:.1f}%) × {q1}\n"
          f"🎯 TP2:   `{tp2:.{d}f}` (+{t2:.1f}%) × {q2}\n"
          f"📦 {qty} contracts · Risk: ${risk:.2f} · ${bal:.2f}\n"
          f"━━━━━━━━━━━━━━━━━━━━")


# ════════════ MAIN SCAN ═══════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    run, mode, vol, reason = should_scan()
    if not run:
        log.info(f"  Scan skipped: {reason}"); return

    deribit    = DeribitClient(os.getenv("DERIBIT_CLIENT_ID",""), os.getenv("DERIBIT_CLIENT_SECRET",""))
    deribit.test_connection()
    pipeline   = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    risk_mult  = get_effective_risk(mode, vol)

    log.info(f"  {mode['label']} | conf≥{thresholds['min_confidence']}% "
             f"| score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{risk_mult:.2f}")

    log.info("\n[0] Balance..."); balance = save_balance(deribit)
    log.info("\n[1] Monitor trades..."); check_open_trades(deribit)
    log.info("\n[2] Stale trade check..."); check_stale_trades(deribit)
    log.info("\n[3] Ghost trade recovery..."); clean_ghost_trades(deribit)
    log.info("\n[3b] Funding rate check..."); check_funding_rates(deribit)
    
    # ── Orphan Position Monitor ──
    log.info("\n[3c] Checking for untracked positions...")
    tracked = set(load_trades().keys())
    for p in deribit.get_positions():
        if float(p.get("size", 0)) == 0:
            continue
        inst  = p.get("instrument_name", "")
        base  = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
        sym   = f"{base}USDT"
        upnl  = float(p.get("floating_profit_loss_usd", 0) or 0)
        size  = float(p.get("size", 0))
        if sym not in tracked:
            log.warning(f"  ⚠️ UNTRACKED POSITION: {sym} size={size} uPnL=${upnl:+.2f} — close manually!")
            _send(f"⚠️ *UNTRACKED POSITION* — {sym}\nSize: `{size}` | uPnL: `${upnl:+.2f}`\nNot in trades.json — close manually on Deribit!")

    save_balance(deribit)

    open_count = len([t for t in load_trades().values() if not t.get("closed",False)])
    log.info(f"\n[4] Scanning {len(SYMBOLS)} coins | Open:{open_count}/{MAX_OPEN_TRADES}")

    found = 0
    for symbol in SYMBOLS:
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.2); continue
        found += 1
        if execute_trade(deribit, sig, risk_mult, balance): time.sleep(1.5)

    save_balance(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | ${balance:.2f}\n{'═'*56}")


if __name__ == "__main__":
    run_execution_scan()
