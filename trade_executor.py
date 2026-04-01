# trade_executor.py
import os, json, time, logging, requests, joblib, random
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

try: import ccxt
except ImportError: raise ImportError("Run: pip install ccxt")

from config import (SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT, RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, LIVE_LIMIT, MODEL_FILE, LOG_FILE, get_tier)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk

# ════════════ THE BRIDGE: PERSISTENCE ════════════
try:
    from persistence import load_json, save_json, sync_all_to_github
    PERSISTENCE_ENABLED = True
except ImportError:
    PERSISTENCE_ENABLED = False
    def load_json(path, default):
        try:
            if Path(path).exists():
                with open(path) as f: return json.load(f)
        except Exception: pass
        return default
    def save_json(path, data):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)

TRADES_FILE, HISTORY_FILE, SIGNALS_FILE, BALANCE_FILE, MODE_FILE = "trades.json", "trade_history.json", "signals.json", "balance.json", "scan_mode.json"
MAX_OPEN_TRADES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()])
log = logging.getLogger(__name__)

load_trades  = lambda: load_json(TRADES_FILE, {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])
load_signals = lambda: load_json(SIGNALS_FILE, [])
def append_history(rec): h=load_history(); h.append(rec); save_json(HISTORY_FILE,h)
def save_signal(sig): s=load_signals(); s.append({**sig,"generated_at":datetime.now(timezone.utc).isoformat()}); save_json(SIGNALS_FILE,s[-500:])


# ════════════ AUTO-PROXY HUNTER (100% FREE) ════════════
def get_working_free_proxy():
    log.info("Hunting for a free proxy to bypass USA block...")
    proxy_list = []
    
    # Source 1: ProxyScrape (Pulls Elite proxies from Europe)
    try:
        r1 = requests.get("https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=DE,FR,GB,NL&ssl=yes&anonymity=elite", timeout=10)
        if r1.status_code == 200:
            proxy_list.extend([p for p in r1.text.strip().split("\r\n") if p])
    except: pass
    
    # Source 2: GitHub Public Proxy List (Fallback)
    if not proxy_list:
        try:
            r2 = requests.get("https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt", timeout=10)
            if r2.status_code == 200:
                proxy_list.extend([p for p in r2.text.strip().split("\n") if p])
        except: pass
        
    if not proxy_list:
        return ""
        
    random.shuffle(proxy_list)
    
    # Rapidly test up to 15 free proxies until one connects to Binance
    for p in proxy_list[:15]: 
        proxy_url = f"http://{p.strip()}"
        try:
            test = requests.get("https://testnet.binance.vision/api/v3/ping", 
                                proxies={"http": proxy_url, "https": proxy_url}, 
                                timeout=4)
            if test.status_code == 200:
                log.info(f"✓ Found working free proxy: {p}")
                return proxy_url
        except:
            continue
            
    log.warning("Could not find a working free proxy in time.")
    return ""


def init_exchange():
    key, secret, proxy_url = os.getenv("BINANCE_API_KEY",""), os.getenv("BINANCE_SECRET",""), os.getenv("PROXY_URL", "")
    if not key or not secret: raise ValueError("BINANCE_API_KEY missing!")
    cfg = {"apiKey": key, "secret": secret, "enableRateLimit": True, "options": {"defaultType": "spot"}, "urls": {"api": {"public": "https://testnet.binance.vision/api", "private": "https://testnet.binance.vision/api"}}}
    
    if proxy_url: cfg["proxies"] = {"http": proxy_url, "https": proxy_url} # 👈 PROXY APPLIED
    
    ex = ccxt.binance(cfg)
    ex.set_sandbox_mode(True)
    log.info("✓ Exchange: Binance TESTNET" + (" (Routed via Proxy 🛡️)" if proxy_url else " (No Proxy)"))
    return ex

def fetch_and_save_balance(ex):
    proxy_url = os.getenv("PROXY_URL", "")
    proxies   = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    try:
        b = ex.fetch_balance()
        usdt = float(b.get("USDT", {}).get("free", 0) or 0)
        assets = [{"asset": a, "free": round(float(v.get("free",0) or 0), 6), "total": round(float(v.get("total",0) or 0), 6)} for a, v in b.items() if isinstance(v, dict) and float(v.get("total", 0) or 0) > 0 and a not in ("info","free","used","total","timestamp","datetime")]
        save_json(BALANCE_FILE, {"usdt": round(usdt, 2), "assets": assets, "updated_at": datetime.now(timezone.utc).isoformat(), "source": "ccxt"})
        log.info(f"✓ Balance (ccxt): {usdt:.2f} USDT")
        return usdt
    except Exception as e:
        log.warning(f"ccxt balance failed ({e}) — trying direct REST...")

    try:
        import hmac, hashlib
        key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET", "")
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        params = f"timestamp={ts}"
        sig = hmac.new(secret.encode(), params.encode(), hashlib.sha256).hexdigest()
        url = f"https://testnet.binance.vision/api/v3/account?{params}&signature={sig}"
        
        # 🛡️ Uses proxy for REST fallback to bypass the block
        r = requests.get(url, headers={"X-MBX-APIKEY": key}, proxies=proxies, timeout=15)
        r.raise_for_status()
        data = r.json()

        usdt, assets = 0.0, []
        for b in data.get("balances", []):
            free, locked = float(b.get("free",0) or 0), float(b.get("locked",0) or 0)
            if (free + locked) > 0:
                if b["asset"] == "USDT": usdt = free
                assets.append({"asset": b["asset"], "free": round(free,6), "total": round(free+locked,6)})

        save_json(BALANCE_FILE, {"usdt": round(usdt, 2), "assets": assets, "updated_at": datetime.now(timezone.utc).isoformat(), "source": "direct_rest"})
        log.info(f"✓ Balance (REST): {usdt:.2f} USDT")
        return usdt
    except Exception as e2:
        log.error(f"Both balance methods failed. REST error: {e2}")
        save_json(BALANCE_FILE, {"usdt": None, "assets": [], "error": str(e2), "updated_at": datetime.now(timezone.utc).isoformat()})
        return 0.0

def load_model():
    p=joblib.load(MODEL_FILE)
    for k in ["ensemble","selector","all_features","label_map"]:
        if k not in p: raise ValueError(f"Model missing key: {k}")
    log.info(f"✓ Model: {len(p['all_features'])} features")
    return p

def get_data(symbol, interval):
    endpoints = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines", "https://api1.binance.com/api/v3/klines"]
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    last_err = None
    for url in endpoints:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
            return df
        except Exception as e:
            last_err = e; continue
    raise last_err

def calc_pos_size(balance,entry,stop):
    dist=abs(entry-stop)
    return round(balance*RISK_PER_TRADE/dist,6) if dist>0 else 0.0

def execute_trade(ex,symbol,signal,entry,atr,confidence,score,reasons):
    trades=load_trades()
    if symbol in trades: log.info(f"  {symbol} already open"); return False
    if len(trades)>=MAX_OPEN_TRADES: return False

    dec=4 if entry<10 else 2
    if signal=="BUY":
        stop=round(entry-atr*ATR_STOP_MULT,dec); tp1=round(entry+atr*ATR_TARGET1_MULT,dec); tp2=round(entry+atr*ATR_TARGET2_MULT,dec)
        side="buy"; sl_side="sell"; tp_side="sell"
    else:
        stop=round(entry+atr*ATR_STOP_MULT,dec); tp1=round(entry-atr*ATR_TARGET1_MULT,dec); tp2=round(entry-atr*ATR_TARGET2_MULT,dec)
        side="sell"; sl_side="buy"; tp_side="buy"

    balance=fetch_and_save_balance(ex)
    if balance<10: return False

    qty=calc_pos_size(balance,entry,stop)
    if qty<=0: return False
    qty_tp1, qty_tp2 = round(qty*0.5,6), round(qty*0.5,6)
    risk_usd=round(balance*RISK_PER_TRADE,2)
    log.info(f"  {signal} {symbol} qty={qty} SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids={}
    try:
        eo=ex.create_order(symbol,"market",side,qty); order_ids["entry"]=eo["id"]; actual_entry=float(eo.get("average",entry) or entry)
        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}"); time.sleep(1.5)
        for ot in ["stop_loss_limit","limit"]:
            try: o=ex.create_order(symbol,ot,sl_side,qty,stop,params={"stopPrice":stop,"timeInForce":"GTC"}); order_ids["stop_loss"]=o["id"]; break
            except: pass
        for ot in ["take_profit_limit","limit"]:
            try: o=ex.create_order(symbol,ot,tp_side,qty_tp1,tp1,params={"stopPrice":tp1,"timeInForce":"GTC"}); order_ids["tp1"]=o["id"]; break
            except: pass
        for ot in ["take_profit_limit","limit"]:
            try: o=ex.create_order(symbol,ot,tp_side,qty_tp2,tp2,params={"stopPrice":tp2,"timeInForce":"GTC"}); order_ids["tp2"]=o["id"]; break
            except: pass
    except Exception as e: log.error(f"  Trade error: {e}"); return False

    rec={"symbol":symbol,"signal":signal,"entry":actual_entry,"stop":stop,"tp1":tp1,"tp2":tp2,
         "qty":qty,"qty_tp1":qty_tp1,"qty_tp2":qty_tp2,"risk_usd":risk_usd,"balance_at_open":balance,
         "order_ids":order_ids,"opened_at":datetime.now(timezone.utc).isoformat(),
         "tp1_hit":False,"tp2_hit":False,"closed":False,"confidence":confidence,"score":score,"reasons":reasons,"tier":get_tier(symbol)}
    trades[symbol]=rec; save_trades(trades); save_signal(rec)
    _send_open_alert(symbol,signal,confidence,score,actual_entry,stop,tp1,tp2,qty,risk_usd,balance,reasons)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}"); return True

def check_open_trades(ex):
    trades=load_trades()
    if not trades: log.info("  No open trades"); return
    to_rm=[]
    for symbol,trade in list(trades.items()):
        if trade.get("closed"): to_rm.append(symbol); continue
        oids=trade.get("order_ids",{}); entry=trade["entry"]
        try:
            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o=ex.fetch_order(oids["tp1"],symbol)
                    if o["status"]=="closed":
                        trade["tp1_hit"]=True; pnl=_pnl(trade,float(o["average"]),"tp1")
                        log.info(f"  🎯 TP1 {symbol} {pnl:+.4f}"); _send_close_alert(symbol,"TP1 HIT 🎯",pnl,entry,float(o["average"]),trade["opened_at"])
                except: pass
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o=ex.fetch_order(oids["tp2"],symbol)
                    if o["status"]=="closed":
                        trade["tp2_hit"]=trade["closed"]=True; pnl=_pnl(trade,float(o["average"]),"tp2")
                        log.info(f"  ✅ TP2 {symbol} {pnl:+.4f}"); _send_close_alert(symbol,"✅ FULL WIN",pnl,entry,float(o["average"]),trade["opened_at"])
                        _record_close(trade,float(o["average"]),pnl,"TP2 hit"); to_rm.append(symbol)
                except: pass
            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o=ex.fetch_order(oids["stop_loss"],symbol)
                    if o["status"]=="closed":
                        trade["closed"]=True; pnl=_pnl(trade,float(o["average"]),"sl")
                        log.info(f"  ❌ SL {symbol} {pnl:+.4f}"); _send_close_alert(symbol,"❌ STOPPED OUT",pnl,entry,float(o["average"]),trade["opened_at"])
                        _record_close(trade,float(o["average"]),pnl,"SL hit")
                        for k in ("tp1","tp2"):
                            if k in oids and not trade.get(f"{k}_hit"):
                                try: ex.cancel_order(oids[k],symbol)
                                except: pass
                        to_rm.append(symbol)
                except: pass
        except: pass
    save_trades(trades)
    for s in set(to_rm): trades.pop(s,None)
    save_trades(trades)

def _pnl(t,cp,ct):
    entry=t["entry"]; qty=t["qty_tp1"] if ct=="tp1" else t["qty_tp2"] if ct=="tp2" else t["qty"]
    return round((cp-entry)*qty if t["signal"]=="BUY" else (entry-cp)*qty,4)

def _record_close(t,cp,pnl,reason):
    append_history({**t,"close_price":cp,"pnl":pnl,"closed_at":datetime.now(timezone.utc).isoformat(),"close_reason":reason})

def generate_signal(symbol,pipeline,thresholds):
    try:
        df_e = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_c = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df_e.empty or len(df_e) < 50: 
            log.info(f"    Not enough data")
            return None
        
        row_c = df_c.iloc[-1] if not df_c.empty else pd.Series(dtype=float)
        
        # ── INJECT 1H FEATURES INTO DF_E BEFORE CHECKING ──
        df_e["rsi_1h"]   = float(row_c.get("rsi", 50))
        df_e["adx_1h"]   = float(row_c.get("adx", 0))
        df_e["trend_1h"] = float(row_c.get("trend", 0))
        
        row_e = df_e.iloc[-1].copy()
        af = pipeline["all_features"]; sel = pipeline["selector"]; ens = pipeline["ensemble"]
        
        miss = [f for f in af if f not in df_e.columns]
        if miss: 
            log.warning(f"    Missing: {miss[:3]}")
            return None
            
        X = pd.DataFrame([row_e[af].values], columns=af); Xs = sel.transform(X)
        pred = ens.predict(Xs)[0]; prob = ens.predict_proba(Xs)[0]
        signal = {0:"BUY",1:"SELL",2:"NO_TRADE"}[pred]
        confidence = round(float(max(prob))*100,1)
        
        log.info(f"    ML → {signal} {confidence:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if signal=="NO_TRADE" or confidence<thresholds["min_confidence"]: return None
        
        adx=float(row_e.get("adx",0))
        log.info(f"    ADX={adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx<thresholds["min_adx"]: return None
        
        score,reasons=_quality_score(row_e,row_c,signal,confidence)
        log.info(f"    Score={score}/6 (need ≥{thresholds['min_score']})")
        
        entry=float(row_e["close"]); atr=float(row_e["atr"])
        stop=round(entry-atr*ATR_STOP_MULT,4) if signal=="BUY" else round(entry+atr*ATR_STOP_MULT,4)
        tp1=round(entry+atr*ATR_TARGET1_MULT,4) if signal=="BUY" else round(entry-atr*ATR_TARGET1_MULT,4)
        tp2=round(entry+atr*ATR_TARGET2_MULT,4) if signal=="BUY" else round(entry-atr*ATR_TARGET2_MULT,4)
        base={"symbol":symbol,"signal":signal,"confidence":confidence,"score":score,
              "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons,"tier":get_tier(symbol)}
        if score<thresholds["min_score"]:
            save_signal({**base,"rejected":True,"reject_reason":f"score {score}<{thresholds['min_score']}"})
            return None
        return base
    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}"); return None

def _quality_score(re,rc,signal,confidence):
    s,r=0,[]
    if confidence>=75: s+=1; r.append(f"High confidence ({confidence:.0f}%)")
    elif confidence>=60: s+=1; r.append(f"Confidence ({confidence:.0f}%)")
    adx=float(re.get("adx",0))
    if adx>25: s+=1; r.append(f"Strong trend ADX {adx:.0f}")
    elif adx>18: s+=1; r.append(f"Moderate trend ADX {adx:.0f}")
    rsi=float(re.get("rsi",50))
    if signal=="BUY" and rsi<45: s+=1; r.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>55: s+=1; r.append(f"RSI bearish ({rsi:.0f})")
    e20,e50=float(re.get("ema20",0)),float(re.get("ema50",0))
    if signal=="BUY" and e20>e50: s+=1; r.append("EMA20>EMA50")
    elif signal=="SELL" and e20<e50: s+=1; r.append("EMA20<EMA50")
    c20,c50=float(rc.get("ema20",0)),float(rc.get("ema50",0))
    if signal=="BUY" and c20>c50: s+=1; r.append("1h confirms")
    elif signal=="SELL" and c20<c50: s+=1; r.append("1h confirms")
    return s,r

def _send(text):
    tok,cid=os.getenv("TELEGRAM_TOKEN",""),os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id":cid,"text":text,"parse_mode":"Markdown"},timeout=10)
    except: pass

def _warn(t): log.warning(t); _send(t)

def check_mode_switch(mode):
    last=load_json(MODE_FILE,{})
    if last.get("mode")!=mode["mode"]:
        msgs={"active":"📈 *Active hours* — conf≥60%","quiet":"🌙 *Quiet hours* — conf≥68%","weekend":"📅 *Weekend* — conf≥65%"}
        _send(msgs.get(mode["mode"],"Mode changed"))
        save_json(MODE_FILE,{"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol,signal,confidence,score,entry,stop,tp1,tp2,qty,risk_usd,balance,reasons):
    emoji="🟢" if signal=="BUY" else "🔴"; stars="⭐"*min(score,5); dec=4 if entry<10 else 2; fp=lambda v:f"{v:,.{dec}f}"
    rlines="\n".join([f"  • {r}" for r in reasons])
    _send(f"🤖 *TESTNET TRADE OPENED*\n━━━━━━━━━━━━━━━━━━━━\n{emoji} *{signal} — {symbol}* {stars}\n"
        f"⚡ *ENTRY:* `{fp(entry)}`\n🛑 *STOP:* `{fp(stop)}`\n🎯 *TP1/TP2:* `{fp(tp1)}` / `{fp(tp2)}`\n"
        f"⚠️ *Risk:* `{risk_usd:.2f} USDT`\n💼 *Bal:* `{balance:.2f} USDT`\n\n📊 *Reasons:*\n{rlines}\n_Binance Testnet_")

def _send_close_alert(symbol,result,pnl,entry,close_price,opened_at):
    emoji="✅" if pnl>0 else "❌"; dec=4 if entry<10 else 2
    _send(f"🤖 *TRADE CLOSED*\n━━━━━━━━━━━━━━━━━━━━\n{emoji} *{result} — {symbol}*\n"
        f"💵 *PnL: `{pnl:+.4f} USDT`*\n_Binance Testnet_")

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\nGitHub Actions US IP — geo-block bypassed\n{'═'*56}")
    run,mode,vol,reason=should_scan()
    check_mode_switch(mode)
    if not run: log.info(f"SKIPPED: {reason}"); return
    vol_warn=vol["message"] if vol.get("warn") else None
    
    # 1. Grab a free proxy and inject it into the environment
    working_proxy = get_working_free_proxy()
    if working_proxy:
        os.environ["PROXY_URL"] = working_proxy

    # 2. Init exchange and FORCE balance check BEFORE anything else
    try:
        exchange = init_exchange()
        get_balance_usdt(exchange)
    except Exception as e:
        log.error(f"Exchange init failed: {e}")
        return
        
    pipeline=load_model(); thresholds=get_mode_thresholds(mode)
    
    log.info("\n[1] Checking open trades...")
    check_open_trades(exchange)
    
    trades=load_trades()
    log.info(f"\n[2] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES} | conf≥{thresholds['min_confidence']}% score≥{thresholds['min_score']} ADX≥{thresholds['min_adx']}")
    
    found=0
    for symbol in SYMBOLS:
        if len(load_trades())>=MAX_OPEN_TRADES: log.info("Max trades reached"); break
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig=generate_signal(symbol,pipeline,thresholds)
        if sig is None: time.sleep(0.4); continue
        found+=1
        if vol_warn: sig["reasons"]=list(sig.get("reasons",[]))+[f"⚠️ {vol_warn}"]
        execute_trade(exchange,sig["symbol"],sig["signal"],sig["entry"],sig["atr"],sig["confidence"],sig["score"],sig["reasons"])
        time.sleep(1)
    
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) found\n{'═'*56}\n")

    # 👈 THE BRIDGE: Pushing data to GitHub so Render can read the balance
    if PERSISTENCE_ENABLED:
        log.info("Pushing data to GitHub so Dashboard can see it...")
        time.sleep(2)
        sync_all_to_github()

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="diagnostic": 
        # Added quick diagnostic if needed
        pass
    else: 
        run_execution_scan()
