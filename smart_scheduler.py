# smart_scheduler.py — Capital Protection & Sentiment Filters Integrated

import logging, requests, json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_DEAD_PCT  = 0.05

def _get_time_risk_mult() -> float:
    """Returns 1.2 during peak hours, 1.0 otherwise."""
    hour = datetime.now(timezone.utc).hour
    if (9 <= hour < 12) or (13 <= hour < 17): return 1.2
    return 1.0

def get_drawdown_ratchet() -> float:
    """Calculates daily PnL and ratchets down risk dynamically."""
    try:
        bal, hist = {}, []
        for p in [Path("balance.json"), Path("data/balance.json")]:
            if p.exists():
                with open(p) as f: bal = json.load(f); break
        for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
            if p.exists():
                with open(p) as f: hist = json.load(f); break
                
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0: return 1.0

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at","") or h.get("opened_at",""))[:10] == today
            and "Ghost" not in h.get("close_reason","")
        )
        
        drawdown_pct = (today_pl / current_balance) * 100
        
        if drawdown_pct <= -5.0:
            log.warning(f"🚨 MAX DRAWDOWN ({drawdown_pct:.1f}%) — Trading Halted")
            return 0.0
        elif drawdown_pct <= -2.0:
            log.warning(f"⚠️ HIGH DRAWDOWN ({drawdown_pct:.1f}%) — Risk Halved")
            return 0.5
        return 1.0
    except Exception:
        return 1.0

def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    time_mult  = _get_time_risk_mult()

    is_active = 8 <= hour < 20
    if is_weekend:
        return {
            "mode": "weekend_active" if is_active else "weekend_quiet", 
            "label": "WEEKEND ACTIVE" if is_active else "WEEKEND QUIET", "emoji": "📅",
            "min_confidence": 50,  # FIXED: Matched to active 45% standard
            "min_score": 3 if is_active else 4, 
            "min_adx": 18 if is_active else 22,
            "interval_min": 15 if is_active else 30, 
            "risk_mult": round((0.85 if is_active else 0.50) * time_mult, 3),
        }

    if is_active:
        return {
            "mode": "active", "label": "ACTIVE HOURS", "emoji": "📈",
            "min_confidence": 45, "min_score": 3, "min_adx": 15,
            "interval_min": 15, "risk_mult": round(1.0 * time_mult, 3),
        }
    return {
        "mode": "quiet", "label": "QUIET HOURS", "emoji": "🌙",
        "min_confidence": 50, "min_score": 3, "min_adx": 18,
        "interval_min": 30, "risk_mult": round(0.5 * time_mult, 3),
    }

def check_fear_and_greed() -> dict:
    """Fetches F&G for both Score Modification AND Hard Regime Blocks."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        val = int(r.json()["data"][0]["value"])
        
        # Original scoring logic (Required by trade_executor for scoring)
        if val >= 75:
            bias, score_mod, msg, label = "SELL", 1, f"Extreme Greed ({val}) — SELL Bias", "Extreme Greed"
        elif val <= 25:
            bias, score_mod, msg, label = "BUY", 1, f"Extreme Fear ({val}) — BUY Bias", "Extreme Fear"
        else:
            bias, score_mod, msg, label = None, 0, f"Neutral ({val})", "Neutral"

        return {
            # Old Keys
            "bias": bias, 
            "score_mod": score_mod, 
            "message": msg,
            # New Keys (For Hard Blocks)
            "value": val,
            "fg_blocks_sell": val <= 20,
            "fg_blocks_buy": val >= 70,
            "label": label
        }
    except Exception as e:
        log.warning(f"F&G fetch failed ({e}) — defaulting neutral")
        return {
            "bias": None, "score_mod": 0, "message": "F&G fetch failed",
            "value": 50, "fg_blocks_sell": False, "fg_blocks_buy": False, "label": "Neutral"
        }

def check_btc_momentum() -> dict:
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "15m", "limit": 2}, timeout=10)
        data = r.json()
        if len(data) >= 2:
            prev_close, curr_close = float(data[0][4]), float(data[1][4])
            pct_change = ((curr_close - prev_close) / prev_close) * 100
            if pct_change >= 1.5: return {"bias": "BUY", "score_mod": 1, "strength": "strong", "message": f"BTC up {pct_change:.2f}%"}
            if pct_change <= -1.5: return {"bias": "SELL", "score_mod": 1, "strength": "strong", "message": f"BTC down {pct_change:.2f}%"}
        return {"bias": None, "score_mod": 0, "strength": "neutral", "message": "BTC neutral"}
    except Exception: return {"bias": None, "score_mod": 0, "strength": "unknown", "message": "BTC momentum unknown"}

def check_btc_volatility() -> dict:
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "15m", "limit": 30}, timeout=10)
        df = pd.DataFrame(r.json(), columns=["open_time","open","high","low","close","volume","close_time","quote_vol","trades","tb_base","tb_quote","ignore"])
        for c in ["high","low","close"]: df[c] = pd.to_numeric(df[c])
        prev_c = df["close"].shift(1)
        df["tr"] = pd.concat([df["high"]-df["low"], (df["high"]-prev_c).abs(), (df["low"]-prev_c).abs()], axis=1).max(axis=1)
        atr = df["tr"].rolling(14).mean().iloc[-1]
        price = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_VERY_HIGH: return {"status":"VERY_HIGH","risk_mult":0.25,"skip":False,"message":f"🚨 EXTREME VOL {atr_pct:.2f}%"}
        if atr_pct > ATR_HIGH_PCT: return {"status":"HIGH","risk_mult":0.5,"skip":False,"message":f"⚠️ HIGH VOL {atr_pct:.2f}%"}
        if atr_pct < ATR_DEAD_PCT: return {"status":"DEAD","risk_mult":0.0,"skip":True,"message":f"😴 Dead market {atr_pct:.2f}%"}
        return {"status":"NORMAL","risk_mult":1.0,"skip":False,"message":f"✓ Normal BTC ATR {atr_pct:.2f}%"}
    except Exception: return {"status":"UNKNOWN","risk_mult":1.0,"skip":False,"message":"Vol check failed"}

def check_daily_pnl_advisory() -> str:
    try:
        bal = {}
        for p in [Path("balance.json"), Path("data/balance.json")]:
            if p.exists():
                with open(p) as f: bal = json.load(f); break
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0: return ""
        hist = []
        for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
            if p.exists():
                with open(p) as f: hist = json.load(f); break
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(float(h.get("pnl", 0) or 0) for h in hist if (h.get("closed_at","") or h.get("opened_at",""))[:10] == today and "Ghost" not in h.get("close_reason","") and "auto-removed" not in h.get("close_reason",""))
        if today_pl < - (current_balance * 0.05): return f"⚠️ Daily loss advisory: {today_pl:.2f} USDT"
        return ""
    except Exception: return ""

def check_correlation(trades: dict, new_signal: str, new_symbol: str) -> bool:
    try:
        from config import MAX_SAME_DIRECTION
    except ImportError:
        MAX_SAME_DIRECTION = 2

    # 1. Global Direction Cap
    same = sum(1 for t in trades.values()
               if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= MAX_SAME_DIRECTION:
        log.info(f"  Correlation filter: {same} {new_signal} already open — skip")
        return False

    # 2. Sector Correlation Cap
    SECTOR_MAP = {
        "L1": ["SOLUSDT","AVAXUSDT","NEARUSDT","APTUSDT","SUIUSDT","MATICUSDT","TRXUSDT","ATOMUSDT"],
        "DeFi": ["UNIUSDT","AAVEUSDT","LINKUSDT"],
        "BTC_family": ["BTCUSDT","LTCUSDT","BCHUSDT"],
    }
    
    for sector, coins in SECTOR_MAP.items():
        if new_symbol in coins:
            sector_open = sum(1 for t in trades.values()
                              if t.get("symbol") in coins and not t.get("closed", False))
            if sector_open >= 2:
                log.info(f"  Sector cap: {sector} already has {sector_open} open — skip {new_symbol}")
                return False

    return True

def should_scan() -> tuple:
    log.info(f"  Scan triggered at {datetime.now(timezone.utc).strftime('%H:%M UTC')} "
             f"| UTC hour={datetime.now(timezone.utc).hour}")
    mode = get_scan_mode()
    vol  = check_btc_volatility()
    log.info(f"  {mode['label']} | conf≥{mode['min_confidence']}% "
             f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} "
             f"| risk_mult={mode['risk_mult']} | {vol['message']}")
             
    if vol.get("skip") or mode.get("risk_mult", 1.0) == 0.0:
        skip_reason = vol["message"] if vol.get("skip") else mode["label"]
        return False, mode, vol, skip_reason
        
    advisory = check_daily_pnl_advisory()
    if advisory: log.warning(advisory)
    
    return True, mode, vol, f"{mode['label']}"

def get_mode_thresholds(mode: dict) -> dict:
    return {"min_confidence": mode["min_confidence"], "min_score": mode["min_score"], "min_adx": mode["min_adx"], "risk_mult": mode.get("risk_mult", 1.0)}

def get_effective_risk(mode: dict, vol: dict) -> float:
    ratchet = get_drawdown_ratchet()
    return max(mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0) * ratchet, 0.25) if ratchet > 0 else 0.0
            
