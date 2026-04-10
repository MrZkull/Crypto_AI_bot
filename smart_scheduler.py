# smart_scheduler.py — Much lower thresholds to actually get trades
# Previous: conf≥55% active — too high for 73.1% model in quiet markets
# Now: conf≥50% active — ML model is good, let it trade

import logging, requests, pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)
ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_LOW_PCT   = 0.05


def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    is_active  = 8 <= hour < 20

    if is_weekend:
        return {"mode":"weekend","label":"WEEKEND MODE","emoji":"📅",
                "min_confidence":50,"min_score":1,"min_adx":15,
                "interval_min":15,"risk_mult":0.75,"description":"Weekend"}
    if is_active:
        return {"mode":"active","label":"ACTIVE HOURS","emoji":"📈",
                "min_confidence":50,"min_score":1,"min_adx":15,
                "interval_min":15,"risk_mult":1.0,"description":"Active 08–20 UTC"}
    return {"mode":"quiet","label":"QUIET HOURS","emoji":"🌙",
            "min_confidence":55,"min_score":2,"min_adx":18,
            "interval_min":30,"risk_mult":0.5,"description":"Quiet 00–08 UTC"}


def check_btc_volatility() -> dict:
    try:
        r      = requests.get("https://data-api.binance.vision/api/v3/klines",
                              params={"symbol":"BTCUSDT","interval":"15m","limit":30},
                              timeout=10)
        data   = r.json()
        df     = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","tb_base","tb_quote","ignore"])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])
        prev_c = df["close"].shift(1)
        df["tr"] = pd.concat([df["high"]-df["low"],
            (df["high"]-prev_c).abs(),(df["low"]-prev_c).abs()],axis=1).max(axis=1)
        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_VERY_HIGH:
            return {"status":"VERY_HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.25,
                    "message":f"🚨 EXTREME VOL {atr_pct:.2f}% — 25% position"}
        elif atr_pct > ATR_HIGH_PCT:
            return {"status":"HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.5,
                    "message":f"⚠️ HIGH VOL {atr_pct:.2f}% — 50% position"}
        elif atr_pct < ATR_LOW_PCT:
            return {"status":"LOW","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":True,"warn":False,"risk_mult":0.0,
                    "message":f"😴 Dead market {atr_pct:.2f}% — skip"}
        else:
            return {"status":"NORMAL","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":False,"risk_mult":1.0,
                    "message":f"✓ Normal vol {atr_pct:.2f}%"}
    except Exception as e:
        log.warning(f"Volatility check failed: {e}")
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"price":0,
                "skip":False,"warn":False,"risk_mult":1.0,"message":"Vol check failed"}


def check_correlation(trades: dict, new_signal: str) -> bool:
    same = sum(1 for t in trades.values()
               if t.get("signal")==new_signal and not t.get("closed",False))
    if same >= 2:
        log.info(f"  Correlation filter: {same} {new_signal} already open — skip")
        return False
    return True


def should_scan() -> tuple:
    mode = get_scan_mode()
    vol  = check_btc_volatility()
    log.info(f"  {mode['label']} | conf≥{mode['min_confidence']}% "
             f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} | {vol['message']}")
    if vol["skip"]:
        return False, mode, vol, vol["message"]
    return True, mode, vol, f"{mode['label']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    return max(mode.get("risk_mult",1.0) * vol.get("risk_mult",1.0), 0.25)
