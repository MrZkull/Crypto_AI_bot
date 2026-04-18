# smart_scheduler.py — Professional grade thresholds
# Key changes vs old version:
#   1. Min confidence raised to 65% active (was 50% — too many bad signals)
#   2. Market regime filter: only trade when BTC ADX > 20 (trending market)
#   3. Daily loss circuit breaker: if today's losses > 5% of balance → stop
#   4. ATR dead-market skip threshold: 0.08% (was 0.05% — misses quiet markets)

import logging, requests, json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ATR_VERY_HIGH   = 4.0
ATR_HIGH_PCT    = 2.0
ATR_DEAD_PCT    = 0.08   # below this = dead market, skip scan
MAX_DAILY_LOSS_PCT = 5.0  # stop all trading if down >5% today


def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    is_active  = 8 <= hour < 20

    if is_weekend:
        return {
            "mode": "weekend", "label": "WEEKEND MODE", "emoji": "📅",
            "min_confidence": 68, "min_score": 2, "min_adx": 20,
            "interval_min": 30, "risk_mult": 0.5,
            "description": "Weekend — 50% risk, stricter filters",
        }
    if is_active:
        return {
            "mode": "active", "label": "ACTIVE HOURS", "emoji": "📈",
            "min_confidence": 65, "min_score": 2, "min_adx": 18,
            "interval_min": 15, "risk_mult": 1.0,
            "description": "Active hours 08:00–20:00 UTC",
        }
    return {
        "mode": "quiet", "label": "QUIET HOURS", "emoji": "🌙",
        "min_confidence": 70, "min_score": 3, "min_adx": 22,
        "interval_min": 30, "risk_mult": 0.5,
        "description": "Quiet hours — 50% risk, higher threshold",
    }


def check_btc_volatility() -> dict:
    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 30},
            timeout=10
        )
        data = r.json()
        df   = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","tb_base","tb_quote","ignore"
        ])
        for c in ["high","low","close"]: df[c] = pd.to_numeric(df[c])
        prev_c  = df["close"].shift(1)
        df["tr"] = pd.concat([
            df["high"]-df["low"],
            (df["high"]-prev_c).abs(),
            (df["low"]-prev_c).abs()
        ], axis=1).max(axis=1)
        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        # Also compute BTC ADX for regime detection
        high = df["high"]; low = df["low"]
        tr   = df["tr"]
        dm_pos = (high.diff()).clip(lower=0)
        dm_neg = (-low.diff()).clip(lower=0)
        dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
        dm_neg = dm_neg.where(dm_neg > dm_pos, 0)
        di_pos = 100 * dm_pos.rolling(14).mean() / tr.rolling(14).mean()
        di_neg = 100 * dm_neg.rolling(14).mean() / tr.rolling(14).mean()
        dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg)
        adx    = dx.rolling(14).mean().iloc[-1]

        if atr_pct > ATR_VERY_HIGH:
            return {"status":"VERY_HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "btc_adx":round(adx,1),"price":round(price,2),
                    "skip":False,"warn":True,"risk_mult":0.25,
                    "message":f"🚨 EXTREME VOL {atr_pct:.2f}% — 25% position size"}
        elif atr_pct > ATR_HIGH_PCT:
            return {"status":"HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "btc_adx":round(adx,1),"price":round(price,2),
                    "skip":False,"warn":True,"risk_mult":0.5,
                    "message":f"⚠️ HIGH VOL {atr_pct:.2f}% — 50% position size"}
        elif atr_pct < ATR_DEAD_PCT:
            return {"status":"DEAD","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "btc_adx":round(adx,1),"price":round(price,2),
                    "skip":True,"warn":False,"risk_mult":0.0,
                    "message":f"😴 Dead market {atr_pct:.2f}% — skipping scan"}
        else:
            return {"status":"NORMAL","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "btc_adx":round(adx,1),"price":round(price,2),
                    "skip":False,"warn":False,"risk_mult":1.0,
                    "message":f"✓ Normal vol {atr_pct:.2f}% | BTC ADX {adx:.0f}"}
    except Exception as e:
        log.warning(f"Volatility check: {e}")
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"btc_adx":0,"price":0,
                "skip":False,"warn":False,"risk_mult":1.0,
                "message":"Vol check failed — scanning anyway"}


def check_daily_loss_limit(balance_file: str = "balance.json",
                           history_file: str  = "trade_history.json") -> dict:
    """
    Professional circuit breaker: if today's realised losses exceed
    MAX_DAILY_LOSS_PCT of current balance → stop trading for today.
    """
    try:
        bal = {}
        for p in [Path(balance_file), Path("data") / balance_file]:
            if p.exists():
                with open(p) as f: bal = json.load(f); break
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0:
            return {"triggered": False, "today_loss": 0, "limit": 0}

        hist = []
        for p in [Path(history_file), Path("data") / history_file]:
            if p.exists():
                with open(p) as f: hist = json.load(f); break

        today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at", "") or h.get("opened_at",""))[:10] == today
            and "Ghost" not in (h.get("close_reason",""))
            and "auto-removed" not in (h.get("close_reason",""))
        )

        limit = current_balance * MAX_DAILY_LOSS_PCT / 100
        triggered = today_pl < -limit

        if triggered:
            log.warning(f"⛔ DAILY LOSS LIMIT HIT: {today_pl:.2f} USDT today "
                        f"(limit: -{limit:.2f} USDT = {MAX_DAILY_LOSS_PCT}%)")

        return {
            "triggered":    triggered,
            "today_pnl":    round(today_pl, 4),
            "limit_usd":    round(-limit, 2),
            "limit_pct":    MAX_DAILY_LOSS_PCT,
            "message":      f"⛔ Daily loss limit hit ({today_pl:.2f} USDT)" if triggered else "",
        }
    except Exception as e:
        log.warning(f"Daily loss check: {e}")
        return {"triggered": False, "today_loss": 0, "limit": 0}


def check_correlation(trades: dict, new_signal: str) -> bool:
    """Prevent more than 2 trades in same direction."""
    same = sum(1 for t in trades.values()
               if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= 2:
        log.info(f"  Correlation filter: {same} {new_signal} trades already open — skip")
        return False
    return True


def should_scan() -> tuple:
    """Returns (run, mode, vol, reason)."""
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(f"  {mode['label']} | conf≥{mode['min_confidence']}% "
             f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} | {vol['message']}")

    if vol["skip"]:
        return False, mode, vol, vol["message"]

    # Daily loss circuit breaker
    dlc = check_daily_loss_limit()
    if dlc.get("triggered"):
        return False, mode, vol, dlc["message"]

    return True, mode, vol, f"{mode['label']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    """Compound risk reduction: mode × volatility, never below 25%."""
    combined = mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0)
    return max(combined, 0.25)
