# smart_scheduler.py — v3: BTC Momentum Bias (S3) + Time-of-Day Risk Weighting (S4)

import logging, requests, json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_DEAD_PCT  = 0.05


# ── Scenario 4: Time-of-day risk multiplier ───────────────────────────
# London open 09:00–12:00 UTC and US overlap 13:00–17:00 UTC have the
# highest volume and cleanest trend completions.  Size up during these
# windows, size down during thin overnight hours.

def _get_time_risk_mult() -> float:
    hour = datetime.now(timezone.utc).hour
    if 9 <= hour < 12:
        return 1.2   # London open — peak liquidity
    elif 13 <= hour < 17:
        return 1.2   # US/London overlap — strongest trends
    elif 8 <= hour < 20:
        return 1.0   # Active but not peak
    elif 20 <= hour < 22 or 6 <= hour < 8:
        return 0.75  # Shoulder hours — reduce slightly
    else:
        return 0.5   # Overnight thin market


def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    is_active  = 8 <= hour < 20
    time_mult  = _get_time_risk_mult()   # S4: per-hour multiplier

    if is_weekend:
        if is_active:
            return {
                "mode": "weekend_active", "label": "WEEKEND MODE", "emoji": "📅",
                "min_confidence": 55, "min_score": 3, "min_adx": 18,
                "interval_min": 15, "risk_mult": round(0.85 * time_mult, 3),
            }
        else:
            return {
                "mode": "weekend_quiet", "label": "WEEKEND QUIET", "emoji": "🌙",
                "min_confidence": 60, "min_score": 4, "min_adx": 22,
                "interval_min": 30, "risk_mult": round(0.50 * time_mult, 3),
            }

    if is_active:
        return {
            "mode": "active", "label": "ACTIVE HOURS", "emoji": "📈",
            "min_confidence": 55,   # lowered: 60→50 to let more signals through
            "min_score": 3, "min_adx": 15,
            "interval_min": 15, "risk_mult": round(1.0 * time_mult, 3),
        }
    return {
        "mode": "quiet", "label": "QUIET HOURS", "emoji": "🌙",
        "min_confidence": 60,   # lowered: 65→60
        "min_score": 3, "min_adx": 18,
        "interval_min": 30, "risk_mult": round(0.5 * time_mult, 3),
    }


# ── Scenario 3: BTC momentum pre-filter ──────────────────────────────
# When BTC makes a strong 15m or 3-candle directional move, alt coins
# follow within 1–3 bars.  Use this as a free directional bias filter:
#   - Strong BTC move UP   → favour BUY signals, penalise SELL
#   - Strong BTC move DOWN → favour SELL signals, penalise BUY
#   - Neutral BTC          → scan all directions normally
#
# Returns a dict consumed by generate_signal() in trade_executor.py.

BTC_MOMENTUM_THRESHOLD  = 1.5   # % single candle to call "strong"
BTC_THREE_CANDLE_THRESH = 2.0   # % over 3 candles to call "trending"


def check_btc_momentum() -> dict:
    """
    Fetches last 5 BTC 15m candles from Binance and returns directional bias.
    Called once per scan in run_execution_scan() — not per-symbol.

    Return schema:
        bias     : "BUY" | "SELL" | None
        strength : "strong" | "moderate" | "neutral"
        pct_move : last-candle % move (signed)
        score_mod: +1 (aligned) or -1 (counter) applied inside generate_signal()
        message  : human-readable log string
    """
    _NEUTRAL = {
        "bias": None, "strength": "neutral",
        "pct_move": 0.0, "score_mod": 0,
        "message": "➡️ BTC neutral — scanning all directions"
    }

    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 5},
            timeout=10
        )
        if r.status_code != 200:
            return _NEUTRAL

        data   = r.json()
        closes = [float(d[4]) for d in data]

        last_pct        = (closes[-1] - closes[-2]) / closes[-2] * 100
        three_candle_pct = (closes[-1] - closes[-3]) / closes[-3] * 100

        if last_pct > BTC_MOMENTUM_THRESHOLD or three_candle_pct > BTC_THREE_CANDLE_THRESH:
            strength = "strong" if last_pct > 2.0 else "moderate"
            return {
                "bias":     "BUY",
                "strength": strength,
                "pct_move": round(last_pct, 3),
                "score_mod": 1,   # +1 score for aligned BUY signals
                "message":  f"📈 BTC +{last_pct:.2f}% ({strength}) — BUY bias"
            }

        elif last_pct < -BTC_MOMENTUM_THRESHOLD or three_candle_pct < -BTC_THREE_CANDLE_THRESH:
            strength = "strong" if last_pct < -2.0 else "moderate"
            return {
                "bias":     "SELL",
                "strength": strength,
                "pct_move": round(last_pct, 3),
                "score_mod": 1,   # +1 score for aligned SELL signals
                "message":  f"📉 BTC {last_pct:.2f}% ({strength}) — SELL bias"
            }

        return {**_NEUTRAL, "pct_move": round(last_pct, 3),
                "message": f"➡️ BTC {last_pct:+.2f}% — neutral, scanning all"}

    except Exception as e:
        log.warning(f"BTC momentum check failed: {e}")
        return _NEUTRAL


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
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])
        prev_c   = df["close"].shift(1)
        df["tr"] = pd.concat([
            df["high"]-df["low"],
            (df["high"]-prev_c).abs(),
            (df["low"]-prev_c).abs()
        ], axis=1).max(axis=1)
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
        elif atr_pct < ATR_DEAD_PCT:
            return {"status":"DEAD","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":True,"warn":False,"risk_mult":0.0,
                    "message":f"😴 Dead market {atr_pct:.2f}% — skip scan"}
        else:
            return {"status":"NORMAL","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":False,"risk_mult":1.0,
                    "message":f"✓ Normal vol — BTC ATR {atr_pct:.2f}%"}
    except Exception as e:
        log.warning(f"Volatility check failed: {e}")
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"price":0,
                "skip":False,"warn":False,"risk_mult":1.0,
                "message":"Vol check failed — scanning anyway"}


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

        today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at","") or h.get("opened_at",""))[:10] == today
            and "Ghost" not in (h.get("close_reason",""))
            and "auto-removed" not in (h.get("close_reason",""))
        )
        limit = current_balance * 5.0 / 100
        if today_pl < -limit:
            msg = (f"⚠️ Daily loss advisory: {today_pl:.2f} USDT today "
                   f"(>{5.0}% of ${current_balance:.0f}) — consider manual review")
            log.warning(msg); return msg
        return ""
    except Exception:
        return ""


def check_correlation(trades: dict, new_signal: str) -> bool:
    try:
        from config import MAX_SAME_DIRECTION
    except ImportError:
        MAX_SAME_DIRECTION = 2

    same = sum(1 for t in trades.values()
               if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= MAX_SAME_DIRECTION:
        log.info(f"  Correlation filter: {same} {new_signal} already open — skip")
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
    if vol.get("skip"):
        return False, mode, vol, vol["message"]
    advisory = check_daily_pnl_advisory()
    if advisory: log.warning(advisory)
    return True, mode, vol, f"{mode['label']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    # S4: time-of-day is already baked into mode["risk_mult"]
    # Vol multiplier is applied on top
    combined = mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0)
    return max(combined, 0.25)
