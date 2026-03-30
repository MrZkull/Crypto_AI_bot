# smart_scheduler.py — Optimized version
# Fixes:
#   1. Quiet hours threshold corrected to match dashboard display
#   2. High volatility reduces position size (not just warns)
#   3. Correlation filter added — max 2 same-direction trades
#   4. Better mode descriptions

import logging
import requests
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Volatility thresholds ─────────────────────────────────────
ATR_HIGH_PCT     = 2.0   # above 2% = high volatility — reduce position size
ATR_VERY_HIGH    = 4.0   # above 4% = very high — warn strongly
ATR_LOW_PCT      = 0.1   # below 0.1% = dead market — skip scan


def get_scan_mode() -> dict:
    """Returns scan mode based on UTC time and day of week."""
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    weekday    = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
    is_weekend = weekday >= 5
    is_active  = (8 <= hour < 20)

    if is_weekend:
        return {
            "mode":           "weekend",
            "label":          "WEEKEND MODE",
            "emoji":          "📅",
            "min_confidence": 65,
            "min_score":      3,
            "min_adx":        20,
            "interval_min":   15 if is_active else 30,
            "risk_mult":      0.75,  # 75% of normal risk on weekends
            "description":    "Weekend — confidence raised to 65%, risk reduced",
        }

    if is_active:
        return {
            "mode":           "active",
            "label":          "ACTIVE HOURS",
            "emoji":          "📈",
            "min_confidence": 65,    # FIXED: raised from 60 to 65
            "min_score":      3,     # FIXED: raised from 2 to 3
            "min_adx":        20,    # FIXED: raised from 18 to 20
            "interval_min":   15,
            "risk_mult":      1.0,   # full risk
            "description":    "Active hours 08:00-20:00 UTC",
        }

    # Quiet hours — FIXED: now matches dashboard display
    return {
        "mode":           "quiet",
        "label":          "QUIET HOURS",
        "emoji":          "🌙",
        "min_confidence": 72,   # FIXED: matches dashboard
        "min_score":      4,    # FIXED: raised from 3 to 4
        "min_adx":        25,   # FIXED: raised to filter weak signals
        "interval_min":   30,
        "risk_mult":      0.5,  # 50% risk during quiet hours
        "description":    "Quiet hours 00:00-08:00 UTC — higher thresholds",
    }


def check_btc_volatility() -> dict:
    """Fetch BTC 15m ATR and classify volatility level."""
    try:
        url    = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 30}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])

        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0
        ), axis=1)

        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_VERY_HIGH:
            return {
                "status":       "VERY_HIGH",
                "atr":          round(atr, 2),
                "atr_pct":      round(atr_pct, 3),
                "price":        round(price, 2),
                "skip":         False,
                "warn":         True,
                "risk_mult":    0.25,  # reduce to 25% position size
                "message":      f"🚨 EXTREME VOLATILITY — BTC ATR {atr_pct:.2f}% — position size reduced to 25%",
            }
        elif atr_pct > ATR_HIGH_PCT:
            return {
                "status":       "HIGH",
                "atr":          round(atr, 2),
                "atr_pct":      round(atr_pct, 3),
                "price":        round(price, 2),
                "skip":         False,
                "warn":         True,
                "risk_mult":    0.5,   # reduce to 50% position size
                "message":      f"⚠️ HIGH VOLATILITY — BTC ATR {atr_pct:.2f}% — position size reduced to 50%",
            }
        elif atr_pct < ATR_LOW_PCT:
            return {
                "status":       "LOW",
                "atr":          round(atr, 2),
                "atr_pct":      round(atr_pct, 3),
                "price":        round(price, 2),
                "skip":         True,
                "warn":         False,
                "risk_mult":    0.0,
                "message":      f"😴 Market dead — BTC ATR {atr_pct:.2f}% — scan skipped",
            }
        else:
            return {
                "status":       "NORMAL",
                "atr":          round(atr, 2),
                "atr_pct":      round(atr_pct, 3),
                "price":        round(price, 2),
                "skip":         False,
                "warn":         False,
                "risk_mult":    1.0,
                "message":      f"✓ Normal vol — BTC ATR {atr_pct:.2f}%",
            }

    except Exception as e:
        log.warning(f"  Volatility check failed: {e} — defaulting to NORMAL")
        return {
            "status":   "UNKNOWN", "atr": 0, "atr_pct": 0, "price": 0,
            "skip":     False, "warn": False, "risk_mult": 1.0,
            "message":  "Volatility check failed — scan running anyway",
        }


def check_correlation(trades: dict, new_signal: str) -> bool:
    """
    Returns True if OK to trade, False if correlation limit reached.
    Max 2 trades in the same direction (BUY or SELL) at once.
    This prevents all-or-nothing scenarios.
    """
    if not trades:
        return True

    same_dir = sum(
        1 for t in trades.values()
        if t.get("signal") == new_signal
        and not t.get("closed", False)
    )

    if same_dir >= 2:
        log.info(f"  Correlation filter: already {same_dir} {new_signal} trades open — skip")
        return False

    return True


def should_scan() -> tuple:
    """
    Returns (should_run, mode, vol, reason).
    Skips only if ATR truly dead (< 0.1%).
    """
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(
        f"  Mode: {mode['label']} | conf≥{mode['min_confidence']}% "
        f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} "
        f"| risk_mult:{mode['risk_mult']} | {vol['message']}"
    )

    if vol["skip"]:
        return False, mode, vol, vol["message"]

    return True, mode, vol, f"{mode['label']} — ATR {vol['status']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    """
    Calculate effective risk multiplier combining mode + volatility.
    Never goes below 0.25 (25% of normal) to ensure some trading.
    Example: active mode (1.0) + high vol (0.5) = 0.5 risk
    Example: weekend mode (0.75) + high vol (0.5) = 0.375 risk
    """
    mode_mult = mode.get("risk_mult", 1.0)
    vol_mult  = vol.get("risk_mult",  1.0)
    effective = mode_mult * vol_mult
    return max(effective, 0.25)
