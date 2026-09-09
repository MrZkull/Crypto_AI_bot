# smart_scheduler.py — Capital Protection, Dynamic Config Sync & Sentiment Filters

import logging
import requests
import json
import importlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)

ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_DEAD_PCT  = 0.05


def _get_live_config():
    """Dynamically reads parameters directly from config.py on every scan."""
    try:
        import config
        importlib.reload(config)
        return {
            "min_confidence": float(getattr(config, "MIN_CONFIDENCE", 50.0)),
            "min_score": int(getattr(config, "MIN_SCORE", 3)),
            "min_adx": float(getattr(config, "MIN_ADX", 15.0)),
            "max_same_direction": int(getattr(config, "MAX_SAME_DIRECTION", 4)),
            "max_open_trades": int(getattr(config, "MAX_OPEN_TRADES", 8)),
        }
    except Exception:
        return {
            "min_confidence": 50.0,
            "min_score": 3,
            "min_adx": 15.0,
            "max_same_direction": 4,
            "max_open_trades": 8,
        }


def _get_time_risk_mult() -> float:
    hour = datetime.now(timezone.utc).hour
    if (9 <= hour < 12) or (13 <= hour < 17):
        return 1.2
    return 1.0


def _read_balance_and_history():
    """Shared reader — was duplicated between get_drawdown_ratchet() and
    check_daily_pnl_advisory(); consolidated so both stay in sync."""
    bal, hist = {}, []
    for p in [Path("balance.json"), Path("data/balance.json")]:
        if p.exists():
            with open(p) as f:
                bal = json.load(f)
                break
    for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
        if p.exists():
            with open(p) as f:
                hist = json.load(f)
                break
    return bal, hist


def get_drawdown_ratchet() -> float:
    try:
        bal, hist = _read_balance_and_history()
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0:
            return 1.0

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at", "") or h.get("opened_at", ""))[:10] == today
            and "Ghost" not in h.get("close_reason", "")
        )

        # NOTE: this denominator is TODAY'S balance (already net of today's
        # PnL), not the balance the day started with. That makes the ratchet
        # self-reinforcing as losses accumulate — each subsequent dollar lost
        # counts for a larger % against an already-shrunken base. At small
        # account sizes this trips the halt/half-risk thresholds off very
        # small absolute losses. Directionally conservative (never dangerous
        # the way it fails), but worth switching to a tracked day-start
        # balance if you want the % to mean what it says, especially once
        # running at the ~$10 scale discussed separately.
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
    cfg        = _get_live_config()

    is_active = 8 <= hour < 20
    base_conf = cfg["min_confidence"]
    base_score = cfg["min_score"]
    base_adx = cfg["min_adx"]

    if is_weekend:
        return {
            "mode": "weekend_active" if is_active else "weekend_quiet",
            "label": "WEEKEND ACTIVE" if is_active else "WEEKEND QUIET",
            "emoji": "📅",
            "min_confidence": base_conf,
            "min_score": base_score if is_active else (base_score + 1),
            "min_adx": (base_adx + 3) if is_active else (base_adx + 7),
            "interval_min": 15 if is_active else 30,
            "risk_mult": round((0.85 if is_active else 0.50) * time_mult, 3),
        }

    if is_active:
        return {
            "mode": "active",
            "label": "ACTIVE HOURS",
            "emoji": "📈",
            "min_confidence": base_conf,
            "min_score": base_score,
            "min_adx": base_adx,
            "interval_min": 15,
            "risk_mult": round(1.0 * time_mult, 3),
        }
    return {
        "mode": "quiet",
        "label": "QUIET HOURS",
        "emoji": "🌙",
        "min_confidence": base_conf + 5.0,
        "min_score": base_score,
        "min_adx": base_adx + 3.0,
        "interval_min": 30,
        "risk_mult": round(0.5 * time_mult, 3),
    }


def check_fear_and_greed() -> dict:
    """Fetches F&G for UI labeling, scoring bias, and extreme exhaustion blocks."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        val = int(r.json()["data"][0]["value"])

        if val >= 75:
            bias, score_mod, label = "SELL", 1, "Extreme Greed"
            msg = f"Extreme Greed ({val}) — SELL Bias"
        elif val >= 56:
            bias, score_mod, label = None, 0, "Greed"
            msg = f"Greed ({val})"
        elif val <= 24:
            bias, score_mod, label = "BUY", 1, "Extreme Fear"
            msg = f"Extreme Fear ({val}) — BUY Bias"
        elif val <= 44:
            bias, score_mod, label = None, 0, "Fear"
            msg = f"Fear ({val})"
        else:
            bias, score_mod, label = None, 0, "Neutral"
            msg = f"Neutral ({val})"

        return {
            "bias": bias,
            "score_mod": score_mod,
            "message": msg,
            "value": val,
            "label": label,
            "fg_blocks_sell": val <= 15,
            "fg_blocks_buy":  val >= 85,
        }
    except Exception as e:
        log.warning(f"F&G fetch failed ({e}) — defaulting neutral")
        return {
            "bias": None, "score_mod": 0, "message": "F&G fetch failed",
            "value": 50, "label": "Neutral", "fg_blocks_sell": False, "fg_blocks_buy": False
        }


def check_btc_momentum() -> dict:
    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 2},
            timeout=10
        )
        data = r.json()
        if len(data) >= 2:
            prev_close, curr_close = float(data[0][4]), float(data[1][4])
            pct_change = ((curr_close - prev_close) / prev_close) * 100
            if pct_change >= 1.5:
                return {"bias": "BUY", "score_mod": 1, "strength": "strong", "message": f"BTC up {pct_change:.2f}%"}
            if pct_change <= -1.5:
                return {"bias": "SELL", "score_mod": 1, "strength": "strong", "message": f"BTC down {pct_change:.2f}%"}
        return {"bias": None, "score_mod": 0, "strength": "neutral", "message": "BTC neutral"}
    except Exception:
        return {"bias": None, "score_mod": 0, "strength": "unknown", "message": "BTC momentum unknown"}


def check_btc_volatility() -> dict:
    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 30},
            timeout=10
        )
        df = pd.DataFrame(
            r.json(),
            columns=["open_time","open","high","low","close","volume","close_time","quote_vol","trades","tb_base","tb_quote","ignore"]
        )
        for c in ["high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])

        prev_c = df["close"].shift(1)
        df["tr"] = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        atr = df["tr"].rolling(14).mean().iloc[-1]
        price = df["close"].iloc[-1]
        atr_pct = (atr / price) * 100

        if atr_pct > ATR_VERY_HIGH:
            return {"status": "VERY_HIGH", "risk_mult": 0.25, "skip": False, "message": f"🚨 EXTREME VOL {atr_pct:.2f}%"}
        if atr_pct > ATR_HIGH_PCT:
            return {"status": "HIGH", "risk_mult": 0.5, "skip": False, "message": f"⚠️ HIGH VOL {atr_pct:.2f}%"}
        if atr_pct < ATR_DEAD_PCT:
            return {"status": "DEAD", "risk_mult": 0.0, "skip": True, "message": f"😴 Dead market {atr_pct:.2f}%"}
        return {"status": "NORMAL", "risk_mult": 1.0, "skip": False, "message": f"✓ Normal BTC ATR {atr_pct:.2f}%"}
    except Exception:
        return {"status": "UNKNOWN", "risk_mult": 1.0, "skip": False, "message": "Vol check failed"}


def check_daily_pnl_advisory() -> str:
    try:
        bal, hist = _read_balance_and_history()
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0:
            return ""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at", "") or h.get("opened_at", ""))[:10] == today
            and "Ghost" not in h.get("close_reason", "")
            and "auto-removed" not in h.get("close_reason", "")
        )
        if today_pl < -(current_balance * 0.05):
            return f"⚠️ Daily loss advisory: {today_pl:.2f} USDT"
        return ""
    except Exception:
        return ""


def check_correlation(trades: dict, new_signal: str, new_symbol: str) -> bool:
    """NOTE: this function is fully correct but was never actually called
    anywhere in trade_executor.py — imported at the top, never invoked. The
    max_same_direction cap it duplicates IS enforced inline in execute_trade(),
    but the sector-diversification cap (max 3 open per sector below) has
    never been active despite being written. Wire this into execute_trade()
    (right alongside the existing same_dir_count check, before order
    placement) if sector diversification is actually wanted — leaving it
    documented-but-dormant here in case that's intentional for now."""
    cfg = _get_live_config()
    max_same_direction = cfg["max_same_direction"]

    same = sum(1 for t in trades.values() if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= max_same_direction:
        log.info(f"  Correlation filter: {same} {new_signal} already open (max {max_same_direction}) — skip {new_symbol}")
        return False

    SECTOR_MAP = {
        "Majors": ["ETHUSDT", "BNBUSDT", "SOLUSDT"],
        "Proven_L1_L2": ["XRPUSDT", "NEARUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ADAUSDT", "TRXUSDT", "XLMUSDT"],
        "DeFi_Infrastructure": ["UNIUSDT", "AAVEUSDT", "LINKUSDT", "CRVUSDT"],
        "High_Beta_Ecosystem": ["ENAUSDT", "DOGEUSDT", "TRUMPUSDT", "PUMPUSDT", "SUIUSDT", "AVAXUSDT", "ZECUSDT", "HBARUSDT", "FILUSDT"],
    }

    for sector, coins in SECTOR_MAP.items():
        if new_symbol in coins:
            sector_open = sum(1 for t in trades.values() if t.get("symbol") in coins and not t.get("closed", False))
            if sector_open >= 3:
                log.info(f"  Sector cap: {sector} already has {sector_open} open — skip {new_symbol}")
                return False

    return True


def should_scan() -> tuple:
    log.info(f"  Scan triggered at {datetime.now(timezone.utc).strftime('%H:%M UTC')} | UTC hour={datetime.now(timezone.utc).hour}")
    mode = get_scan_mode()
    vol  = check_btc_volatility()
    log.info(
        f"  {mode['label']} | conf≥{mode['min_confidence']:.1f}% "
        f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} "
        f"| risk_mult={mode['risk_mult']} | {vol['message']}"
    )

    if vol.get("skip") or mode.get("risk_mult", 1.0) == 0.0:
        skip_reason = vol["message"] if vol.get("skip") else mode["label"]
        return False, mode, vol, skip_reason

    advisory = check_daily_pnl_advisory()
    if advisory:
        log.warning(advisory)

    return True, mode, vol, f"{mode['label']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score": mode["min_score"],
        "min_adx": mode["min_adx"],
        "risk_mult": mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    """FIX: the previous `max(combined, 0.25)` floor actively fought the
    volatility de-risking it sits downstream of. In extreme volatility
    (vol.risk_mult=0.25) during quiet hours (mode.risk_mult=0.5), the intended
    combined multiplier of 0.125 was being forced back UP to 0.25 — i.e. risk
    was doubled relative to what check_btc_volatility() explicitly asked for,
    exactly in the scenario (extreme vol) where de-risking matters most.
    check_btc_volatility() and get_scan_mode() already carry their own
    internal floors (0.25 and 0.5 respectively) — an extra outer floor on top
    of both was redundant at best. The only real halt condition is the
    drawdown ratchet hitting 0, which is preserved below."""
    ratchet = get_drawdown_ratchet()
    if ratchet <= 0:
        return 0.0
    return mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0) * ratchet
