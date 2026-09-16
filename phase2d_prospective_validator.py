#!/usr/bin/env python3
"""
Phase 2D — Prospective Shadow Validation

Research-only. Never places orders and never modifies the production model.
Each run:
  1) Fetches the latest completed Deribit 15m/1h/4h candles.
  2) Builds live features with the same canonical feature engineering.
  3) Scores the locked HTF-removal candidate and current production model.
  4) Records only qualifying BUY/SELL predictions using each model's locked thresholds.
  5) Resolves pending predictions after the same 24-bar barrier horizon used in training.
  6) Saves compact state/results for the next scheduled run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

from feature_engineering import add_indicators

SYMBOLS = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
    "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
    "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
    "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT",
    "FILUSDT",
]

ENTRY_INTERVAL = "15m"
ENTRY_MS = 15 * 60 * 1000
HTF_1H_MS = 60 * 60 * 1000
HTF_4H_MS = 4 * 60 * 60 * 1000
LOOKAHEAD = 24
BUY_TP_R = 3.5
BUY_SL_R = 2.5
SELL_TP_R = 3.5
SELL_SL_R = 2.5
FRICTION_R = 0.12

CANDIDATE_MODEL = Path("candidate_model.pkl")
PRODUCTION_MODEL = Path("pro_crypto_ai_model.pkl")
STATE_FILE = Path("research_outputs/phase2d_state.json")
RESULTS_FILE = Path("research_outputs/phase2d_results.json")
SNAPSHOT_FILE = Path("research_outputs/phase2d_latest_snapshot.json")

REQUEST_TIMEOUT = 12
CANDLE_LIMIT_15M = 300
CANDLE_LIMIT_HTF = 120


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x, default=0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _deribit_symbol(symbol: str) -> str:
    return f"{symbol.replace('USDT', '').upper()}_USDC-PERPETUAL"


def fetch_deribit_ohlcv(symbol: str, resolution: int, limit: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    span_ms = resolution * 60 * 1000 * (limit + 5)
    start_ms = now_ms - span_ms
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    r = requests.get(
        url,
        params={
            "instrument_name": _deribit_symbol(symbol),
            "resolution": str(resolution),
            "start_timestamp": start_ms,
            "end_timestamp": now_ms,
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    result = r.json().get("result", {})
    ticks = result.get("ticks", [])
    if not ticks:
        return pd.DataFrame()

    fields = {
        "open_time": ticks,
        "open": result.get("open", []),
        "high": result.get("high", []),
        "low": result.get("low", []),
        "close": result.get("close", []),
        "volume": result.get("volume", []),
    }
    n = min(len(v) for v in fields.values())
    df = pd.DataFrame({k: v[:n] for k, v in fields.items()})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    # The live production helper uses a neutral 50% proxy because Deribit
    # tradingview data does not expose taker-buy volume.
    df["taker_buy_base_vol"] = df["volume"] * 0.5
    df = df.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # Never score a candle that has not fully closed.
    interval_ms = resolution * 60 * 1000
    closed_before = int(time.time() * 1000) - interval_ms
    return df[df["open_time"] + interval_ms <= closed_before].tail(limit).reset_index(drop=True)


def funding_asof(symbol: str, candle_close_ms: int) -> Optional[float]:
    base = symbol.replace("USDT", "").upper()
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    try:
        r = requests.get(
            url,
            params={"symbol": f"{base}USDT", "limit": 100},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
        vals = []
        for row in rows:
            ts = int(row.get("fundingTime", 0))
            rate = safe_float(row.get("fundingRate"), math.nan)
            if ts <= candle_close_ms and math.isfinite(rate):
                vals.append((ts, rate))
        if not vals:
            return None
        return vals[-1][1]
    except Exception:
        return None


def add_live_relational_features(df15: pd.DataFrame, btc15: pd.DataFrame) -> pd.DataFrame:
    df = df15.copy()
    if btc15 is None or btc15.empty:
        df["btc_corr_20"] = 0.0
        df["btc_beta_20"] = 1.0
        df["btc_rel_strength"] = 0.0
        return df

    btc = btc15[["open_time", "close"]].rename(columns={"close": "btc_close"}).copy()
    # Align on completed 15m candle open time; both streams are already closed.
    df = pd.merge_asof(
        df.sort_values("open_time"),
        btc.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    btc_ret = df["btc_close"].pct_change()
    coin_ret = df["close"].pct_change()
    roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
    roll_var = btc_ret.rolling(20, min_periods=10).var()
    df["btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
    df["btc_beta_20"] = roll_cov / roll_var.replace(0, np.nan)
    df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100
    df["btc_corr_20"] = df["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df["btc_beta_20"] = df["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)
    return df


def latest_completed_htf(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[-1]


def build_symbol_row(symbol: str, btc15: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    raw15 = fetch_deribit_ohlcv(symbol, 15, CANDLE_LIMIT_15M)
    raw1h = fetch_deribit_ohlcv(symbol, 60, CANDLE_LIMIT_HTF)
    raw4h = fetch_deribit_ohlcv(symbol, 240, CANDLE_LIMIT_HTF)
    if len(raw15) < 80 or raw1h.empty or raw4h.empty:
        raise RuntimeError(f"{symbol}: insufficient live candles")

    df15 = add_indicators(raw15)
    df1h = add_indicators(raw1h)
    df4h = add_indicators(raw4h)
    df15 = add_live_relational_features(df15, btc15)

    row = df15.iloc[-1].copy()
    r1h = latest_completed_htf(df1h)
    r4h = latest_completed_htf(df4h)

    row["rsi_1h"] = safe_float(r1h.get("rsi", 50), 50)
    row["adx_1h"] = safe_float(r1h.get("adx", 0), 0)
    row["trend_1h"] = safe_float(r1h.get("trend", 0), 0)
    row["rsi_4h"] = safe_float(r4h.get("rsi", 50), 50)
    row["trend_4h"] = safe_float(r4h.get("trend", 0), 0)

    candle_close_ms = int(row["open_time"]) + ENTRY_MS
    fr = funding_asof(symbol, candle_close_ms)
    row["fundingRate"] = fr if fr is not None else 0.0
    # Keep a separate provenance flag in research state; model input itself
    # remains numeric. Missing funding is a rejected sample, not silently used.
    if fr is None:
        raise RuntimeError(f"{symbol}: funding unavailable for completed candle")

    row["symbol"] = symbol
    return row, raw15


def model_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


def score_model(model, row: pd.Series) -> dict:
    active = list(model["all_features"])
    for feature in active:
        if feature not in row.index:
            raise RuntimeError(f"Model feature missing from live row: {feature}")

    X = pd.DataFrame([[row[f] for f in active]], columns=active)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    Xs = model["selector"].transform(X)
    prob = model["ensemble"].predict_proba(Xs)[0]
    pred = int(model["ensemble"].predict(Xs)[0])
    label = model["label_map"][pred]
    class_names = {int(k): v for k, v in model["label_map"].items()}
    buy_idx = next((k for k, v in class_names.items() if v == "BUY"), None)
    sell_idx = next((k for k, v in class_names.items() if v == "SELL"), None)
    nt_idx = next((k for k, v in class_names.items() if v == "NO_TRADE"), None)
    p_buy = safe_float(prob[buy_idx]) if buy_idx is not None else 0.0
    p_sell = safe_float(prob[sell_idx]) if sell_idx is not None else 0.0
    p_nt = safe_float(prob[nt_idx]) if nt_idx is not None else 0.0

    th_buy = safe_float(model.get("recommended_threshold_buy", 1.01), 1.01)
    th_sell = safe_float(model.get("recommended_threshold_sell", 1.01), 1.01)
    if p_buy >= th_buy and p_buy > p_sell:
        signal = "BUY"
    elif p_sell >= th_sell and p_sell > p_buy:
        signal = "SELL"
    else:
        signal = "NO_TRADE"

    return {
        "signal": signal,
        "prediction_label": label,
        "confidence": round(max(p_buy, p_sell, p_nt) * 100.0, 3),
        "p_buy": round(p_buy, 6),
        "p_sell": round(p_sell, 6),
        "p_no_trade": round(p_nt, 6),
        "threshold_buy": th_buy,
        "threshold_sell": th_sell,
        "n_features": len(active),
        "trained_at": model.get("trained_at"),
    }


def resolve_prediction(pred: dict, df: pd.DataFrame) -> Optional[dict]:
    ts = int(pred["open_time"])
    idxs = np.where(df["open_time"].values == ts)[0]
    if len(idxs) == 0:
        return None
    i = int(idxs[-1])
    if i + LOOKAHEAD >= len(df):
        return None

    entry = safe_float(pred["entry"])
    atr = safe_float(pred["atr"])
    if entry <= 0 or atr <= 0:
        return {**pred, "status": "INVALID", "resolved_at": utc_now()}

    side = pred["signal"]
    if side == "BUY":
        tp = entry + atr * BUY_TP_R
        sl = entry - atr * BUY_SL_R
    else:
        tp = entry - atr * SELL_TP_R
        sl = entry + atr * SELL_SL_R

    for k in range(1, LOOKAHEAD + 1):
        bar = df.iloc[i + k]
        hit_tp = safe_float(bar["high"]) >= tp if side == "BUY" else safe_float(bar["low"]) <= tp
        hit_sl = safe_float(bar["low"]) <= sl if side == "BUY" else safe_float(bar["high"]) >= sl
        if hit_tp and hit_sl:
            return {
                **pred,
                "status": "AMBIGUOUS",
                "outcome_bar": k,
                "resolved_at": utc_now(),
                "gross_r": None,
                "net_r": None,
            }
        if hit_tp:
            gross = BUY_TP_R if side == "BUY" else SELL_TP_R
            return {**pred, "status": "TP", "outcome_bar": k, "resolved_at": utc_now(), "gross_r": gross, "net_r": gross - FRICTION_R}
        if hit_sl:
            gross = -BUY_SL_R if side == "BUY" else -SELL_SL_R
            return {**pred, "status": "SL", "outcome_bar": k, "resolved_at": utc_now(), "gross_r": gross, "net_r": gross - FRICTION_R}

    # Same horizon as training: unresolved barrier at bar 24 is an expiry.
    # Research Net-R applies the same friction cost that was used in threshold selection.
    return {
        **pred,
        "status": "EXPIRED",
        "outcome_bar": LOOKAHEAD,
        "resolved_at": utc_now(),
        "gross_r": 0.0,
        "net_r": -FRICTION_R,
    }


def summarize(resolved: List[dict]) -> dict:
    out = {
        "resolved": len(resolved),
        "tp": 0,
        "sl": 0,
        "expired": 0,
        "ambiguous": 0,
        "invalid": 0,
        "precision_excluding_ambiguous": None,
        "mean_net_r": None,
        "cum_net_r": 0.0,
        "max_drawdown_r": 0.0,
    }
    valid = []
    for r in resolved:
        status = r.get("status")
        out[status.lower()] = out.get(status.lower(), 0) + 1
        if status in {"TP", "SL", "EXPIRED"} and r.get("net_r") is not None:
            valid.append(r)
    if valid:
        wins = sum(1 for r in valid if r["status"] == "TP")
        out["precision_excluding_ambiguous"] = wins / len(valid)
        values = [safe_float(r["net_r"]) for r in valid]
        out["mean_net_r"] = float(np.mean(values))
        out["cum_net_r"] = float(np.sum(values))
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for v in values:
            equity += v
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        out["max_drawdown_r"] = float(max_dd)
    return out


def main() -> int:
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError("candidate_model.pkl not found. Download the locked candidate artifact first.")
    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError("pro_crypto_ai_model.pkl not found.")

    state = load_json(STATE_FILE, {"pending": {"candidate": [], "production": []}, "resolved": {"candidate": [], "production": []}, "seen": {"candidate": [], "production": []}})
    state.setdefault("pending", {"candidate": [], "production": []})
    state.setdefault("resolved", {"candidate": [], "production": []})
    state.setdefault("seen", {"candidate": [], "production": []})

    candidate = load_model(CANDIDATE_MODEL)
    production = load_model(PRODUCTION_MODEL)
    print(f"Phase 2D | candidate sha={model_sha256(CANDIDATE_MODEL)}")
    print(f"Phase 2D | production sha={model_sha256(PRODUCTION_MODEL)}")

    btc15 = fetch_deribit_ohlcv("BTCUSDT", 15, CANDLE_LIMIT_15M)
    if len(btc15) < 80:
        raise RuntimeError("BTCUSDT: insufficient live candles")

    models = {"candidate": candidate, "production": production}
    current_rows: Dict[str, pd.DataFrame] = {}
    run_snapshot = {"timestamp": utc_now(), "signals": {}}

    for symbol in SYMBOLS:
        try:
            row, df15 = build_symbol_row(symbol, btc15)
            current_rows[symbol] = df15
            ts = int(row["open_time"])
            entry = safe_float(row["close"])
            atr = safe_float(row.get("atr", 0))
            if entry <= 0 or atr <= 0:
                continue

            for name, model in models.items():
                scored = score_model(model, row)
                run_snapshot["signals"].setdefault(name, []).append({"symbol": symbol, "open_time": ts, **scored})
                if scored["signal"] == "NO_TRADE":
                    continue
                key = f"{symbol}:{ts}"
                if key in set(state["seen"].get(name, [])):
                    continue
                pred = {
                    "id": f"{name}:{symbol}:{ts}",
                    "model": name,
                    "symbol": symbol,
                    "open_time": ts,
                    "signal": scored["signal"],
                    "confidence": scored["confidence"],
                    "p_buy": scored["p_buy"],
                    "p_sell": scored["p_sell"],
                    "threshold_buy": scored["threshold_buy"],
                    "threshold_sell": scored["threshold_sell"],
                    "entry": entry,
                    "atr": atr,
                    "created_at": utc_now(),
                }
                state["pending"].setdefault(name, []).append(pred)
                state["seen"].setdefault(name, []).append(key)
        except Exception as e:
            print(f"WARN {symbol}: {e}")

    # Resolve old pending predictions using the same symbol's current 15m window.
    for name in ("candidate", "production"):
        still_pending = []
        for pred in state["pending"].get(name, []):
            df15 = current_rows.get(pred["symbol"])
            if df15 is None:
                try:
                    df15 = fetch_deribit_ohlcv(pred["symbol"], 15, CANDLE_LIMIT_15M)
                except Exception:
                    still_pending.append(pred)
                    continue
            result = resolve_prediction(pred, df15)
            if result is None:
                still_pending.append(pred)
            else:
                state["resolved"].setdefault(name, []).append(result)
        state["pending"][name] = still_pending

    # Bound state to the latest 500 resolved trades per model and 500 seen keys.
    for name in ("candidate", "production"):
        state["resolved"][name] = state["resolved"].get(name, [])[-500:]
        state["seen"][name] = state["seen"].get(name, [])[-500:]

    summary = {
        "updated_at": utc_now(),
        "candidate": summarize(state["resolved"]["candidate"]),
        "production": summarize(state["resolved"]["production"]),
        "pending": {k: len(v) for k, v in state["pending"].items()},
        "candidate_sha256": model_sha256(CANDIDATE_MODEL),
        "production_sha256": model_sha256(PRODUCTION_MODEL),
        "candidate_trained_at": candidate.get("trained_at"),
        "production_trained_at": production.get("trained_at"),
        "candidate_thresholds": {
            "buy": candidate.get("recommended_threshold_buy"),
            "sell": candidate.get("recommended_threshold_sell"),
        },
        "production_thresholds": {
            "buy": production.get("recommended_threshold_buy"),
            "sell": production.get("recommended_threshold_sell"),
        },
        "methodology": {
            "horizon_bars": LOOKAHEAD,
            "tp_r": BUY_TP_R,
            "sl_r": BUY_SL_R,
            "friction_r": FRICTION_R,
            "ambiguous_excluded_from_precision": True,
            "expired_net_r": -FRICTION_R,
        },
    }

    state["last_run"] = summary
    save_json(STATE_FILE, state)
    save_json(RESULTS_FILE, summary)
    save_json(SNAPSHOT_FILE, run_snapshot)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
