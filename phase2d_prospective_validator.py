#!/usr/bin/env python3
"""
Phase 2D — Prospective Shadow Validation v2

Research-only. Never places orders and never modifies production.

Key design:
  - Candidate uses the current canonical feature_engineering.py.
  - Production uses a locked compatibility reconstruction of the exact
    best_features stored in the production model artifact.
  - Both models score the SAME completed 15m candle stream.
  - No 4h API call is made. Four-hour context is built locally from
    completed 1h candles.
  - Pending predictions are resolved after the same 24-bar barrier horizon.
  - State is persisted between scheduled runs and is reset only if the locked
    model identities or validator schema change.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests

from feature_engineering import add_indicators as add_current_indicators

SYMBOLS = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
    "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
    "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
    "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT",
    "FILUSDT",
]

ENTRY_MS = 15 * 60 * 1000
HOUR_MS = 60 * 60 * 1000
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
CANDLE_LIMIT_1H = 300
STATE_SCHEMA = 4

LOCKED_CANDIDATE_SHA = (
    "45064fddd32f2e23eb4c34ed56bcbe73a8a1bffc"
    "90b8b9b6d82fc3838c4308c2"
)

LOCKED_PRODUCTION_SHA = (
    "f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x, default=0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def model_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    model = joblib.load(path)
    if not isinstance(model, dict):
        raise RuntimeError(f"{path}: unexpected model type {type(model).__name__}")
    required = {"all_features", "best_features", "ensemble", "label_map"}
    missing = required - set(model)
    if missing:
        raise RuntimeError(f"{path}: missing model keys: {sorted(missing)}")
    return model


def _deribit_symbol(symbol: str) -> str:
    return f"{symbol.replace('USDT', '').upper()}_USDC-PERPETUAL"


def fetch_deribit(symbol: str, resolution: str = "15", limit: int = CANDLE_LIMIT_15M) -> pd.DataFrame:
    if resolution not in {"15", "60"}:
        raise ValueError(f"Unsupported resolution: {resolution}")
    now_ms = int(time.time() * 1000)
    interval_ms = ENTRY_MS if resolution == "15" else HOUR_MS
    start_ms = now_ms - interval_ms * (limit + 20)
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    r = requests.get(url, params={
        "instrument_name": _deribit_symbol(symbol),
        "resolution": resolution,
        "start_timestamp": start_ms,
        "end_timestamp": now_ms,
    }, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    result = r.json().get("result", {})
    ticks = result.get("ticks", [])
    if not ticks:
        return pd.DataFrame()
    raw = pd.DataFrame({
        "open_time": ticks,
        "open": result.get("open", []),
        "high": result.get("high", []),
        "low": result.get("low", []),
        "close": result.get("close", []),
        "volume": result.get("volume", []),
    })
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna().copy()
    if raw.empty:
        return raw
    raw["open_time"] = raw["open_time"].astype("int64")
    raw = raw.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    raw["close_time"] = raw["open_time"] + interval_ms
    raw = raw[raw["close_time"] <= now_ms]
    raw["taker_buy_base_vol"] = raw["volume"] * 0.5
    return raw.tail(limit).reset_index(drop=True)

def fetch_deribit_15m(symbol: str, limit: int = CANDLE_LIMIT_15M) -> pd.DataFrame:
    return fetch_deribit(symbol, "15", limit)

def fetch_deribit_1h(symbol: str, limit: int = CANDLE_LIMIT_1H) -> pd.DataFrame:
    return fetch_deribit(symbol, "60", limit)

def aggregate_1h_to_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    if df1h.empty:
        return pd.DataFrame()
    x = df1h.sort_values("open_time").copy()
    x["bucket"] = x["open_time"] // (4 * HOUR_MS)
    g = x.groupby("bucket", sort=True).agg(
        open_time=("open_time", "min"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bars=("open_time", "count"),
    ).reset_index(drop=True)
    g = g[g["bars"] == 4].drop(columns=["bars"]).copy()
    g["close_time"] = g["open_time"] + 4 * HOUR_MS
    g["taker_buy_base_vol"] = g["volume"] * 0.5
    return g.reset_index(drop=True)
def current_row(raw15: pd.DataFrame, symbol: str) -> pd.Series:
    df = add_current_indicators(raw15.copy())
    if df.empty:
        raise RuntimeError(f"{symbol}: current feature engineering returned no rows")
    row = df.iloc[-1].copy()
    row["symbol"] = symbol
    return row


def build_production_row(raw15: pd.DataFrame, symbol: str, btc15: pd.DataFrame) -> pd.Series:
    if len(raw15) < 220:
        raise RuntimeError(f"{symbol}: insufficient 15m history")
    d = add_current_indicators(raw15.copy())
    c = d["close"].astype(float)
    e9 = c.ewm(span=9, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    d["ema9"], d["ema20"], d["ema50"], d["ema200"] = e9, e20, e50, e200
    d["ema20_slope"] = e20 / e20.shift(5) - 1.0
    d["ema50_slope"] = e50 / e50.shift(5) - 1.0
    d["price_vs_ema50"] = c / e50.replace(0, np.nan) - 1.0
    d["price_vs_ema200"] = c / e200.replace(0, np.nan) - 1.0
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std()
    d["bb_high"] = sma20 + 2.0 * std20
    d["bb_low"] = sma20 - 2.0 * std20
    vol = c.pct_change().rolling(20, min_periods=5).std()
    vol_med = vol.rolling(100, min_periods=20).median()
    d["vol_regime"] = vol / vol_med.replace(0, np.nan)
    d["regime_transitional"] = (d["trend"].rolling(4, min_periods=4).mean().abs() < 1.0).astype(float)

    h1_raw = fetch_deribit_1h(symbol)
    if len(h1_raw) < 80:
        raise RuntimeError(f"{symbol}: insufficient 1h history")
    h1 = add_current_indicators(h1_raw.copy())
    h4_raw = aggregate_1h_to_4h(h1_raw)
    if len(h4_raw) < 20:
        raise RuntimeError(f"{symbol}: insufficient 4h history")
    h4 = add_current_indicators(h4_raw.copy())

    def asof_features(base, higher, cols, names):
        left = base[["close_time"]].sort_values("close_time")
        right = higher[["close_time"] + cols].sort_values("close_time").rename(columns=dict(zip(cols, names)))
        out = pd.merge_asof(left, right, on="close_time", direction="backward")
        return out[names].reset_index(drop=True)

    h1a = asof_features(d, h1, ["rsi", "adx", "trend"], ["rsi_1h", "adx_1h", "trend_1h"])
    h4a = asof_features(d, h4, ["rsi", "trend"], ["rsi_4h", "trend_4h"])
    d = d.reset_index(drop=True)
    d["rsi_1h"], d["adx_1h"], d["trend_1h"] = h1a["rsi_1h"], h1a["adx_1h"], h1a["trend_1h"]
    d["rsi_4h"], d["trend_4h"] = h4a["rsi_4h"], h4a["trend_4h"]

    dt = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek
    d["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    d["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    d["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    d["regime_transitional"] = (d["trend_1h"] != d["trend_4h"]).astype(float)

    if btc15.empty:
        raise RuntimeError("BTC reference data unavailable")
    btc = btc15[["open_time", "close"]].rename(columns={"close": "btc_close"}).sort_values("open_time")
    d = pd.merge_asof(d.sort_values("open_time"), btc, on="open_time", direction="backward")
    cr = d["close"].pct_change()
    br = d["btc_close"].pct_change()
    d["btc_corr_20"] = cr.rolling(20, min_periods=10).corr(br).clip(-1, 1)
    d["btc_beta_20"] = (cr.rolling(20, min_periods=10).cov(br) / br.rolling(20, min_periods=10).var().replace(0, np.nan)).clip(-5, 5)

    d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d.iloc[-1].copy()


def model_features(model) -> List[str]:
    """Return the exact feature set used to fit the trained ensemble."""
    best = model.get("best_features")
    if not best:
        raise RuntimeError(
            "Model is missing best_features; refusing to infer the model schema."
        )
    features = [str(x) for x in best]
    if len(features) != len(set(features)):
        raise RuntimeError("Model best_features contains duplicate feature names.")
    return features


def score_model(model, row: pd.Series) -> dict:
    active = model_features(model)
    missing = [f for f in active if f not in row.index]
    if missing:
        raise RuntimeError(
            f"feature schema mismatch: model requires {len(active)} features; "
            f"row has {len(row.index)} columns; missing={missing[:12]}"
        )

    X = pd.DataFrame([[row[f] for f in active]], columns=active)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # The ensemble was fitted directly on best_features.
    # Do not call the serialized selector here; older pickles may not retain
    # selected_features even though best_features is present in the model dict.
    prob = model["ensemble"].predict_proba(X)[0]
    pred = int(model["ensemble"].predict(X)[0])

    label_map = {int(k): v for k, v in model["label_map"].items()}
    buy_idx = next((k for k, v in label_map.items() if v == "BUY"), None)
    sell_idx = next((k for k, v in label_map.items() if v == "SELL"), None)
    nt_idx = next((k for k, v in label_map.items() if v == "NO_TRADE"), None)

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
        "prediction_label": label_map.get(pred, str(pred)),
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
    tp = entry + atr * BUY_TP_R if side == "BUY" else entry - atr * SELL_TP_R
    sl = entry - atr * BUY_SL_R if side == "BUY" else entry + atr * SELL_SL_R

    for k in range(1, LOOKAHEAD + 1):
        bar = df.iloc[i + k]
        high = safe_float(bar["high"])
        low = safe_float(bar["low"])
        hit_tp = high >= tp if side == "BUY" else low <= tp
        hit_sl = low <= sl if side == "BUY" else high >= sl
        if hit_tp and hit_sl:
            return {**pred, "status": "AMBIGUOUS", "outcome_bar": k, "resolved_at": utc_now(), "gross_r": None, "net_r": None}
        if hit_tp:
            gross = BUY_TP_R if side == "BUY" else SELL_TP_R
            return {**pred, "status": "TP", "outcome_bar": k, "resolved_at": utc_now(), "gross_r": gross, "net_r": gross - FRICTION_R}
        if hit_sl:
            gross = -BUY_SL_R if side == "BUY" else -SELL_SL_R
            return {**pred, "status": "SL", "outcome_bar": k, "resolved_at": utc_now(), "gross_r": gross, "net_r": gross - FRICTION_R}

    return {**pred, "status": "EXPIRED", "outcome_bar": LOOKAHEAD, "resolved_at": utc_now(), "gross_r": 0.0, "net_r": -FRICTION_R}


def summarize(resolved: List[dict]) -> dict:
    out = {
        "resolved": len(resolved), "tp": 0, "sl": 0, "expired": 0,
        "ambiguous": 0, "invalid": 0, "precision_excluding_ambiguous": None,
        "mean_net_r": None, "cum_net_r": 0.0, "max_drawdown_r": 0.0,
    }
    valid = []
    for r in resolved:
        status = r.get("status", "INVALID")
        out[status.lower()] = out.get(status.lower(), 0) + 1
        if status in {"TP", "SL", "EXPIRED"} and r.get("net_r") is not None:
            valid.append(r)
    if valid:
        wins = sum(1 for r in valid if r["status"] == "TP")
        out["precision_excluding_ambiguous"] = wins / len(valid)
        values = [safe_float(r["net_r"]) for r in valid]
        out["mean_net_r"] = float(np.mean(values))
        equity = peak = max_dd = 0.0
        for v in values:
            equity += v
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        out["cum_net_r"] = float(equity)
        out["max_drawdown_r"] = float(max_dd)
    return out


def empty_state(candidate_sha: str, production_sha: str) -> dict:
    return {
        "schema_version": STATE_SCHEMA,
        "locked_models": {"candidate_sha256": candidate_sha, "production_sha256": production_sha},
        "pending": {"candidate": [], "production": []},
        "resolved": {"candidate": [], "production": []},
        "seen": {"candidate": [], "production": []},
    }


def load_compatible_state(candidate_sha: str, production_sha: str) -> dict:
    raw = load_json(STATE_FILE, {})
    expected = empty_state(candidate_sha, production_sha)
    if raw.get("schema_version") != STATE_SCHEMA:
        print("Phase 2D | state schema changed -> starting fresh")
        return expected
    locked = raw.get("locked_models", {})
    if locked.get("candidate_sha256") != candidate_sha or locked.get("production_sha256") != production_sha:
        print("Phase 2D | locked model identity changed -> starting fresh")
        return expected
    for key in ("pending", "resolved", "seen"):
        raw.setdefault(key, {"candidate": [], "production": []})
        raw[key].setdefault("candidate", [])
        raw[key].setdefault("production", [])
    return raw


def main() -> int:
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError("candidate_model.pkl not found.")
    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError("pro_crypto_ai_model.pkl not found.")

    candidate_sha = model_sha256(CANDIDATE_MODEL)
    production_sha = model_sha256(PRODUCTION_MODEL)

    if candidate_sha != LOCKED_CANDIDATE_SHA:
        raise RuntimeError(
            "Candidate SHA256 mismatch: "
            f"expected {LOCKED_CANDIDATE_SHA}, got {candidate_sha}"
        )

    if production_sha != LOCKED_PRODUCTION_SHA:
        raise RuntimeError(
            "Production SHA256 mismatch: "
            f"expected {LOCKED_PRODUCTION_SHA}, got {production_sha}"
        )

    candidate = load_model(CANDIDATE_MODEL)
    production = load_model(PRODUCTION_MODEL)

    print(f"Phase 2D | candidate sha={candidate_sha}")
    print(f"Phase 2D | production sha={production_sha}")
    print(f"Phase 2D | candidate declared features={len(candidate['all_features'])}")
    print(f"Phase 2D | production declared features={len(production['all_features'])}")

    # Hard compatibility check before any research result is recorded.
    btc15 = fetch_deribit_15m("BTCUSDT")
    if len(btc15) < 80:
        raise RuntimeError("BTCUSDT: insufficient completed 15m candles")
    probe = fetch_deribit_15m("ETHUSDT")
    if len(probe) < 80:
        raise RuntimeError("ETHUSDT: insufficient completed 15m candles")
    c_probe = current_row(probe, "ETHUSDT")
    btc_probe = fetch_deribit_15m("BTCUSDT")
    p_probe = build_production_row(probe, "ETHUSDT", btc_probe)
    candidate_features = model_features(candidate)
    production_features = model_features(production)

    c_missing = [f for f in candidate_features if f not in c_probe.index]
    p_missing = [f for f in production_features if f not in p_probe.index]

    print(f"Phase 2D | candidate selected features: {len(candidate_features)}")
    print(f"Phase 2D | production selected features: {len(production_features)}")
    print(f"Phase 2D | candidate schema check missing={len(c_missing)}")
    print(f"Phase 2D | production schema check missing={len(p_missing)}")
    if c_missing:
        raise RuntimeError(f"Candidate schema mismatch: {c_missing}")
    if p_missing:
        print(f"Phase 2D | production missing selected features: {p_missing[:25]}")
        raise RuntimeError("Production selected feature schema cannot be reproduced.")

    state = load_compatible_state(candidate_sha, production_sha)
    models = {"candidate": candidate, "production": production}
    run_snapshot = {"timestamp": utc_now(), "signals": {"candidate": [], "production": []}}
    current_rows: Dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        try:
            raw15 = fetch_deribit_15m(symbol)
            if len(raw15) < 80:
                raise RuntimeError("insufficient completed 15m candles")
            c_row = current_row(raw15, symbol)
            p_row = build_production_row(raw15, symbol, btc15)
            rows_by_model = {"candidate": c_row, "production": p_row}
            df15 = raw15
            current_rows[symbol] = df15
            ts = int(c_row["open_time"])
            entry = safe_float(c_row["close"])
            c_atr = safe_float(c_row.get("atr", 0))
            p_atr = safe_float(p_row.get("atr", c_row.get("atr", 0)))
            if entry <= 0 or c_atr <= 0 or p_atr <= 0:
                continue

            for name, model in models.items():
                row = rows_by_model[name]
                scored = score_model(model, row)
                run_snapshot["signals"][name].append({"symbol": symbol, "open_time": ts, **scored})
                if scored["signal"] == "NO_TRADE":
                    continue
                key = f"{symbol}:{ts}"
                if key in set(state["seen"].get(name, [])):
                    continue
                pred = {
                    "id": f"{name}:{symbol}:{ts}", "model": name, "symbol": symbol,
                    "open_time": ts, "signal": scored["signal"],
                    "confidence": scored["confidence"], "p_buy": scored["p_buy"],
                    "p_sell": scored["p_sell"], "threshold_buy": scored["threshold_buy"],
                    "threshold_sell": scored["threshold_sell"], "entry": entry,
                    "atr": c_atr if name == "candidate" else p_atr, "created_at": utc_now(),
                }
                state["pending"][name].append(pred)
                state["seen"][name].append(key)
        except Exception as e:
            print(f"WARN {symbol}: {e}")

    for name in ("candidate", "production"):
        still_pending = []
        for pred in state["pending"].get(name, []):
            df15 = current_rows.get(pred["symbol"])
            if df15 is None:
                try:
                    df15 = fetch_deribit_15m(pred["symbol"])
                except Exception as e:
                    print(f"WARN {pred['symbol']} resolution fetch: {e}")
                    still_pending.append(pred)
                    continue
            result = resolve_prediction(pred, df15)
            if result is None:
                still_pending.append(pred)
            else:
                state["resolved"][name].append(result)
        state["pending"][name] = still_pending
        state["resolved"][name] = state["resolved"][name][-500:]
        state["seen"][name] = state["seen"][name][-1000:]

    summary = {
        "updated_at": utc_now(),
        "candidate": summarize(state["resolved"]["candidate"]),
        "production": summarize(state["resolved"]["production"]),
        "pending": {k: len(v) for k, v in state["pending"].items()},
        "candidate_sha256": candidate_sha,
        "production_sha256": production_sha,
        "candidate_trained_at": candidate.get("trained_at"),
        "production_trained_at": production.get("trained_at"),
        "candidate_thresholds": {"buy": candidate.get("recommended_threshold_buy"), "sell": candidate.get("recommended_threshold_sell")},
        "production_thresholds": {"buy": production.get("recommended_threshold_buy"), "sell": production.get("recommended_threshold_sell")},
        "methodology": {
            "horizon_bars": LOOKAHEAD, "tp_r": BUY_TP_R, "sl_r": BUY_SL_R,
            "friction_r": FRICTION_R, "ambiguous_excluded_from_precision": True,
            "expired_net_r": -FRICTION_R, "candidate_feature_source": "current canonical",
            "production_feature_source": "legacy compatibility reconstruction from selected 35 features",
            "same_15m_market_stream": True,
            "no_240m_deribit_request": True,
            "four_hour_source": "completed 1h candles aggregated locally",
            "candidate_selected_feature_count": len(candidate_features),
            "production_selected_feature_count": len(production_features),
            "promotion_ready": False,
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
