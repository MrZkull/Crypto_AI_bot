#!/usr/bin/env python3
"""
Phase 2D — Prospective Shadow Validation v2

Research-only. Never places orders and never modifies production.

Key design:
  - Candidate uses the current canonical feature_engineering.py.
  - Production uses the exact v2.0 release feature_engineering.py so the
    legacy 63-feature production model is evaluated with its native schema.
  - Both models score the SAME completed 15m candle stream.
  - No 4h API call is made. Phase 2D does not need HTF features for the
    locked 25-feature candidate, and the legacy production model gets its
    native 15m feature row from the v2.0 release code.
  - Pending predictions are resolved after the same 24-bar barrier horizon.
  - State is persisted between scheduled runs and is reset only if the locked
    model identities or validator schema change.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

LEGACY_FE_PATH = Path(os.getenv("PHASE2D_LEGACY_FE", "legacy_v20/feature_engineering.py"))
REQUEST_TIMEOUT = 12
CANDLE_LIMIT_15M = 300
STATE_SCHEMA = 2


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
    required = {"all_features", "selector", "ensemble", "label_map"}
    missing = required - set(model)
    if missing:
        raise RuntimeError(f"{path}: missing model keys: {sorted(missing)}")
    return model


def load_legacy_feature_engineering():
    if not LEGACY_FE_PATH.exists():
        raise FileNotFoundError(f"Legacy v2.0 feature_engineering.py not found: {LEGACY_FE_PATH}")
    spec = importlib.util.spec_from_file_location("phase2d_legacy_feature_engineering", LEGACY_FE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy feature engineering module: {LEGACY_FE_PATH}")
    root = str(LEGACY_FE_PATH.parent.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "add_indicators"):
        raise RuntimeError("Legacy v2.0 feature_engineering.py has no add_indicators()")
    return module


def _deribit_symbol(symbol: str) -> str:
    return f"{symbol.replace('USDT', '').upper()}_USDC-PERPETUAL"


def fetch_deribit_15m(symbol: str, limit: int = CANDLE_LIMIT_15M) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    interval_ms = ENTRY_MS
    start_ms = now_ms - interval_ms * (limit + 5)
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    r = requests.get(
        url,
        params={
            "instrument_name": _deribit_symbol(symbol),
            "resolution": "15",
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

    raw = pd.DataFrame({
        "open_time": ticks,
        "open": result.get("open", []),
        "high": result.get("high", []),
        "low": result.get("low", []),
        "close": result.get("close", []),
        "volume": result.get("volume", []),
    })
    raw["taker_buy_base_vol"] = pd.to_numeric(raw["volume"], errors="coerce") * 0.5
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base_vol"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna().copy()
    raw["open_time"] = raw["open_time"].astype("int64")
    raw = raw.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # Keep candles whose close time has passed. This preserves the latest fully
    # closed candle and avoids the prior accidental two-candle lag.
    raw = raw[raw["open_time"] + interval_ms <= now_ms]
    return raw.tail(limit).reset_index(drop=True)


def current_row(raw15: pd.DataFrame, symbol: str) -> pd.Series:
    df = add_current_indicators(raw15.copy())
    if df.empty:
        raise RuntimeError(f"{symbol}: current feature engineering returned no rows")
    row = df.iloc[-1].copy()
    row["symbol"] = symbol
    return row


def legacy_row(raw15: pd.DataFrame, symbol: str, legacy_module) -> pd.Series:
    df = legacy_module.add_indicators(raw15.copy())
    if df is None or len(df) == 0:
        raise RuntimeError(f"{symbol}: legacy v2.0 feature engineering returned no rows")
    row = df.iloc[-1].copy()
    row["symbol"] = symbol
    return row


def score_model(model, row: pd.Series) -> dict:
    active = [str(f) for f in model["all_features"]]
    missing = [f for f in active if f not in row.index]
    if missing:
        raise RuntimeError(
            f"feature schema mismatch: model requires {len(active)} features; "
            f"row has {len(row.index)} columns; missing={missing[:12]}"
        )

    X = pd.DataFrame([[row[f] for f in active]], columns=active)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xs = model["selector"].transform(X)
    prob = model["ensemble"].predict_proba(Xs)[0]
    pred = int(model["ensemble"].predict(Xs)[0])

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
    candidate = load_model(CANDIDATE_MODEL)
    production = load_model(PRODUCTION_MODEL)
    legacy_fe = load_legacy_feature_engineering()

    print(f"Phase 2D v2 | candidate sha={candidate_sha}")
    print(f"Phase 2D v2 | production sha={production_sha}")
    print(f"Phase 2D v2 | candidate features={len(candidate['all_features'])}")
    print(f"Phase 2D v2 | production features={len(production['all_features'])}")
    print(f"Phase 2D v2 | legacy FE={LEGACY_FE_PATH}")

    # Hard compatibility check before any research result is recorded.
    probe = fetch_deribit_15m("ETHUSDT")
    if len(probe) < 80:
        raise RuntimeError("ETHUSDT: insufficient completed 15m candles")
    c_probe = current_row(probe, "ETHUSDT")
    p_probe = legacy_row(probe, "ETHUSDT", legacy_fe)
    c_missing = [f for f in candidate["all_features"] if f not in c_probe.index]
    p_missing = [f for f in production["all_features"] if f not in p_probe.index]
    print(f"Phase 2D v2 | candidate schema check missing={len(c_missing)}")
    print(f"Phase 2D v2 | production legacy schema check missing={len(p_missing)}")
    if c_missing:
        raise RuntimeError(f"Candidate schema mismatch: {c_missing}")
    if p_missing:
        print(f"Phase 2D v2 | production missing features: {p_missing[:25]}")
        raise RuntimeError("Production model cannot be reproduced from the locked v2.0 feature engineering source.")

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
            p_row = legacy_row(raw15, symbol, legacy_fe)
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
            "production_feature_source": "release v2.0 feature_engineering.py",
            "same_15m_market_stream": True,
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
