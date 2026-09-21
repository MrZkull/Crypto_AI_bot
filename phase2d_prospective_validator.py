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
from phase2d_promotion_gate import evaluate_promotion

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
BACKFILL_BARS = 8  # 2h of 15m bars; covers hourly cadence with slack

LOCKED_CANDIDATE_SHA = "7213987874543e5d6e18b0bac32556647d40080b3e4222532e40a225a1bf08a3"

LOCKED_PRODUCTION_SHA = (
    "f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21"
)

LOCKED_PRODUCTION_SELECTED_FEATURES = [
    "volume_ratio",
    "volume_spike",
    "obv_slope",
    "bb_width",
    "atr_pct",
    "volatility",
    "vwap_dev",
    "trend_1h",
    "rsi_4h",
    "dow_sin",
    "dow_cos",
    "trend_4h",
    "price_vs_ema200",
    "hour_cos",
    "hour_sin",
    "rsi_1h",
    "regime_transitional",
    "adx",
    "ema20_vs_ema50",
    "adx_1h",
    "ema200",
    "bb_high",
    "ema50_slope",
    "bb_low",
    "vol_regime",
    "ema50",
    "ema20",
    "btc_beta_20",
    "macd_signal",
    "price_vs_ema50",
    "btc_corr_20",
    "ema9",
    "atr",
    "ema20_slope",
    "macd_hist",
]



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


def current_features(raw15: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = add_current_indicators(raw15.copy())
    if df.empty:
        raise RuntimeError(f"{symbol}: current feature engineering returned no rows")
    df = df.copy()
    df["symbol"] = symbol
    return df.reset_index(drop=True)


# ── Locked Production Compatibility Feature Construction ─────────────────

def _align_htf_point_in_time(ltf: pd.DataFrame, htf: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """Strict point-in-time backward merge: HTF candle must be closed on or before LTF close."""
    if htf.empty or ltf.empty:
        for target in col_map.values():
            ltf[target] = 0.0
        return ltf

    available_cols = [c for c in col_map.keys() if c in htf.columns]
    if not available_cols:
        for target in col_map.values():
            ltf[target] = 0.0
        return ltf

    htf_sub = htf[["close_time"] + available_cols].dropna().sort_values("close_time").copy()
    htf_sub = htf_sub.rename(columns=col_map)

    merged = pd.merge_asof(
        ltf.sort_values("close_time"),
        htf_sub,
        on="close_time",
        direction="backward"
    )

    for target in col_map.values():
        if target not in merged.columns:
            merged[target] = 0.0
        else:
            merged[target] = merged[target].fillna(0.0)

    return merged.sort_values("open_time").reset_index(drop=True)


def _add_btc_cross_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if btc_df is not None and not btc_df.empty and "close" in btc_df.columns:
        btc_sub = btc_df[["close_time", "close"]].dropna().sort_values("close_time").rename(columns={"close": "btc_close"})
        merged = pd.merge_asof(
            df.sort_values("close_time"),
            btc_sub,
            on="close_time",
            direction="backward"
        )
        df["btc_close"] = merged["btc_close"].ffill().bfill()
    else:
        df["btc_close"] = np.nan

    if "btc_close" in df.columns and df["btc_close"].notna().sum() > 30:
        btc_ret = df["btc_close"].pct_change()
        coin_ret = df["close"].pct_change()
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
        roll_var = btc_ret.rolling(20, min_periods=10).var()
        df["btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
        df["btc_beta_20"] = roll_cov / roll_var.replace(0, np.nan)
        df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100
    else:
        df["btc_corr_20"] = 0.0
        df["btc_beta_20"] = 1.0
        df["btc_rel_strength"] = 0.0

    df["btc_corr_20"] = df["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df["btc_beta_20"] = df["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)
    return df


def _legacy_production_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical feature formulas used by locked production v2.0."""
    if df is None or df.empty:
        return df

    df = df.copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)

    # Legacy EMAs / price relationships
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff(3) / df["ema20"].shift(3) * 100
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3) * 100
    df["price_vs_ema20"] = (c - df["ema20"]) / df["ema20"] * 100
    df["price_vs_ema50"] = (c - df["ema50"]) / df["ema50"] * 100
    df["price_vs_ema200"] = (c - df["ema200"]) / df["ema200"] * 100
    df["ema20_vs_ema50"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100

    # Legacy RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)

    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    df["rsi_slope"] = df["rsi"].diff(3)

    avg_g7 = gain.ewm(com=6, adjust=False).mean()
    avg_l7 = loss.ewm(com=6, adjust=False).mean()
    rs7 = avg_g7 / avg_l7.replace(0, np.nan)

    df["rsi_fast"] = (100 - 100 / (1 + rs7)).fillna(50)

    # Legacy stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()

    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Legacy MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_slope"] = df["macd"].diff(3)

    # Legacy ATR / ADX
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    dm_pos = (h.diff()).clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)

    atr14 = tr.ewm(span=14, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)

    df["adx"] = dx.ewm(span=14, adjust=False).mean()
    df["adx_pos"] = di_pos
    df["adx_neg"] = di_neg
    df["di_diff"] = di_pos - di_neg

    # Legacy Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    bb_high = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    bb_width = bb_high - bb_low

    df["bb_high"] = bb_high
    df["bb_low"] = bb_low
    df["bb_pct"] = (c - bb_low) / (bb_width + 1e-10)
    df["bb_width"] = bb_width / sma20 * 100

    # Legacy volume / VWAP
    vol_ma20 = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma20.replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(5) / (vol_ma20 * 5 + 1e-10)

    df["vwap"] = (c * v).cumsum() / (v.cumsum() + 1e-10)
    df["vwap_dev"] = (c - df["vwap"]) / df["vwap"] * 100

    # Legacy price-action
    df["price_change"] = c.pct_change(1) * 100
    df["price_change3"] = c.pct_change(3) * 100
    df["price_change6"] = c.pct_change(6) * 100
    df["high_low_pct"] = (h - l) / c * 100
    df["body_pct"] = (c - o).abs() / (h - l + 1e-10)
    df["momentum"] = c - c.shift(10)
    df["volatility"] = c.rolling(14).std() / c * 100

    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    df["pivot_dev"] = (c - pivot) / pivot * 100

    # Legacy candlestick patterns
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    rng = h - l + 1e-10

    df["bullish_candle"] = ((c > o) & (body > rng * 0.6)).astype(int)
    df["doji"] = (body < rng * 0.1).astype(int)
    df["hammer"] = ((lower_wick > body * 2) & (upper_wick < body)).astype(int)

    # Legacy trend
    df["trend"] = np.where(
        df["ema20"] > df["ema50"], 1,
        np.where(df["ema20"] < df["ema50"], -1, 0)
    )

    # Legacy volatility regime
    atr_smooth = df["atr_pct"].rolling(5).mean()
    adx_smooth = df["adx"].rolling(3).mean()
    df["vol_regime"] = np.where(
        (atr_smooth > 2.0) & (adx_smooth > 25), 2,
        np.where((atr_smooth < 0.5) | (adx_smooth < 15), 0, 1)
    ).astype(float)

    # Legacy order flow
    if "taker_buy_base_vol" in df.columns:
        vol_safe = v.replace(0, np.nan)
        df["taker_buy_ratio"] = (
            df["taker_buy_base_vol"].astype(float) / vol_safe
        ).fillna(0.5).clip(0, 1)
    else:
        df["taker_buy_ratio"] = 0.5

    # Legacy calendar encoding
    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        hour = ts.dt.hour.fillna(0)
        dow = ts.dt.dayofweek.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0

    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _legacy_production_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical feature formulas used by locked production v2.0."""
    if df is None or df.empty:
        return df

    df = df.copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)

    # Legacy EMAs / price relationships
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff(3) / df["ema20"].shift(3) * 100
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3) * 100
    df["price_vs_ema20"] = (c - df["ema20"]) / df["ema20"] * 100
    df["price_vs_ema50"] = (c - df["ema50"]) / df["ema50"] * 100
    df["price_vs_ema200"] = (c - df["ema200"]) / df["ema200"] * 100
    df["ema20_vs_ema50"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100

    # Legacy RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)

    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    df["rsi_slope"] = df["rsi"].diff(3)

    avg_g7 = gain.ewm(com=6, adjust=False).mean()
    avg_l7 = loss.ewm(com=6, adjust=False).mean()
    rs7 = avg_g7 / avg_l7.replace(0, np.nan)

    df["rsi_fast"] = (100 - 100 / (1 + rs7)).fillna(50)

    # Legacy stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()

    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Legacy MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_slope"] = df["macd"].diff(3)

    # Legacy ATR / ADX
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    dm_pos = (h.diff()).clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)

    atr14 = tr.ewm(span=14, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)

    df["adx"] = dx.ewm(span=14, adjust=False).mean()
    df["adx_pos"] = di_pos
    df["adx_neg"] = di_neg
    df["di_diff"] = di_pos - di_neg

    # Legacy Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    bb_high = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    bb_width = bb_high - bb_low

    df["bb_high"] = bb_high
    df["bb_low"] = bb_low
    df["bb_pct"] = (c - bb_low) / (bb_width + 1e-10)
    df["bb_width"] = bb_width / sma20 * 100

    # Legacy volume / VWAP
    vol_ma20 = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma20.replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(5) / (vol_ma20 * 5 + 1e-10)

    df["vwap"] = (c * v).cumsum() / (v.cumsum() + 1e-10)
    df["vwap_dev"] = (c - df["vwap"]) / df["vwap"] * 100

    # Legacy price-action
    df["price_change"] = c.pct_change(1) * 100
    df["price_change3"] = c.pct_change(3) * 100
    df["price_change6"] = c.pct_change(6) * 100
    df["high_low_pct"] = (h - l) / c * 100
    df["body_pct"] = (c - o).abs() / (h - l + 1e-10)
    df["momentum"] = c - c.shift(10)
    df["volatility"] = c.rolling(14).std() / c * 100

    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    df["pivot_dev"] = (c - pivot) / pivot * 100

    # Legacy candlestick patterns
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    rng = h - l + 1e-10

    df["bullish_candle"] = ((c > o) & (body > rng * 0.6)).astype(int)
    df["doji"] = (body < rng * 0.1).astype(int)
    df["hammer"] = ((lower_wick > body * 2) & (upper_wick < body)).astype(int)

    # Legacy trend
    df["trend"] = np.where(
        df["ema20"] > df["ema50"], 1,
        np.where(df["ema20"] < df["ema50"], -1, 0)
    )

    # Legacy volatility regime
    atr_smooth = df["atr_pct"].rolling(5).mean()
    adx_smooth = df["adx"].rolling(3).mean()
    df["vol_regime"] = np.where(
        (atr_smooth > 2.0) & (adx_smooth > 25), 2,
        np.where((atr_smooth < 0.5) | (adx_smooth < 15), 0, 1)
    ).astype(float)

    # Legacy order flow
    if "taker_buy_base_vol" in df.columns:
        vol_safe = v.replace(0, np.nan)
        df["taker_buy_ratio"] = (
            df["taker_buy_base_vol"].astype(float) / vol_safe
        ).fillna(0.5).clip(0, 1)
    else:
        df["taker_buy_ratio"] = 0.5

    # Legacy calendar encoding
    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        hour = ts.dt.hour.fillna(0)
        dow = ts.dt.dayofweek.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0

    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def build_production_features(
    raw15: pd.DataFrame,
    symbol: str,
    btc15: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct the locked v2.0 production scoring frame."""
    df15 = raw15.copy()
    if df15.empty:
        raise RuntimeError(f"{symbol}: empty 15m closed candles")

    df15 = _legacy_production_indicators(df15)

    raw1h = fetch_deribit_1h(symbol)
    if not raw1h.empty:
        df1h_feat = _legacy_production_indicators(raw1h.copy())
        df15 = _align_htf_point_in_time(
            df15, df1h_feat,
            {"rsi": "rsi_1h", "adx": "adx_1h", "trend": "trend_1h"}
        )

        raw4h = aggregate_1h_to_4h(raw1h)
        if not raw4h.empty:
            df4h_feat = _legacy_production_indicators(raw4h.copy())
            df15 = _align_htf_point_in_time(
                df15, df4h_feat,
                {"rsi": "rsi_4h", "trend": "trend_4h"}
            )
        else:
            df15["rsi_4h"] = 50.0
            df15["trend_4h"] = 0.0
    else:
        for c in ["rsi_1h", "adx_1h", "trend_1h", "rsi_4h", "trend_4h"]:
            df15[c] = 0.0

    df15 = _add_btc_cross_features(df15, btc15)

    if "fundingRate" not in df15.columns:
        df15["fundingRate"] = 0.0
    if "regime_transitional" not in df15.columns:
        df15["regime_transitional"] = (df15["trend"] == 0).astype(float)

    df15["symbol"] = symbol
    return df15.reset_index(drop=True)
# ── Inference and Resolution ───────────────────────────────────────────────


def assert_locked_production_schema(production: dict, production_frame: pd.DataFrame) -> None:
    """Fail closed if locked v2.0 production schema cannot be reproduced."""
    actual = model_features(production)
    expected = LOCKED_PRODUCTION_SELECTED_FEATURES
    if len(expected) != 35:
        raise RuntimeError(f"Internal production schema contract is invalid: expected 35 features, got {len(expected)}")
    if actual != expected:
        raise RuntimeError(f"Locked production artifact feature order changed. Expected={expected} Actual={actual}")
    missing = [feature for feature in expected if feature not in production_frame.index]
    if missing:
        raise RuntimeError(f"Locked production compatibility frame is missing {len(missing)} required features: {missing}")
    print(f"Phase 2D | production compatibility schema regression check=PASS ({len(expected)}/{len(expected)} selected features reproduced)")

def model_features(model) -> List[str]:
    best = model.get("best_features")
    if not best:
        raise RuntimeError("Model is missing best_features; refusing to infer the model schema.")
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

    # Models were fitted on NumPy arrays, not named DataFrames.
    # Preserve the locked feature order while avoiding sklearn's repeated
    # "X has feature names" warning during prospective scoring.
    X = np.asarray([[row[f] for f in active]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

    btc15 = fetch_deribit_15m("BTCUSDT")
    if len(btc15) < 80:
        raise RuntimeError("BTCUSDT: insufficient completed 15m candles")
    probe = fetch_deribit_15m("ETHUSDT")
    if len(probe) < 80:
        raise RuntimeError("ETHUSDT: insufficient completed 15m candles")

    c_probe = current_features(probe, "ETHUSDT").iloc[-1].copy()
    p_probe = build_production_features(probe, "ETHUSDT", btc15).iloc[-1].copy()
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
    seen_sets = {name: set(state["seen"].get(name, [])) for name in models}
    new_predictions = {name: 0 for name in models}

    for symbol in SYMBOLS:
        try:
            raw15 = fetch_deribit_15m(symbol)
            if len(raw15) < 80:
                raise RuntimeError("insufficient completed 15m candles")

            c_df = current_features(raw15, symbol)
            p_df = build_production_features(raw15, symbol, btc15)
            if len(c_df) != len(p_df):
                raise RuntimeError(f"candidate/production feature-frame length mismatch: {len(c_df)} != {len(p_df)}")
            current_rows[symbol] = raw15

            max_offset = min(BACKFILL_BARS, len(c_df), len(p_df))
            for offset in range(1, max_offset + 1):
                c_row = c_df.iloc[-offset]
                p_row = p_df.iloc[-offset]
                ts = int(c_row["open_time"])
                entry = safe_float(c_row["close"])
                c_atr = safe_float(c_row.get("atr", 0))
                p_atr = safe_float(p_row.get("atr", c_row.get("atr", 0)))
                if entry <= 0 or c_atr <= 0 or p_atr <= 0:
                    continue

                rows_by_model = {"candidate": c_row, "production": p_row}
                for name, model in models.items():
                    row = rows_by_model[name]
                    scored = score_model(model, row)
                    run_snapshot["signals"][name].append({"symbol": symbol, "open_time": ts, **scored})
                    if scored["signal"] == "NO_TRADE":
                        continue
                    key = f"{symbol}:{ts}"
                    if key in seen_sets[name]:
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
                    seen_sets[name].add(key)
                    new_predictions[name] += 1
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

    promotion_gate = evaluate_promotion(
        state["resolved"]["candidate"],
        state["resolved"]["production"],
    )

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
        "promotion_gate": promotion_gate,
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
            "backfill_bars": BACKFILL_BARS,
            "new_predictions": new_predictions,
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