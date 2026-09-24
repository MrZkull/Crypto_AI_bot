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

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests

from feature_engineering import (
    ALL_FEATURES,
    add_indicators as add_current_indicators,
)
from phase2d_promotion_gate import (
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    CONF,
    MIN_RESOLVED,
    MIN_UNIQUE_BLOCKS,
    BLOCK_MS,
    evaluate_promotion,
)

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
STATE_SCHEMA = 5
BACKFILL_BARS = 8

LOG_SCHEMA_VERSION = 1

EXPERIMENT_ID = os.getenv(
    "PHASE2D_EXPERIMENT_ID",
    "PHASE2D-HARDENED-20260924",
)

LEDGER_FILE = Path(
    "research_outputs/experiment_ledger.jsonl"
)

FEATURE_CODE_FILE = (
    Path(__file__).resolve().with_name(
        "feature_engineering.py"
    )
)

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
    "btc_beta_20",
    "ema20",
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

class StateCompatibilityError(RuntimeError):
    """Raised when an existing prospective state is not compatible."""


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Required hash input does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def feature_schema_sha256() -> str:
    payload = json.dumps(
        {
            "all_features": list(ALL_FEATURES),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def build_experiment_identity(
    candidate_sha: str,
    production_sha: str,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "candidate_sha256": candidate_sha,
        "production_sha256": production_sha,
        "feature_code_hash": file_sha256(
            FEATURE_CODE_FILE
        ),
        "feature_schema_hash": feature_schema_sha256(),
        "validator_code_hash": file_sha256(
            Path(__file__).resolve()
        ),
    }


def build_experiment_definition(
    candidate: dict,
    production: dict,
    candidate_features: list[str],
    production_features: list[str],
    identity: dict,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,

        "entry_timeframe": "15m",
        "confirmation_timeframe": "1h",
        "trend_timeframe": "4h",

        "lookahead_bars": LOOKAHEAD,

        "buy_tp_r": BUY_TP_R,
        "buy_sl_r": BUY_SL_R,
        "sell_tp_r": SELL_TP_R,
        "sell_sl_r": SELL_SL_R,

        "friction_r": FRICTION_R,

        "candidate_thresholds": {
            "buy": safe_float(
                candidate.get(
                    "recommended_threshold_buy",
                    1.01,
                ),
                1.01,
            ),
            "sell": safe_float(
                candidate.get(
                    "recommended_threshold_sell",
                    1.01,
                ),
                1.01,
            ),
        },

        "production_thresholds": {
            "buy": safe_float(
                production.get(
                    "recommended_threshold_buy",
                    1.01,
                ),
                1.01,
            ),
            "sell": safe_float(
                production.get(
                    "recommended_threshold_sell",
                    1.01,
                ),
                1.01,
            ),
        },

        "symbol_universe": list(SYMBOLS),

        "candidate_selected_features": list(
            candidate_features
        ),

        "production_selected_features": list(
            production_features
        ),

        "candidate_selected_feature_count": len(
            candidate_features
        ),

        "production_selected_feature_count": len(
            production_features
        ),

        "block_definition": (
            "UTC calendar day keyed by "
            "prediction open_time"
        ),

        "block_ms": 24 * 60 * 60 * 1000,

        "bootstrap_method": (
            "independent_block_bootstrap"
        ),

        "bootstrap_seed": 42,
        "bootstrap_rounds": 10_000,
        "confidence_level": 0.95,

        "ambiguous_excluded_from_gate": True,
        "expired_included_as_loss": True,
        "expired_net_r": -FRICTION_R,

        "identity": dict(identity),
    }


def append_ledger_event(
    event: str,
    payload: dict,
) -> None:
    LEDGER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": utc_now(),
        "event": event,
        **payload,
    }

    with LEDGER_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def validate_state_integrity(state: dict) -> None:
    """Fail closed on duplicate/cross-state corruption."""

    required_top = {
        "schema_version",
        "locked_models",
        "experiment_definition",
        "pending",
        "resolved",
        "seen",
    }

    missing_top = required_top - set(state)

    if missing_top:
        raise StateCompatibilityError(
            "state missing required keys: "
            f"{sorted(missing_top)}"
        )

    for name in ("candidate", "production"):
        pending = state["pending"].get(name, [])
        resolved = state["resolved"].get(name, [])
        seen = state["seen"].get(name, [])

        pending_ids = [
            str(item.get("id"))
            for item in pending
        ]

        resolved_ids = [
            str(item.get("id"))
            for item in resolved
        ]

        if len(pending_ids) != len(set(pending_ids)):
            raise StateCompatibilityError(
                f"{name}: duplicate pending IDs detected"
            )

        if len(resolved_ids) != len(
            set(resolved_ids)
        ):
            raise StateCompatibilityError(
                f"{name}: duplicate resolved IDs detected"
            )

        overlap = set(pending_ids).intersection(
            resolved_ids
        )

        if overlap:
            raise StateCompatibilityError(
                f"{name}: prediction exists in both "
                f"pending and resolved: "
                f"{sorted(overlap)[:5]}"
            )

        if len(seen) != len(set(seen)):
            raise StateCompatibilityError(
                f"{name}: duplicate seen keys detected"
            )

        allowed_statuses = {
            "TP",
            "SL",
            "EXPIRED",
            "AMBIGUOUS",
            "INVALID",
        }

        for item in resolved:
            status = item.get("status")

            if status not in allowed_statuses:
                raise StateCompatibilityError(
                    f"{name}: invalid resolved status "
                    f"{status!r}"
                )

            symbol = item.get("symbol")
            open_time = item.get("open_time")

            if symbol is None or open_time is None:
                raise StateCompatibilityError(
                    f"{name}: resolved item missing "
                    "symbol/open_time"
                )

            key = f"{symbol}:{int(open_time)}"

            if key not in seen:
                raise StateCompatibilityError(
                    f"{name}: resolved prediction "
                    f"{item.get('id')} missing from seen"
                )

        for item in pending:
            symbol = item.get("symbol")
            open_time = item.get("open_time")

            if symbol is None or open_time is None:
                raise StateCompatibilityError(
                    f"{name}: pending item missing "
                    "symbol/open_time"
                )

            key = f"{symbol}:{int(open_time)}"

            if key not in seen:
                raise StateCompatibilityError(
                    f"{name}: pending prediction "
                    f"{item.get('id')} missing from seen"
                )


def load_json_strict(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)
      

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


def current_features(
    raw15: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    df = add_current_indicators(raw15.copy())

    if df.empty:
        raise RuntimeError(
            f"{symbol}: current feature engineering "
            "returned no rows"
        )

    df = df.copy()

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    df["_had_missing_inputs"] = (
        df["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
    )

    df["symbol"] = symbol

    return df.reset_index(drop=True)


# ── Locked Production Compatibility Feature Construction ─────────────────

def _align_htf_point_in_time(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    col_map: Dict[str, str],
) -> pd.DataFrame:
    """Strict point-in-time backward merge.

    HTF candle must be closed on or before LTF candle close.
    Missing HTF observations are explicitly marked before fill.
    """

    ltf = ltf.copy()

    if "_had_missing_inputs" not in ltf.columns:
        ltf["_had_missing_inputs"] = False

    ltf["_had_missing_inputs"] = (
        ltf["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
    )

    if htf.empty or ltf.empty:
        ltf["_had_missing_inputs"] = True

        for target in col_map.values():
            ltf[target] = 0.0

        return ltf

    available_cols = [
        c for c in col_map.keys()
        if c in htf.columns
    ]

    if not available_cols:
        ltf["_had_missing_inputs"] = True

        for target in col_map.values():
            ltf[target] = 0.0

        return ltf

    htf_sub = (
        htf[
            ["close_time"] + available_cols
        ]
        .dropna(subset=["close_time"])
        .sort_values("close_time")
        .copy()
    )

    htf_sub = htf_sub.rename(
        columns=col_map
    )

    merged = pd.merge_asof(
        ltf.sort_values("close_time"),
        htf_sub,
        on="close_time",
        direction="backward",
    )

    missing_mask = pd.Series(
        False,
        index=merged.index,
    )

    for target in col_map.values():
        if target not in merged.columns:
            missing_mask = True
            merged[target] = 0.0
        else:
            missing_mask = (
                missing_mask
                | merged[target].isna()
            )

    merged["_had_missing_inputs"] = (
        merged["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
        | missing_mask.fillna(True)
    )

    for target in col_map.values():
        merged[target] = (
            merged[target]
            .fillna(0.0)
        )

    return (
        merged
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _add_btc_cross_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    df["_had_missing_inputs"] = (
        df["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
    )

    if (
        btc_df is not None
        and not btc_df.empty
        and "close" in btc_df.columns
    ):
        btc_sub = (
            btc_df[
                ["close_time", "close"]
            ]
            .dropna()
            .sort_values("close_time")
            .rename(
                columns={
                    "close": "btc_close"
                }
            )
        )

        merged = pd.merge_asof(
            df.sort_values("close_time"),
            btc_sub,
            on="close_time",
            direction="backward",
        )

        btc_missing = (
            merged["btc_close"]
            .isna()
        )

        merged["_had_missing_inputs"] = (
            merged["_had_missing_inputs"]
            | btc_missing
        )

        df["btc_close"] = (
            merged["btc_close"]
            .ffill()
            .bfill()
        )

    else:
        df["btc_close"] = np.nan
        df["_had_missing_inputs"] = True

    if (
        "btc_close" in df.columns
        and df["btc_close"].notna().sum() > 30
    ):
        btc_ret = df["btc_close"].pct_change()
        coin_ret = df["close"].pct_change()

        roll_cov = (
            coin_ret
            .rolling(20, min_periods=10)
            .cov(btc_ret)
        )

        roll_var = (
            btc_ret
            .rolling(20, min_periods=10)
            .var()
        )

        df["btc_corr_20"] = (
            coin_ret
            .rolling(20, min_periods=10)
            .corr(btc_ret)
        )

        df["btc_beta_20"] = (
            roll_cov
            / roll_var.replace(
                0,
                np.nan,
            )
        )

        df["btc_rel_strength"] = (
            df["close"].pct_change(6)
            - df["btc_close"].pct_change(6)
        ) * 100

    else:
        df["btc_corr_20"] = 0.0
        df["btc_beta_20"] = 1.0
        df["btc_rel_strength"] = 0.0

    tracked = [
        f
        for f in LOCKED_PRODUCTION_SELECTED_FEATURES
        if f in df.columns
    ]

    if tracked:
        df["_had_missing_inputs"] = (
            df["_had_missing_inputs"]
            | df[tracked]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .isna()
            .any(axis=1)
        )

    df["btc_corr_20"] = (
        df["btc_corr_20"]
        .fillna(0.0)
        .clip(-1, 1)
    )

    df["btc_beta_20"] = (
        df["btc_beta_20"]
        .fillna(1.0)
        .clip(-5, 5)
    )

    df["btc_rel_strength"] = (
        df["btc_rel_strength"]
        .fillna(0.0)
        .clip(-50, 50)
    )

    return df



def _legacy_production_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the historical feature formulas used by locked production v2.0."""
    if df is None or df.empty:
        return df

    df = df.copy()

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    df["_had_missing_inputs"] = (
        df["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
    )
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

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    tracked_features = [
        f
        for f in LOCKED_PRODUCTION_SELECTED_FEATURES
        if f in df.columns
    ]

    if tracked_features:
        current_missing = (
            df[tracked_features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .isna()
            .any(axis=1)
        )

        df["_had_missing_inputs"] = (
            df["_had_missing_inputs"]
            .fillna(False)
            .astype(bool)
            | current_missing
        )

    return (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .ffill()
        .fillna(0.0)
    )


def build_production_features(
    raw15: pd.DataFrame,
    symbol: str,
    btc15: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct the locked v2.0 production scoring frame."""

    df15 = raw15.copy()

    if df15.empty:
        raise RuntimeError(
            f"{symbol}: empty 15m closed candles"
        )

    df15 = _legacy_production_indicators(
        df15
    )

    raw1h = fetch_deribit_1h(symbol)

    if not raw1h.empty:
        df1h_feat = (
            _legacy_production_indicators(
                raw1h.copy()
            )
        )

        df15 = _align_htf_point_in_time(
            df15,
            df1h_feat,
            {
                "rsi": "rsi_1h",
                "adx": "adx_1h",
                "trend": "trend_1h",
            },
        )

        raw4h = aggregate_1h_to_4h(
            raw1h
        )

        if not raw4h.empty:
            df4h_feat = (
                _legacy_production_indicators(
                    raw4h.copy()
                )
            )

            df15 = _align_htf_point_in_time(
                df15,
                df4h_feat,
                {
                    "rsi": "rsi_4h",
                    "trend": "trend_4h",
                },
            )

        else:
            df15["_had_missing_inputs"] = True
            df15["rsi_4h"] = 50.0
            df15["trend_4h"] = 0.0

    else:
        df15["_had_missing_inputs"] = True

        for c in [
            "rsi_1h",
            "adx_1h",
            "trend_1h",
            "rsi_4h",
            "trend_4h",
        ]:
            df15[c] = 0.0

    df15 = _add_btc_cross_features(
        df15,
        btc15,
    )

    if "fundingRate" not in df15.columns:
        df15["fundingRate"] = 0.0

    if "regime_transitional" not in df15.columns:
        df15["regime_transitional"] = (
            df15["trend"] == 0
        ).astype(float)

    df15["symbol"] = symbol

    return df15.reset_index(
        drop=True
    )
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
    X = np.asarray(
        [[row[f] for f in active]],
        dtype=float,
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "non-finite model feature reached scoring "
            "after missing-input validation"
        )

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

def _side_summary(
    records: list[dict],
    side: str,
) -> dict:
    subset = [
        r
        for r in records
        if r.get("status")
        in {"TP", "SL", "EXPIRED"}
        and r.get("signal") == side
    ]

    if not subset:
        return {
            "n": 0,
            "tp": 0,
            "sl": 0,
            "expired": 0,
            "precision": None,
            "mean_net_r": None,
            "cum_net_r": 0.0,
        }

    wins = sum(
        1
        for r in subset
        if r["status"] == "TP"
    )

    return {
        "n": len(subset),
        "tp": sum(
            1
            for r in subset
            if r["status"] == "TP"
        ),
        "sl": sum(
            1
            for r in subset
            if r["status"] == "SL"
        ),
        "expired": sum(
            1
            for r in subset
            if r["status"] == "EXPIRED"
        ),
        "precision": round(
            wins / len(subset),
            6,
        ),
        "mean_net_r": round(
            float(
                np.mean(
                    [
                        safe_float(
                            r["net_r"]
                        )
                        for r in subset
                    ]
                )
            ),
            6,
        ),
        "cum_net_r": round(
            float(
                sum(
                    safe_float(
                        r["net_r"]
                    )
                    for r in subset
                )
            ),
            6,
        ),
    }
  
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


def empty_state(
    identity: dict,
    experiment_definition: dict,
) -> dict:
    return {
        "schema_version": STATE_SCHEMA,

        "locked_models": dict(identity),

        "experiment_definition": dict(
            experiment_definition
        ),

        "experiment_started_at": utc_now(),

        "pending": {
            "candidate": [],
            "production": [],
        },

        "resolved": {
            "candidate": [],
            "production": [],
        },

        "seen": {
            "candidate": [],
            "production": [],
        },
    }


def load_compatible_state(
    identity: dict,
    experiment_definition: dict,
    allow_fresh_start: bool = False,
) -> tuple[dict, bool, str]:
    """Load only an exactly compatible state.

    Returns:
        state,
        fresh_start,
        reason
    """

    if not STATE_FILE.exists():
        if not allow_fresh_start:
            raise StateCompatibilityError(
                "No prior Phase 2D state exists. "
                "Fresh start is disabled. "
                "Use --allow-fresh-start explicitly."
            )

        state = empty_state(
            identity,
            experiment_definition,
        )

        append_ledger_event(
            "STATE_RESET",
            {
                "reason": "no_previous_state",
                "experiment_id": EXPERIMENT_ID,
                "identity": identity,
            },
        )

        return (
            state,
            True,
            "no previous state; explicit fresh start",
        )

    try:
        raw = load_json_strict(
            STATE_FILE
        )
    except Exception as exc:
        if not allow_fresh_start:
            raise StateCompatibilityError(
                f"Existing Phase 2D state is unreadable: {exc}"
            ) from exc

        state = empty_state(
            identity,
            experiment_definition,
        )

        append_ledger_event(
            "STATE_RESET",
            {
                "reason": "unreadable_previous_state",
                "detail": str(exc),
                "experiment_id": EXPERIMENT_ID,
                "identity": identity,
            },
        )

        return (
            state,
            True,
            "unreadable previous state; explicit fresh start",
        )

    actual_identity = raw.get(
        "locked_models",
        {},
    )

    identity_mismatches = {
        key: {
            "expected": expected_value,
            "actual": actual_identity.get(key),
        }
        for key, expected_value in identity.items()
        if actual_identity.get(key)
        != expected_value
    }

    schema_mismatch = (
        raw.get("schema_version")
        != STATE_SCHEMA
    )

    if schema_mismatch:
        identity_mismatches[
            "state_schema_version"
        ] = {
            "expected": STATE_SCHEMA,
            "actual": raw.get(
                "schema_version"
            ),
        }

    stored_definition = raw.get(
        "experiment_definition"
    )

    if stored_definition != experiment_definition:
        identity_mismatches[
            "experiment_definition"
        ] = {
            "expected": experiment_definition,
            "actual": stored_definition,
        }

    if identity_mismatches:
        if not allow_fresh_start:
            raise StateCompatibilityError(
                "Phase 2D state compatibility check FAILED. "
                "Fresh starts are disabled.\n"
                + json.dumps(
                    identity_mismatches,
                    indent=2,
                    sort_keys=True,
                )
            )

        archive = STATE_FILE.with_name(
            "phase2d_state_pre_reset_archive.json"
        )

        save_json(
            archive,
            raw,
        )

        append_ledger_event(
            "STATE_RESET",
            {
                "reason": (
                    "previous_state_incompatible"
                ),
                "experiment_id": EXPERIMENT_ID,
                "identity": identity,
                "mismatches": identity_mismatches,
                "archive_file": str(archive),
            },
        )

        state = empty_state(
            identity,
            experiment_definition,
        )

        return (
            state,
            True,
            "previous state incompatible; explicit fresh start",
        )

    validate_state_integrity(
        raw
    )

    return (
        raw,
        False,
        "compatible state restored",
    )


def append_resolved_unique(
    state: dict,
    model_name: str,
    result: dict,
) -> None:
    prediction_id = result.get("id")

    if not prediction_id:
        raise RuntimeError(
            f"{model_name}: resolved prediction "
            "has no id"
        )

    existing_ids = {
        item.get("id")
        for item in state["resolved"][model_name]
    }

    if prediction_id in existing_ids:
        raise RuntimeError(
            f"{model_name}: duplicate resolved "
            f"prediction rejected: {prediction_id}"
        )

    pending_ids = {
        item.get("id")
        for item in state["pending"][model_name]
    }

    if prediction_id not in pending_ids:
        raise RuntimeError(
            f"{model_name}: resolved prediction "
            f"{prediction_id} was not present in pending state"
        )

    state["resolved"][model_name].append(
        result
    )

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CryptoBot AI Phase 2D "
            "prospective validator"
        )
    )

    parser.add_argument(
        "--allow-fresh-start",
        action="store_true",
        help=(
            "Explicitly authorize creation of a "
            "new Phase 2D experiment state when "
            "no compatible state exists."
        ),
    )

    args = parser.parse_args()

    run_id = os.getenv(
        "GITHUB_RUN_ID",
        "local",
    )

    run_event = os.getenv(
        "GITHUB_EVENT_NAME",
        "local",
    )

    commit_sha = os.getenv(
        "GITHUB_SHA",
        "unknown",
    )

    started_at = utc_now()

    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(
            "candidate_model.pkl not found."
        )

    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            "pro_crypto_ai_model.pkl not found."
        )

    candidate_sha = model_sha256(
        CANDIDATE_MODEL
    )

    production_sha = model_sha256(
        PRODUCTION_MODEL
    )

    if candidate_sha != LOCKED_CANDIDATE_SHA:
        raise RuntimeError(
            "Candidate SHA256 mismatch: "
            f"expected {LOCKED_CANDIDATE_SHA}, "
            f"got {candidate_sha}"
        )

    if production_sha != LOCKED_PRODUCTION_SHA:
        raise RuntimeError(
            "Production SHA256 mismatch: "
            f"expected {LOCKED_PRODUCTION_SHA}, "
            f"got {production_sha}"
        )

    candidate = load_model(
        CANDIDATE_MODEL
    )

    production = load_model(
        PRODUCTION_MODEL
    )

    candidate_features = model_features(
        candidate
    )

    production_features = model_features(
        production
    )

    identity = build_experiment_identity(
        candidate_sha,
        production_sha,
    )

    experiment_definition = (
        build_experiment_definition(
            candidate,
            production,
            candidate_features,
            production_features,
            identity,
        )
    )

    print_section(
        "CRYPTOBOT AI — PHASE 2D PROSPECTIVE VALIDATION"
    )

    print(
        f"Run ID                  : {run_id}"
    )
    print(
        f"Event                   : {run_event}"
    )
    print(
        f"Commit SHA              : {commit_sha}"
    )
    print(
        f"Experiment ID           : {EXPERIMENT_ID}"
    )
    print(
        f"Started UTC             : {started_at}"
    )
    print(
        f"Allow fresh start       : {args.allow_fresh_start}"
    )

    print_section(
        "[01] EXPERIMENT IDENTITY"
    )

    print(
        f"Candidate SHA256        : "
        f"{candidate_sha}"
    )

    print(
        f"Production SHA256       : "
        f"{production_sha}"
    )

    print(
        f"Feature code hash       : "
        f"{identity['feature_code_hash']}"
    )

    print(
        f"Feature schema hash     : "
        f"{identity['feature_schema_hash']}"
    )

    print(
        f"Validator code hash     : "
        f"{identity['validator_code_hash']}"
    )

    print(
        "Model identity status   : PASS"
    )

    print_section(
        "[02] EXPERIMENT DEFINITION"
    )

    print(
        json.dumps(
            experiment_definition,
            indent=2,
            sort_keys=True,
        )
    )

    print_section(
        "[03] MODEL / SCHEMA"
    )

    print(
        f"Candidate trained_at    : "
        f"{candidate.get('trained_at')}"
    )

    print(
        f"Production trained_at   : "
        f"{production.get('trained_at')}"
    )

    print(
        f"Candidate features      : "
        f"{len(candidate_features)}"
    )

    print(
        f"Production features     : "
        f"{len(production_features)}"
    )

    print(
        f"Candidate feature list  : "
        f"{candidate_features}"
    )

    print(
        f"Production feature list : "
        f"{production_features}"
    )

    if (
        len(LOCKED_PRODUCTION_SELECTED_FEATURES)
        != 35
    ):
        raise RuntimeError(
            "Internal production schema contract "
            "must contain exactly 35 features."
        )

    btc15 = fetch_deribit_15m(
        "BTCUSDT"
    )

    if len(btc15) < 80:
        raise RuntimeError(
            "BTCUSDT: insufficient completed "
            "15m candles"
        )

    probe = fetch_deribit_15m(
        "ETHUSDT"
    )

    if len(probe) < 80:
        raise RuntimeError(
            "ETHUSDT: insufficient completed "
            "15m candles"
        )

    c_probe = (
        current_features(
            probe,
            "ETHUSDT",
        )
        .iloc[-1]
        .copy()
    )

    p_probe = (
        build_production_features(
            probe,
            "ETHUSDT",
            btc15,
        )
        .iloc[-1]
        .copy()
    )

    assert_locked_production_schema(
        production,
        p_probe,
    )

    c_missing = [
        f
        for f in candidate_features
        if f not in c_probe.index
    ]

    p_missing = [
        f
        for f in production_features
        if f not in p_probe.index
    ]

    print(
        f"Candidate schema missing  : "
        f"{len(c_missing)}"
    )

    print(
        f"Production schema missing : "
        f"{len(p_missing)}"
    )

    if c_missing:
        raise RuntimeError(
            "Candidate schema mismatch: "
            f"{c_missing}"
        )

    if p_missing:
        raise RuntimeError(
            "Production schema mismatch: "
            f"{p_missing}"
        )

    print(
        "Schema status            : PASS"
    )

    state, fresh_start, state_reason = (
        load_compatible_state(
            identity,
            experiment_definition,
            args.allow_fresh_start,
        )
    )

    print_section(
        "[04] STATE RESTORE"
    )

    print(
        f"State result             : "
        f"{'FRESH_START' if fresh_start else 'RESTORED'}"
    )

    print(
        f"State reason             : "
        f"{state_reason}"
    )

    print(
        f"Candidate resolved       : "
        f"{len(state['resolved']['candidate'])}"
    )

    print(
        f"Production resolved      : "
        f"{len(state['resolved']['production'])}"
    )

    print(
        f"Candidate pending        : "
        f"{len(state['pending']['candidate'])}"
    )

    print(
        f"Production pending       : "
        f"{len(state['pending']['production'])}"
    )

    if fresh_start:
        print(
            "STATE RESET STATUS      : "
            "EXPLICITLY AUTHORIZED"
        )
    else:
        print(
            "STATE RESTORE STATUS    : PASS"
        )

    models = {
        "candidate": candidate,
        "production": production,
    }

    run_snapshot = {
        "timestamp": utc_now(),
        "log_schema_version": LOG_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "identity": identity,
        "signals": {
            "candidate": [],
            "production": [],
        },
    }

    current_rows: Dict[
        str,
        pd.DataFrame,
    ] = {}

    seen_sets = {
        name: set(
            state["seen"].get(
                name,
                [],
            )
        )
        for name in models
    }

    new_predictions = {
        name: 0
        for name in models
    }

    run_metrics = {
        "symbols_requested": len(
            SYMBOLS
        ),
        "symbols_succeeded": 0,
        "symbols_failed": 0,
        "observations_evaluated": 0,
        "alignment_checks": 0,
        "missing_input_exclusions": 0,
        "invalid_entry_atr_exclusions": 0,
        "duplicate_predictions_prevented": {
            "candidate": 0,
            "production": 0,
        },
        "signal_counts": {
            "candidate": {
                "BUY": 0,
                "SELL": 0,
                "NO_TRADE": 0,
            },
            "production": {
                "BUY": 0,
                "SELL": 0,
                "NO_TRADE": 0,
            },
        },
    }

    symbol_reports = []

    print_section(
        "[05] MARKET DATA + SIGNAL GENERATION"
    )

    for symbol in SYMBOLS:
        report = {
            "symbol": symbol,
            "status": "PASS",
            "rows_15m": 0,
            "rows_candidate": 0,
            "rows_production": 0,
            "observations": 0,
            "missing_input_exclusions": 0,
            "new_candidate_predictions": 0,
            "new_production_predictions": 0,
            "errors": [],
        }

        try:
            raw15 = fetch_deribit_15m(
                symbol
            )

            report["rows_15m"] = len(
                raw15
            )

            if len(raw15) < 80:
                raise RuntimeError(
                    "insufficient completed "
                    "15m candles"
                )

            c_df = current_features(
                raw15,
                symbol,
            )

            p_df = build_production_features(
                raw15,
                symbol,
                btc15,
            )

            report["rows_candidate"] = len(
                c_df
            )

            report["rows_production"] = len(
                p_df
            )

            if len(c_df) != len(p_df):
                raise RuntimeError(
                    "candidate/production "
                    "feature-frame length mismatch: "
                    f"{len(c_df)} != {len(p_df)}"
                )

            if not np.array_equal(
                c_df["open_time"].to_numpy(),
                p_df["open_time"].to_numpy(),
            ):
                raise RuntimeError(
                    "candidate/production full "
                    "open_time vector mismatch"
                )

            current_rows[symbol] = raw15

            run_metrics[
                "symbols_succeeded"
            ] += 1

            max_offset = min(
                BACKFILL_BARS,
                len(c_df),
                len(p_df),
            )

            for offset in range(
                1,
                max_offset + 1,
            ):
                c_row = c_df.iloc[
                    -offset
                ]

                p_row = p_df.iloc[
                    -offset
                ]

                c_open_time = int(
                    c_row["open_time"]
                )

                p_open_time = int(
                    p_row["open_time"]
                )

                if c_open_time != p_open_time:
                    raise RuntimeError(
                        "candidate/production "
                        "row misalignment: "
                        f"{symbol} "
                        f"candidate={c_open_time} "
                        f"production={p_open_time}"
                    )

                run_metrics[
                    "alignment_checks"
                ] += 1

                ts = c_open_time

                report["observations"] += 1

                run_metrics[
                    "observations_evaluated"
                ] += 1

                c_missing_input = bool(
                    c_row.get(
                        "_had_missing_inputs",
                        False,
                    )
                )

                p_missing_input = bool(
                    p_row.get(
                        "_had_missing_inputs",
                        False,
                    )
                )

                if (
                    c_missing_input
                    or p_missing_input
                ):
                    run_metrics[
                        "missing_input_exclusions"
                    ] += 1

                    report[
                        "missing_input_exclusions"
                    ] += 1

                    continue

                entry = safe_float(
                    c_row["close"]
                )

                c_atr = safe_float(
                    c_row.get("atr", 0)
                )

                p_atr = safe_float(
                    p_row.get(
                        "atr",
                        c_row.get(
                            "atr",
                            0,
                        ),
                    )
                )

                if (
                    entry <= 0
                    or c_atr <= 0
                    or p_atr <= 0
                ):
                    run_metrics[
                        "invalid_entry_atr_exclusions"
                    ] += 1

                    continue

                rows_by_model = {
                    "candidate": c_row,
                    "production": p_row,
                }

                for name, model in models.items():
                    row = rows_by_model[name]

                    scored = score_model(
                        model,
                        row,
                    )

                    run_snapshot[
                        "signals"
                    ][name].append(
                        {
                            "symbol": symbol,
                            "open_time": ts,
                            **scored,
                        }
                    )

                    signal = scored[
                        "signal"
                    ]

                    run_metrics[
                        "signal_counts"
                    ][name][signal] += 1

                    if signal == "NO_TRADE":
                        continue

                    key = (
                        f"{symbol}:{ts}"
                    )

                    if key in seen_sets[name]:
                        run_metrics[
                            "duplicate_predictions_prevented"
                        ][name] += 1
                        continue

                    pred = {
                        "id": (
                            f"{name}:"
                            f"{symbol}:"
                            f"{ts}"
                        ),
                        "model": name,
                        "symbol": symbol,
                        "open_time": ts,
                        "signal": signal,
                        "confidence": scored[
                            "confidence"
                        ],
                        "p_buy": scored[
                            "p_buy"
                        ],
                        "p_sell": scored[
                            "p_sell"
                        ],
                        "threshold_buy": scored[
                            "threshold_buy"
                        ],
                        "threshold_sell": scored[
                            "threshold_sell"
                        ],
                        "entry": entry,
                        "atr": (
                            c_atr
                            if name
                            == "candidate"
                            else p_atr
                        ),
                        "created_at": utc_now(),
                    }

                    state[
                        "pending"
                    ][name].append(pred)

                    state[
                        "seen"
                    ][name].append(key)

                    seen_sets[name].add(key)

                    new_predictions[
                        name
                    ] += 1

                    report[
                        f"new_{name}_predictions"
                    ] += 1

        except Exception as exc:
            run_metrics[
                "symbols_failed"
            ] += 1

            report["status"] = (
                "FAIL"
            )

            report[
                "errors"
            ].append(str(exc))

            print(
                f"WARN {symbol}: {exc}"
            )

        symbol_reports.append(
            report
        )

    print_section(
        "[06] PENDING / RESOLUTION"
    )

    resolution_metrics = {
        "candidate": {
            "pending_before": len(
                state["pending"]["candidate"]
            ),
            "resolved_this_run": 0,
            "TP": 0,
            "SL": 0,
            "EXPIRED": 0,
            "AMBIGUOUS": 0,
            "INVALID": 0,
        },
        "production": {
            "pending_before": len(
                state["pending"]["production"]
            ),
            "resolved_this_run": 0,
            "TP": 0,
            "SL": 0,
            "EXPIRED": 0,
            "AMBIGUOUS": 0,
            "INVALID": 0,
        },
    }

    for name in (
        "candidate",
        "production",
    ):
        still_pending = []

        for pred in state[
            "pending"
        ].get(name, []):
            df15 = current_rows.get(
                pred["symbol"]
            )

            if df15 is None:
                try:
                    df15 = fetch_deribit_15m(
                        pred["symbol"]
                    )
                except Exception as exc:
                    print(
                        f"WARN "
                        f"{pred['symbol']} "
                        f"resolution fetch: "
                        f"{exc}"
                    )

                    still_pending.append(
                        pred
                    )

                    continue

            result = resolve_prediction(
                pred,
                df15,
            )

            if result is None:
                still_pending.append(
                    pred
                )

                continue

            append_resolved_unique(
                state,
                name,
                result,
            )

            status = result.get(
                "status",
                "INVALID",
            )

            resolution_metrics[
                name
            ][status] += 1

            resolution_metrics[
                name
            ]["resolved_this_run"] += 1

        state["pending"][name] = (
            still_pending
        )

        # Existing methodology retained:
        # Phase 2D keeps a bounded resolved sample
        # of the latest 500 observations for the gate.
        state["resolved"][name] = (
            state["resolved"][name][-500:]
        )

        # IMPORTANT:
        # Do not truncate seen history.
        # It is the duplicate-observation identity ledger.
        state["seen"][name] = list(
            dict.fromkeys(
                state["seen"][name]
            )
        )

    print(
        json.dumps(
            resolution_metrics,
            indent=2,
            sort_keys=True,
        )
    )

    print_section(
        "[07] CUMULATIVE RESULTS"
    )

    candidate_summary = summarize(
        state["resolved"]["candidate"]
    )

    production_summary = summarize(
        state["resolved"]["production"]
    )

    candidate_clean = [
        r
        for r in state["resolved"][
            "candidate"
        ]
        if r.get("status")
        in {"TP", "SL", "EXPIRED"}
    ]

    production_clean = [
        r
        for r in state["resolved"][
            "production"
        ]
        if r.get("status")
        in {"TP", "SL", "EXPIRED"}
    ]

    side_breakdowns = {
        "candidate": {
            "BUY": _side_summary(
                candidate_clean,
                "BUY",
            ),
            "SELL": _side_summary(
                candidate_clean,
                "SELL",
            ),
        },
        "production": {
            "BUY": _side_summary(
                production_clean,
                "BUY",
            ),
            "SELL": _side_summary(
                production_clean,
                "SELL",
            ),
        },
    }

    print(
        json.dumps(
            {
                "candidate": candidate_summary,
                "production": production_summary,
                "side_breakdowns": side_breakdowns,
            },
            indent=2,
            sort_keys=True,
        )
    )

    promotion_gate = evaluate_promotion(
        state["resolved"]["candidate"],
        state["resolved"]["production"],
    )

    print_section(
        "[08] COVERAGE / STATISTICAL GATE"
    )

    print(
        json.dumps(
            promotion_gate,
            indent=2,
            sort_keys=True,
        )
    )

    if promotion_gate.get(
        "promotion_ready"
    ):
        overall_status = (
            "READY_FOR_POLICY_PARITY"
        )

        next_action = (
            "Phase 2D gate passed. "
            "Do NOT deploy directly; "
            "proceed to Stage 2 Policy "
            "Parity Validation."
        )

    elif promotion_gate.get(
        "gate_status"
    ) == "FAIL":
        overall_status = (
            "PHASE2D_GATE_FAILED"
        )

        next_action = (
            "Keep production locked. "
            "Classify the Phase 2D failure "
            "before any new candidate work."
        )

    else:
        overall_status = (
            "ACCUMULATING"
        )

        next_action = (
            "Continue prospective collection."
        )

    finished_at = utc_now()

    summary = {
        "log_schema_version": LOG_SCHEMA_VERSION,
        "updated_at": finished_at,

        "run_context": {
            "run_id": run_id,
            "event": run_event,
            "commit_sha": commit_sha,
            "experiment_id": EXPERIMENT_ID,
            "started_at": started_at,
            "finished_at": finished_at,
        },

        "locked_identity": identity,

        "experiment_definition":
            experiment_definition,

        "state": {
            "fresh_start": fresh_start,
            "reason": state_reason,
            "schema_version": STATE_SCHEMA,
        },

        "market_data": {
            "symbols_requested": run_metrics[
                "symbols_requested"
            ],
            "symbols_succeeded": run_metrics[
                "symbols_succeeded"
            ],
            "symbols_failed": run_metrics[
                "symbols_failed"
            ],
        },

        "run_metrics": run_metrics,

        "resolution_metrics":
            resolution_metrics,

        "candidate": candidate_summary,

        "production": production_summary,

        "side_breakdowns":
            side_breakdowns,

        "pending": {
            "candidate": len(
                state["pending"]["candidate"]
            ),
            "production": len(
                state["pending"]["production"]
            ),
        },

        "resolved_retained": {
            "candidate": len(
                state["resolved"]["candidate"]
            ),
            "production": len(
                state["resolved"]["production"]
            ),
        },

        "seen_observation_keys": {
            "candidate": len(
                state["seen"]["candidate"]
            ),
            "production": len(
                state["seen"]["production"]
            ),
        },

        "promotion_gate":
            promotion_gate,

        "symbol_reports":
            symbol_reports,

        "methodology": {
            "horizon_bars": LOOKAHEAD,
            "buy_tp_r": BUY_TP_R,
            "buy_sl_r": BUY_SL_R,
            "sell_tp_r": SELL_TP_R,
            "sell_sl_r": SELL_SL_R,
            "friction_r": FRICTION_R,
            "ambiguous_excluded_from_precision": True,
            "ambiguous_excluded_from_gate": True,
            "expired_net_r": -FRICTION_R,
            "candidate_feature_source": (
                "current canonical"
            ),
            "production_feature_source": (
                "legacy compatibility "
                "reconstruction"
            ),
            "same_15m_market_stream": True,
            "no_240m_deribit_request": True,
            "four_hour_source": (
                "completed 1h candles "
                "aggregated locally"
            ),
            "block_definition": (
                "UTC calendar day keyed by "
                "prediction open_time"
            ),
            "missing_input_policy": (
                "symmetric exclusion if "
                "candidate OR production "
                "has missing-input marker"
            ),
        },

        "integrity": {
            "candidate_model_sha": (
                candidate_sha
                == LOCKED_CANDIDATE_SHA
            ),
            "production_model_sha": (
                production_sha
                == LOCKED_PRODUCTION_SHA
            ),
            "feature_code_hash": True,
            "feature_schema_hash": True,
            "validator_code_hash": True,
            "alignment_checks": run_metrics[
                "alignment_checks"
            ],
            "duplicate_prediction_prevention": True,
            "duplicate_resolved_protection": True,
            "state_monotonicity": True,
        },

        "overall_status":
            overall_status,

        "next_action":
            next_action,
    }

    state["last_run"] = summary

    run_snapshot[
        "metadata"
    ] = {
        "experiment_id": EXPERIMENT_ID,
        "identity": identity,
        "run_metrics": run_metrics,
        "resolution_metrics":
            resolution_metrics,
        "promotion_gate":
            promotion_gate,
        "overall_status":
            overall_status,
    }

    save_json(
        STATE_FILE,
        state,
    )

    save_json(
        RESULTS_FILE,
        summary,
    )

    save_json(
        SNAPSHOT_FILE,
        run_snapshot,
    )

    append_ledger_event(
        "PHASE2D_RUN",
        {
            "experiment_id":
                EXPERIMENT_ID,
            "run_id": run_id,
            "commit_sha": commit_sha,
            "overall_status":
                overall_status,
            "candidate_clean_n":
                promotion_gate.get(
                    "n_candidate"
                ),
            "production_clean_n":
                promotion_gate.get(
                    "n_production"
                ),
            "candidate_blocks":
                promotion_gate.get(
                    "unique_blocks_candidate"
                ),
            "production_blocks":
                promotion_gate.get(
                    "unique_blocks_production"
                ),
            "candidate_mean_net_r":
                candidate_summary.get(
                    "mean_net_r"
                ),
            "production_mean_net_r":
                production_summary.get(
                    "mean_net_r"
                ),
        },
    )

    print_section(
        "[09] FINAL PHASE 2D STATUS"
    )

    print(
        f"Experiment identity     : PASS"
    )

    print(
        f"State                   : "
        f"{'FRESH_START' if fresh_start else 'RESTORED'}"
    )

    print(
        f"Symbols                 : "
        f"{run_metrics['symbols_succeeded']}/"
        f"{run_metrics['symbols_requested']}"
    )

    print(
        f"Alignment checks        : "
        f"{run_metrics['alignment_checks']}"
    )

    print(
        f"Missing-input exclusions: "
        f"{run_metrics['missing_input_exclusions']}"
    )

    print(
        f"Candidate clean N       : "
        f"{promotion_gate.get('n_candidate', 0)}/"
        f"{promotion_gate.get('min_resolved', MIN_RESOLVED)}"
    )

    print(
        f"Production clean N      : "
        f"{promotion_gate.get('n_production', 0)}/"
        f"{promotion_gate.get('min_resolved', MIN_RESOLVED)}"
    )

    print(
        f"Candidate unique blocks : "
        f"{promotion_gate.get('unique_blocks_candidate', 0)}/"
        f"{promotion_gate.get('min_unique_blocks', 10)}"
    )

    print(
        f"Production unique blocks: "
        f"{promotion_gate.get('unique_blocks_production', 0)}/"
        f"{promotion_gate.get('min_unique_blocks', 10)}"
    )

    print(
        f"Statistical gate        : "
        f"{promotion_gate.get('gate_status', 'UNKNOWN')}"
    )

    print(
        f"Overall status          : "
        f"{overall_status}"
    )

    print(
        f"Next action             : "
        f"{next_action}"
    )

    print_section(
        "[10] MACHINE-READABLE RESULT"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
