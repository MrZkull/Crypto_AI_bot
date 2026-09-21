#!/usr/bin/env python3
# train_model.py — Canonical Training Engine, Anti-Leakage Audit & Manifest Pipeline

import os
import sys
import json
import time
import uuid
import logging
import hashlib
import argparse
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

# Scikit-Learn 1.6+ compatibility patch for XGBoost in VotingClassifier
XGBClassifier._estimator_type = "classifier"
try:
    from sklearn.utils._tags import ClassifierTags, Tags
    def _xgb_sklearn_tags(self):
        try:
            tags = super(XGBClassifier, self).__sklearn_tags__()
        except Exception:
            tags = Tags()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags
    XGBClassifier.__sklearn_tags__ = _xgb_sklearn_tags
except (ImportError, AttributeError):
    pass

from feature_engineering import add_indicators, ALL_FEATURES, ImportanceSelector
from market_data_integrity import sanitize_closed_candles, merge_completed_htf, interval_ms

try:
    from execution_policy import (
        get_policy_hash, get_feature_code_hash, get_feature_schema_hash,
        build_candidate_config, get_config_hash
    )
except ImportError:
    def get_policy_hash(): return hashlib.sha256(b"default_policy").hexdigest()
    def get_feature_code_hash(): return hashlib.sha256(b"default_code").hexdigest()
    def get_feature_schema_hash(features): return hashlib.sha256(json.dumps(sorted(features)).encode()).hexdigest()
    def build_candidate_config(): return {
        "evaluation_balance_usd": 10000.0,
        "risk_mult": 1.0,
        "atr_stop_mult": 2.5,
        "atr_target1_mult": 3.5,
        "atr_target2_mult": 7.5,
        "friction_r": 0.12
    }
    def get_config_hash(cfg): return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

try:
    from config import SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT
except ImportError:
    SYMBOLS = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
        "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
        "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
        "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT", "FILUSDT"
    ]
    ATR_STOP_MULT    = 2.5
    ATR_TARGET1_MULT = 3.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TEST_SPLIT                 = 0.20
CALIB_SPLIT                = 0.15
EMBARGO_BARS               = 24
MODEL_FILE                 = Path("pro_crypto_ai_model.pkl")
CANDIDATE_MODEL_FILE       = Path("candidate_model.pkl")
CANDIDATE_MANIFEST         = Path("candidate_manifest.json")
N_FEATURES                 = 35
MIN_BARS                   = 100
UNDERSAMPLE_RATIO          = 1.0
THRESHOLD_VALIDATION_SPLIT = 0.40
THRESHOLD_GATE_SPLIT       = 0.50
MIN_THRESHOLD_GATE_TRADES  = 30

MIN_BUY_THRESHOLD_FLOOR  = 0.36
MIN_SELL_THRESHOLD_FLOOR = 0.36

HISTORICAL_DATA_DIR = Path("data/historical")
BASE_FEATURES = list(dict.fromkeys(ALL_FEATURES))
FULL_FEATURES = list(dict.fromkeys(BASE_FEATURES + ["fundingRate", "btc_corr_20", "btc_beta_20", "btc_rel_strength"]))


def _write_json_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


MAX_MODEL_BYTES = 95 * 1024 * 1024


def _dump_model_atomic(model, path: Path, compress: int = 3) -> int:
    """Write a compressed joblib model atomically and enforce a size guard."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        joblib.dump(model, tmp, compress=compress)
        size_bytes = tmp.stat().st_size

        if size_bytes <= 0:
            raise ValueError(f"CRITICAL: model artifact is empty: {tmp}")

        if size_bytes > MAX_MODEL_BYTES:
            raise ValueError(
                f"CRITICAL: compressed model artifact exceeds the "
                f"{MAX_MODEL_BYTES / (1024 * 1024):.1f} MiB safety ceiling: "
                f"{size_bytes / (1024 * 1024):.1f} MiB"
            )

        os.replace(tmp, path)
        return int(size_bytes)

    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def audit_anti_leakage(ds: pd.DataFrame) -> dict:
    """Fail-closed provenance audit for every temporal feature source."""
    required = ["open_time", "close_time"]
    missing = [c for c in required if c not in ds.columns]
    if missing:
        raise ValueError(f"CRITICAL LEAKAGE AUDIT: missing required columns: {missing}")

    out = {"rows": int(len(ds)), "violations": {}, "total_violations": 0}
    obs = pd.to_numeric(ds["close_time"], errors="coerce")
    if obs.isna().any():
        raise ValueError("CRITICAL LEAKAGE AUDIT: invalid observation close_time")

    provenance_cols = [
        c for c in ds.columns
        if c.endswith("_source_close_time") or c == "funding_source_time"
    ]
    for col in provenance_cols:
        src = pd.to_numeric(ds[col], errors="coerce")
        invalid = int(src.notna().sum() - np.isfinite(src.dropna().to_numpy(dtype=float)).sum())
        future = int((src.notna() & (src > obs)).sum())
        if invalid:
            raise ValueError(f"CRITICAL LEAKAGE AUDIT: non-finite provenance in {col}: {invalid}")
        out["violations"][col] = future
        out["total_violations"] += future
        if future:
            raise ValueError(f"CRITICAL LEAKAGE: {col} exceeds observation close_time: {future} rows")

    opens = pd.to_numeric(ds["open_time"], errors="coerce")
    closes = pd.to_numeric(ds["close_time"], errors="coerce")
    bad_order = int((closes < opens).sum())
    if bad_order:
        raise ValueError(f"CRITICAL LEAKAGE AUDIT: close_time precedes open_time: {bad_order} rows")

    log.info("ANTI-LEAKAGE AUDIT OK | rows=%d | provenance_cols=%d | violations=0", len(ds), len(provenance_cols))
    return out


def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if df1h.empty or len(df1h) < 5 or df15.empty:
        return pd.DataFrame()
    h = merge_completed_htf(df15, df1h, ["rsi", "adx", "trend"], prefix="htf1h")
    return h.rename(columns={"htf1h_rsi": "rsi_1h", "htf1h_adx": "adx_1h", "htf1h_trend": "trend_1h"})


def _align_4h_to_15m(df4h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if df4h.empty or len(df4h) < 5 or df15.empty:
        return pd.DataFrame()
    h = merge_completed_htf(df15, df4h, ["rsi", "trend"], prefix="htf4h")
    return h.rename(columns={"htf4h_rsi": "rsi_4h", "htf4h_trend": "trend_4h"})


def _align_btc_to_15m(btc_df15: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if btc_df15 is None or btc_df15.empty or "close" not in btc_df15.columns:
        df15["btc_close"] = np.nan
        return df15
    try:
        out = merge_completed_htf(df15, btc_df15, ["close"], prefix="btc")
        return out.rename(columns={"btc_close": "btc_close"})
    except Exception as e:
        log.warning(f"_align_btc_to_15m failed ({e})")
        df15["btc_close"] = np.nan
        return df15


def load_parquet_segment(symbol: str, interval: str) -> pd.DataFrame:
    path = HISTORICAL_DATA_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        clean = sanitize_closed_candles(df, candle_duration_ms=interval_ms(interval))
        return pd.DataFrame(clean).sort_values("open_time").reset_index(drop=True)
    except Exception as e:
        log.warning(f"Failed loading Parquet segment {path}: {e}")
        return pd.DataFrame()


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def _merge_funding_to_15m(df15: pd.DataFrame, symbol: str) -> pd.DataFrame:
    funding_path = HISTORICAL_DATA_DIR / "funding" / f"{symbol}_funding.parquet"
    if not funding_path.exists():
        raise FileNotFoundError(f"CRITICAL: Missing funding archive for {symbol}: {funding_path}")

    fdf = pd.read_parquet(funding_path)
    if fdf.empty:
        raise ValueError(f"CRITICAL: Funding archive for {symbol} is empty")

    time_col = next((c for c in ("funding_time", "timestamp", "open_time") if c in fdf.columns), None)
    rate_col = next((c for c in ("fundingRate", "funding_rate") if c in fdf.columns), None)

    if time_col is None or rate_col is None:
        raise ValueError(f"CRITICAL: Missing expected funding columns for {symbol}. Found: {fdf.columns.tolist()}")

    fdf = fdf.rename(columns={time_col: "funding_time", rate_col: "fundingRate"}).copy()
    fdf["funding_time"] = pd.to_numeric(fdf["funding_time"], errors="coerce")
    fdf["fundingRate"]  = pd.to_numeric(fdf["fundingRate"], errors="coerce")

    if fdf["funding_time"].isna().any() or fdf["fundingRate"].isna().any():
        raise ValueError(f"CRITICAL: Invalid timestamps/rates in funding source for {symbol}")

    if fdf["funding_time"].duplicated().any():
        raise ValueError(f"CRITICAL: Duplicate funding timestamps detected for {symbol}")

    fdf = fdf.sort_values("funding_time").reset_index(drop=True)
    source_std = float(fdf["fundingRate"].std())
    source_unique = int(fdf["fundingRate"].nunique(dropna=True))

    if source_unique < 2 or not np.isfinite(source_std) or source_std == 0.0:
        raise ValueError(f"CRITICAL: Source funding archive for {symbol} has zero variance")

    if "close_time" not in df15.columns:
        raise ValueError(f"CRITICAL: 15m dataset for {symbol} is missing close_time")

    base = df15.copy()
    base["close_time"] = pd.to_numeric(base["close_time"], errors="coerce")
    base = base.sort_values("close_time").reset_index(drop=True)

    merged = pd.merge_asof(
        base,
        fdf[["funding_time", "fundingRate"]],
        left_on="close_time",
        right_on="funding_time",
        direction="backward",
    )

    if merged["fundingRate"].isna().any():
        raise ValueError(f"CRITICAL: Missing funding coverage after merge for {symbol}")

    if (merged["funding_time"] > merged["close_time"]).any():
        raise ValueError(f"CRITICAL: Lookahead leakage detected for {symbol}")

    merged = merged.rename(columns={"funding_time": "funding_source_time"})
    return merged.sort_values("open_time").reset_index(drop=True)


def _build_features(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, regime: str, btc_df15=None) -> pd.DataFrame:
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()

    df15 = _merge_funding_to_15m(df15, symbol)
    taker_col = df15[["open_time", "taker_buy_base_vol"]].copy() if "taker_buy_base_vol" in df15.columns else None

    df15 = add_indicators(df15)
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:
        df15 = df15.merge(taker_col, on="open_time", how="left")

    if not df1h.empty:
        df15 = _align_1h_to_15m(add_indicators(df1h), df15)
        if df15.empty: return pd.DataFrame()
    else:
        return pd.DataFrame()

    if not df4h.empty:
        df15 = _align_4h_to_15m(add_indicators(df4h), df15)
        if df15.empty: return pd.DataFrame()
    else:
        return pd.DataFrame()

    df15 = _align_btc_to_15m(btc_df15, df15)
    if "htf1h_source_close_time" not in df15.columns or "htf4h_source_close_time" not in df15.columns:
        return pd.DataFrame()

    df15 = _add_extra_features(df15)
    if "fundingRate" not in df15.columns:
        raise ValueError(f"CRITICAL: Funding attachment failed for {symbol}")

    df15["symbol"] = symbol
    df15["regime"] = regime
    return df15.copy()


def preload_symbol_features() -> list[pd.DataFrame]:
    preloaded = []
    btc_df15 = load_parquet_segment("BTCUSDT", "15m")

    for symbol in SYMBOLS:
        df15 = load_parquet_segment(symbol, "15m")
        df1h = load_parquet_segment(symbol, "1h")
        df4h = load_parquet_segment(symbol, "4h")

        seg = _build_features(symbol, df15, df1h, df4h, regime="historical_canonical", btc_df15=btc_df15)
        if not seg.empty:
            preloaded.append(seg)

    if not preloaded:
        raise ValueError("CRITICAL: No valid Parquet datasets qualified.")
    return preloaded


def make_targets(df: pd.DataFrame, sell_tp_mult: float = 3.5, sell_sl_mult: float = 2.5) -> pd.Series:
    n = len(df)
    labels = np.full(n, "NO_TRADE", dtype=object)
    lookahead = 24
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values if "atr" in df.columns else np.zeros(n)

    for i in range(n - lookahead):
        entry = closes[i]
        atr = atrs[i]
        if atr <= 0 or np.isnan(atr):
            continue

        buy_tp = entry + atr * ATR_TARGET1_MULT
        buy_sl = entry - atr * ATR_STOP_MULT
        sell_tp = entry - atr * sell_tp_mult
        sell_sl = entry + atr * sell_sl_mult

        def first_barrier(tp, sl, up):
            for k in range(1, lookahead + 1):
                h, l = highs[i + k], lows[i + k]
                hit_tp = (h >= tp) if up else (l <= tp)
                hit_sl = (l <= sl) if up else (h >= sl)
                if hit_tp and hit_sl: return "BOTH"
                if hit_tp: return "TP"
                if hit_sl: return "SL"
            return "NONE"

        b = first_barrier(buy_tp, buy_sl, True)
        s = first_barrier(sell_tp, sell_sl, False)

        if b == "BOTH" or s == "BOTH" or (b == "TP" and s == "TP"):
            labels[i] = "AMBIGUOUS"
        elif b == "TP" and s != "TP": labels[i] = "BUY"
        elif s == "TP" and b != "TP": labels[i] = "SELL"

    return pd.Series(labels, index=df.index)


def _apply_targets(df: pd.DataFrame, sell_tp_mult: float, sell_sl_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["target"] = make_targets(df, sell_tp_mult, sell_sl_mult)
    if len(df) <= 24:
        return pd.DataFrame()
    df = df.iloc[:-24].copy()
    return df[df["target"] != "AMBIGUOUS"].copy()


def build_dataset_from_local_parquet(sell_tp_mult: float = 3.5, sell_sl_mult: float = 2.5) -> pd.DataFrame:
    raw_list = preload_symbol_features()
    processed = [_apply_targets(df, sell_tp_mult, sell_sl_mult) for df in raw_list]
    return pd.concat(processed, ignore_index=True)


def temporal_symbol_split(ds: pd.DataFrame, test_split: float, calib_split: float, embargo: int):
    train_parts, calib_parts, test_parts = [], [], []
    for sym, grp in ds.groupby("symbol", sort=False):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        n = len(grp)
        test_n = int(n * test_split)
        calib_n = int(n * calib_split)
        test_start = n - test_n
        calib_end = test_start - embargo
        calib_start = calib_end - calib_n
        train_end = calib_start - embargo

        if test_n <= 0 or calib_n <= 0 or train_end <= 0:
            train_parts.append(grp)
            continue

        train_parts.append(grp.iloc[:train_end])
        calib_parts.append(grp.iloc[calib_start:calib_end])
        test_parts.append(grp.iloc[test_start:])

    base = ds.iloc[:0].copy()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else base.copy()
    calib_df = pd.concat(calib_parts, ignore_index=True) if calib_parts else base.copy()
    test_df  = pd.concat(test_parts, ignore_index=True) if test_parts else base.copy()
    return tuple(x.sort_values(["open_time", "symbol"]).reset_index(drop=True) for x in (train_df, calib_df, test_df))


def split_calibration_threshold_validation(
    calib_df: pd.DataFrame,
    validation_fraction: float = THRESHOLD_VALIDATION_SPLIT,
    gate_fraction: float = THRESHOLD_GATE_SPLIT,
    embargo: int = EMBARGO_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the calibration era into calibrator-fit, threshold-tune, and threshold-gate eras.

    The locked test remains untouched. The latest calibration-era slice is reserved as
    a final threshold-gate holdout: it can reject a tuned threshold, but it is never
    used to choose among thresholds. This reduces threshold-selection overfitting.
    """
    if calib_df.empty:
        raise ValueError("CRITICAL: Empty calibration era; cannot select thresholds.")
    if not 0.20 <= validation_fraction <= 0.50:
        raise ValueError("CRITICAL: validation_fraction must be between 0.20 and 0.50.")
    if not 0.25 <= gate_fraction <= 0.75:
        raise ValueError("CRITICAL: gate_fraction must be between 0.25 and 0.75.")

    fit_parts: list[pd.DataFrame] = []
    tune_parts: list[pd.DataFrame] = []
    gate_parts: list[pd.DataFrame] = []

    for symbol, grp in calib_df.groupby("symbol", sort=False):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        n = len(grp)
        holdout_n = max(1, int(n * validation_fraction))
        gate_n = max(1, int(holdout_n * gate_fraction))
        tune_n = holdout_n - gate_n
        gate_start = n - gate_n
        tune_start = gate_start - tune_n
        fit_end = tune_start - embargo

        if tune_n <= 0 or fit_end <= 0:
            continue

        fit_parts.append(grp.iloc[:fit_end])
        tune_parts.append(grp.iloc[tune_start:gate_start])
        gate_parts.append(grp.iloc[gate_start:])

    if not fit_parts or not tune_parts or not gate_parts:
        raise ValueError(
            "CRITICAL: Calibration era too small for chronological threshold "
            "tuning/gating with embargo."
        )

    fit_df = pd.concat(fit_parts, ignore_index=True)
    tune_df = pd.concat(tune_parts, ignore_index=True)
    gate_df = pd.concat(gate_parts, ignore_index=True)

    fit_df = fit_df.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    tune_df = tune_df.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    gate_df = gate_df.sort_values(["open_time", "symbol"]).reset_index(drop=True)

    required = {"BUY", "SELL", "NO_TRADE"}
    for name, frame in (("fit", fit_df), ("tune", tune_df), ("gate", gate_df)):
        labels = set(frame["target"].dropna().unique())
        missing = sorted(required - labels)
        if missing:
            raise ValueError(
                f"CRITICAL: Threshold {name} era missing classes: {missing}"
            )

    return fit_df, tune_df, gate_df


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Two-sided 95% Wilson lower confidence bound for a binomial proportion."""
    if trials <= 0:
        return 0.0
    phat = float(successes) / float(trials)
    zz = z * z
    denom = 1.0 + zz / trials
    center = phat + zz / (2.0 * trials)
    spread = z * np.sqrt((phat * (1.0 - phat) / trials) + (zz / (4.0 * trials * trials)))
    return float((center - spread) / denom)


def _gate_selected_threshold(
    side: str,
    threshold: float,
    probas: np.ndarray,
    y_true: np.ndarray,
    buy_idx: int,
    sell_idx: int,
    tp_mult: float,
    sl_mult: float,
    friction_r: float,
) -> dict:
    """Evaluate one pre-selected threshold on the untouched threshold-gate era."""
    if side == "BUY":
        signal = (probas[:, buy_idx] >= threshold) & (probas[:, buy_idx] > probas[:, sell_idx])
        actual_label = buy_idx
    elif side == "SELL":
        signal = (probas[:, sell_idx] >= threshold) & (probas[:, sell_idx] > probas[:, buy_idx])
        actual_label = sell_idx
    else:
        raise ValueError(f"Unknown side: {side}")

    n = int(signal.sum())
    successes = int((y_true[signal] == actual_label).sum()) if n else 0
    precision = float(successes / n) if n else 0.0
    ev = (
        precision * tp_mult
        - (1.0 - precision) * sl_mult
        - friction_r
        if n else -sl_mult - friction_r
    )
    lower = _wilson_lower_bound(successes, n)
    be = (sl_mult + friction_r) / (tp_mult + sl_mult)

    return {
        "side": side,
        "threshold": float(threshold),
        "trades": n,
        "precision": precision,
        "ev_r": float(ev),
        "precision_wilson_lower_95": lower,
        "break_even_precision": float(be),
        "passes": bool(
            n >= MIN_THRESHOLD_GATE_TRADES
            and ev > 0.0
            and lower > be
        ),
    }


def _audit_calibration_directional_preservation(
    raw_probas: np.ndarray,
    calibrated_probas: np.ndarray,
    buy_idx: int,
    sell_idx: int,
    min_raw_dominance: int = 100,
) -> dict:
    """Fail closed on catastrophic calibration-induced side-ranking collapse."""
    raw_sell_dom = int((raw_probas[:, sell_idx] > raw_probas[:, buy_idx]).sum())
    cal_sell_dom = int((calibrated_probas[:, sell_idx] > calibrated_probas[:, buy_idx]).sum())

    raw_buy_dom = int((raw_probas[:, buy_idx] > raw_probas[:, sell_idx]).sum())
    cal_buy_dom = int((calibrated_probas[:, buy_idx] > calibrated_probas[:, sell_idx]).sum())

    if raw_sell_dom >= min_raw_dominance and cal_sell_dom == 0:
        raise ValueError(
            "CRITICAL CALIBRATION COLLAPSE: SELL-vs-BUY ranking vanished after "
            f"calibration (raw={raw_sell_dom}, calibrated={cal_sell_dom})."
        )
    if raw_buy_dom >= min_raw_dominance and cal_buy_dom == 0:
        raise ValueError(
            "CRITICAL CALIBRATION COLLAPSE: BUY-vs-SELL ranking vanished after "
            f"calibration (raw={raw_buy_dom}, calibrated={cal_buy_dom})."
        )

    return {
        "raw_sell_dominance": raw_sell_dom,
        "calibrated_sell_dominance": cal_sell_dom,
        "raw_buy_dominance": raw_buy_dom,
        "calibrated_buy_dominance": cal_buy_dom,
    }


def undersample_no_trade(X_train: pd.DataFrame, y_train: np.ndarray, nt_idx: int, ratio: float = UNDERSAMPLE_RATIO) -> tuple:
    signal_mask  = y_train != nt_idx
    signal_idx   = np.where(signal_mask)[0]
    no_trade_idx = np.where(~signal_mask)[0]
    target_nt    = min(int(len(signal_idx) * ratio), len(no_trade_idx))

    rng = np.random.default_rng(42)
    sampled_nt = rng.choice(no_trade_idx, size=target_nt, replace=False)
    keep = np.sort(np.concatenate([signal_idx, sampled_nt]))
    return X_train.iloc[keep].reset_index(drop=True), y_train[keep]


def train(
    ds: pd.DataFrame,
    active_features: list = None,
    sell_tp_mult: float = 3.5,
    sell_sl_mult: float = 2.5,
    candidate_mode: bool = True,
    reserved_features: list = None
):
    ds = ds.loc[:, ~ds.columns.duplicated()].copy()
    audit_anti_leakage(ds)

    if active_features is None:
        active_features = [f for f in FULL_FEATURES if f in ds.columns]
        if "fundingRate" in active_features:
            fr_series = ds["fundingRate"].dropna()
            if len(fr_series) == 0 or (fr_series == 0).all() or fr_series.std() == 0:
                log.warning("⚠️ 'fundingRate' is all-zero. Pruning from active features.")
                active_features = [f for f in active_features if f != "fundingRate"]
            else:
                log.info("✓ 'fundingRate' contains active variance. Retaining in active features.")

    for f in active_features:
        if f not in ds.columns: ds[f] = 0.0

    le = LabelEncoder()
    le.fit(ds["target"])
    classes = list(le.classes_)

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    train_df, calib_df, test_df = temporal_symbol_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)

    train_df = train_df.sort_values("open_time").reset_index(drop=True)
    X_train_raw = train_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_raw = le.transform(train_df["target"])

    X_test  = test_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test  = le.transform(test_df["target"]) if len(test_df) > 0 else np.array([])

    scanner = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train_raw, y_train_raw)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in active_features]

    for f in (reserved_features or []):
        if f in active_features and f not in selected:
            selected.append(f)

    for i in top_idx:
        feat = active_features[i]
        if feat not in selected: selected.append(feat)
        if len(selected) >= min(N_FEATURES, len(active_features)): break

    # Calibration split into: Calib-Fit, Threshold-Tune, and Threshold-Gate
    calib_fit_df, threshold_tune_df, threshold_gate_df = split_calibration_threshold_validation(calib_df)

    X_train_raw_sel = X_train_raw[selected]
    Xte = X_test[selected].values
    Xcal_fit = calib_fit_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xtune = threshold_tune_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xgate = threshold_gate_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    Xcal_fit = Xcal_fit[selected].values
    Xtune = Xtune[selected].values
    Xgate = Xgate[selected].values

    y_calib_fit = le.transform(calib_fit_df["target"])
    y_tune = le.transform(threshold_tune_df["target"])
    y_gate = le.transform(threshold_gate_df["target"])

    X_train_sel, y_train = undersample_no_trade(X_train_raw_sel, y_train_raw, nt_idx)
    Xtr = X_train_sel.values

    n_buy, n_sell = max((y_train == buy_idx).sum(), 1), max((y_train == sell_idx).sum(), 1)
    sell_weight = float(n_buy / n_sell) * 2.0
    sw_asym = np.ones(len(y_train))
    sw_asym[y_train == buy_idx]  = 2.0
    sw_asym[y_train == sell_idx] = sell_weight

    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.03, subsample=0.75,
        colsample_bytree=0.75, min_child_weight=5, reg_alpha=1.0, reg_lambda=2.0,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(Xtr, y_train, sample_weight=sw_asym)

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=7, min_samples_leaf=10, max_features="sqrt",
        random_state=42, n_jobs=-1, class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: sell_weight},
    )
    rf.fit(Xtr, y_train)

    gb = HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.03, min_samples_leaf=10,
        l2_regularization=1.5, random_state=42, class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: sell_weight},
    )
    gb.fit(Xtr, y_train)

    base_ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[3, 2, 1],
    )
    base_ensemble.fit(Xtr, y_train)

    raw_tune_probas = base_ensemble.predict_proba(Xtune)

    calibrated_ensemble = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_ensemble),
        method="temperature",
    )
    calibrated_ensemble.fit(Xcal_fit, y_calib_fit)

    tune_probas = calibrated_ensemble.predict_proba(Xtune)
    gate_probas = calibrated_ensemble.predict_proba(Xgate)

    directional_audit = _audit_calibration_directional_preservation(
        raw_tune_probas,
        tune_probas,
        buy_idx,
        sell_idx,
    )

    ensemble = calibrated_ensemble

    tune_buy_n = int((y_tune == buy_idx).sum())
    tune_sell_n = int((y_tune == sell_idx).sum())
    friction_r = 0.12

    be_buy  = (ATR_STOP_MULT + friction_r) / (ATR_TARGET1_MULT + ATR_STOP_MULT)
    be_sell = (sell_sl_mult + friction_r) / (sell_tp_mult + sell_sl_mult)

    sweep_thresholds = np.round(np.arange(0.32, 0.62, 0.02), 2)
    best_thresh_buy, best_score_buy   = MIN_BUY_THRESHOLD_FLOOR, 0.0
    best_thresh_sell, best_score_sell = MIN_SELL_THRESHOLD_FLOOR, 0.0

    log.info(f"\n{'='*102}")
    log.info(f"THRESHOLD TUNE + INDEPENDENT GATE (Friction={friction_r}R | Target={ATR_TARGET1_MULT}R | Stop={ATR_STOP_MULT}R)")
    log.info(f"BUY Geometry:  {ATR_TARGET1_MULT}R / {ATR_STOP_MULT}R (Req Prec > {be_buy*100:.1f}%)")
    log.info(f"SELL Geometry: {sell_tp_mult}R / {sell_sl_mult}R (Req Prec > {be_sell*100:.1f}%)")
    log.info(
        f"Calibration fit rows: {len(calib_fit_df)} | "
        f"Threshold tune rows: {len(threshold_tune_df)} | "
        f"Threshold gate rows: {len(threshold_gate_df)} | "
        f"Embargo bars: {EMBARGO_BARS}"
    )
    log.info(f"Calibration→threshold directional audit: {directional_audit}")
    log.info(f"{'='*102}")
    log.info(
        f"{'Thresh':<7} | {'BUY N':<6} {'BUY Prec':<9} {'BUY Rec':<8} {'BUY EV':<8} {'Score':<7} | "
        f"{'SELL N':<7} {'SELL Prec':<10} {'SELL Rec':<9} {'SELL EV':<8} {'Score':<7}"
    )
    log.info(f"{'-'*102}")

    for thresh in sweep_thresholds:
        yp = []
        for p in tune_probas:
            p_buy, p_sell = p[buy_idx], p[sell_idx]
            if p_buy >= thresh and p_buy > p_sell: yp.append(buy_idx)
            elif p_sell >= thresh and p_sell > p_buy: yp.append(sell_idx)
            else: yp.append(nt_idx)

        yp = np.array(yp)
        bm, sm = (yp == buy_idx), (yp == sell_idx)

        pb = float((y_tune[bm] == buy_idx).mean())  if bm.sum() > 0 else 0.0
        ps = float((y_tune[sm] == sell_idx).mean()) if sm.sum() > 0 else 0.0
        rb = float((yp[y_tune == buy_idx] == buy_idx).mean())   if tune_buy_n > 0 else 0.0
        rs = float((yp[y_tune == sell_idx] == sell_idx).mean()) if tune_sell_n > 0 else 0.0

        buy_ev  = pb * ATR_TARGET1_MULT - (1.0 - pb) * ATR_STOP_MULT - friction_r
        sell_ev = ps * sell_tp_mult - (1.0 - ps) * sell_sl_mult - friction_r

        buy_score  = buy_ev * rb * np.sqrt(max(bm.sum(), 1))  if buy_ev > 0 else 0.0
        sell_score = sell_ev * rs * np.sqrt(max(sm.sum(), 1)) if sell_ev > 0 else 0.0

        log.info(
            f"{thresh:<7.2f} | {bm.sum():<6} {pb*100:>7.1f}%  {rb*100:>6.1f}%  {buy_ev:>+6.2f}R {buy_score:>7.2f} | "
            f"{sm.sum():<7} {ps*100:>8.1f}%  {rs*100:>7.1f}%  {sell_ev:>+6.2f}R {sell_score:>7.2f}"
        )

        if thresh >= MIN_BUY_THRESHOLD_FLOOR and buy_score > best_score_buy and bm.sum() > 15:
            best_score_buy, best_thresh_buy = buy_score, thresh
        if thresh >= MIN_SELL_THRESHOLD_FLOOR and sell_score > best_score_sell and sm.sum() > 15:
            best_score_sell, best_thresh_sell = sell_score, thresh

    # Independent final threshold gate
    gate_buy = _gate_selected_threshold(
        "BUY", best_thresh_buy, gate_probas, y_gate, buy_idx, sell_idx,
        ATR_TARGET1_MULT, ATR_STOP_MULT, friction_r,
    ) if best_score_buy > 0 else {"passes": False, "threshold": best_thresh_buy, "trades": 0}
    gate_sell = _gate_selected_threshold(
        "SELL", best_thresh_sell, gate_probas, y_gate, buy_idx, sell_idx,
        sell_tp_mult, sell_sl_mult, friction_r,
    ) if best_score_sell > 0 else {"passes": False, "threshold": best_thresh_sell, "trades": 0}

    log.info(f"Threshold gate BUY: {gate_buy}")
    log.info(f"Threshold gate SELL: {gate_sell}")

    if not gate_buy.get("passes", False):
        log.warning("⚠️ BUY threshold rejected by independent threshold gate; disabling BUY for this candidate.")
        best_thresh_buy = 1.01
        best_score_buy = 0.0

    if not gate_sell.get("passes", False):
        log.warning("⚠️ SELL threshold rejected by independent threshold gate; disabling SELL for this candidate.")
        best_thresh_sell = 1.01
        best_score_sell = 0.0

    # Fail-closed sentinel check (1.01 disables non-viable sides)
    best_thresh_buy = max(MIN_BUY_THRESHOLD_FLOOR, best_thresh_buy) if best_score_buy > 0.0 else 1.01
    best_thresh_sell = max(MIN_SELL_THRESHOLD_FLOOR, best_thresh_sell) if best_score_sell > 0.0 else 1.01

    log.info(f"{'-'*102}")
    log.info(f"✅ Selected Optimal Thresholds -> BUY: {best_thresh_buy:.2f} (Score: {best_score_buy:.2f}) | SELL: {best_thresh_sell:.2f} (Score: {best_score_sell:.2f})")
    log.info(f"{'='*102}\n")

    probas = ensemble.predict_proba(Xte)
    y_pred_tuned = []
    for p in probas:
        p_buy, p_sell = p[buy_idx], p[sell_idx]
        if p_buy >= best_thresh_buy and p_buy > p_sell: y_pred_tuned.append(buy_idx)
        elif p_sell >= best_thresh_sell and p_sell > p_buy: y_pred_tuned.append(sell_idx)
        else: y_pred_tuned.append(nt_idx)

    y_pred_tuned = np.array(y_pred_tuned)
    acc = accuracy_score(y_test, y_pred_tuned)
    train_acc = accuracy_score(y_train, ensemble.predict(Xtr))
    log.info(f"Generalization Check: In-Sample (Train)={train_acc*100:.1f}% | Out-of-Sample (Test)={acc*100:.1f}%")

    report = classification_report(y_test, y_pred_tuned, target_names=classes, output_dict=True, zero_division=0)
    active_thresh_list = [t for t in (best_thresh_buy, best_thresh_sell) if t <= 1.0]
    scalar_recommended = min(active_thresh_list) if active_thresh_list else 1.01

    now_utc = datetime.now(timezone.utc)
    pipeline = {
        "ensemble":                   ensemble,
        "selector":                   ImportanceSelector(selected),
        "all_features":               active_features,
        "best_features":              selected,
        "label_map":                  {i: c for i, c in enumerate(classes)},
        "label_encoder":              le,
        "accuracy":                   round(acc * 100, 1),
        "trained_at":                 now_utc.isoformat(),
        "symbols":                    SYMBOLS,
        "n_features":                 len(active_features),
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "recommended_threshold":      scalar_recommended,
        "threshold_gate":             {"buy": gate_buy, "sell": gate_sell},
        "calibrated":                 True,
        "buy_tp_mult":                ATR_TARGET1_MULT,
        "buy_sl_mult":                ATR_STOP_MULT,
        "sell_tp_mult":               sell_tp_mult,
        "sell_sl_mult":               sell_sl_mult,
    }

    target_output_file = (
        CANDIDATE_MODEL_FILE
        if candidate_mode
        else MODEL_FILE
    )

    model_size_bytes = _dump_model_atomic(
        pipeline,
        Path(target_output_file),
        compress=3,
    )

    log.info(
        f"✅ Exported compressed model artifact: {target_output_file} "
        f"({model_size_bytes / (1024 * 1024):.1f} MiB)"
    )

    with open(target_output_file, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()

    candidate_cfg = build_candidate_config()
    candidate_id = f"cand_{uuid.uuid4().hex}"
    manifest = {
        "artifact_version": 1,
        "candidate_id": candidate_id,
        "model_sha256": model_sha256,
        "model_size_bytes": model_size_bytes,
        "feature_schema_hash": get_feature_schema_hash(active_features),
        "feature_code_hash": get_feature_code_hash(),
        "execution_policy_hash": get_policy_hash(),
        "config_hash": get_config_hash(candidate_cfg),
        "config": candidate_cfg,
        "candidate_created_at": now_utc.isoformat(),
        "candidate_expiry_at": (now_utc + timedelta(days=30)).isoformat(),
        "status": "AWAITING_PROSPECTIVE_EVIDENCE" if candidate_mode else "ACTIVE_PRODUCTION",
        "recommended_threshold_buy": best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "recommended_threshold": scalar_recommended,
        "accuracy": round(acc * 100, 1),
    }

    if candidate_mode:
        _write_json_atomic(CANDIDATE_MANIFEST, manifest)
        log.info(f"✅ Generated candidate manifest: {CANDIDATE_MANIFEST} (ID: {candidate_id})")

    perf = {
        "candidate_id":               candidate_id,
        "model_size_bytes":           model_size_bytes,
        "accuracy":                   round(acc * 100, 1),
        "test_accuracy":              f"{round(acc * 100, 1)}%",
        "train_accuracy":             f"{round(train_acc * 100, 1)}%",
        "n_train":                    int(len(X_train_raw)),
        "n_calib":                    int(len(calib_df)),
        "n_calib_fit":                int(len(calib_fit_df)),
        "n_threshold_tune":           int(len(threshold_tune_df)),
        "n_threshold_gate":           int(len(threshold_gate_df)),
        "n_train_sampled":            int(len(y_train)),
        "n_test":                     int(len(X_test)),
        "features":                   active_features,
        "selected":                   selected,
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "recommended_threshold":      scalar_recommended,
        "threshold_gate":             {"buy": gate_buy, "sell": gate_sell},
        "buy_precision":              round(report.get("BUY", {}).get("precision", 0), 4),
        "sell_precision":             round(report.get("SELL", {}).get("precision", 0), 4),
        "no_trade_precision":         round(report.get("NO_TRADE", {}).get("precision", 0), 4),
    }
    _write_json_atomic(Path("model_performance.json"), perf)

    return acc, best_score_buy, best_score_sell


def _parse_args():
    parser = argparse.ArgumentParser(
        description="CryptoBot AI Canonical Model Training Engine."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--candidate",
        action="store_true",
        help="Build a prospective candidate artifact (default).",
    )
    mode.add_argument(
        "--production",
        action="store_true",
        help=(
            "Explicitly export to pro_crypto_ai_model.pkl. "
            "Use only from the controlled promotion workflow."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    is_candidate = not bool(args.production)
    t0 = time.time()
    run_mode = "CANDIDATE" if is_candidate else "PRODUCTION"
    export_target = CANDIDATE_MODEL_FILE if is_candidate else MODEL_FILE

    log.info(f"Starting model training pipeline (Mode: {run_mode})...")
    log.info(f"Model export target: {export_target}")

    dataset = build_dataset_from_local_parquet(
        sell_tp_mult=3.5,
        sell_sl_mult=2.5,
    )
    acc, score_b, score_s = train(
        dataset,
        sell_tp_mult=3.5,
        sell_sl_mult=2.5,
        candidate_mode=is_candidate
    )
    log.info(f"✅ Training completed in {(time.time()-t0)/60:.1f} min | Test Accuracy: {acc*100:.1f}%")
