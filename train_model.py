#!/usr/bin/env python3
# train_model.py — Canonical Parquet Rebuild Engine & Target Geometry Experiment Runner

import os
import sys
import json
import time
import uuid
import logging
import hashlib
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

# ── Scikit-Learn 1.6+ Compatibility Patch for XGBoost in VotingClassifier ──
XGBClassifier._estimator_type = "classifier"
try:
    from sklearn.utils._tags import ClassifierTags
    def _xgb_sklearn_tags(self):
        try:
            tags = super(XGBClassifier, self).__sklearn_tags__()
        except Exception:
            from sklearn.utils._tags import Tags
            tags = Tags()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags
    XGBClassifier.__sklearn_tags__ = _xgb_sklearn_tags
except (ImportError, AttributeError):
    pass

from feature_engineering import add_indicators, ALL_FEATURES, ImportanceSelector
from market_data_integrity import sanitize_closed_candles, merge_completed_htf
from execution_policy import (
    get_file_hash,
    get_policy_hash,
    get_feature_code_hash,
    get_feature_schema_hash,
    build_candidate_config,
    get_config_hash,
)

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

TEST_SPLIT         = 0.20
CALIB_SPLIT        = 0.15
EMBARGO_BARS       = 24
CANDIDATE_MODEL_FILE = "candidate_model.pkl"
CANDIDATE_MANIFEST = Path("candidate_manifest.json")
N_FEATURES         = 30
MIN_BARS           = 100
UNDERSAMPLE_RATIO  = 1.0

MIN_BUY_THRESHOLD_FLOOR  = 0.36
MIN_SELL_THRESHOLD_FLOOR = 0.36

HISTORICAL_DATA_DIR = Path("data/historical")

INTERVAL_MS_MAP = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}

FULL_FEATURES = list(dict.fromkeys(ALL_FEATURES))


# ── Strict HTF Alignment & Base Feature Generation ────────────────────

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
        duration_ms = INTERVAL_MS_MAP.get(interval, 15 * 60 * 1000)
        clean = sanitize_closed_candles(df, candle_duration_ms=duration_ms)
        out = pd.DataFrame(clean)
        return out.sort_values("open_time").reset_index(drop=True)
    except Exception as e:
        log.warning(f"Failed loading Parquet segment {path}: {e}")
        return pd.DataFrame()


def _build_features(symbol, df15, df1h, df4h, regime, btc_df15=None) -> pd.DataFrame:
    """Calculates all indicators and HTF alignments. Does NOT apply targets or truncate lookahead."""
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()

    taker_col = None
    if "taker_buy_base_vol" in df15.columns:
        taker_col = df15[["open_time", "taker_buy_base_vol"]].copy()

    df15 = add_indicators(df15)
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:
        df15 = df15.merge(taker_col, on="open_time", how="left")

    if not df1h.empty:
        df1h_feat = add_indicators(df1h)
        df15 = _align_1h_to_15m(df1h_feat, df15)
        if df15.empty: return pd.DataFrame()
    else:
        return pd.DataFrame()

    if not df4h.empty:
        df4h_feat = add_indicators(df4h)
        df15 = _align_4h_to_15m(df4h_feat, df15)
        if df15.empty: return pd.DataFrame()
    else:
        return pd.DataFrame()

    df15 = _align_btc_to_15m(btc_df15, df15)
    if "htf1h_source_close_time" not in df15.columns or "htf4h_source_close_time" not in df15.columns:
        return pd.DataFrame()

    df15["symbol"] = symbol
    df15["regime"] = regime
    return df15.copy()


def preload_symbol_features() -> list[pd.DataFrame]:
    if not HISTORICAL_DATA_DIR.exists() or len(list(HISTORICAL_DATA_DIR.glob("*.parquet"))) < 10:
        log.warning("data/historical/ incomplete. Invoking download_training_data.py...")
        from download_training_data import build_canonical_archive
        build_canonical_archive()

    log.info(f"Preloading Base Indicator Features across {len(SYMBOLS)} symbols (I/O Heavy)...")
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


# ── Dynamic Target Generation (Phase 1 Variable Isolation) ────────────

def make_targets(df: pd.DataFrame, sell_tp_mult: float, sell_sl_mult: float) -> pd.Series:
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
            
        # BUY target geometry strictly held constant across experiments
        buy_tp = entry + atr * ATR_TARGET1_MULT
        buy_sl = entry - atr * ATR_STOP_MULT
        
        # SELL target geometry dynamically adjusted for isolation testing
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
    """Applies target logic dynamically, truncates lookahead bars, and drops ambiguous setups."""
    df = df.copy()
    df["target"] = make_targets(df, sell_tp_mult, sell_sl_mult)
    if len(df) <= 24:
        return pd.DataFrame()
    df = df.iloc[:-24].copy()
    return df[df["target"] != "AMBIGUOUS"].copy()


def build_dataset_from_local_parquet(sell_tp_mult=3.5, sell_sl_mult=2.5) -> pd.DataFrame:
    """Legacy wrapper for train_meta_model.py backward compatibility."""
    raw_list = preload_symbol_features()
    processed = [_apply_targets(df, sell_tp_mult, sell_sl_mult) for df in raw_list]
    return pd.concat(processed, ignore_index=True)


# ── Training & Chronological Evaluation ───────────────────────────────

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
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else base.copy()
    return tuple(x.sort_values(["open_time", "symbol"]).reset_index(drop=True) for x in (train_df, calib_df, test_df))


def undersample_no_trade(X_train: pd.DataFrame, y_train: np.ndarray, nt_idx: int, ratio: float = UNDERSAMPLE_RATIO) -> tuple:
    signal_mask  = y_train != nt_idx
    signal_idx   = np.where(signal_mask)[0]
    no_trade_idx = np.where(~signal_mask)[0]
    target_nt    = min(int(len(signal_idx) * ratio), len(no_trade_idx))

    rng = np.random.default_rng(42)
    sampled_nt = rng.choice(no_trade_idx, size=target_nt, replace=False)
    keep = np.sort(np.concatenate([signal_idx, sampled_nt]))
    return X_train.iloc[keep].reset_index(drop=True), y_train[keep]


def train(ds: pd.DataFrame, sell_tp_mult: float, sell_sl_mult: float, export_artifact: bool = True):
    ds = ds.loc[:, ~ds.columns.duplicated()].copy()
    for f in FULL_FEATURES:
        if f not in ds.columns: ds[f] = 0.0

    le = LabelEncoder()
    le.fit(ds["target"])
    classes = list(le.classes_)

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    train_df, calib_df, test_df = temporal_symbol_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)
    
    if export_artifact:
        n = len(ds)
        b, s, nt = (ds.target == "BUY").sum(), (ds.target == "SELL").sum(), (ds.target == "NO_TRADE").sum()
        log.info(f"DATASET DISTRIBUTION: {n:,} rows | BUY: {b:,} ({b/n*100:.1f}%) | SELL: {s:,} ({s/n*100:.1f}%) | NO_TRADE: {nt:,} ({nt/n*100:.1f}%)")

    train_df = train_df.sort_values("open_time").reset_index(drop=True)
    X_train_raw = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_raw = le.transform(train_df["target"])

    X_calib = calib_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_calib = le.transform(calib_df["target"]) if len(calib_df) > 0 else np.array([])
    X_test = test_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = le.transform(test_df["target"]) if len(test_df) > 0 else np.array([])

    scanner = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train_raw, y_train_raw)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in FULL_FEATURES]
    for i in top_idx:
        feat = FULL_FEATURES[i]
        if feat not in selected: selected.append(feat)
        if len(selected) >= N_FEATURES: break

    X_train_raw_sel = X_train_raw[selected]
    Xte, Xcal = X_test[selected].values, X_calib[selected].values
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

    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)], voting="soft", weights=[3, 2, 1])
    ensemble.fit(Xtr, y_train)

    calibrated_ensemble = CalibratedClassifierCV(estimator=FrozenEstimator(ensemble), method="isotonic")
    calibrated_ensemble.fit(Xcal, y_calib)
    ensemble = calibrated_ensemble

    calib_probas = ensemble.predict_proba(Xcal)
    calib_buy_n, calib_sell_n = (y_calib == buy_idx).sum(), (y_calib == sell_idx).sum()

    best_thresh_buy, best_score_buy   = MIN_BUY_THRESHOLD_FLOOR, 0.0
    best_thresh_sell, best_score_sell = MIN_SELL_THRESHOLD_FLOOR, 0.0
    friction_r = 0.12

    # Break-Even Pre-Calculations
    be_buy = (ATR_STOP_MULT + friction_r) / (ATR_TARGET1_MULT + ATR_STOP_MULT)
    be_sell = (sell_sl_mult + friction_r) / (sell_tp_mult + sell_sl_mult)
    
    if export_artifact:
        log.info(f"\n{'='*95}")
        log.info(f"CALIBRATION THRESHOLD SWEEP (Friction={friction_r}R)")
        log.info(f"BUY Geometry:  {ATR_TARGET1_MULT}R / {ATR_STOP_MULT}R (Req Prec > {be_buy*100:.1f}%)")
        log.info(f"SELL Geometry: {sell_tp_mult}R / {sell_sl_mult}R (Req Prec > {be_sell*100:.1f}%)")
        log.info(f"{'='*95}")
        log.info(
            f"{'Thresh':<7} | {'BUY N':<6} {'BUY Prec':<9} {'BUY Rec':<8} {'BUY EV':<8} {'Score':<7} | "
            f"{'SELL N':<7} {'SELL Prec':<10} {'SELL Rec':<9} {'SELL EV':<8} {'Score':<7}"
        )
        log.info(f"{'-'*95}")

    sweep_thresholds = np.round(np.arange(0.32, 0.62, 0.02), 2)
    for thresh in sweep_thresholds:
        yp = []
        for p in calib_probas:
            p_buy, p_sell = p[buy_idx], p[sell_idx]
            if p_buy >= thresh and p_buy > p_sell: yp.append(buy_idx)
            elif p_sell >= thresh and p_sell > p_buy: yp.append(sell_idx)
            else: yp.append(nt_idx)
                
        yp = np.array(yp)
        bm, sm = (yp == buy_idx), (yp == sell_idx)

        pb = float((y_calib[bm] == buy_idx).mean())  if bm.sum() > 0 else 0.0
        ps = float((y_calib[sm] == sell_idx).mean()) if sm.sum() > 0 else 0.0
        rb = float((yp[y_calib == buy_idx] == buy_idx).mean())   if calib_buy_n > 0 else 0.0
        rs = float((yp[y_calib == sell_idx] == sell_idx).mean()) if calib_sell_n > 0 else 0.0

        buy_ev  = pb * ATR_TARGET1_MULT - (1.0 - pb) * ATR_STOP_MULT - friction_r
        sell_ev = ps * sell_tp_mult - (1.0 - ps) * sell_sl_mult - friction_r

        buy_score  = buy_ev * rb * np.sqrt(max(bm.sum(), 1))  if buy_ev > 0 else 0.0
        sell_score = sell_ev * rs * np.sqrt(max(sm.sum(), 1)) if sell_ev > 0 else 0.0

        if export_artifact:
            log.info(
                f"{thresh:<7.2f} | {bm.sum():<6} {pb*100:>7.1f}%  {rb*100:>6.1f}%  {buy_ev:>+6.2f}R {buy_score:>7.2f} | "
                f"{sm.sum():<7} {ps*100:>8.1f}%  {rs*100:>7.1f}%  {sell_ev:>+6.2f}R {sell_score:>7.2f}"
            )

        if thresh >= MIN_BUY_THRESHOLD_FLOOR and buy_score > best_score_buy and bm.sum() > 15:
            best_score_buy, best_thresh_buy = buy_score, thresh
        if thresh >= MIN_SELL_THRESHOLD_FLOOR and sell_score > best_score_sell and sm.sum() > 15:
            best_score_sell, best_thresh_sell = sell_score, thresh

    best_thresh_buy  = max(MIN_BUY_THRESHOLD_FLOOR, best_thresh_buy)
    best_thresh_sell = max(MIN_SELL_THRESHOLD_FLOOR, best_thresh_sell)

    probas = ensemble.predict_proba(Xte)
    y_pred_tuned = []
    for p in probas:
        p_buy, p_sell = p[buy_idx], p[sell_idx]
        if p_buy >= best_thresh_buy and p_buy > p_sell: y_pred_tuned.append(buy_idx)
        elif p_sell >= best_thresh_sell and p_sell > p_buy: y_pred_tuned.append(sell_idx)
        else: y_pred_tuned.append(nt_idx)
            
    y_pred_tuned = np.array(y_pred_tuned)
    acc = accuracy_score(y_test, y_pred_tuned)

    if not export_artifact:
        return acc, best_score_buy, best_score_sell

    train_acc = accuracy_score(y_train, ensemble.predict(Xtr))
    report = classification_report(y_test, y_pred_tuned, target_names=classes, output_dict=True, zero_division=0)
    
    log.info(f"{'-'*95}")
    log.info(f"✅ Selected Optimal Thresholds -> BUY: {best_thresh_buy:.2f} (Score: {best_score_buy:.2f}) | SELL: {best_thresh_sell:.2f} (Score: {best_score_sell:.2f})")
    log.info(f"{'='*95}\n")
    log.info(f"Generalization Check: In-Sample (Train)={train_acc*100:.1f}% | Out-of-Sample (Test)={acc*100:.1f}%")

    now_utc = datetime.now(timezone.utc)
    pipeline = {
        "ensemble":                   ensemble,
        "selector":                   ImportanceSelector(selected),
        "all_features":               FULL_FEATURES,
        "best_features":              selected,
        "label_map":                  {i: c for i, c in enumerate(classes)},
        "label_encoder":              le,
        "accuracy":                   round(acc * 100, 1),
        "trained_at":                 now_utc.isoformat(),
        "symbols":                    SYMBOLS,
        "n_features":                 len(FULL_FEATURES),
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "recommended_threshold":      max(best_thresh_buy, best_thresh_sell),
        "calibrated":                 True,
    }

    joblib.dump(pipeline, CANDIDATE_MODEL_FILE, compress=3)
    log.info(f"✅ Exported compressed candidate binary: {CANDIDATE_MODEL_FILE}")

    with open(CANDIDATE_MODEL_FILE, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()

    candidate_id = f"cand_{uuid.uuid4().hex}"
    manifest = {
        "artifact_version": 1,
        "candidate_id": candidate_id,
        "model_sha256": model_sha256,
        "feature_schema_hash": get_feature_schema_hash(FULL_FEATURES),
        "feature_code_hash": get_feature_code_hash(),
        "execution_policy_hash": get_policy_hash(),
        "decision_policy_hash": "",
        "config_hash": get_config_hash(build_candidate_config()),
        "candidate_created_at": now_utc.isoformat(),
        "candidate_expiry_at": (now_utc + timedelta(days=30)).isoformat(),
        "status": "AWAITING_PROSPECTIVE_EVIDENCE",
    }

    with open(CANDIDATE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    perf = {
        "candidate_id":               candidate_id,
        "accuracy":                   round(acc * 100, 1),
        "test_accuracy":              f"{round(acc * 100, 1)}%",
        "train_accuracy":             f"{round(train_acc * 100, 1)}%",
        "n_train":                    int(len(X_train_raw)),
        "n_calib":                    int(len(X_calib)),
        "n_train_sampled":            int(len(y_train)),
        "n_test":                     int(len(X_test)),
        "features":                   FULL_FEATURES,
        "selected":                   selected,
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "buy_precision":              round(report.get("BUY", {}).get("precision", 0), 4),
        "sell_precision":             round(report.get("SELL", {}).get("precision", 0), 4),
    }

    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)

    return acc, best_score_buy, best_score_sell


if __name__ == "__main__":
    t0 = time.time()
    log.info("Starting Target Geometry Experiment (Phase 1)...")
    
    preloaded_symbols = preload_symbol_features()
    
    experiment_grid = [
        {"name": "Current Baseline", "tp": 3.5, "sl": 2.5},
        {"name": "Moderate Relax",   "tp": 3.0, "sl": 2.5},
        {"name": "Symmetric",        "tp": 2.5, "sl": 2.5},
        {"name": "Aggressive Relax", "tp": 2.0, "sl": 2.0},
    ]
    
    results = []
    
    for config in experiment_grid:
        log.info(f"\n{'='*95}")
        log.info(f"🚀 RUNNING GEOMETRY EXPERIMENT: {config['name']} (SELL TP: {config['tp']}R | SELL SL: {config['sl']}R)")
        log.info(f"{'='*95}")
        
        ds_parts = [_apply_targets(df, config["tp"], config["sl"]) for df in preloaded_symbols]
        ds = pd.concat(ds_parts, ignore_index=True)
        
        acc, best_score_buy, best_score_sell = train(
            ds, 
            sell_tp_mult=config["tp"], 
            sell_sl_mult=config["sl"], 
            export_artifact=False
        )
        
        results.append({
            "Config": config["name"],
            "TP/SL": f"{config['tp']}R/{config['sl']}R",
            "BUY Score": f"{best_score_buy:.2f}",
            "SELL Score": f"{best_score_sell:.2f}",
            "Accuracy": f"{acc*100:.1f}%",
        })
        
    log.info(f"\n{'='*80}")
    log.info("🎯 TARGET GEOMETRY EXPERIMENT SUMMARY (Chronological Lock, Undersampling=1.0)")
    log.info(f"{'='*80}")
    log.info(f"{'Config':<20} | {'TP/SL':<12} | {'BUY Score':<10} | {'SELL Score':<10} | {'Accuracy':<10}")
    log.info("-" * 80)
    for res in results:
        log.info(f"{res['Config']:<20} | {res['TP/SL']:<12} | {res['BUY Score']:<10} | {res['SELL Score']:<10} | {res['Accuracy']:<10}")
    log.info("=" * 80)
    
    log.info("\nExecuting Final Export Run to satisfy CI pipeline schema (Baseline 3.5R/2.5R)...")
    ds_final = pd.concat([_apply_targets(df, 3.5, 2.5) for df in preloaded_symbols], ignore_index=True)
    acc_final, _, _ = train(ds_final, sell_tp_mult=3.5, sell_sl_mult=2.5, export_artifact=True)
    
    log.info(f"Phase 1 Experiment & Build complete in {(time.time()-t0)/60:.1f} min")
