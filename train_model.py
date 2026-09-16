#!/usr/bin/env python3
# train_model.py — Canonical Parquet Rebuild Engine, Feature Ablation & Sentinel Validation

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

# Scikit-Learn 1.6+ compatibility patch for XGBoost in VotingClassifier
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
from market_data_integrity import sanitize_closed_candles, merge_completed_htf, interval_ms

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
MODEL_FILE         = "pro_crypto_ai_model.pkl"
CANDIDATE_MODEL_FILE = "candidate_model.pkl"
N_FEATURES         = 30
MIN_BARS           = 100
UNDERSAMPLE_RATIO  = 1.0

MIN_BUY_THRESHOLD_FLOOR  = 0.36
MIN_SELL_THRESHOLD_FLOOR = 0.36

HISTORICAL_DATA_DIR = Path("data/historical")
BASE_FEATURES = list(dict.fromkeys(ALL_FEATURES))
FULL_FEATURES = list(dict.fromkeys(BASE_FEATURES + ["fundingRate", "btc_corr_20", "btc_beta_20", "btc_rel_strength"]))


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


def _build_features(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, regime: str, btc_df15=None) -> pd.DataFrame:
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()

    taker_col = df15[["open_time", "taker_buy_base_vol"]].copy() if "taker_buy_base_vol" in df15.columns else None
    funding_col = df15[["open_time", "fundingRate"]].copy() if "fundingRate" in df15.columns else None

    df15 = add_indicators(df15)
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:
        df15 = df15.merge(taker_col, on="open_time", how="left")
    if funding_col is not None and "fundingRate" not in df15.columns:
        df15 = df15.merge(funding_col, on="open_time", how="left")

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

    if "btc_close" in df15.columns and df15["btc_close"].notna().sum() > 30:
        btc_ret = df15["btc_close"].pct_change()
        coin_ret = df15["close"].pct_change()
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
        roll_var = btc_ret.rolling(20, min_periods=10).var()
        df15["btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
        df15["btc_beta_20"] = roll_cov / roll_var.replace(0, np.nan)
        df15["btc_rel_strength"] = (df15["close"].pct_change(6) - df15["btc_close"].pct_change(6)) * 100
    else:
        df15["btc_corr_20"] = 0.0
        df15["btc_beta_20"] = 1.0
        df15["btc_rel_strength"] = 0.0

    df15["btc_corr_20"] = df15["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df15["btc_beta_20"] = df15["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df15["btc_rel_strength"] = df15["btc_rel_strength"].fillna(0.0).clip(-50, 50)
    if "fundingRate" not in df15.columns:
        df15["fundingRate"] = 0.0

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


def train(ds: pd.DataFrame, active_features: list, sell_tp_mult: float, sell_sl_mult: float, export_artifact: bool = True, reserved_features: list = None):
    ds = ds.loc[:, ~ds.columns.duplicated()].copy()
    for f in active_features:
        if f not in ds.columns: ds[f] = 0.0

    le = LabelEncoder()
    le.fit(ds["target"])
    classes = list(le.classes_)

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2
    
    # ── FATAL GUARD: Prevent Fabricated Negative Results ──
    if "fundingRate" in active_features:
        if "fundingRate" not in ds.columns or ds["fundingRate"].notna().sum() == 0 or (ds["fundingRate"] == 0).all():
            raise ValueError(
                "FATAL: Funding archive empty, all-NaN, or all-zero. "
                "Phase 2 variants would be numerically identical to BASELINE, "
                "producing a false negative result. Aborting."
            )

    train_df, calib_df, test_df = temporal_symbol_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)

    train_df = train_df.sort_values("open_time").reset_index(drop=True)
    X_train_raw = train_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_raw = le.transform(train_df["target"])

    X_calib = calib_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_calib = le.transform(calib_df["target"]) if len(calib_df) > 0 else np.array([])
    X_test = test_df[active_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = le.transform(test_df["target"]) if len(test_df) > 0 else np.array([])

    scanner = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train_raw, y_train_raw)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in active_features]
    
    # ── BUG FIX: Unconditionally Reserve Ablation Variables Before Cutoff ──
    for f in (reserved_features or []):
        if f in active_features and f not in selected:
            selected.append(f)

    for i in top_idx:
        feat = active_features[i]
        if feat not in selected: selected.append(feat)
        if len(selected) >= min(N_FEATURES, len(active_features)): break

    # Fail loudly rather than reporting a collapsed variant
    for f in (reserved_features or []):
        if f in active_features and f not in selected:
            raise ValueError(
                f"FATAL: reserved ablation feature '{f}' was not selected. "
                f"Variant would be numerically identical to baseline."
            )

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

    be_buy = (ATR_STOP_MULT + friction_r) / (ATR_TARGET1_MULT + ATR_STOP_MULT)
    be_sell = (sell_sl_mult + friction_r) / (sell_tp_mult + sell_sl_mult)

    sweep_thresholds = np.round(np.arange(0.32, 0.62, 0.02), 2)
    raw_sell_candidates = (calib_probas[:, sell_idx] > calib_probas[:, buy_idx]).sum()

    lowest_thresh = sweep_thresholds[0]
    p_sell_mask = (calib_probas[:, sell_idx] >= lowest_thresh) & (calib_probas[:, sell_idx] > calib_probas[:, buy_idx])
    
    # ── BUG FIX: Correctly Evaluate Mean After Casting Condition Array ──
    base_sell_prec = float((y_calib[p_sell_mask] == sell_idx).mean()) if p_sell_mask.sum() > 0 else 0.0

    if export_artifact:
        log.info(f"\n{'='*95}")
        log.info(f"CALIBRATION THRESHOLD SWEEP (Friction={friction_r}R)")
        log.info(f"BUY Geometry:  {ATR_TARGET1_MULT}R / {ATR_STOP_MULT}R (Req Prec > {be_buy*100:.1f}%)")
        log.info(f"SELL Geometry: {sell_tp_mult}R / {sell_sl_mult}R (Req Prec > {be_sell*100:.1f}%)")
        log.info(f"DIAGNOSTIC: Raw SELL calibration candidates (p_sell > p_buy): {raw_sell_candidates}")
        log.info(f"DIAGNOSTIC: SELL Precision @ Floor {lowest_thresh:.2f}: {base_sell_prec*100:.1f}% (n={p_sell_mask.sum()})")
        log.info(f"{'='*95}")

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

        if thresh >= MIN_BUY_THRESHOLD_FLOOR and buy_score > best_score_buy and bm.sum() > 15:
            best_score_buy, best_thresh_buy = buy_score, thresh
        if thresh >= MIN_SELL_THRESHOLD_FLOOR and sell_score > best_score_sell and sm.sum() > 15:
            best_score_sell, best_thresh_sell = sell_score, thresh

    # Fail-closed sentinel check (disables broken sides entirely)
    best_thresh_buy = max(MIN_BUY_THRESHOLD_FLOOR, best_thresh_buy) if best_score_buy > 0.0 else 1.01
    best_thresh_sell = max(MIN_SELL_THRESHOLD_FLOOR, best_thresh_sell) if best_score_sell > 0.0 else 1.01

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
        "recommended_threshold":      max(best_thresh_buy, best_thresh_sell),
        "calibrated":                 True,
        "buy_tp_mult":                ATR_TARGET1_MULT,
        "buy_sl_mult":                ATR_STOP_MULT,
        "sell_tp_mult":               sell_tp_mult,
        "sell_sl_mult":               sell_sl_mult,
    }

    joblib.dump(pipeline, MODEL_FILE)
    joblib.dump(pipeline, CANDIDATE_MODEL_FILE, compress=3)
    log.info(f"✅ Exported candidate binary: {CANDIDATE_MODEL_FILE}")

    perf = {
        "accuracy":                   round(acc * 100, 1),
        "test_accuracy":              f"{round(acc * 100, 1)}%",
        "train_accuracy":             f"{round(train_acc * 100, 1)}%",
        "n_train":                    int(len(X_train_raw)),
        "n_calib":                    int(len(X_calib)),
        "n_train_sampled":            int(len(y_train)),
        "n_test":                     int(len(X_test)),
        "features":                   active_features,
        "selected":                   selected,
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "buy_precision":              round(report.get("BUY", {}).get("precision", 0), 4),
        "sell_precision":             round(report.get("SELL", {}).get("precision", 0), 4),
        "no_trade_precision":         round(report.get("NO_TRADE", {}).get("precision", 0), 4),
    }

    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)

    return acc, best_score_buy, best_score_sell


if __name__ == "__main__":
    t0 = time.time()
    preloaded_symbols = preload_symbol_features()
    ds_baseline = pd.concat([_apply_targets(df, 3.5, 2.5) for df in preloaded_symbols], ignore_index=True)

    experiment_grid = [
        {"name": "Baseline (No Funding/BTC)", "features": BASE_FEATURES},
        {"name": "Baseline + Funding Rate",   "features": BASE_FEATURES + ["fundingRate"]},
        {"name": "Baseline + BTC Metrics",    "features": BASE_FEATURES + ["btc_corr_20", "btc_beta_20", "btc_rel_strength"]},
        {"name": "Full Kitchen Sink",         "features": BASE_FEATURES + ["fundingRate", "btc_corr_20", "btc_beta_20", "btc_rel_strength"]},
    ]

    results = []
    for config in experiment_grid:
        log.info(f"🚀 Running Feature Ablation: {config['name']}")
        
        # ── VARIANT INTEGRITY CHECK ──
        if config["name"] != "Baseline (No Funding/BTC)":
            if set(config["features"]) == set(BASE_FEATURES):
                raise ValueError(f"FATAL: {config['name']} feature set silently collapsed to BASELINE.")
                
        # Inject reserved parameters cleanly for non-baseline configurations
        reserved = [f for f in config["features"] if f not in BASE_FEATURES]
        
        acc, score_b, score_s = train(
            ds_baseline,
            active_features=config["features"],
            sell_tp_mult=3.5,
            sell_sl_mult=2.5,
            export_artifact=False,
            reserved_features=reserved
        )
        results.append({
            "Config": config["name"],
            "BUY Score": f"{score_b:.2f}",
            "SELL Score": f"{score_s:.2f}",
            "Accuracy": f"{acc*100:.1f}%",
        })

    log.info(f"\n{'='*80}")
    log.info("🎯 FEATURE ABLATION EXPERIMENT SUMMARY (Chronological Lock, Target=3.5/2.5)")
    log.info(f"{'='*80}")
    log.info(f"{'Config':<30} | {'BUY Score':<10} | {'SELL Score':<10} | {'Accuracy':<10}")
    log.info("-" * 80)
    for res in results:
        log.info(f"{res['Config']:<30} | {res['BUY Score']:<10} | {res['SELL Score']:<10} | {res['Accuracy']:<10}")
    log.info("=" * 80)

    # Export final model with full feature set (reserving the entire kitchen sink)
    reserved_export = [f for f in experiment_grid[-1]["features"] if f not in BASE_FEATURES]
    train(
        ds_baseline, 
        active_features=experiment_grid[-1]["features"], 
        sell_tp_mult=3.5, 
        sell_sl_mult=2.5, 
        export_artifact=True,
        reserved_features=reserved_export
    )
    log.info(f"Done in {(time.time()-t0)/60:.1f} min")
