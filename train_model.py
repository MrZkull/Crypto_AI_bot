# train_model.py — V3.9: Canonical Parquet Rebuild Engine, EV Friction & Provenance Invariants[cite: 6]

import os, json, time, logging, joblib, requests[cite: 6]
from pathlib import Path
import pandas as pd[cite: 6]
import numpy as np[cite: 6]
from datetime import datetime, timezone[cite: 6]
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier[cite: 6]
from sklearn.calibration import CalibratedClassifierCV[cite: 6]
from sklearn.metrics import classification_report, accuracy_score[cite: 6]
from sklearn.preprocessing import LabelEncoder[cite: 6]
from xgboost import XGBClassifier[cite: 6]
from sklearn.frozen import FrozenEstimator[cite: 6]

from feature_engineering import add_indicators, ALL_FEATURES, ImportanceSelector[cite: 6]
from market_data_integrity import sanitize_closed_candles, merge_completed_htf[cite: 3, 6]

try:
    from config import (
        SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT
    )[cite: 6]
except ImportError:
    SYMBOLS = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
        "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
        "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
        "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT", "FILUSDT"
    ][cite: 6]
    ATR_STOP_MULT    = 2.5[cite: 6]
    ATR_TARGET1_MULT = 3.5[cite: 6]
    ATR_TARGET2_MULT = 7.5[cite: 6]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")[cite: 6]
log = logging.getLogger(__name__)[cite: 6]

TEST_SPLIT         = 0.20[cite: 6]
CALIB_SPLIT        = 0.15[cite: 6]
EMBARGO_BARS       = 24[cite: 6]
MODEL_FILE         = "pro_crypto_ai_model.pkl"[cite: 6]
N_FEATURES         = 35[cite: 6]
MIN_BARS           = 100[cite: 6]
UNDERSAMPLE_RATIO  = 1.0[cite: 6]

MIN_BUY_THRESHOLD_FLOOR  = 0.40[cite: 6]
MIN_SELL_THRESHOLD_FLOOR = 0.45[cite: 6]

HISTORICAL_DATA_DIR = Path("data/historical")

NEW_FEATURES = [
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
][cite: 6]
FULL_FEATURES = ALL_FEATURES + NEW_FEATURES[cite: 6]


# ── Feature Alignment & Target Engineering ─────────────────────────────

def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    defaults = {"rsi_1h": 50.0, "adx_1h": 0.0, "trend_1h": 0.0}[cite: 6]
    if df1h.empty or df15.empty:[cite: 6]
        for c, v in defaults.items():[cite: 6]
            df15[c] = v[cite: 6]
        return df15[cite: 6]
    h = merge_completed_htf(df15, df1h, ["rsi", "adx", "trend"], prefix="htf1h")[cite: 6]
    h = h.rename(columns={"htf1h_rsi": "rsi_1h", "htf1h_adx": "adx_1h", "htf1h_trend": "trend_1h"})[cite: 6]
    for c, v in defaults.items():[cite: 6]
        h[c] = h[c].fillna(v) if c in h else v[cite: 6]
    return h[cite: 6]


def _align_4h_to_15m(df4h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    defaults = {"rsi_4h": 50.0, "trend_4h": 0.0}[cite: 6]
    if df4h.empty or df15.empty:[cite: 6]
        for c, v in defaults.items():[cite: 6]
            df15[c] = v[cite: 6]
        return df15[cite: 6]
    h = merge_completed_htf(df15, df4h, ["rsi", "trend"], prefix="htf4h")[cite: 6]
    h = h.rename(columns={"htf4h_rsi": "rsi_4h", "htf4h_trend": "trend_4h"})[cite: 6]
    for c, v in defaults.items():[cite: 6]
        h[c] = h[c].fillna(v) if c in h else v[cite: 6]
    return h[cite: 6]


def _align_btc_to_15m(btc_df15: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if btc_df15 is None or btc_df15.empty or "close" not in btc_df15.columns:[cite: 6]
        df15["btc_close"] = np.nan[cite: 6]
        return df15[cite: 6]
    try:
        out = merge_completed_htf(df15, btc_df15, ["close"], prefix="btc")[cite: 6]
        out = out.rename(columns={"btc_close": "btc_close"})[cite: 6]
        return out[cite: 6]
    except Exception as e:[cite: 6]
        log.warning(f"_align_btc_to_15m failed ({e})")[cite: 6]
        df15["btc_close"] = np.nan[cite: 6]
        return df15[cite: 6]


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rolling BTC correlation, beta, and relative strength with NaN guards."""
    df = df.copy()[cite: 6]
    if "btc_close" in df.columns and df["btc_close"].notna().sum() > 30:[cite: 6]
        btc_ret = df["btc_close"].pct_change()[cite: 6]
        coin_ret = df["close"].pct_change()[cite: 6]
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)[cite: 6]
        roll_var = btc_ret.rolling(20, min_periods=10).var()[cite: 6]
        df["btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)[cite: 6]
        df["btc_beta_20"] = roll_cov / roll_var.replace(0, np.nan)[cite: 6]
        df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100[cite: 6]
    else:
        df["btc_corr_20"] = 0.0[cite: 6]
        df["btc_beta_20"] = 1.0[cite: 6]
        df["btc_rel_strength"] = 0.0[cite: 6]

    df["btc_corr_20"] = df["btc_corr_20"].fillna(0.0).clip(-1, 1)[cite: 6]
    df["btc_beta_20"] = df["btc_beta_20"].fillna(1.0).clip(-5, 5)[cite: 6]
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)[cite: 6]
    return df[cite: 6]


def make_targets(df: pd.DataFrame) -> pd.Series:
    """Triple-barrier labeling. Same-candle TP+SL collisions are labeled AMBIGUOUS."""
    n = len(df)[cite: 6]
    labels = np.full(n, "NO_TRADE", dtype=object)[cite: 6]
    lookahead = 24[cite: 6]
    highs = df["high"].values[cite: 6]
    lows = df["low"].values[cite: 6]
    closes = df["close"].values[cite: 6]
    atrs = df["atr"].values if "atr" in df.columns else np.zeros(n)[cite: 6]

    for i in range(n - lookahead):[cite: 6]
        entry = closes[i][cite: 6]
        atr = atrs[i][cite: 6]
        if atr <= 0 or np.isnan(atr):[cite: 6]
            continue[cite: 6]
        buy_tp = entry + atr * ATR_TARGET1_MULT[cite: 6]
        buy_sl = entry - atr * ATR_STOP_MULT[cite: 6]
        sell_tp = entry - atr * ATR_TARGET1_MULT[cite: 6]
        sell_sl = entry + atr * ATR_STOP_MULT[cite: 6]

        def first_barrier(tp, sl, up):
            for k in range(1, lookahead + 1):[cite: 6]
                h, l = highs[i + k], lows[i + k][cite: 6]
                hit_tp = (h >= tp) if up else (l <= tp)[cite: 6]
                hit_sl = (l <= sl) if up else (h >= sl)[cite: 6]
                if hit_tp and hit_sl: return "BOTH"[cite: 6]
                if hit_tp: return "TP"[cite: 6]
                if hit_sl: return "SL"[cite: 6]
            return "NONE"[cite: 6]

        b = first_barrier(buy_tp, buy_sl, True)[cite: 6]
        s = first_barrier(sell_tp, sell_sl, False)[cite: 6]
        if b == "BOTH" or s == "BOTH" or (b == "TP" and s == "TP"):[cite: 6]
            labels[i] = "AMBIGUOUS"[cite: 6]
        elif b == "TP" and s != "TP": labels[i] = "BUY"[cite: 6]
        elif s == "TP" and b != "TP": labels[i] = "SELL"[cite: 6]
    return pd.Series(labels, index=df.index)[cite: 6]


def _process_segment(symbol, df15, df1h, df4h, regime, btc_df15=None):
    if df15.empty or len(df15) < MIN_BARS:[cite: 6]
        return pd.DataFrame()[cite: 6]

    taker_col = None[cite: 6]
    if "taker_buy_base_vol" in df15.columns:[cite: 6]
        taker_col = df15[["open_time", "taker_buy_base_vol"]].copy()[cite: 6]

    df15 = add_indicators(df15)[cite: 6]
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:[cite: 6]
        df15 = df15.merge(taker_col, on="open_time", how="left")[cite: 6]

    if not df1h.empty:[cite: 6]
        df1h_feat = add_indicators(df1h)[cite: 6]
        df15 = _align_1h_to_15m(df1h_feat, df15)[cite: 6]
    else:
        df15["rsi_1h"] = 50.0[cite: 6]
        df15["adx_1h"] = 0.0[cite: 6]
        df15["trend_1h"] = 0.0[cite: 6]

    if not df4h.empty:[cite: 6]
        df4h_feat = add_indicators(df4h)[cite: 6]
        df15 = _align_4h_to_15m(df4h_feat, df15)[cite: 6]
    else:
        df15["rsi_4h"] = 50.0[cite: 6]
        df15["trend_4h"] = 0.0[cite: 6]

    df15 = _align_btc_to_15m(btc_df15, df15)[cite: 6]
    if "htf1h_source_close_time" not in df15.columns or "htf4h_source_close_time" not in df15.columns:[cite: 6]
        log.warning(f"[{symbol}] Missing HTF source-close provenance — rejecting segment")[cite: 6]
        return pd.DataFrame()[cite: 6]

    df15 = _add_extra_features(df15)[cite: 6]
    df15["symbol"] = symbol[cite: 6]
    df15["target"] = make_targets(df15)[cite: 6]
    df15["regime"] = regime[cite: 6]

    # Invariant: Trim unevaluated forward horizon BEFORE filtering out AMBIGUOUS targets[cite: 6]
    if len(df15) <= 24:[cite: 6]
        return pd.DataFrame()[cite: 6]
    df15 = df15.iloc[:-24].copy()[cite: 6]
    return df15[df15["target"] != "AMBIGUOUS"].copy()[cite: 6]


# ── Canonical Local Parquet Ingestion ──────────────────────────────────

def load_parquet_segment(symbol: str, interval: str) -> pd.DataFrame:
    """Loads historical candles from canonical local storage and sanitizes closed bars."""
    path = HISTORICAL_DATA_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        clean = sanitize_closed_candles(df)[cite: 3]
        out = pd.DataFrame(clean)
        return out.sort_values("open_time").reset_index(drop=True)
    except Exception as e:
        log.warning(f"Failed loading Parquet segment {path}: {e}")
        return pd.DataFrame()


def build_dataset_from_local_parquet() -> pd.DataFrame:
    """Constructs the master training dataset exclusively from verified local Parquet files."""
    log.info(f"Building DATASET FROM CANONICAL PARQUET — {len(SYMBOLS)} symbols")[cite: 6]
    all_rows = [][cite: 6]
    btc_df15 = load_parquet_segment("BTCUSDT", "15m")

    for symbol in SYMBOLS:[cite: 6]
        df15 = load_parquet_segment(symbol, "15m")
        df1h = load_parquet_segment(symbol, "1h")
        df4h = load_parquet_segment(symbol, "4h")

        if df15.empty or len(df15) < MIN_BARS:[cite: 6]
            log.warning(f"  [{symbol}] Insufficient Parquet data — skipping symbol.")
            continue

        seg = _process_segment(symbol, df15, df1h, df4h, regime="historical_canonical", btc_df15=btc_df15)[cite: 6]
        if not seg.empty:[cite: 6]
            all_rows.append(seg)[cite: 6]

    if not all_rows:[cite: 6]
        raise ValueError(f"CRITICAL: No valid Parquet data found in {HISTORICAL_DATA_DIR}")

    ds = pd.concat(all_rows, ignore_index=True)[cite: 6]
    n = len(ds)[cite: 6]
    b = (ds.target == "BUY").sum()[cite: 6]
    s = (ds.target == "SELL").sum()[cite: 6]
    nt = (ds.target == "NO_TRADE").sum()[cite: 6]

    log.info(f"\n{'='*60}")[cite: 6]
    log.info(f"CANONICAL PARQUET DATASET: {n:,} rows | BUY: {b:,} ({b/n*100:.1f}%) | SELL: {s:,} ({s/n*100:.1f}%) | NO_TRADE: {nt:,} ({nt/n*100:.1f}%)")[cite: 6]
    log.info(f"{'='*60}")[cite: 6]
    return ds[cite: 6]


# Backward-compatible alias for dataset construction
build_dataset = build_dataset_from_local_parquet


# ── Temporal Partitioning & Sampling ───────────────────────────────────

def temporal_symbol_split(ds: pd.DataFrame, test_split: float, calib_split: float, embargo: int):
    """Chronological per-symbol train/calibration/test split with boundary embargos."""
    if ds is None or ds.empty:[cite: 6]
        empty = ds.iloc[:0].copy() if isinstance(ds, pd.DataFrame) else pd.DataFrame()[cite: 6]
        return empty, empty.copy(), empty.copy()[cite: 6]

    train_parts, calib_parts, test_parts = [], [], [][cite: 6]
    for sym, grp in ds.groupby("symbol", sort=False):[cite: 6]
        grp = grp.sort_values("open_time").reset_index(drop=True)[cite: 6]
        n = len(grp)[cite: 6]
        test_n = int(n * test_split)[cite: 6]
        calib_n = int(n * calib_split)[cite: 6]
        test_start = n - test_n[cite: 6]
        calib_end = test_start - embargo[cite: 6]
        calib_start = calib_end - calib_n[cite: 6]
        train_end = calib_start - embargo[cite: 6]

        if test_n <= 0 or calib_n <= 0 or train_end <= 0:[cite: 6]
            train_parts.append(grp)[cite: 6]
            continue[cite: 6]

        train_parts.append(grp.iloc[:train_end])[cite: 6]
        calib_parts.append(grp.iloc[calib_start:calib_end])[cite: 6]
        test_parts.append(grp.iloc[test_start:])[cite: 6]

    base = ds.iloc[:0].copy()[cite: 6]
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else base.copy()[cite: 6]
    calib_df = pd.concat(calib_parts, ignore_index=True) if calib_parts else base.copy()[cite: 6]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else base.copy()[cite: 6]
    return tuple(x.sort_values(["open_time", "symbol"]).reset_index(drop=True) for x in (train_df, calib_df, test_df))[cite: 6]


def undersample_no_trade(X_train: pd.DataFrame, y_train: np.ndarray, nt_idx: int, ratio: float = UNDERSAMPLE_RATIO) -> tuple:
    signal_mask  = y_train != nt_idx[cite: 6]
    signal_idx   = np.where(signal_mask)[0][cite: 6]
    no_trade_idx = np.where(~signal_mask)[0][cite: 6]
    target_nt    = min(int(len(signal_idx) * ratio), len(no_trade_idx))[cite: 6]

    rng = np.random.default_rng(42)[cite: 6]
    sampled_nt = rng.choice(no_trade_idx, size=target_nt, replace=False)[cite: 6]
    keep = np.sort(np.concatenate([signal_idx, sampled_nt]))[cite: 6]
    return X_train.iloc[keep].reset_index(drop=True), y_train[keep][cite: 6]


# ── Training & Calibration Engine ──────────────────────────────────────

def train(ds: pd.DataFrame) -> float:
    for f in FULL_FEATURES:[cite: 6]
        if f not in ds.columns:[cite: 6]
            ds[f] = 0.0[cite: 6]

    le = LabelEncoder()[cite: 6]
    le.fit(ds["target"])[cite: 6]
    classes = list(le.classes_)[cite: 6]

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1[cite: 6]
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0[cite: 6]
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2[cite: 6]

    train_df, calib_df, test_df = temporal_symbol_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)[cite: 6]

    # Invariant: sort globally by open_time so walk-forward slices strictly advance in forward time
    train_df = train_df.sort_values("open_time").reset_index(drop=True)

    X_train_raw = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)[cite: 6]
    y_train_raw = le.transform(train_df["target"])[cite: 6]

    X_calib = calib_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)[cite: 6]
    y_calib = le.transform(calib_df["target"]) if len(calib_df) > 0 else np.array([])[cite: 6]

    X_test = test_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)[cite: 6]
    y_test = le.transform(test_df["target"]) if len(test_df) > 0 else np.array([])[cite: 6]

    log.info(
        f"Chronological Split (embargo={EMBARGO_BARS} bars): "
        f"train={len(X_train_raw):,}  calib={len(X_calib):,}  test={len(X_test):,}"
    )

    log.info("Running feature importance scan...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")[cite: 6]
    scanner.fit(X_train_raw, y_train_raw)[cite: 6]
    top_idx  = np.argsort(scanner.feature_importances_)[::-1][cite: 6]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"][cite: 6]
    selected  = [f for f in essential if f in FULL_FEATURES][cite: 6]

    for i in top_idx:[cite: 6]
        if FULL_FEATURES[i] not in selected:[cite: 6]
            selected.append(FULL_FEATURES[i])[cite: 6]
        if len(selected) >= N_FEATURES:[cite: 6]
            break[cite: 6]

    X_train_raw_sel = X_train_raw[selected][cite: 6]
    Xte             = X_test[selected].values[cite: 6]
    Xcal            = X_calib[selected].values[cite: 6]

    X_train_sel, y_train = undersample_no_trade(X_train_raw_sel, y_train_raw, nt_idx)[cite: 6]
    Xtr                  = X_train_sel.values[cite: 6]

    sw_asym = np.ones(len(y_train))[cite: 6]
    sw_asym[y_train == buy_idx]  = 2.0[cite: 6]
    sw_asym[y_train == sell_idx] = 2.0[cite: 6]

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )[cite: 6]
    xgb.fit(Xtr, y_train, sample_weight=sw_asym)[cite: 6]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.0},
    )[cite: 6]
    rf.fit(Xtr, y_train)[cite: 6]

    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.04,
        min_samples_leaf=3, random_state=42,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.0},
    )[cite: 6]
    gb.fit(Xtr, y_train)[cite: 6]

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft", weights=[3, 2, 1],
    )[cite: 6]
    ensemble.fit(Xtr, y_train)[cite: 6]

    # Chronological Walk-Forward Slices
    wf_scores = [][cite: 6]
    window = len(Xtr) // 5[cite: 6]
    wf_embargo = min(EMBARGO_BARS, max(window // 10, 1))[cite: 6]

    for i in range(4):[cite: 6]
        wf_train_end  = (i + 1) * window[cite: 6]
        wf_test_start = wf_train_end + wf_embargo[cite: 6]
        wf_test_end   = wf_test_start + window[cite: 6]
        if wf_test_end > len(Xtr):[cite: 6]
            break[cite: 6]
        probe = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss", n_jobs=-1)[cite: 6]
        probe.fit(Xtr[:wf_train_end], y_train[:wf_train_end])[cite: 6]
        acc_wf = accuracy_score(y_train[wf_test_start:wf_test_end], probe.predict(Xtr[wf_test_start:wf_test_end]))[cite: 6]
        wf_scores.append(acc_wf)[cite: 6]

    wf_mean = np.mean(wf_scores) if wf_scores else 0.0[cite: 6]
    wf_std  = np.std(wf_scores) if wf_scores else 0.0[cite: 6]
    log.info(f"Walk-forward Accuracy: {wf_mean*100:.1f}% ± {wf_std*100:.1f}%")[cite: 6]

    calibrated_ensemble = CalibratedClassifierCV(estimator=FrozenEstimator(ensemble), method="isotonic")[cite: 6]
    calibrated_ensemble.fit(Xcal, y_calib)[cite: 6]
    ensemble = calibrated_ensemble[cite: 6]

    # Friction-Penalized Threshold Optimization Sweep (0.12R round-trip friction)
    calib_probas = ensemble.predict_proba(Xcal)[cite: 6]
    calib_buy_n  = (y_calib == buy_idx).sum()[cite: 6]
    calib_sell_n = (y_calib == sell_idx).sum()[cite: 6]

    best_thresh_buy, best_score_buy   = MIN_BUY_THRESHOLD_FLOOR, 0.0[cite: 6]
    best_thresh_sell, best_score_sell = MIN_SELL_THRESHOLD_FLOOR, 0.0[cite: 6]

    friction_r = 0.12
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:[cite: 6]
        yp = [np.argmax(p) if np.argmax(p) != nt_idx and p[np.argmax(p)] >= thresh else nt_idx for p in calib_probas][cite: 6]
        yp = np.array(yp)[cite: 6]
        bm, sm = (yp == buy_idx), (yp == sell_idx)[cite: 6]

        pb = (y_calib[bm] == buy_idx).mean()  if bm.sum() > 0 else 0[cite: 6]
        ps = (y_calib[sm] == sell_idx).mean() if sm.sum() > 0 else 0[cite: 6]
        rb = (yp[y_calib == buy_idx] == buy_idx).mean()   if calib_buy_n > 0 else 0[cite: 6]
        rs = (yp[y_calib == sell_idx] == sell_idx).mean() if calib_sell_n > 0 else 0[cite: 6]

        buy_ev  = pb * ATR_TARGET1_MULT - (1 - pb) * ATR_STOP_MULT - friction_r
        sell_ev = ps * ATR_TARGET1_MULT - (1 - ps) * ATR_STOP_MULT - friction_r

        buy_score  = buy_ev * rb * np.sqrt(max(bm.sum(), 1))  if buy_ev > 0 else 0.0[cite: 6]
        sell_score = sell_ev * rs * np.sqrt(max(sm.sum(), 1)) if sell_ev > 0 else 0.0[cite: 6]

        if thresh >= MIN_BUY_THRESHOLD_FLOOR and buy_score > best_score_buy and bm.sum() > 15:[cite: 6]
            best_score_buy, best_thresh_buy = buy_score, thresh[cite: 6]
        if thresh >= MIN_SELL_THRESHOLD_FLOOR and sell_score > best_score_sell and sm.sum() > 15:[cite: 6]
            best_score_sell, best_thresh_sell = sell_score, thresh[cite: 6]

    best_thresh_buy  = max(MIN_BUY_THRESHOLD_FLOOR, best_thresh_buy)[cite: 6]
    best_thresh_sell = max(MIN_SELL_THRESHOLD_FLOOR, best_thresh_sell)[cite: 6]

    probas = ensemble.predict_proba(Xte)[cite: 6]
    y_pred_tuned = [][cite: 6]
    for p in probas:[cite: 6]
        pred_c = np.argmax(p)[cite: 6]
        if pred_c == buy_idx and p[pred_c] >= best_thresh_buy:[cite: 6]
            y_pred_tuned.append(buy_idx)[cite: 6]
        elif pred_c == sell_idx and p[pred_c] >= best_thresh_sell:[cite: 6]
            y_pred_tuned.append(sell_idx)[cite: 6]
        else:
            y_pred_tuned.append(nt_idx)[cite: 6]
    y_pred_tuned = np.array(y_pred_tuned)[cite: 6]

    acc = accuracy_score(y_test, y_pred_tuned)[cite: 6]
    report = classification_report(y_test, y_pred_tuned, target_names=classes, output_dict=True, zero_division=0)[cite: 6]

    pipeline = {
        "ensemble":                   ensemble,[cite: 6]
        "selector":                   ImportanceSelector(selected),[cite: 6]
        "all_features":               FULL_FEATURES,[cite: 6]
        "best_features":              selected,[cite: 6]
        "label_map":                  {i: c for i, c in enumerate(classes)},[cite: 6]
        "label_encoder":              le,[cite: 6]
        "accuracy":                   round(acc * 100, 1),[cite: 6]
        "trained_at":                 datetime.now(timezone.utc).isoformat(),[cite: 6]
        "symbols":                    SYMBOLS,[cite: 6]
        "n_features":                 len(FULL_FEATURES),[cite: 6]
        "recommended_threshold_buy":  best_thresh_buy,[cite: 6]
        "recommended_threshold_sell": best_thresh_sell,[cite: 6]
        "recommended_threshold":      max(best_thresh_buy, best_thresh_sell),[cite: 6]
        "calibrated":                 True,[cite: 6]
    }
    joblib.dump(pipeline, MODEL_FILE)[cite: 6]
    log.info(f"✅ Saved primary model artifact: {MODEL_FILE}")[cite: 6]

    perf = {
        "accuracy":                   round(acc * 100, 1),[cite: 6]
        "test_accuracy":              f"{round(acc * 100, 1)}%",[cite: 6]
        "wf_mean":                    round(wf_mean * 100, 1),[cite: 6]
        "wf_std":                     round(wf_std * 100, 1),[cite: 6]
        "n_train":                    int(len(X_train_raw)),[cite: 6]
        "n_calib":                    int(len(X_calib)),[cite: 6]
        "n_train_sampled":            int(len(y_train)),[cite: 6]
        "n_test":                     int(len(X_test)),[cite: 6]
        "features":                   FULL_FEATURES,[cite: 6]
        "selected":                   selected,[cite: 6]
        "recommended_threshold_buy":  best_thresh_buy,[cite: 6]
        "recommended_threshold_sell": best_thresh_sell,[cite: 6]
        "buy_precision":              round(report.get("BUY", {}).get("precision", 0), 4),[cite: 6]
        "sell_precision":             round(report.get("SELL", {}).get("precision", 0), 4),[cite: 6]
        "no_trade_precision":         round(report.get("NO_TRADE", {}).get("precision", 0), 4),[cite: 6]
    }
    with open("model_performance.json", "w") as f:[cite: 6]
        json.dump(perf, f, indent=2)[cite: 6]

    return acc[cite: 6]


if __name__ == "__main__":
    t0 = time.time()[cite: 6]
    dataset = build_dataset_from_local_parquet()
    acc = train(dataset)[cite: 6]
    log.info(f"Primary rebuild complete in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")[cite: 6]
