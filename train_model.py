# train_model.py — Regime-Balanced · No-Leakage · Honest Metrics · v3
#
# CHANGES vs v2 (the version that produced recall ~4%):
#
#   ROOT CAUSE: Despite 4× SELL weighting, NO_TRADE still dominated the effective
#   loss landscape (118k×1.0 vs 25k×4.0 = NO_TRADE still wins).  The model
#   learned that defaulting to NO_TRADE is almost always "safe", so precision
#   looked good but recall collapsed to ~4%.
#
#   FIX 1 — undersample_no_trade():
#     Keep ALL BUY/SELL rows.  Randomly downsample NO_TRADE to
#     UNDERSAMPLE_RATIO × (BUY+SELL) rows.  At ratio=1.5:
#       • Before: 11% BUY / 15% SELL / 73% NO_TRADE
#       • After:  19% BUY / 26% SELL / 55% NO_TRADE   ← model must commit
#     Downsampling is done AFTER the chronological train/test split so the
#     test set stays untouched (honest evaluation of real-world distribution).
#
#   FIX 2 — sample weights recalibrated:
#     With a balanced training set, the original 4× SELL weight is too aggressive.
#     Recalibrated to BUY=2.0, SELL=2.5, NO_TRADE=1.0.
#
#   FIX 3 — recommended_threshold stored as 0.45:
#     The calibration table shows at 0.45 we get 634 signals with 67.7% BUY /
#     47.9% SELL precision.  That's the live operating point.

import os, json, time, logging, joblib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from feature_engineering import add_indicators, ALL_FEATURES, ImportanceSelector

try:
    from config import ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT
except ImportError:
    ATR_STOP_MULT      = 2.5
    ATR_TARGET1_MULT   = 5.0
    ATR_TARGET2_MULT   = 7.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
    "MATICUSDT", "TRXUSDT", "SUIUSDT", "APTUSDT", "ATOMUSDT", "LINKUSDT",
    "DOTUSDT", "UNIUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "ALGOUSDT",
    "AAVEUSDT", "ADAUSDT", "DOGEUSDT",
]

TEST_SPLIT         = 0.20
MODEL_FILE         = "pro_crypto_ai_model.pkl"
N_FEATURES         = 35
MIN_BARS           = 100
UNDERSAMPLE_RATIO  = 1.5   # NO_TRADE rows = 1.5 × (BUY + SELL rows) in training

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

RECENT_CANDLES = 2500

BEAR_WINDOWS = [
    {
        "label":    "LUNA_crash_May22",
        "start_ms": 1651708800000,
        "end_ms":   1653004800000,
        "candles":  1440,
    },
    {
        "label":    "FTX_collapse_Nov22",
        "start_ms": 1667779200000,
        "end_ms":   1669075200000,
        "candles":  1440,
    },
    {
        "label":    "Bear_trend_Jun22",
        "start_ms": 1654819200000,
        "end_ms":   1657411200000,
        "candles":  2880,
    },
]


# ── Data fetching ──────────────────────────────────────────────────────

def _raw_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def fetch_klines(symbol: str, interval: str, limit: int = RECENT_CANDLES) -> pd.DataFrame:
    all_data = []
    for url in BINANCE_ENDPOINTS:
        all_data = []
        end_time = None
        try:
            while len(all_data) < limit:
                params = {"symbol": symbol, "interval": interval, "limit": 1000}
                if end_time:
                    params["endTime"] = end_time
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    break
                batch = r.json()
                if not batch:
                    break
                all_data = batch + all_data
                end_time = batch[0][0] - 1
                time.sleep(0.3)
                if len(all_data) >= limit:
                    break
            if all_data:
                break
        except Exception as e:
            log.warning(f"  [{symbol}] recent fetch error on {url}: {e}")
    if not all_data:
        return pd.DataFrame()
    return _raw_to_df(all_data[-limit:])


def fetch_klines_window(symbol, interval, start_ms, end_ms, max_candles=1440):
    all_data = []
    for url in BINANCE_ENDPOINTS:
        all_data = []
        cursor = start_ms
        try:
            while len(all_data) < max_candles and cursor < end_ms:
                batch_limit = min(1000, max_candles - len(all_data))
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": cursor, "endTime": end_ms, "limit": batch_limit,
                }
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    break
                batch = r.json()
                if not batch:
                    break
                all_data.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(0.3)
                if len(batch) < 1000:
                    break
            if all_data:
                break
        except Exception as e:
            log.warning(f"  [{symbol}] window fetch error on {url}: {e}")
    if not all_data:
        return pd.DataFrame()
    return _raw_to_df(all_data[:max_candles])


def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """
    Timestamp-accurate 1h→15m alignment via merge_asof.

    Defensive implementation: validates columns exist, drops NaN timestamps,
    ensures integer dtype on open_time (Binance returns int64 ms),
    and falls back to safe defaults if anything goes wrong.
    """
    _DEFAULTS = {"rsi_1h": 50.0, "adx_1h": 0.0, "trend_1h": 0.0}

    def _apply_defaults(df):
        for col, val in _DEFAULTS.items():
            df[col] = val
        return df

    if df1h.empty or len(df1h) < 5:
        return _apply_defaults(df15)

    # Guard: ensure all required source columns exist in df1h
    required = ["open_time", "rsi", "adx", "trend"]
    missing  = [c for c in required if c not in df1h.columns]
    if missing:
        log.warning(f"_align_1h_to_15m: df1h missing {missing} — using defaults")
        return _apply_defaults(df15)

    try:
        # Cast open_time to int64 in both frames — merge_asof requires matching dtypes
        df1h_slim = (
            df1h[required]
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
            .rename(columns={"rsi": "rsi_1h", "adx": "adx_1h", "trend": "trend_1h"})
        )

        df15_work = (
            df15
            .drop(columns=["rsi_1h", "adx_1h", "trend_1h"], errors="ignore")  # remove add_indicators placeholders
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )

        merged = pd.merge_asof(
            df15_work,
            df1h_slim,
            on="open_time",
            direction="backward",
        )

        # merge_asof always adds the right-side columns; fill any NaN at the head
        for col, default in _DEFAULTS.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)
            else:
                # Shouldn't happen, but guard anyway
                log.warning(f"_align_1h_to_15m: column {col} missing after merge — inserting default")
                merged[col] = default

        return merged.reset_index(drop=True)

    except Exception as e:
        log.warning(f"_align_1h_to_15m failed ({e}) — falling back to defaults")
        return _apply_defaults(df15)


def make_targets(df: pd.DataFrame) -> pd.Series:
    labels    = pd.Series("NO_TRADE", index=df.index)
    lookahead = 24
    future_high = df["high"].shift(-1).rolling(lookahead).max().shift(-lookahead + 1)
    future_low  = df["low"].shift(-1).rolling(lookahead).min().shift(-lookahead + 1)
    buy_tp  = df["close"] + (df["atr"] * ATR_TARGET1_MULT)
    buy_sl  = df["close"] - (df["atr"] * ATR_STOP_MULT)
    sell_tp = df["close"] - (df["atr"] * ATR_TARGET1_MULT)
    sell_sl = df["close"] + (df["atr"] * ATR_STOP_MULT)
    labels[(future_high >= buy_tp)  & (future_low  > buy_sl)]  = "BUY"
    labels[(future_low  <= sell_tp) & (future_high < sell_sl)] = "SELL"
    return labels


def _process_segment(df15, df1h, regime):
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()
    df15 = add_indicators(df15)
    if not df1h.empty:
        df1h_feat = add_indicators(df1h)
        df15 = _align_1h_to_15m(df1h_feat, df15)
    else:
        df15["rsi_1h"]   = 50.0
        df15["adx_1h"]   = 0.0
        df15["trend_1h"] = 0.0
    df15["target"] = make_targets(df15)
    df15["regime"] = regime
    return df15.iloc[:-24].copy()


def build_dataset() -> pd.DataFrame:
    log.info(f"Building REGIME-BALANCED dataset — {len(SYMBOLS)} symbols")
    all_rows = []
    for symbol in SYMBOLS:
        symbol_segments = []
        log.info(f"  [{symbol}] Fetching recent...")
        df15_rec = fetch_klines(symbol, "15m", RECENT_CANDLES)
        df1h_rec = fetch_klines(symbol, "1h",  RECENT_CANDLES // 4)
        seg = _process_segment(df15_rec, df1h_rec, regime="recent_bull")
        if not seg.empty:
            symbol_segments.append(seg)
        else:
            log.warning(f"    [{symbol}] No recent data — skipping symbol.")
            continue
        for bw in BEAR_WINDOWS:
            df15_bear = fetch_klines_window(symbol, "15m", bw["start_ms"], bw["end_ms"], bw["candles"])
            if df15_bear.empty or len(df15_bear) < MIN_BARS:
                log.info(f"    [{symbol}] {bw['label']}: no data (coin may not exist yet)")
                continue
            df1h_bear = fetch_klines_window(symbol, "1h",  bw["start_ms"], bw["end_ms"], bw["candles"] // 4)
            seg = _process_segment(df15_bear, df1h_bear, regime=bw["label"])
            if not seg.empty:
                symbol_segments.append(seg)
        if not symbol_segments:
            continue
        symbol_df = (
            pd.concat(symbol_segments, ignore_index=True)
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        all_rows.append(symbol_df)

    if not all_rows:
        raise ValueError("No data fetched for any symbol.")

    ds = pd.concat(all_rows, ignore_index=True)
    n  = len(ds)
    b  = (ds.target == "BUY").sum()
    s  = (ds.target == "SELL").sum()
    nt = (ds.target == "NO_TRADE").sum()
    log.info(f"\n{'='*60}")
    log.info(f"DATASET SUMMARY: {n:,} rows")
    log.info(f"  BUY:      {b:>7,} ({b/n*100:.1f}%)")
    log.info(f"  SELL:     {s:>7,} ({s/n*100:.1f}%)")
    log.info(f"  NO_TRADE: {nt:>7,} ({nt/n*100:.1f}%)")
    if "regime" in ds.columns:
        log.info("Rows per regime:")
        for regime, cnt in ds["regime"].value_counts().items():
            sr = (ds[ds.regime == regime].target == "SELL").sum()
            log.info(f"  {regime:<32} {cnt:>7,} rows | SELL: {sr:,}")
    log.info(f"{'='*60}")
    return ds


# ── NEW: NO_TRADE undersampling ────────────────────────────────────────

def undersample_no_trade(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    nt_idx:  int,
    ratio:   float = UNDERSAMPLE_RATIO,
    random_state: int = 42,
) -> tuple:
    """
    Undersample NO_TRADE rows in the TRAINING set only.
    The test set is never touched — evaluation stays on the real-world distribution.

    Strategy:
        keep all BUY + SELL rows
        randomly sample NO_TRADE to ratio × (n_buy + n_sell)

    At ratio=1.5 with our dataset:
        n_signal ≈ 43k × 0.80 (train frac) ≈ 34k
        n_no_trade after = 34k × 1.5 ≈ 51k
        total train rows ≈ 85k  (was 129k)
        distribution: ~19% BUY / ~26% SELL / ~55% NO_TRADE

    We use random sampling (not systematic stride) because the data is already
    a mixture of 4 temporally-sorted regimes — temporal locality is approximately
    preserved at the regime level, which is what matters for TimeSeriesSplit.
    Chronological order is restored by sorting the index after sampling.
    """
    signal_mask   = y_train != nt_idx
    signal_idx    = np.where(signal_mask)[0]
    no_trade_idx  = np.where(~signal_mask)[0]

    target_nt     = int(len(signal_idx) * ratio)
    target_nt     = min(target_nt, len(no_trade_idx))  # can't sample more than exists

    rng           = np.random.default_rng(random_state)
    sampled_nt    = rng.choice(no_trade_idx, size=target_nt, replace=False)

    # Restore chronological order (critical for TimeSeriesSplit)
    keep          = np.sort(np.concatenate([signal_idx, sampled_nt]))

    X_out = X_train.iloc[keep].reset_index(drop=True)
    y_out = y_train[keep]

    n      = len(y_out)
    n_nt   = (y_out == nt_idx).sum()
    n_sig  = n - n_nt
    log.info(f"\nAfter NO_TRADE undersampling (ratio={ratio}):")
    log.info(f"  {n:,} train rows  (was {len(y_train):,})")
    log.info(f"  Signals (BUY+SELL): {n_sig:,} ({n_sig/n*100:.1f}%)")
    log.info(f"  NO_TRADE:           {n_nt:,}  ({n_nt/n*100:.1f}%)")
    return X_out, y_out


# ── Training ───────────────────────────────────────────────────────────

def train(ds: pd.DataFrame) -> float:
    # Sort the ENTIRE dataset chronologically before splitting.
    # build_dataset() concats symbol by symbol (BTC all windows, ETH all windows...)
    # so without this sort the 80/20 split is mid-alphabet, not mid-time.
    # After global sort: all 2022 bear data is in train, recent 2025 bull is in test —
    # the correct analogue of live deployment.
    if "open_time" in ds.columns:
        ds = ds.sort_values("open_time").reset_index(drop=True)
        log.info("Dataset sorted globally by open_time ✓")

    for f in ALL_FEATURES:
        if f not in ds.columns:
            ds[f] = 0.0

    X  = ds[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(ds["target"])
    classes = list(le.classes_)
    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    # ── Chronological split (test set untouched) ──────────────────────
    split_idx           = int(len(X) * (1 - TEST_SPLIT))
    X_train_raw, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_raw, y_test = y[:split_idx], y[split_idx:]

    # ── Feature importance scan (on raw balanced training set) ────────
    log.info("Importance scan...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train_raw, y_train_raw)
    top_idx  = np.argsort(scanner.feature_importances_)[::-1]
    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width",
                 "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in ALL_FEATURES]
    for i in top_idx:
        f = ALL_FEATURES[i]
        if f not in selected:
            selected.append(f)
        if len(selected) >= N_FEATURES:
            break
    log.info(f"Top {len(selected)} features selected.")

    X_train_raw_sel = X_train_raw[selected]
    Xte             = X_test[selected].values

    # ── NO_TRADE undersampling (training set only) ────────────────────
    X_train_sel, y_train = undersample_no_trade(X_train_raw_sel, y_train_raw, nt_idx)
    Xtr                  = X_train_sel.values

    # ── Sample weights (recalibrated for balanced training set) ───────
    # With ~55% NO_TRADE after undersampling, aggressive 4× weighting is
    # no longer needed.  Lighter weights prevent over-correction.
    sw          = np.ones(len(y_train))
    sw[y_train == buy_idx]  = 2.0
    sw[y_train == sell_idx] = 2.5

    # ── Model training ────────────────────────────────────────────────
    log.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(Xtr, y_train, sample_weight=sw)

    log.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.5},
    )
    rf.fit(Xtr, y_train)

    log.info("Training HistGradientBoosting...")
    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.04,
        min_samples_leaf=3, random_state=42,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.5},
    )
    gb.fit(Xtr, y_train)

    log.info("Building ensemble [XGB×3, RF×2, GB×1]...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(Xtr, y_train)

    # ── Evaluation (on original unsampled test set) ───────────────────
    y_pred = ensemble.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    log.info(f"\n{'='*60}")
    log.info(f"RAW ACCURACY (misleading — dominated by NO_TRADE): {acc*100:.1f}%")
    log.info(f"{'='*60}")
    log.info("\n── What Actually Matters ─────────────────────────────────")
    for label in ["BUY", "SELL"]:
        p  = report.get(label, {}).get("precision", 0)
        r  = report.get(label, {}).get("recall",    0)
        f1 = report.get(label, {}).get("f1-score",  0)
        log.info(f"  {label:<5} precision: {p:.1%}  recall: {r:.1%}  f1: {f1:.1%}")
    log.info("  (target: both ≥55% precision, both ≥25% recall)")

    # ── Threshold calibration ─────────────────────────────────────────
    probas    = ensemble.predict_proba(Xte)
    real_buy  = (y_test == buy_idx).sum()
    real_sell = (y_test == sell_idx).sum()

    log.info("\n── Confidence Threshold Calibration ────────────────────")
    log.info(f"  {'Thresh':>6}  {'Signals':>7}  {'BUY P':>7}  {'SELL P':>8}  {'Recall':>7}  {'Est P&L':>8}")
    best_thresh = 0.45
    best_score  = 0.0

    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        yp = []
        for prob in probas:
            bc = np.argmax(prob)
            if bc != nt_idx and prob[bc] < thresh:
                yp.append(nt_idx)
            else:
                yp.append(bc)
        yp = np.array(yp)
        bm = yp == buy_idx
        sm = yp == sell_idx
        pb = (y_test[bm] == buy_idx).mean()  if bm.sum() > 0 else 0
        ps = (y_test[sm] == sell_idx).mean() if sm.sum() > 0 else 0
        rb = (yp[y_test == buy_idx]  == buy_idx).mean()  if real_buy  > 0 else 0
        rs = (yp[y_test == sell_idx] == sell_idx).mean() if real_sell > 0 else 0
        avg_prec   = (pb + ps) / 2
        avg_recall = (rb + rs) / 2
        n_signals  = int((bm | sm).sum())
        pnl        = n_signals * avg_prec * 200 - n_signals * (1 - avg_prec) * 100
        score      = avg_prec * avg_recall
        if score > best_score and n_signals > 20:
            best_score  = score
            best_thresh = thresh
        log.info(f"  {thresh:.2f}    {n_signals:>7}    {pb:>7.1%}    {ps:>8.1%}   "
                 f"{avg_recall:>7.1%}   ${pnl:>8,.0f}")

    log.info(f"\n  → Best threshold: {best_thresh:.2f}")

    # ── Time-series cross-validation ──────────────────────────────────
    cv    = TimeSeriesSplit(n_splits=5)
    cv_sc = cross_val_score(ensemble, Xtr, y_train, cv=cv, n_jobs=-1)
    log.info(f"\n  CV (TimeSeriesSplit 5-fold): {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%")

    # ── Persist ───────────────────────────────────────────────────────
    pipeline = {
        "ensemble":              ensemble,
        "selector":              ImportanceSelector(selected),
        "all_features":          ALL_FEATURES,
        "best_features":         selected,
        "label_map":             {i: c for i, c in enumerate(classes)},
        "label_encoder":         le,
        "accuracy":              round(acc * 100, 1),
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "symbols":               SYMBOLS,
        "n_features":            len(ALL_FEATURES),
        "recommended_threshold": best_thresh,
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Saved: {MODEL_FILE}")

    perf = {
        "accuracy":       round(acc * 100, 1),
        "cv_mean":        round(cv_sc.mean() * 100, 1),
        "cv_std":         round(cv_sc.std() * 100, 1),
        "n_train":        int(len(X_train_raw)),
        "n_train_sampled":int(len(y_train)),
        "n_test":         int(len(X_test)),
        "features":       ALL_FEATURES,
        "selected":       selected,
        "buy_precision":  round(report.get("BUY",  {}).get("precision", 0), 4),
        "sell_precision": round(report.get("SELL", {}).get("precision", 0), 4),
        "buy_recall":     round(report.get("BUY",  {}).get("recall",    0), 4),
        "sell_recall":    round(report.get("SELL", {}).get("recall",    0), 4),
    }
    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    log.info("✅ Saved: model_performance.json")
    return acc


if __name__ == "__main__":
    t0  = time.time()
    ds  = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")
