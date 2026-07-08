# train_model.py — Stable Core Reversion · Phase 3.4 (Leakage Fix + New Features)
#
# REVERT ACTION: Completely removed LABEL_MULT.
# Target generation matches live execution 1:1.
# Retains symmetric weights (2.0/2.0).
# PHASE 2.1: Dynamic Break-Even Thresholding.
# PHASE 3: Walk-forward validation + Asymmetric XGBoost Loss.
# PHASE 3.1: Added Recovery_Jan23 to patch Window 3 transition gap.
# PHASE 3.2: Strict 1.0 Undersampling ratio for balanced class weights.
# PHASE 3.3: EV-based threshold selector to maximize actual R-yield.
#
# PHASE 3.4 CHANGES (this version):
#   FIX 1 — Calibration leakage: isotonic calibration was previously fit AND
#           evaluated on the same test set (Xte/y_test), inflating reported
#           accuracy/confidence. Now uses a dedicated, embargoed calibration
#           split (Xcal/y_calib) that the final test set never touches.
#   FIX 2 — Overlapping-label leakage: each label looks 24 bars into the
#           future, so rows within 24 bars of a split boundary leak info
#           across train/calib/test. Now embargoes (drops) EMBARGO_BARS rows
#           at every split boundary, and applies an approximate embargo gap
#           inside the walk-forward loop too.
#   NEW FEATURES — taker_buy_ratio (order flow), btc_corr_20 / btc_beta_20 /
#           btc_rel_strength (cross-asset context vs BTC), funding_rate_pct /
#           funding_rate_chg (derivatives positioning). These are computed
#           directly in this file (feature_engineering.py is untouched) and
#           appended to ALL_FEATURES via FULL_FEATURES.

import os, json, time, logging, joblib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.frozen import FrozenEstimator

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
CALIB_SPLIT        = 0.15   # NEW: dedicated calibration slice, carved out of what used to all be "train"
EMBARGO_BARS       = 24     # NEW: matches label lookahead — dropped at every split boundary
MODEL_FILE         = "pro_crypto_ai_model.pkl"
N_FEATURES         = 35
MIN_BARS           = 100

# Strict 1.0 ratio ensures enough NO_TRADE samples exist to correctly balance the classes
UNDERSAMPLE_RATIO  = 1.0

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
FUTURES_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

RECENT_CANDLES = 5000

BEAR_WINDOWS = [
    {"label": "LUNA_crash_May22",   "start_ms": 1651708800000, "end_ms": 1653004800000, "candles": 1440},
    {"label": "FTX_collapse_Nov22", "start_ms": 1667779200000, "end_ms": 1669075200000, "candles": 1440},
    {"label": "Bear_trend_Jun22",   "start_ms": 1654819200000, "end_ms": 1657411200000, "candles": 2880},
    {"label": "Aug2023_dip",        "start_ms": 1690848000000, "end_ms": 1692057600000, "candles": 1440},
    {"label": "Apr2024_halving",    "start_ms": 1713225600000, "end_ms": 1714435200000, "candles": 1440},
    {"label": "Bull_peak_Oct21",    "start_ms": 1633046400000, "end_ms": 1638316800000, "candles": 2880},
    {"label": "Recovery_Jan23",     "start_ms": 1672531200000, "end_ms": 1675209600000, "candles": 2880},
]

# ── New engineered features (appended on top of feature_engineering.ALL_FEATURES) ──
# NOTE: taker_buy_ratio + hour/dow cyclical features now live in feature_engineering.py's
# add_indicators() itself (added there so trade_executor.py gets them for free with zero
# extra live-side merging). Only the features that need EXTERNAL data (BTC benchmark,
# funding history) are computed here.
NEW_FEATURES = [
    "btc_corr_20",         # rolling 20-bar correlation of returns vs BTC
    "btc_beta_20",         # rolling 20-bar beta vs BTC
    "btc_rel_strength",    # coin's 6-bar return minus BTC's 6-bar return
    "funding_rate_pct",    # current perpetual funding rate (%)
    "funding_rate_chg",    # change in funding rate over the last 3 fundings
]
FULL_FEATURES = ALL_FEATURES + NEW_FEATURES

# ── Data fetching ──────────────────────────────────────────────────────

def _raw_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"]
    df.columns = cols[:df.shape[1]]
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base_vol"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ["open_time", "open", "high", "low", "close", "volume", "taker_buy_base_vol"]
            if c in df.columns]
    return df[keep].reset_index(drop=True)

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

def fetch_funding_history(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Historical perpetual funding rate from Binance USDM futures (public endpoint).
    Gracefully returns an empty frame if the symbol has no futures listing or the
    request fails — callers must default funding_rate to 0.0 in that case."""
    all_data = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": limit}
            r = requests.get(FUTURES_FUNDING_URL, params=params, timeout=10)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            cursor = int(batch[-1]["fundingTime"]) + 1
            time.sleep(0.2)
            if len(batch) < limit:
                break
    except Exception as e:
        log.debug(f"  funding history fetch failed [{symbol}]: {e}")

    if not all_data:
        return pd.DataFrame(columns=["open_time", "funding_rate"])

    df = pd.DataFrame(all_data).rename(columns={"fundingTime": "open_time", "fundingRate": "funding_rate"})
    df["open_time"]    = df["open_time"].astype("int64")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
    return df[["open_time", "funding_rate"]].sort_values("open_time").reset_index(drop=True)


# ── Feature Alignment ──────────────────────────────────────────────────

def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    _DEFAULTS = {"rsi_1h": 50.0, "adx_1h": 0.0, "trend_1h": 0.0}

    def _apply_defaults(df):
        for col, val in _DEFAULTS.items():
            df[col] = val
        return df

    if df1h.empty or len(df1h) < 5:
        return _apply_defaults(df15)

    required = ["open_time", "rsi", "adx", "trend"]
    missing  = [c for c in required if c not in df1h.columns]
    if missing:
        return _apply_defaults(df15)

    try:
        df1h_slim = (
            df1h[required]
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
            .rename(columns={"rsi": "rsi_1h", "adx": "adx_1h", "trend": "trend_1h"})
        )

        df15_work = (
            df15
            .drop(columns=["rsi_1h", "adx_1h", "trend_1h"], errors="ignore")
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )

        merged = pd.merge_asof(df15_work, df1h_slim, on="open_time", direction="backward")

        for col, default in _DEFAULTS.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)
            else:
                merged[col] = default

        return merged.reset_index(drop=True)

    except Exception as e:
        log.warning(f"_align_1h_to_15m failed ({e}) — falling back to defaults")
        return _apply_defaults(df15)


def _align_4h_to_15m(df4h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    _DEFAULTS = {"rsi_4h": 50.0, "trend_4h": 0.0}

    def _apply_defaults(df):
        for col, val in _DEFAULTS.items():
            df[col] = val
        return df

    if df4h.empty or len(df4h) < 5:
        return _apply_defaults(df15)

    required = ["open_time", "rsi", "trend"]
    missing  = [c for c in required if c not in df4h.columns]
    if missing:
        return _apply_defaults(df15)

    try:
        df4h_slim = (
            df4h[required]
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
            .rename(columns={"rsi": "rsi_4h", "trend": "trend_4h"})
        )

        df15_work = (
            df15
            .drop(columns=["rsi_4h", "trend_4h"], errors="ignore")
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )

        merged = pd.merge_asof(df15_work, df4h_slim, on="open_time", direction="backward")

        for col, default in _DEFAULTS.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)
            else:
                merged[col] = default

        return merged.reset_index(drop=True)

    except Exception as e:
        log.warning(f"_align_4h_to_15m failed ({e}) — falling back to defaults")
        return _apply_defaults(df15)


def _align_btc_to_15m(btc_df15: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """NEW: merges BTC's close price onto every row as 'btc_close' for cross-asset features."""
    if btc_df15 is None or btc_df15.empty or "close" not in btc_df15.columns:
        df15["btc_close"] = np.nan
        return df15
    try:
        btc_slim = (
            btc_df15[["open_time", "close"]]
            .rename(columns={"close": "btc_close"})
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )
        df15_work = (
            df15.drop(columns=["btc_close"], errors="ignore")
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )
        merged = pd.merge_asof(df15_work, btc_slim, on="open_time", direction="backward")
        return merged.reset_index(drop=True)
    except Exception as e:
        log.warning(f"_align_btc_to_15m failed ({e})")
        df15["btc_close"] = np.nan
        return df15


def _align_funding_to_15m(funding_df: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """NEW: merges the last-known perpetual funding rate onto every row as 'funding_rate'."""
    if funding_df is None or funding_df.empty:
        df15["funding_rate"] = 0.0
        return df15
    try:
        f_slim = funding_df.sort_values("open_time")
        df15_work = (
            df15.drop(columns=["funding_rate"], errors="ignore")
            .dropna(subset=["open_time"])
            .assign(open_time=lambda d: d["open_time"].astype("int64"))
            .sort_values("open_time")
        )
        merged = pd.merge_asof(df15_work, f_slim, on="open_time", direction="backward")
        merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
        return merged.reset_index(drop=True)
    except Exception as e:
        log.warning(f"_align_funding_to_15m failed ({e})")
        df15["funding_rate"] = 0.0
        return df15


def _add_extra_features(df15: pd.DataFrame) -> pd.DataFrame:
    """NEW: computes NEW_FEATURES from raw columns already merged onto df15.
    taker_buy_ratio/hour_sin/hour_cos/dow_sin/dow_cos are handled inside
    feature_engineering.add_indicators() instead — not duplicated here."""
    df = df15.copy()

    # Cross-asset context vs BTC
    if "btc_close" in df.columns and df["btc_close"].notna().sum() > 30:
        btc_ret  = df["btc_close"].pct_change()
        coin_ret = df["close"].pct_change()
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
        roll_var = btc_ret.rolling(20, min_periods=10).var()
        df["btc_corr_20"]      = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
        df["btc_beta_20"]      = roll_cov / roll_var.replace(0, np.nan)
        df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100
    else:
        df["btc_corr_20"]      = 0.0
        df["btc_beta_20"]      = 1.0
        df["btc_rel_strength"] = 0.0

    df["btc_corr_20"]      = df["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df["btc_beta_20"]      = df["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)

    # Derivatives positioning
    if "funding_rate" in df.columns:
        df["funding_rate_pct"] = (df["funding_rate"] * 100).fillna(0.0)
        df["funding_rate_chg"] = df["funding_rate_pct"].diff(3).fillna(0.0)
    else:
        df["funding_rate_pct"] = 0.0
        df["funding_rate_chg"] = 0.0

    return df


def make_targets(df: pd.DataFrame) -> pd.Series:
    labels    = pd.Series("NO_TRADE", index=df.index)
    lookahead = 24

    future_high = df["high"].shift(-1).rolling(lookahead).max().shift(-lookahead + 1)
    future_low  = df["low"].shift(-1).rolling(lookahead).min().shift(-lookahead + 1)

    LABEL_TARGET_MULT = ATR_TARGET1_MULT * 1.0

    buy_tp  = df["close"] + (df["atr"] * LABEL_TARGET_MULT)
    buy_sl  = df["close"] - (df["atr"] * ATR_STOP_MULT)
    sell_tp = df["close"] - (df["atr"] * LABEL_TARGET_MULT)
    sell_sl = df["close"] + (df["atr"] * ATR_STOP_MULT)

    labels[(future_high >= buy_tp)  & (future_low  > buy_sl)]  = "BUY"
    labels[(future_low  <= sell_tp) & (future_high < sell_sl)] = "SELL"
    return labels

def _process_segment(df15, df1h, df4h, regime, btc_df15=None, funding_df=None):
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()

    taker_col = None
    if "taker_buy_base_vol" in df15.columns:
        taker_col = df15[["open_time", "taker_buy_base_vol"]].copy()

    df15 = add_indicators(df15)

    # add_indicators() is only guaranteed to preserve OHLCV — restore this if it was dropped
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:
        df15 = df15.merge(taker_col, on="open_time", how="left")

    # 1H Alignment
    if not df1h.empty:
        df1h_feat = add_indicators(df1h)
        df15 = _align_1h_to_15m(df1h_feat, df15)
    else:
        df15["rsi_1h"] = 50.0
        df15["adx_1h"] = 0.0
        df15["trend_1h"] = 0.0

    # 4H Alignment
    if not df4h.empty:
        df4h_feat = add_indicators(df4h)
        df15 = _align_4h_to_15m(df4h_feat, df15)
    else:
        df15["rsi_4h"] = 50.0
        df15["trend_4h"] = 0.0

    # NEW: BTC benchmark + funding rate alignment, then derived features
    df15 = _align_btc_to_15m(btc_df15, df15)
    df15 = _align_funding_to_15m(funding_df, df15)
    df15 = _add_extra_features(df15)

    df15["target"] = make_targets(df15)
    df15["regime"] = regime
    return df15.iloc[:-24].copy()

def build_dataset() -> pd.DataFrame:
    log.info(f"Building REGIME-BALANCED dataset — {len(SYMBOLS)} symbols")
    all_rows = []

    log.info("  Fetching BTC benchmark (recent)...")
    btc_df15_rec = fetch_klines("BTCUSDT", "15m", RECENT_CANDLES)
    btc_bear_cache = {}

    for symbol in SYMBOLS:
        symbol_segments = []
        log.info(f"  [{symbol}] Fetching recent...")
        df15_rec = fetch_klines(symbol, "15m", RECENT_CANDLES)
        df1h_rec = fetch_klines(symbol, "1h",  RECENT_CANDLES // 4)
        df4h_rec = fetch_klines(symbol, "4h",  RECENT_CANDLES // 16)

        funding_rec = pd.DataFrame()
        if not df15_rec.empty:
            try:
                f_start = int(df15_rec["open_time"].min())
                f_end   = int(df15_rec["open_time"].max()) + 1
                funding_rec = fetch_funding_history(symbol, f_start, f_end)
            except Exception as e:
                log.debug(f"    [{symbol}] funding history (recent) skipped: {e}")

        seg = _process_segment(df15_rec, df1h_rec, df4h_rec, regime="recent_bull",
                                btc_df15=btc_df15_rec, funding_df=funding_rec)
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
            df4h_bear = fetch_klines_window(symbol, "4h",  bw["start_ms"], bw["end_ms"], bw["candles"] // 16)

            if bw["label"] not in btc_bear_cache:
                btc_bear_cache[bw["label"]] = fetch_klines_window(
                    "BTCUSDT", "15m", bw["start_ms"], bw["end_ms"], bw["candles"]
                )
            btc_bear_df15 = btc_bear_cache[bw["label"]]

            funding_bear = pd.DataFrame()
            try:
                funding_bear = fetch_funding_history(symbol, bw["start_ms"], bw["end_ms"])
            except Exception as e:
                log.debug(f"    [{symbol}] funding history ({bw['label']}) skipped: {e}")

            seg = _process_segment(df15_bear, df1h_bear, df4h_bear, regime=bw["label"],
                                    btc_df15=btc_bear_df15, funding_df=funding_bear)
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


# ── NO_TRADE undersampling ─────────────────────────────────────────────

def undersample_no_trade(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    nt_idx:  int,
    ratio:   float = UNDERSAMPLE_RATIO,
    random_state: int = 42,
) -> tuple:
    signal_mask   = y_train != nt_idx
    signal_idx    = np.where(signal_mask)[0]
    no_trade_idx  = np.where(~signal_mask)[0]

    target_nt     = int(len(signal_idx) * ratio)
    target_nt     = min(target_nt, len(no_trade_idx))

    rng           = np.random.default_rng(random_state)
    sampled_nt    = rng.choice(no_trade_idx, size=target_nt, replace=False)

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
    if "open_time" in ds.columns:
        ds = ds.sort_values("open_time").reset_index(drop=True)
        log.info("Dataset sorted globally by open_time ✓")

    for f in FULL_FEATURES:
        if f not in ds.columns:
            ds[f] = 0.0

    X  = ds[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(ds["target"])
    classes = list(le.classes_)

    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    # ── FIX: three-way split (train / calib / test) with embargo gaps ──
    # Embargo prevents label leakage across split boundaries: each label looks
    # EMBARGO_BARS candles into the future, so rows straddling a boundary can
    # otherwise "see" price action from the other side of the split.
    n_total    = len(X)
    test_size  = int(n_total * TEST_SPLIT)
    calib_size = int(n_total * CALIB_SPLIT)

    test_start  = n_total - test_size
    calib_end   = test_start - EMBARGO_BARS
    calib_start = calib_end - calib_size
    train_end   = calib_start - EMBARGO_BARS

    if train_end <= 0:
        raise ValueError(
            "Not enough rows for an embargoed train/calib/test split — "
            "reduce CALIB_SPLIT/TEST_SPLIT or gather more data."
        )

    X_train_raw, y_train_raw = X.iloc[:train_end], y[:train_end]
    X_calib,     y_calib     = X.iloc[calib_start:calib_end], y[calib_start:calib_end]
    X_test,      y_test      = X.iloc[test_start:], y[test_start:]

    log.info(
        f"Split (embargo={EMBARGO_BARS} bars/boundary): "
        f"train={len(X_train_raw):,}  calib={len(X_calib):,}  test={len(X_test):,}  "
        f"(dropped {n_total - len(X_train_raw) - len(X_calib) - len(X_test):,} embargoed rows)"
    )
    if len(X_calib) < 200:
        log.warning("  ⚠ Calibration split is small (<200 rows) — isotonic calibration may be noisy.")

    log.info("Importance scan...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train_raw, y_train_raw)
    top_idx  = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in FULL_FEATURES]

    for i in top_idx:
        if FULL_FEATURES[i] not in selected:
            selected.append(FULL_FEATURES[i])
        if len(selected) >= N_FEATURES:
            break

    log.info(f"Top {len(selected)} features selected.")
    log.info(f"  New features included: {[f for f in NEW_FEATURES if f in selected]}")

    X_train_raw_sel = X_train_raw[selected]
    Xte             = X_test[selected].values
    Xcal            = X_calib[selected].values

    X_train_sel, y_train = undersample_no_trade(X_train_raw_sel, y_train_raw, nt_idx)
    Xtr                  = X_train_sel.values

    log.info("Building asymmetric sample weights for XGBoost...")
    sw_asym = np.ones(len(y_train))
    sw_asym[y_train == buy_idx]  = 2.0
    sw_asym[y_train == sell_idx] = 2.0

    if "trend_1h" in selected:
        trend_idx = selected.index("trend_1h")
        downtrend_mask = Xtr[:, trend_idx] < 0
        sw_asym[(y_train == buy_idx) & downtrend_mask] *= 2.0
        log.info("  -> Asymmetric penalty applied to counter-trend BUYs.")

    log.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(Xtr, y_train, sample_weight=sw_asym)

    log.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.0},
    )
    rf.fit(Xtr, y_train)

    log.info("Training HistGradientBoosting...")
    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.04,
        min_samples_leaf=3, random_state=42,
        class_weight={nt_idx: 1.0, buy_idx: 2.0, sell_idx: 2.0},
    )
    gb.fit(Xtr, y_train)

    log.info("Building ensemble [XGB×3, RF×2, GB×1]...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(Xtr, y_train)

    log.info("\nRunning Walk-Forward Validation (4 chronological windows)...")
    wf_scores = []
    window = len(Xtr) // 5
    wf_embargo = min(EMBARGO_BARS, max(window // 10, 1))  # approximate — Xtr is undersampled/reindexed

    for i in range(4):
        wf_train_end  = (i + 1) * window
        wf_test_start = wf_train_end + wf_embargo
        wf_test_end   = wf_test_start + window

        if wf_test_end > len(Xtr):
            break

        probe = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss", n_jobs=-1)
        probe.fit(Xtr[:wf_train_end], y_train[:wf_train_end])

        acc_wf = accuracy_score(y_train[wf_test_start:wf_test_end], probe.predict(Xtr[wf_test_start:wf_test_end]))
        wf_scores.append(acc_wf)

    wf_mean = np.mean(wf_scores) if wf_scores else 0.0
    wf_std  = np.std(wf_scores) if wf_scores else 0.0
    log.info(f"  Walk-forward Accuracy: {wf_mean*100:.1f}% ± {wf_std*100:.1f}%")

    # ── FIX: calibrate on the dedicated calibration split, NOT on the test set ──
    log.info("\nCalibrating probability estimates (isotonic regression) on held-out calibration split...")
    calibrated_ensemble = CalibratedClassifierCV(estimator=FrozenEstimator(ensemble), method="isotonic")
    calibrated_ensemble.fit(Xcal, y_calib)

    ensemble = calibrated_ensemble

    # Final evaluation on Xte/y_test — untouched by both training AND calibration
    y_pred = ensemble.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    log.info(f"\n{'='*60}")
    log.info(f"RAW ACCURACY (misleading — dominated by NO_TRADE): {acc*100:.1f}%")
    log.info(f"{'='*60}")

    log.info("\n── What Actually Matters ─────────────────────────────────")
    for label in ["BUY", "SELL"]:
        log.info(f"  {label:<5} precision: {report.get(label, {}).get('precision', 0):.1%}  "
                 f"recall: {report.get(label, {}).get('recall', 0):.1%}  "
                 f"f1: {report.get(label, {}).get('f1-score', 0):.1%}")

    probas    = ensemble.predict_proba(Xte)
    real_buy  = (y_test == buy_idx).sum()
    real_sell = (y_test == sell_idx).sum()

    log.info("\n── Confidence Threshold Calibration ────────────────────")
    best_thresh = 0.45
    best_score  = 0.0

    # ── Expected Value Scoring Logic ──
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

        # EV calculation
        buy_ev  = pb * ATR_TARGET1_MULT - (1 - pb) * ATR_STOP_MULT
        sell_ev = ps * ATR_TARGET1_MULT - (1 - ps) * ATR_STOP_MULT

        if buy_ev <= 0 or sell_ev <= 0:
            score = 0.0  # reject — either direction loses money
        else:
            score = (buy_ev + sell_ev) * avg_recall * n_signals

        if score > best_score and n_signals > 20:
            best_score  = score
            best_thresh = thresh

        log.info(f"  {thresh:.2f}    {n_signals:>7}    {pb:>7.1%}    {ps:>8.1%}   {avg_recall:>7.1%}   ${pnl:>8,.0f}")

    log.info(f"\n  → Best EV threshold: {best_thresh:.2f}")

    pipeline = {
        "ensemble":              ensemble,
        "selector":              ImportanceSelector(selected),
        "all_features":          FULL_FEATURES,
        "best_features":         selected,
        "label_map":             {i: c for i, c in enumerate(classes)},
        "label_encoder":         le,
        "accuracy":              round(acc * 100, 1),
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "symbols":               SYMBOLS,
        "n_features":            len(FULL_FEATURES),
        "recommended_threshold": best_thresh,
        "calibrated":            True,
        "calibration_method":    "isotonic",
        "calibration_split":     "dedicated_embargoed",  # marker so you can tell old vs new models apart
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Saved: {MODEL_FILE}")

    # Verified output mapping to fix github action log
    perf = {
        "accuracy":       round(acc * 100, 1),
        "wf_mean":        round(wf_mean * 100, 1),
        "wf_std":         round(wf_std * 100, 1),
        "n_train":        int(len(X_train_raw)),
        "n_calib":        int(len(X_calib)),
        "n_train_sampled":int(len(y_train)),
        "n_test":         int(len(X_test)),
        "features":       FULL_FEATURES,
        "selected":       selected,
        "buy_precision":  round(report.get("BUY",  {}).get("precision", 0), 4),
        "sell_precision": round(report.get("SELL", {}).get("precision", 0), 4),
        "buy_recall":     round(report.get("BUY",  {}).get("recall",    0), 4),
        "sell_recall":    round(report.get("SELL", {}).get("recall",    0), 4),
    }

    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)

    return acc

if __name__ == "__main__":
    t0  = time.time()
    ds  = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")
