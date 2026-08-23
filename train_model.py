# train_model.py — V3.6: Institutional Training Pipeline with Threshold Floor & Zero Data Leakage

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

# ── Dynamic Configuration Sync ────────────────────────────────────────
try:
    from config import (
        SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT
    )
except ImportError:
    SYMBOLS = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
        "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
        "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
        "ADAUSDT", "TRXUSDT", "ZECUSDT", "TAOUSDT", "XLMUSDT", "HBARUSDT",
        "PENDLEUSDT", "WIFUSDT", "CRVUSDT", "RENDERUSDT"
    ]
    ATR_STOP_MULT    = 2.5
    ATR_TARGET1_MULT = 3.5
    ATR_TARGET2_MULT = 7.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TEST_SPLIT         = 0.20
CALIB_SPLIT        = 0.15
EMBARGO_BARS       = 24
MODEL_FILE         = "pro_crypto_ai_model.pkl"
N_FEATURES         = 35
MIN_BARS           = 100
UNDERSAMPLE_RATIO  = 1.0

# Strict Precision Guardrails
MIN_BUY_THRESHOLD_FLOOR  = 0.40  # Prevents BUY barrier from collapsing
MIN_SELL_THRESHOLD_FLOOR = 0.45  # Prevents SELL precision regression below 55%+

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

RECENT_CANDLES = 5000
PINNED_RECENT_WINDOW = None

BEAR_WINDOWS = [
    {"label": "LUNA_crash_May22",   "start_ms": 1651708800000, "end_ms": 1653004800000, "candles": 1440},
    {"label": "FTX_collapse_Nov22", "start_ms": 1667779200000, "end_ms": 1669075200000, "candles": 1440},
    {"label": "Bear_trend_Jun22",   "start_ms": 1654819200000, "end_ms": 1657411200000, "candles": 2880},
    {"label": "Aug2023_dip",        "start_ms": 1690848000000, "end_ms": 1692057600000, "candles": 1440},
    {"label": "Apr2024_halving",    "start_ms": 1713225600000, "end_ms": 1714435200000, "candles": 1440},
    {"label": "Bull_peak_Oct21",    "start_ms": 1633046400000, "end_ms": 1638316800000, "candles": 2880},
    {"label": "Recovery_Jan23",     "start_ms": 1672531200000, "end_ms": 1675209600000, "candles": 2880},
]

NEW_FEATURES = [
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
]
FULL_FEATURES = ALL_FEATURES + NEW_FEATURES


# ── Data Fetching ──────────────────────────────────────────────────────

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


def _fetch_deribit_klines(symbol: str, limit: int) -> pd.DataFrame:
    try:
        base = symbol.replace("USDT", "").upper()
        inst = f"{base}_USDC-PERPETUAL"
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (limit * 15 * 60 * 1000)
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
            params={"instrument_name": inst, "resolution": "15", "start_timestamp": start_ms, "end_timestamp": now_ms},
            timeout=10
        )
        if r.status_code == 200:
            res = r.json().get("result", {})
            ticks = res.get("ticks", [])
            if ticks:
                df = pd.DataFrame({
                    "open_time": ticks,
                    "open": [float(x) for x in res.get("open", [])],
                    "high": [float(x) for x in res.get("high", [])],
                    "low": [float(x) for x in res.get("low", [])],
                    "close": [float(x) for x in res.get("close", [])],
                    "volume": [float(x) for x in res.get("volume", [])],
                    "taker_buy_base_vol": [float(x) * 0.5 for x in res.get("volume", [])]
                })
                return df.sort_values("open_time").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_klines(symbol: str, interval: str, limit: int = RECENT_CANDLES) -> pd.DataFrame:
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
                time.sleep(0.15)
                if len(all_data) >= limit:
                    break
            if all_data:
                return _raw_to_df(all_data[-limit:])
        except Exception:
            pass

    # Deribit fallback for secondary pairs
    deribit_df = _fetch_deribit_klines(symbol, limit)
    if not deribit_df.empty:
        log.info(f"  [{symbol}] Fetched {len(deribit_df)} candles via Deribit fallback endpoint.")
        return deribit_df

    return pd.DataFrame()


def fetch_klines_window(symbol, interval, start_ms, end_ms, max_candles=1440):
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
                time.sleep(0.15)
                if len(batch) < 1000:
                    break
            if all_data:
                return _raw_to_df(all_data[:max_candles])
        except Exception:
            pass
    return pd.DataFrame()


# ── Feature Alignment & Target Engineering ─────────────────────────────

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
        log.warning(f"_align_1h_to_15m failed ({e}) — fallback to defaults")
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
        log.warning(f"_align_4h_to_15m failed ({e}) — fallback to defaults")
        return _apply_defaults(df15)


def _align_btc_to_15m(btc_df15: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
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


def _add_extra_features(df15: pd.DataFrame) -> pd.DataFrame:
    df = df15.copy()
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

    return df


def make_targets(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    labels = np.full(n, "NO_TRADE", dtype=object)
    lookahead = 24

    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    atrs  = df["atr"].values if "atr" in df.columns else np.zeros(n)

    target_mult = ATR_TARGET1_MULT

    for i in range(n - lookahead):
        entry = closes[i]
        atr = atrs[i]
        if atr <= 0 or np.isnan(atr):
            continue

        buy_tp = entry + (atr * target_mult)
        buy_sl = entry - (atr * ATR_STOP_MULT)
        sell_tp = entry - (atr * target_mult)
        sell_sl = entry + (atr * ATR_STOP_MULT)

        # Path evaluation for BUY
        buy_success = False
        for k in range(1, lookahead + 1):
            idx = i + k
            if lows[idx] <= buy_sl:
                break
            if highs[idx] >= buy_tp:
                buy_success = True
                break

        # Path evaluation for SELL
        sell_success = False
        for k in range(1, lookahead + 1):
            idx = i + k
            if highs[idx] >= sell_sl:
                break
            if lows[idx] <= sell_tp:
                sell_success = True
                break

        if buy_success and not sell_success:
            labels[i] = "BUY"
        elif sell_success and not buy_success:
            labels[i] = "SELL"

    return pd.Series(labels, index=df.index)


def _process_segment(symbol, df15, df1h, df4h, regime, btc_df15=None):
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
    else:
        df15["rsi_1h"] = 50.0
        df15["adx_1h"] = 0.0
        df15["trend_1h"] = 0.0

    if not df4h.empty:
        df4h_feat = add_indicators(df4h)
        df15 = _align_4h_to_15m(df4h_feat, df15)
    else:
        df15["rsi_4h"] = 50.0
        df15["trend_4h"] = 0.0

    df15 = _align_btc_to_15m(btc_df15, df15)
    df15 = _add_extra_features(df15)

    df15["symbol"] = symbol
    df15["target"] = make_targets(df15)
    df15["regime"] = regime
    return df15.iloc[:-24].copy()


def _fetch_recent(symbol: str, interval: str, limit_divisor: int = 1) -> pd.DataFrame:
    if PINNED_RECENT_WINDOW is not None:
        return fetch_klines_window(
            symbol, interval,
            PINNED_RECENT_WINDOW["start_ms"], PINNED_RECENT_WINDOW["end_ms"],
            max(PINNED_RECENT_WINDOW["candles"] // limit_divisor, 50),
        )
    return fetch_klines(symbol, interval, RECENT_CANDLES // limit_divisor)


def build_dataset() -> pd.DataFrame:
    log.info(f"Building REGIME-BALANCED dataset — {len(SYMBOLS)} symbols")
    all_rows = []

    log.info("  Fetching BTC benchmark (recent)...")
    btc_df15_rec = _fetch_recent("BTCUSDT", "15m")
    btc_bear_cache = {}

    for symbol in SYMBOLS:
        symbol_segments = []
        log.info(f"  [{symbol}] Fetching candles...")
        df15_rec = _fetch_recent(symbol, "15m")
        df1h_rec = _fetch_recent(symbol, "1h", 4)
        df4h_rec = _fetch_recent(symbol, "4h", 16)

        seg = _process_segment(symbol, df15_rec, df1h_rec, df4h_rec, regime="recent_bull",
                                btc_df15=btc_df15_rec)
        if not seg.empty:
            symbol_segments.append(seg)
        else:
            log.warning(f"    [{symbol}] No recent data — skipping symbol.")
            continue

        for bw in BEAR_WINDOWS:
            df15_bear = fetch_klines_window(symbol, "15m", bw["start_ms"], bw["end_ms"], bw["candles"])
            if df15_bear.empty or len(df15_bear) < MIN_BARS:
                continue
            df1h_bear = fetch_klines_window(symbol, "1h",  bw["start_ms"], bw["end_ms"], bw["candles"] // 4)
            df4h_bear = fetch_klines_window(symbol, "4h",  bw["start_ms"], bw["end_ms"], bw["candles"] // 16)

            if bw["label"] not in btc_bear_cache:
                btc_bear_cache[bw["label"]] = fetch_klines_window(
                    "BTCUSDT", "15m", bw["start_ms"], bw["end_ms"], bw["candles"]
                )
            btc_bear_df15 = btc_bear_cache[bw["label"]]

            seg = _process_segment(symbol, df15_bear, df1h_bear, df4h_bear, regime=bw["label"],
                                    btc_df15=btc_bear_df15)
            if not seg.empty:
                symbol_segments.append(seg)

        if symbol_segments:
            all_rows.append(pd.concat(symbol_segments, ignore_index=True).sort_values("open_time").reset_index(drop=True))

    if not all_rows:
        raise ValueError("No data fetched for any symbol.")

    ds = pd.concat(all_rows, ignore_index=True)
    n  = len(ds)
    b  = (ds.target == "BUY").sum()
    s  = (ds.target == "SELL").sum()
    nt = (ds.target == "NO_TRADE").sum()

    log.info(f"\n{'='*60}")
    log.info(f"DATASET SUMMARY: {n:,} rows | BUY: {b:,} ({b/n*100:.1f}%) | SELL: {s:,} ({s/n*100:.1f}%) | NO_TRADE: {nt:,}")
    log.info(f"{'='*60}")

    return ds


# ── NO_TRADE Undersampling ─────────────────────────────────────────────

def undersample_no_trade(X_train: pd.DataFrame, y_train: np.ndarray, nt_idx: int, ratio: float = UNDERSAMPLE_RATIO) -> tuple:
    signal_mask  = y_train != nt_idx
    signal_idx   = np.where(signal_mask)[0]
    no_trade_idx = np.where(~signal_mask)[0]
    target_nt    = min(int(len(signal_idx) * ratio), len(no_trade_idx))

    rng = np.random.default_rng(42)
    sampled_nt = rng.choice(no_trade_idx, size=target_nt, replace=False)
    keep = np.sort(np.concatenate([signal_idx, sampled_nt]))

    return X_train.iloc[keep].reset_index(drop=True), y_train[keep]


# ── Training & Calibration Engine ──────────────────────────────────────

def train(ds: pd.DataFrame) -> float:
    for f in FULL_FEATURES:
        if f not in ds.columns:
            ds[f] = 0.0

    le = LabelEncoder()
    le.fit(ds["target"])
    classes = list(le.classes_)

    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    train_parts, calib_parts, test_parts = [], [], []
    dropped_total = 0

    for (sym, regime), grp in ds.groupby(["symbol", "regime"], sort=False):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        n_r = len(grp)

        test_size_r  = int(n_r * TEST_SPLIT)
        calib_size_r = int(n_r * CALIB_SPLIT)

        test_start_r  = n_r - test_size_r
        calib_end_r   = test_start_r - EMBARGO_BARS
        calib_start_r = calib_end_r - calib_size_r
        train_end_r   = calib_start_r - EMBARGO_BARS

        if train_end_r <= 0:
            train_parts.append(grp)
            continue

        train_parts.append(grp.iloc[:train_end_r])
        calib_parts.append(grp.iloc[calib_start_r:calib_end_r])
        test_parts.append(grp.iloc[test_start_r:])
        dropped_total += (n_r - train_end_r - (calib_end_r - calib_start_r) - test_size_r)

    train_df = pd.concat(train_parts, ignore_index=True)
    calib_df = pd.concat(calib_parts, ignore_index=True) if calib_parts else train_df.iloc[:0]
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else train_df.iloc[:0]

    X_train_raw = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_raw = le.transform(train_df["target"])

    X_calib = calib_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_calib = le.transform(calib_df["target"]) if len(calib_df) > 0 else np.array([])

    X_test = test_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = le.transform(test_df["target"]) if len(test_df) > 0 else np.array([])

    log.info(
        f"True (Symbol, Regime) Split (embargo={EMBARGO_BARS} bars/boundary): "
        f"train={len(X_train_raw):,}  calib={len(X_calib):,}  test={len(X_test):,}  "
        f"(dropped ~{dropped_total:,} embargoed rows across all symbol timelines)"
    )

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

    log.info("Building sample weights for XGBoost...")
    sw_asym = np.ones(len(y_train))
    sw_asym[y_train == buy_idx]  = 2.0
    sw_asym[y_train == sell_idx] = 2.0

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
    wf_embargo = min(EMBARGO_BARS, max(window // 10, 1))

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

    calibration_method = "isotonic"
    log.info(f"\nCalibrating probability estimates ({calibration_method}) on held-out calibration split...")
    calibrated_ensemble = CalibratedClassifierCV(estimator=FrozenEstimator(ensemble), method=calibration_method)
    calibrated_ensemble.fit(Xcal, y_calib)
    ensemble = calibrated_ensemble

    # ── Threshold Tuning with Precision Guardrail Floor ──
    calib_probas = ensemble.predict_proba(Xcal)
    calib_buy_n  = (y_calib == buy_idx).sum()
    calib_sell_n = (y_calib == sell_idx).sum()

    best_thresh_buy, best_score_buy   = MIN_BUY_THRESHOLD_FLOOR, 0.0
    best_thresh_sell, best_score_sell = MIN_SELL_THRESHOLD_FLOOR, 0.0

    # Search strictly at or above institutional confidence floors
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        yp = [np.argmax(p) if np.argmax(p) != nt_idx and p[np.argmax(p)] >= thresh else nt_idx for p in calib_probas]
        yp = np.array(yp)
        bm, sm = (yp == buy_idx), (yp == sell_idx)

        pb = (y_calib[bm] == buy_idx).mean()  if bm.sum() > 0 else 0
        ps = (y_calib[sm] == sell_idx).mean() if sm.sum() > 0 else 0
        rb = (yp[y_calib == buy_idx] == buy_idx).mean()   if calib_buy_n > 0 else 0
        rs = (yp[y_calib == sell_idx] == sell_idx).mean() if calib_sell_n > 0 else 0

        buy_ev  = pb * ATR_TARGET1_MULT - (1 - pb) * ATR_STOP_MULT
        sell_ev = ps * ATR_TARGET1_MULT - (1 - ps) * ATR_STOP_MULT

        buy_score  = buy_ev * rb * np.sqrt(max(bm.sum(), 1))  if buy_ev > 0 else 0.0
        sell_score = sell_ev * rs * np.sqrt(max(sm.sum(), 1)) if sell_ev > 0 else 0.0

        if thresh >= MIN_BUY_THRESHOLD_FLOOR and buy_score > best_score_buy and bm.sum() > 15:
            best_score_buy, best_thresh_buy = buy_score, thresh
        if thresh >= MIN_SELL_THRESHOLD_FLOOR and sell_score > best_score_sell and sm.sum() > 15:
            best_score_sell, best_thresh_sell = sell_score, thresh

    # Final assertion to guarantee precision floor preservation
    best_thresh_buy  = max(MIN_BUY_THRESHOLD_FLOOR, best_thresh_buy)
    best_thresh_sell = max(MIN_SELL_THRESHOLD_FLOOR, best_thresh_sell)

    log.info(f"\n  → Selected BUY Threshold:  {best_thresh_buy:.2f} (Floor: {MIN_BUY_THRESHOLD_FLOOR:.2f})")
    log.info(f"  → Selected SELL Threshold: {best_thresh_sell:.2f} (Floor: {MIN_SELL_THRESHOLD_FLOOR:.2f})")

    # ── Final Evaluation on Untouched Test Split ──
    probas = ensemble.predict_proba(Xte)
    y_pred_tuned = []
    for p in probas:
        pred_c = np.argmax(p)
        if pred_c == buy_idx and p[pred_c] >= best_thresh_buy:
            y_pred_tuned.append(buy_idx)
        elif pred_c == sell_idx and p[pred_c] >= best_thresh_sell:
            y_pred_tuned.append(sell_idx)
        else:
            y_pred_tuned.append(nt_idx)
    y_pred_tuned = np.array(y_pred_tuned)

    acc    = accuracy_score(y_test, y_pred_tuned)
    report = classification_report(y_test, y_pred_tuned, target_names=classes, output_dict=True, zero_division=0)

    log.info(f"\n{'='*60}")
    log.info(f"RAW ACCURACY: {acc*100:.1f}%")
    log.info(f"{'='*60}")

    for label in ["BUY", "SELL"]:
        log.info(f"  {label:<5} precision: {report.get(label, {}).get('precision', 0):.1%}  "
                 f"recall: {report.get(label, {}).get('recall', 0):.1%}  "
                 f"f1: {report.get(label, {}).get('f1-score', 0):.1%}")

    pipeline = {
        "ensemble":                   ensemble,
        "selector":                   ImportanceSelector(selected),
        "all_features":               FULL_FEATURES,
        "best_features":              selected,
        "label_map":                  {i: c for i, c in enumerate(classes)},
        "label_encoder":              le,
        "accuracy":                   round(acc * 100, 1),
        "trained_at":                 datetime.now(timezone.utc).isoformat(),
        "symbols":                    SYMBOLS,
        "n_features":                 len(FULL_FEATURES),
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "recommended_threshold":      max(best_thresh_buy, best_thresh_sell),
        "calibrated":                 True,
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Saved: {MODEL_FILE}")

    perf = {
        "accuracy":                   round(acc * 100, 1),
        "test_accuracy":              f"{round(acc * 100, 1)}%",
        "wf_mean":                    round(wf_mean * 100, 1),
        "wf_std":                     round(wf_std * 100, 1),
        "n_train":                    int(len(X_train_raw)),
        "n_calib":                    int(len(X_calib)),
        "n_train_sampled":            int(len(y_train)),
        "n_test":                     int(len(X_test)),
        "features":                   FULL_FEATURES,
        "selected":                   selected,
        "recommended_threshold_buy":  best_thresh_buy,
        "recommended_threshold_sell": best_thresh_sell,
        "buy_precision":              round(report.get("BUY",  {}).get("precision", 0), 4),
        "sell_precision":             round(report.get("SELL", {}).get("precision", 0), 4),
        "buy_recall":                 round(report.get("BUY",  {}).get("recall",    0), 4),
        "sell_recall":                round(report.get("SELL", {}).get("recall",    0), 4),
    }

    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)

    return acc


if __name__ == "__main__":
    t0  = time.time()
    ds  = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")
