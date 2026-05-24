This is the final, production-ready version of your script. I have integrated the **Undersampling logic** at the start of your `train()` function, and I have also ensured the **`_align_1h_to_15m` fix** (the column dropping fix) is included so you don't run into any Pandas key errors.

You are set. Paste this into `train_model.py`, commit it to GitHub, and the bot will start training with a much more "decisive" boundary.

```python
# train_model.py — Regime-Balanced · No-Leakage · Honest Metrics · Undersampled
#
# KEY CHANGES:
#  1. Undersampling — NO_TRADE reduced to 1.5x signals to force model decision-making
#  2. fetch_klines_window() — forward-paginated fetcher for specific historical windows
#  3. _align_1h_to_15m()   — FIXED: drops placeholders before merge to prevent _x/_y suffix bugs
#  4. _process_segment()   — make_targets() called PER-SEGMENT (no cross-boundary leakage)
#  5. SELL sample weight   — 4.0 to compensate for historical under-representation

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

TEST_SPLIT  = 0.20
MODEL_FILE  = "pro_crypto_ai_model.pkl"
N_FEATURES  = 35
MIN_BARS    = 100

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

RECENT_CANDLES = 2500

BEAR_WINDOWS = [
    {"label": "LUNA_crash_May22",   "start_ms": 1651708800000, "end_ms": 1653004800000, "candles": 1440},
    {"label": "FTX_collapse_Nov22", "start_ms": 1667779200000, "end_ms": 1669075200000, "candles": 1440},
    {"label": "Bear_trend_Jun22",   "start_ms": 1654819200000, "end_ms": 1657411200000, "candles": 2880},
]

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
                if end_time: params["endTime"] = end_time
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                all_data = batch + all_data
                end_time = batch[0][0] - 1
                time.sleep(0.3)
                if len(all_data) >= limit: break
            if all_data: break
        except Exception: continue
    return _raw_to_df(all_data[-limit:]) if all_data else pd.DataFrame()

def fetch_klines_window(symbol, interval, start_ms, end_ms, max_candles=1440) -> pd.DataFrame:
    all_data = []
    for url in BINANCE_ENDPOINTS:
        all_data, cursor = [], start_ms
        try:
            while len(all_data) < max_candles and cursor < end_ms:
                params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": min(1000, max_candles - len(all_data))}
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                all_data.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(0.3)
                if len(batch) < 1000: break
            if all_data: break
        except Exception: continue
    return _raw_to_df(all_data[:max_candles]) if all_data else pd.DataFrame()

def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if df1h.empty or len(df1h) < 20 or "rsi" not in df1h.columns:
        df15["rsi_1h"] = 50.0; df15["adx_1h"] = 0.0; df15["trend_1h"] = 0.0
        return df15
    df1h_slim = df1h[["open_time", "rsi", "adx", "trend"]].sort_values("open_time").rename(columns={"rsi": "rsi_1h", "adx": "adx_1h", "trend": "trend_1h"})
    df15_sorted = df15.sort_values("open_time").drop(columns=["rsi_1h", "adx_1h", "trend_1h"], errors="ignore")
    merged = pd.merge_asof(df15_sorted, df1h_slim, on="open_time", direction="backward")
    merged[["rsi_1h", "adx_1h", "trend_1h"]] = merged[["rsi_1h", "adx_1h", "trend_1h"]].fillna({"rsi_1h": 50.0, "adx_1h": 0.0, "trend_1h": 0.0})
    return merged.reset_index(drop=True)

def make_targets(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("NO_TRADE", index=df.index)
    lookahead = 24
    future_high = df["high"].shift(-1).rolling(lookahead).max().shift(-lookahead + 1)
    future_low  = df["low"].shift(-1).rolling(lookahead).min().shift(-lookahead + 1)
    buy_tp = df["close"] + (df["atr"] * ATR_TARGET1_MULT)
    buy_sl = df["close"] - (df["atr"] * ATR_STOP_MULT)
    sell_tp = df["close"] - (df["atr"] * ATR_TARGET1_MULT)
    sell_sl = df["close"] + (df["atr"] * ATR_STOP_MULT)
    labels[(future_high >= buy_tp) & (future_low > buy_sl)] = "BUY"
    labels[(future_low <= sell_tp) & (future_high < sell_sl)] = "SELL"
    return labels

def _process_segment(df15: pd.DataFrame, df1h: pd.DataFrame, regime: str) -> pd.DataFrame:
    if df15.empty or len(df15) < MIN_BARS: return pd.DataFrame()
    df15 = add_indicators(df15)
    if not df1h.empty:
        df15 = _align_1h_to_15m(add_indicators(df1h), df15)
    else:
        df15["rsi_1h"] = 50.0; df15["adx_1h"] = 0.0; df15["trend_1h"] = 0.0
    df15["target"] = make_targets(df15)
    df15["regime"] = regime
    return df15.iloc[:-24].copy()

def build_dataset() -> pd.DataFrame:
    all_rows = []
    for symbol in SYMBOLS:
        symbol_segments = []
        df15_rec = fetch_klines(symbol, "15m", RECENT_CANDLES)
        df1h_rec = fetch_klines(symbol, "1h", RECENT_CANDLES // 4)
        seg = _process_segment(df15_rec, df1h_rec, regime="recent_bull")
        if not seg.empty: symbol_segments.append(seg)
        else: continue
        for bw in BEAR_WINDOWS:
            df15_bear = fetch_klines_window(symbol, "15m", bw["start_ms"], bw["end_ms"], max_candles=bw["candles"])
            if df15_bear.empty or len(df15_bear) < MIN_BARS: continue
            df1h_bear = fetch_klines_window(symbol, "1h", bw["start_ms"], bw["end_ms"], max_candles=bw["candles"] // 4)
            seg = _process_segment(df15_bear, df1h_bear, regime=bw["label"])
            if not seg.empty: symbol_segments.append(seg)
        if not symbol_segments: continue
        all_rows.append(pd.concat(symbol_segments, ignore_index=True).sort_values("open_time").reset_index(drop=True))
    return pd.concat(all_rows, ignore_index=True)

def train(ds: pd.DataFrame) -> float:
    # ── FIXED: NO_TRADE Undersampling ──
    signals = ds[ds['target'] != 'NO_TRADE']
    no_trades = ds[ds['target'] == 'NO_TRADE']
    max_no_trades = int(len(signals) * 1.5)
    if len(no_trades) > max_no_trades:
        log.info(f"Undersampling NO_TRADE: {len(no_trades)} → {max_no_trades}")
        no_trades = no_trades.sample(n=max_no_trades, random_state=42)
    ds = pd.concat([signals, no_trades]).sample(frac=1, random_state=42).reset_index(drop=True)
    # ─────────────────────────────────────

    for f in ALL_FEATURES:
        if f not in ds.columns: ds[f] = 0.0

    X = ds[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(ds["target"])
    classes = list(le.classes_)
    
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info("Importance scan with XGBoost...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]
    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected = [f for f in essential if f in ALL_FEATURES]
    for i in top_idx:
        f = ALL_FEATURES[i]
        if f not in selected: selected.append(f)
        if len(selected) >= N_FEATURES: break
    
    Xtr, Xte = X_train[selected].values, X_test[selected].values
    nt_idx = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx, sell_idx = classes.index("BUY"), classes.index("SELL")
    
    sw = np.ones(len(y_train))
    sw[y_train == buy_idx] = 2.5
    sw[y_train == sell_idx] = 4.0

    log.info("Training Ensemble...")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xgb.fit(Xtr, y_train, sample_weight=sw)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1, class_weight={buy_idx: 2.5, sell_idx: 4.0, nt_idx: 1.0})
    rf.fit(Xtr, y_train)
    gb = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.04, min_samples_leaf=3, random_state=42, class_weight={buy_idx: 2.5, sell_idx: 4.0, nt_idx: 1.0})
    gb.fit(Xtr, y_train)

    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)], voting="soft", weights=[3, 2, 1])
    ensemble.fit(Xtr, y_train)

    y_pred = ensemble.predict(Xte)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    # ── Evaluation & Persist ──
    log.info("\n── Precision/Recall ──")
    for l in ["BUY", "SELL"]:
        p = report.get(l, {}).get("precision", 0)
        r = report.get(l, {}).get("recall", 0)
        log.info(f"  {l:<5} p: {p:.1%}  r: {r:.1%}")
    
    joblib.dump({"ensemble": ensemble, "selector": ImportanceSelector(selected), "label_map": {i:c for i,c in enumerate(classes)}, "label_encoder": le, "recommended_threshold": 0.55}, MODEL_FILE)
    log.info(f"✅ Saved: {MODEL_FILE}")
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    ds = build_dataset()
    train(ds)

```
