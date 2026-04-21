# train_model.py — Final Accuracy Fix (Feature Importance + Git Sync)
import os, json, time, logging, joblib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

from feature_engineering import add_indicators, ALL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── FIX: Picklable Importance Selector ──────────────────────────────
class ImportanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Works for both DataFrames and Numpy arrays
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        return X  # During live trade, X is already processed

# ── Config ────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    "SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "MATICUSDT", "ATOMUSDT",
    "LINKUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT",
    "FETUSDT", "RENDERUSDT", "ADAUSDT", "INJUSDT", "ARBUSDT", "OPUSDT", "SEIUSDT"
]
LIMIT        = 1500
TARGET_BARS  = 6
TARGET_PCT   = 0.005 
TEST_SPLIT   = 0.20
MODEL_FILE   = "pro_crypto_ai_model.pkl"

def fetch_klines(symbol, interval, limit=1500):
    for url in ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as e: log.warning(f"  {symbol}: {e}")
    return pd.DataFrame()

def make_targets(df):
    future_close = df["close"].shift(-TARGET_BARS)
    pct_change   = (future_close - df["close"]) / df["close"]
    labels = pd.Series("NO_TRADE", index=df.index)
    labels[pct_change >  TARGET_PCT] = "BUY"
    labels[pct_change < -TARGET_PCT] = "SELL"
    return labels

def build_dataset():
    log.info(f"Building dataset from {len(SYMBOLS)} symbols")
    all_rows = []
    for symbol in SYMBOLS:
        df15 = fetch_klines(symbol, "15m", LIMIT)
        df1h = fetch_klines(symbol, "1h", LIMIT // 4)
        if df15.empty or len(df15) < 100: continue
        time.sleep(0.1)
        df15 = add_indicators(df15)
        if not df1h.empty:
            df1h = add_indicators(df1h)
            df15["rsi_1h"] = df1h["rsi"].reindex(df15.index, method="ffill").fillna(50)
            df15["adx_1h"] = df1h["adx"].reindex(df15.index, method="ffill").fillna(0)
            df15["trend_1h"] = df1h["trend"].reindex(df15.index, method="ffill").fillna(0)
        df15["target"] = make_targets(df15)
        all_rows.append(df15.iloc[:-TARGET_BARS])
    return pd.concat(all_rows, ignore_index=True)

def balance_dataset(dataset):
    buy_rows = dataset[dataset.target == "BUY"]
    sell_rows = dataset[dataset.target == "SELL"]
    notrade_rows = dataset[dataset.target == "NO_TRADE"]
    signal_avg = (len(buy_rows) + len(sell_rows)) // 2
    notrade_rows = notrade_rows.sample(n=min(len(notrade_rows), int(signal_avg * 1.5)), random_state=42)
    return pd.concat([buy_rows, sell_rows, notrade_rows]).sample(frac=1, random_state=42)

def train(dataset):
    X = dataset[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = LabelEncoder().fit_transform(dataset["target"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)

    # ── ACCURACY FIX: Feature Importance Selection ──────────────────
    log.info("Running Importance Scan...")
    selector_model = XGBClassifier(n_estimators=100, random_state=42)
    selector_model.fit(X_train, y_train)
    importances = selector_model.feature_importances_
    indices = np.argsort(importances)[-30:] # Take top 30
    selected_feats = [ALL_FEATURES[i] for i in indices]
    log.info(f"Top Features: {selected_feats}")

    X_train_sel = X_train[selected_feats]
    X_test_sel  = X_test[selected_feats]

    # ── Training ────────────────────────────────────────────────────
    xgb = XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)

    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)], voting='soft', weights=[3, 2, 1])
    ensemble.fit(X_train_sel, y_train)

    acc = accuracy_score(y_test, ensemble.predict(X_test_sel))
    log.info(f"FINAL ACCURACY: {acc*100:.1f}%")

    # ── Save Pipeline ───────────────────────────────────────────────
    pipeline = {
        "ensemble": ensemble,
        "selector": ImportanceSelector(selected_feats),
        "all_features": ALL_FEATURES,
        "label_map": {0: "BUY", 1: "NO_TRADE", 2: "SELL"},
        "accuracy": round(acc * 100, 1),
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    joblib.dump(pipeline, MODEL_FILE)
    return acc

if __name__ == "__main__":
    ds = balance_dataset(build_dataset())
    train(ds)
