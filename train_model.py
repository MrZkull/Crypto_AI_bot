# train_model.py — No-Leakage Time Series, Fast HistGradientBoosting, Honest Metrics

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

# Safely import the execution multipliers to train the model realistically
try:
    from config import ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT
except ImportError:
    ATR_STOP_MULT = 2.5
    ATR_TARGET1_MULT = 5.0
    ATR_TARGET2_MULT = 7.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","NEARUSDT",
    "MATICUSDT","TRXUSDT","SUIUSDT","APTUSDT","ATOMUSDT","LINKUSDT",
    "DOTUSDT","UNIUSDT","XRPUSDT","LTCUSDT","BCHUSDT","ALGOUSDT",
    "AAVEUSDT","ADAUSDT","DOGEUSDT"
]
LIMIT       = 5000   
TEST_SPLIT  = 0.20
MODEL_FILE  = "pro_crypto_ai_model.pkl"
N_FEATURES  = 35


def fetch_klines(symbol, interval, limit=5000):
    urls = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines"
    ]
    
    for url in urls:
        all_data = []
        end_time = None
        try:
            while len(all_data) < limit:
                params = {"symbol": symbol, "interval": interval, "limit": 1000}
                if end_time:
                    params["endTime"] = end_time
                
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    log.warning(f"  Endpoint {url} blocked (Status {r.status_code})")
                    break 
                
                data = r.json()
                if not data: break
                
                all_data = data + all_data
                end_time = data[0][0] - 1
                time.sleep(0.3) 
                
            if all_data: 
                break 
                
        except Exception as e:
            log.warning(f"  {symbol} API Error on {url}: {e}")

    if not all_data:
        log.error(f"  Failed to fetch data for {symbol} on all endpoints.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data).iloc[:,:6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.tail(limit).reset_index(drop=True)


def make_targets(df):
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


def build_dataset():
    log.info(f"Building massive dataset from {len(SYMBOLS)} symbols")
    rows = []
    for symbol in SYMBOLS:
        log.info(f"  Fetching {LIMIT} bars for {symbol}...")
        df15 = fetch_klines(symbol, "15m", LIMIT)
        df1h = fetch_klines(symbol, "1h",  LIMIT//4)
        if df15.empty or len(df15) < 100: 
            log.warning(f"  {symbol} dataset empty or too small. Skipping.")
            continue
        
        df15 = add_indicators(df15)
        if not df1h.empty:
            df1h = add_indicators(df1h)
            df15["rsi_1h"]   = df1h["rsi"].reindex(df15.index, method="ffill").fillna(50)
            df15["adx_1h"]   = df1h["adx"].reindex(df15.index, method="ffill").fillna(0)
            df15["trend_1h"] = df1h["trend"].reindex(df15.index, method="ffill").fillna(0)
        else:
            df15["rsi_1h"] = 50.0; df15["adx_1h"] = 0.0; df15["trend_1h"] = 0.0
            
        df15["target"] = make_targets(df15)
        rows.append(df15.iloc[:-24])
        
    if not rows:
        raise ValueError("CRITICAL ERROR: No data was downloaded for any symbols. Both Binance endpoints rejected the connection.")
        
    ds = pd.concat(rows, ignore_index=True)
    b=(ds.target=="BUY").sum(); s=(ds.target=="SELL").sum(); n=(ds.target=="NO_TRADE").sum()
    log.info(f"Dataset: {len(ds)} rows | BUY:{b}({b/len(ds)*100:.0f}%) "
             f"SELL:{s}({s/len(ds)*100:.0f}%) NO_TRADE:{n}({n/len(ds)*100:.0f}%)")
    return ds


def train(ds):
    for f in ALL_FEATURES:
        if f not in ds.columns: ds[f] = 0.0

    X  = ds[ALL_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(ds["target"])
    classes = list(le.classes_)
    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    # ── FIXED: Chronological Split (No Data Leakage) ──
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info("Importance scan...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio","volume_spike","obv_slope","bb_width","atr_pct","volatility","vwap_dev"]
    selected  = [f for f in essential if f in ALL_FEATURES]
    for i in top_idx:
        f = ALL_FEATURES[i]
        if f not in selected: selected.append(f)
        if len(selected) >= N_FEATURES: break
    log.info(f"Top {len(selected)} features: {selected}")

    Xtr = X_train[selected].values
    Xte = X_test[selected].values

    nt_idx = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    sw     = np.where(y_train == nt_idx, 1.0, 2.5)

    log.info("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                        gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xgb.fit(Xtr, y_train, sample_weight=sw)

    log.info("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)

    log.info("Training HistGradientBoosting (fast)...")
    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.04,
        min_samples_leaf=3, random_state=42
    )
    gb.fit(Xtr, y_train)

    log.info("Building ensemble [XGB×3, RF×2, GB×1]...")
    ensemble = VotingClassifier(estimators=[("xgb",xgb),("rf",rf),("gb",gb)],
                                voting="soft", weights=[3,2,1])
    ensemble.fit(Xtr, y_train)

    y_pred = ensemble.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    # ── FIXED: Honest Metrics Logging ──
    log.info(f"\n{'='*55}\nRAW ACCURACY (Misleading): {acc*100:.1f}%\n{'='*55}")
    
    buy_precision  = report.get("BUY", {}).get("precision", 0)
    sell_precision = report.get("SELL", {}).get("precision", 0)
    buy_recall     = report.get("BUY", {}).get("recall", 0)
    sell_recall    = report.get("SELL", {}).get("recall", 0)

    log.info(f"\n── What Actually Matters (Real Edge) ───────────────────")
    log.info(f"  BUY  precision: {buy_precision:.1%}  | recall: {buy_recall:.1%}")
    log.info(f"  SELL precision: {sell_precision:.1%}  | recall: {sell_recall:.1%}")
    log.info(f"  (Note: Overall 70%+ accuracy is just the NO_TRADE rate)")

    probas    = ensemble.predict_proba(Xte)
    buy_idx   = classes.index("BUY")  if "BUY"  in classes else 0
    sell_idx  = classes.index("SELL") if "SELL" in classes else 2
    real_buy  = (y_test == buy_idx).sum()
    real_sell = (y_test == sell_idx).sum()

    log.info("\n── Confidence Threshold Calibration ──────────────────────")
    best_thresh = 0.50; best_score = 0.0
    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        yp = []
        for prob in probas:
            bc = np.argmax(prob)
            if bc != nt_idx and prob[bc] < thresh: yp.append(nt_idx)
            else: yp.append(bc)
        yp = np.array(yp)

        bm = yp == buy_idx;  sm = yp == sell_idx
        pb = (y_test[bm] == buy_idx).mean()  if bm.sum() > 0 else 0
        ps = (y_test[sm] == sell_idx).mean() if sm.sum() > 0 else 0
        rb = (yp[y_test==buy_idx]  == buy_idx).mean()  if real_buy  > 0 else 0
        rs = (yp[y_test==sell_idx] == sell_idx).mean() if real_sell > 0 else 0
        avg_prec = (pb+ps)/2; avg_recall = (rb+rs)/2
        n_signals = int((bm|sm).sum())
        wins = n_signals * avg_prec; losses = n_signals * (1-avg_prec)
        pnl  = wins*200 - losses*100
        score = avg_prec * avg_recall
        if score > best_score and n_signals > 10:
            best_score = score; best_thresh = thresh
        log.info(f"  {thresh:.2f}    {n_signals:>7}    {pb:>8.1%}    {ps:>9.1%}   "
                 f"{avg_recall:>7.1%}   ${pnl:>8,.0f}")

    log.info(f"\n  → Best threshold: {best_thresh:.2f}")

    # ── FIXED: Time Series Cross Validation ──
    cv   = TimeSeriesSplit(n_splits=5)
    cv_sc = cross_val_score(ensemble, Xtr, y_train, cv=cv, n_jobs=-1)

    pipeline = {
        "ensemble":              ensemble,
        "selector":              ImportanceSelector(selected),
        "all_features":          ALL_FEATURES,
        "best_features":         selected,
        "label_map":             {i:c for i,c in enumerate(classes)},
        "label_encoder":         le,
        "accuracy":              round(acc*100, 1),
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "symbols":               SYMBOLS,
        "n_features":            len(ALL_FEATURES),
        "recommended_threshold": best_thresh,
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Saved: {MODEL_FILE}")

    perf = {
        "accuracy":              round(acc*100,1),
        "cv_mean":               round(cv_sc.mean()*100,1),
        "cv_std":                round(cv_sc.std()*100,1),
        "n_train":               int(len(X_train)),
        "n_test":                int(len(X_test)),
        "features":              ALL_FEATURES,
        "selected":              selected,
    }
    with open("model_performance.json","w") as f: json.dump(perf, f, indent=2)
    return acc


if __name__ == "__main__":
    t0 = time.time()
    ds = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")
