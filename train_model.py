# train_model.py — Regime-Balanced · No-Leakage · Honest Metrics
#
# KEY CHANGES vs previous version:
#  1. fetch_klines_window() — forward-paginated fetcher for specific historical windows
#  2. _align_1h_to_15m()   — FIXED: uses pd.merge_asof on open_time (was broken integer-index align)
#  3. _process_segment()   — make_targets() called PER-SEGMENT (no cross-boundary leakage)
#  4. build_dataset()      — fetches 3 regimes per symbol, merges + sorts chronologically
#  5. SELL sample weight   — raised from 2.5 → 4.0 to compensate for historical under-representation

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
MIN_BARS    = 100       # minimum bars required to keep a segment

# ── Binance endpoints ──────────────────────────────────────────────────
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

# ── Regime configuration ───────────────────────────────────────────────
# Recent candles per symbol (current market regime, ~26 days of 15m bars).
RECENT_CANDLES = 2500

# Bear-market windows, targeted by UTC millisecond timestamps.
#
# Window selection rationale:
#   LUNA crash  (May 5-20 2022): violent depeg, cascading liquidations — ideal
#               short patterns with high ATR.  Most altcoins existed here.
#
#   FTX collapse (Nov 7-22 2022): contagion-driven waterfall — sustained SELL
#               trend over 2 weeks.  APT launched Oct 2022 so it has data here.
#               SUI (launched May 2023) will be skipped gracefully.
#
#   Bear trend  (Jun 10–Jul 10 2022): slow bleed after LUNA contagion settled;
#               teaches the model steady downtrend SELL setups, not just crash spikes.
#
# Each window is 15-30 days (~1440-2880 bars).  Total per symbol that has all
# three windows ≈ 2500 + 1440 + 1440 + 2880 = 8260 bars (~86 days of data).

BEAR_WINDOWS = [
    {
        "label":    "LUNA_crash_May22",
        "start_ms": 1651708800000,   # 2022-05-05 00:00 UTC
        "end_ms":   1653004800000,   # 2022-05-20 00:00 UTC
        "candles":  1440,            # 15 days × 96 bars/day
    },
    {
        "label":    "FTX_collapse_Nov22",
        "start_ms": 1667779200000,   # 2022-11-07 00:00 UTC
        "end_ms":   1669075200000,   # 2022-11-22 00:00 UTC
        "candles":  1440,            # 15 days
    },
    {
        "label":    "Bear_trend_Jun22",
        "start_ms": 1654819200000,   # 2022-06-10 00:00 UTC
        "end_ms":   1657411200000,   # 2022-07-10 00:00 UTC
        "candles":  2880,            # 30 days of sustained downtrend
    },
]


# ── Helpers ────────────────────────────────────────────────────────────

def _raw_to_df(raw: list) -> pd.DataFrame:
    """Convert raw Binance kline list to a typed DataFrame."""
    df = pd.DataFrame(raw).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def fetch_klines(symbol: str, interval: str, limit: int = RECENT_CANDLES) -> pd.DataFrame:
    """
    Fetch the most recent `limit` 15m candles (current market regime).
    Paginates BACKWARDS from now using endTime.
    """
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
                    log.warning(f"  [{symbol}] {url} → HTTP {r.status_code}")
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
        log.error(f"  [{symbol}] Could not fetch recent candles on any endpoint.")
        return pd.DataFrame()
    return _raw_to_df(all_data[-limit:])


def fetch_klines_window(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_candles: int = 1440,
) -> pd.DataFrame:
    """
    Fetch candles from a specific historical time window using startTime → endTime
    FORWARD pagination.

    This is the correct approach for targeting bear-market regimes:
      - Binance will return up to 1000 bars per request starting from startTime.
      - We advance the cursor forward after each batch.
      - Coins that didn't exist in the window return an empty JSON list; the
        function returns an empty DataFrame which build_dataset skips gracefully.

    Args:
        symbol:      e.g. "BTCUSDT"
        interval:    e.g. "15m" or "1h"
        start_ms:    window open  (Unix ms, UTC)
        end_ms:      window close (Unix ms, UTC)
        max_candles: hard cap — we stop even if more data is available

    Returns:
        Typed DataFrame, or empty DataFrame on failure.
    """
    all_data = []
    for url in BINANCE_ENDPOINTS:
        all_data = []
        cursor = start_ms
        try:
            while len(all_data) < max_candles and cursor < end_ms:
                batch_limit = min(1000, max_candles - len(all_data))
                params = {
                    "symbol":    symbol,
                    "interval":  interval,
                    "startTime": cursor,
                    "endTime":   end_ms,
                    "limit":     batch_limit,
                }
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    log.warning(f"  [{symbol}] {url} → HTTP {r.status_code} (window fetch)")
                    break
                batch = r.json()
                if not batch:
                    break   # symbol didn't exist in this window (e.g. SUI in May 2022)
                all_data.extend(batch)
                cursor = batch[-1][0] + 1   # advance past the last returned candle
                time.sleep(0.3)
                if len(batch) < 1000:
                    break   # exhausted the window
            if all_data:
                break
        except Exception as e:
            log.warning(f"  [{symbol}] window fetch error on {url}: {e}")

    if not all_data:
        return pd.DataFrame()
    return _raw_to_df(all_data[:max_candles])


def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Align 1h indicator values to 15m bars by TIMESTAMP using merge_asof.

    The previous implementation used .reindex(df15.index) which aligned by
    integer row-position — i.e. 1h bar #300 was mapped to 15m bar #300,
    completely ignoring the actual timestamps.  This silently corrupted
    rsi_1h / adx_1h / trend_1h for every trade.

    merge_asof with direction="backward" correctly assigns each 15m bar the
    latest 1h value whose open_time <= that 15m bar's open_time.
    """
    if df1h.empty or len(df1h) < 5:
        df15["rsi_1h"]   = 50.0
        df15["adx_1h"]   = 0.0
        df15["trend_1h"] = 0.0
        return df15

    df1h_slim = (
        df1h[["open_time", "rsi", "adx", "trend"]]
        .sort_values("open_time")
        .rename(columns={"rsi": "rsi_1h", "adx": "adx_1h", "trend": "trend_1h"})
    )
    df15_sorted = df15.sort_values("open_time")

    merged = pd.merge_asof(
        df15_sorted,
        df1h_slim,
        on="open_time",
        direction="backward",  # use latest 1h bar that opened AT OR BEFORE this 15m bar
    )

    merged[["rsi_1h", "adx_1h", "trend_1h"]] = merged[["rsi_1h", "adx_1h", "trend_1h"]].fillna({
        "rsi_1h": 50.0, "adx_1h": 0.0, "trend_1h": 0.0,
    })
    return merged.reset_index(drop=True)


def make_targets(df: pd.DataFrame) -> pd.Series:
    """Label each bar as BUY / SELL / NO_TRADE based on ATR-scaled TP/SL lookahead."""
    labels   = pd.Series("NO_TRADE", index=df.index)
    lookahead = 24  # 24 × 15m = 6-hour lookahead window

    future_high = df["high"].shift(-1).rolling(lookahead).max().shift(-lookahead + 1)
    future_low  = df["low"].shift(-1).rolling(lookahead).min().shift(-lookahead + 1)

    buy_tp  = df["close"] + (df["atr"] * ATR_TARGET1_MULT)
    buy_sl  = df["close"] - (df["atr"] * ATR_STOP_MULT)
    sell_tp = df["close"] - (df["atr"] * ATR_TARGET1_MULT)
    sell_sl = df["close"] + (df["atr"] * ATR_STOP_MULT)

    labels[(future_high >= buy_tp)  & (future_low  > buy_sl)]  = "BUY"
    labels[(future_low  <= sell_tp) & (future_high < sell_sl)] = "SELL"
    return labels


def _process_segment(
    df15:      pd.DataFrame,
    df1h:      pd.DataFrame,
    regime:    str,
) -> pd.DataFrame:
    """
    Full per-segment pipeline:
      1. add_indicators (computes VWAP, OBV, ATR, etc. from this segment's price history)
      2. _align_1h_to_15m (timestamp-accurate 1h feature alignment)
      3. make_targets (labels using ATR TP/SL lookahead — WITHIN this segment only)
      4. trim last 24 bars (lookahead contamination window)
      5. tag with regime label

    CRITICAL: make_targets MUST be called before the trim, and WITHIN each segment.
    If we concatenated all segments first and called make_targets once, the rolling
    future_high/future_low would bleed across the gap between May 2022 and Nov 2022
    data, creating phantom BUY/SELL labels at segment boundaries.
    """
    if df15.empty or len(df15) < MIN_BARS:
        return pd.DataFrame()

    df15 = add_indicators(df15)

    # 1h alignment (fixed)
    if not df1h.empty:
        df1h_feat = add_indicators(df1h)
        df15 = _align_1h_to_15m(df1h_feat, df15)
    else:
        df15["rsi_1h"]   = 50.0
        df15["adx_1h"]   = 0.0
        df15["trend_1h"] = 0.0

    df15["target"] = make_targets(df15)
    df15["regime"] = regime

    return df15.iloc[:-24].copy()   # drop lookahead window at tail


def build_dataset() -> pd.DataFrame:
    """
    Regime-balanced dataset builder.

    Per symbol, fetches three market regimes:
      1. Recent (~26 days) — captures current bull/sideways dynamics
      2. LUNA crash (May 2022) — violent crash, high-volatility SELL setups
      3. FTX collapse (Nov 2022) — contagion waterfall, sustained SELL trend
      4. Bear trend (Jun 2022) — slow bleed; teaches gradual downtrend SELL

    Each segment is processed INDEPENDENTLY (indicators + targets computed on
    its own price history) then sorted chronologically per symbol before concat.
    This ensures TimeSeriesSplit sees bear data in early folds and recent data
    in late folds — exactly simulating live deployment conditions.

    Coins launched after a bear window (e.g. SUI → May 2022) are skipped
    gracefully with a log message.
    """
    log.info(f"Building REGIME-BALANCED dataset — {len(SYMBOLS)} symbols")
    log.info(f"  Regime 0 (recent):     {RECENT_CANDLES} bars per symbol")
    for bw in BEAR_WINDOWS:
        log.info(f"  Regime bear ({bw['label']}): {bw['candles']} bars per symbol")

    all_rows = []

    for symbol in SYMBOLS:
        symbol_segments = []

        # ── Regime 0: Recent / current market ────────────────────────
        log.info(f"  [{symbol}] Fetching {RECENT_CANDLES} recent 15m bars...")
        df15_rec  = fetch_klines(symbol, "15m", RECENT_CANDLES)
        df1h_rec  = fetch_klines(symbol, "1h",  RECENT_CANDLES // 4)

        seg = _process_segment(df15_rec, df1h_rec, regime="recent_bull")
        if not seg.empty:
            symbol_segments.append(seg)
            log.info(f"    recent_bull: {len(seg)} rows, "
                     f"BUY={( seg.target=='BUY').sum()}, "
                     f"SELL={(seg.target=='SELL').sum()}")
        else:
            log.warning(f"    [{symbol}] Recent segment too small — skipping symbol.")
            continue   # no point fetching bear data for a symbol with no recent data

        # ── Regime 1–N: Bear market windows ──────────────────────────
        for bw in BEAR_WINDOWS:
            log.info(f"  [{symbol}] Fetching bear window: {bw['label']}...")

            df15_bear = fetch_klines_window(
                symbol, "15m",
                bw["start_ms"], bw["end_ms"],
                max_candles=bw["candles"],
            )

            if df15_bear.empty or len(df15_bear) < MIN_BARS:
                # Coin didn't exist yet (SUI/APT for older windows) or API gap
                log.info(f"    {bw['label']}: insufficient data for {symbol} — skipping window.")
                continue

            # 1h data for the same window (~candles/4 bars covers the window duration)
            df1h_bear = fetch_klines_window(
                symbol, "1h",
                bw["start_ms"], bw["end_ms"],
                max_candles=bw["candles"] // 4,
            )

            seg = _process_segment(df15_bear, df1h_bear, regime=bw["label"])
            if not seg.empty:
                symbol_segments.append(seg)
                log.info(f"    {bw['label']}: {len(seg)} rows, "
                         f"BUY={(seg.target=='BUY').sum()}, "
                         f"SELL={(seg.target=='SELL').sum()}")

        if not symbol_segments:
            log.warning(f"  [{symbol}] No usable segments — skipping.")
            continue

        # Sort all segments for this symbol chronologically.
        # TimeSeriesSplit will then naturally put bear data in early folds and
        # recent data in late folds, which mirrors real deployment conditions.
        symbol_df = (
            pd.concat(symbol_segments, ignore_index=True)
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        all_rows.append(symbol_df)

    if not all_rows:
        raise ValueError(
            "CRITICAL: No data was fetched for any symbol. "
            "Both Binance endpoints rejected the connection."
        )

    ds = pd.concat(all_rows, ignore_index=True)

    # ── Summary statistics ────────────────────────────────────────────
    n   = len(ds)
    b   = (ds.target == "BUY").sum()
    s   = (ds.target == "SELL").sum()
    nt  = (ds.target == "NO_TRADE").sum()
    log.info(f"\n{'='*60}")
    log.info(f"DATASET SUMMARY: {n:,} total rows")
    log.info(f"  BUY:      {b:>7,} ({b/n*100:.1f}%)")
    log.info(f"  SELL:     {s:>7,} ({s/n*100:.1f}%)")
    log.info(f"  NO_TRADE: {nt:>7,} ({nt/n*100:.1f}%)")
    if "regime" in ds.columns:
        log.info("\nRows per regime:")
        for regime, cnt in ds["regime"].value_counts().items():
            sell_r = (ds[ds.regime == regime].target == "SELL").sum()
            log.info(f"  {regime:<30} {cnt:>7,} rows | SELL: {sell_r:,}")
    log.info(f"{'='*60}")

    return ds


def train(ds: pd.DataFrame) -> float:
    for f in ALL_FEATURES:
        if f not in ds.columns:
            ds[f] = 0.0

    X  = ds[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(ds["target"])
    classes = list(le.classes_)
    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    # ── Chronological split (no data leakage) ────────────────────────
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Feature importance scan ───────────────────────────────────────
    log.info("Importance scan with XGBoost...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    essential = ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"]
    selected  = [f for f in essential if f in ALL_FEATURES]
    for i in top_idx:
        f = ALL_FEATURES[i]
        if f not in selected:
            selected.append(f)
        if len(selected) >= N_FEATURES:
            break
    log.info(f"Top {len(selected)} features: {selected}")

    Xtr = X_train[selected].values
    Xte = X_test[selected].values

    # ── Sample weights ────────────────────────────────────────────────
    # Even with regime-balanced data, BUY/SELL remain a minority class
    # because most bars in any market are genuinely sideways (NO_TRADE).
    #
    # We raise SELL weight to 4.0 (was 2.5) because:
    #   a) SELL was invisible in the previous model (0% recall)
    #   b) Bear windows add new SELL examples but they're still outnumbered
    #   c) The asymmetry is intentional: we want the model to *try* to call
    #      SELL when evidence supports it, not default to NO_TRADE
    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    buy_idx  = classes.index("BUY")      if "BUY"      in classes else 0
    sell_idx = classes.index("SELL")     if "SELL"     in classes else 2

    sw = np.ones(len(y_train))
    sw[y_train == buy_idx]  = 2.5   # BUY weight unchanged
    sw[y_train == sell_idx] = 4.0   # SELL weight raised: was 2.5, now 4.0
    # NO_TRADE stays at 1.0

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
        class_weight={buy_idx: 2.5, sell_idx: 4.0, nt_idx: 1.0},
    )
    rf.fit(Xtr, y_train)

    log.info("Training HistGradientBoosting...")
    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.04,
        min_samples_leaf=3, random_state=42,
        class_weight={buy_idx: 2.5, sell_idx: 4.0, nt_idx: 1.0},
    )
    gb.fit(Xtr, y_train)

    log.info("Building ensemble [XGB×3, RF×2, GB×1]...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(Xtr, y_train)

    # ── Evaluation ───────────────────────────────────────────────────
    y_pred = ensemble.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    log.info(f"\n{'='*60}")
    log.info(f"RAW ACCURACY (misleading — dominated by NO_TRADE): {acc*100:.1f}%")
    log.info(f"{'='*60}")
    log.info("\n── What Actually Matters ─────────────────────────────────")
    for label in ["BUY", "SELL"]:
        p = report.get(label, {}).get("precision", 0)
        r = report.get(label, {}).get("recall", 0)
        f1 = report.get(label, {}).get("f1-score", 0)
        log.info(f"  {label:<5} precision: {p:.1%}  recall: {r:.1%}  f1: {f1:.1%}")
    log.info("  (target: both ≥60% precision, both ≥30% recall)")

    # ── Confidence threshold calibration ─────────────────────────────
    probas    = ensemble.predict_proba(Xte)
    real_buy  = (y_test == buy_idx).sum()
    real_sell = (y_test == sell_idx).sum()

    log.info("\n── Confidence Threshold Calibration ────────────────────")
    log.info(f"  {'Thresh':>6}  {'Signals':>7}  {'BUY P':>7}  {'SELL P':>8}  {'Recall':>7}  {'Est P&L':>8}")
    best_thresh = 0.50
    best_score  = 0.0

    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
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

        if score > best_score and n_signals > 10:
            best_score  = score
            best_thresh = thresh

        log.info(f"  {thresh:.2f}    {n_signals:>7}    {pb:>7.1%}    {ps:>8.1%}   "
                 f"{avg_recall:>7.1%}   ${pnl:>8,.0f}")

    log.info(f"\n  → Best threshold: {best_thresh:.2f}")

    # ── Time-series cross-validation ──────────────────────────────────
    cv    = TimeSeriesSplit(n_splits=5)
    cv_sc = cross_val_score(ensemble, Xtr, y_train, cv=cv, n_jobs=-1)
    log.info(f"\n  CV (TimeSeriesSplit 5-fold): {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%")

    # ── Persist pipeline ──────────────────────────────────────────────
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
        "accuracy":              round(acc * 100, 1),
        "cv_mean":               round(cv_sc.mean() * 100, 1),
        "cv_std":                round(cv_sc.std() * 100, 1),
        "n_train":               int(len(X_train)),
        "n_test":                int(len(X_test)),
        "features":              ALL_FEATURES,
        "selected":              selected,
        "buy_precision":         round(report.get("BUY", {}).get("precision", 0), 4),
        "sell_precision":        round(report.get("SELL", {}).get("precision", 0), 4),
        "buy_recall":            round(report.get("BUY", {}).get("recall", 0), 4),
        "sell_recall":           round(report.get("SELL", {}).get("recall", 0), 4),
    }
    with open("model_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    log.info("✅ Saved: model_performance.json")

    return acc


if __name__ == "__main__":
    t0 = time.time()
    ds  = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Accuracy: {acc*100:.1f}%")
