# feature_engineering.py
# 40 professional features used by real quant traders

import pandas as pd
import numpy as np
import ta
from config import RAW_DATA_FILE, FEATURES_FILE, FEATURES

def add_indicators(df):
    df = df.copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── EMA Trend ──────────────────────────────────────────
    df["ema9"]        = ta.trend.ema_indicator(c, window=9)
    df["ema20"]       = ta.trend.ema_indicator(c, window=20)
    df["ema50"]       = ta.trend.ema_indicator(c, window=50)
    df["ema200"]      = ta.trend.ema_indicator(c, window=200)
    df["ema20_slope"] = df["ema20"].diff(5) / df["ema20"].shift(5)
    df["ema50_slope"] = df["ema50"].diff(5) / df["ema50"].shift(5)

    # Price vs EMA ratios — where is price relative to trend
    df["price_vs_ema20"]  = (c - df["ema20"]) / df["ema20"]
    df["price_vs_ema50"]  = (c - df["ema50"]) / df["ema50"]
    df["price_vs_ema200"] = (c - df["ema200"]) / df["ema200"]
    df["ema20_vs_ema50"]  = (df["ema20"] - df["ema50"]) / df["ema50"]

    # ── RSI ─────────────────────────────────────────────────
    df["rsi"]         = ta.momentum.rsi(c, window=14)
    df["rsi_slope"]   = df["rsi"].diff(5)
    df["rsi_fast"]    = ta.momentum.rsi(c, window=7)   # faster RSI

    # ── MACD ────────────────────────────────────────────────
    macd              = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_slope"]  = df["macd_hist"].diff(3)

    # ── Stochastic ──────────────────────────────────────────
    stoch             = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"]     = stoch.stoch()
    df["stoch_d"]     = stoch.stoch_signal()

    # ── ADX trend strength ──────────────────────────────────
    df["adx"]         = ta.trend.adx(h, l, c, window=14)
    df["adx_pos"]     = ta.trend.adx_pos(h, l, c, window=14)  # +DI
    df["adx_neg"]     = ta.trend.adx_neg(h, l, c, window=14)  # -DI
    df["di_diff"]     = df["adx_pos"] - df["adx_neg"]          # DI difference

    # ── Bollinger Bands ─────────────────────────────────────
    bb                = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_high"]     = bb.bollinger_hband()
    df["bb_low"]      = bb.bollinger_lband()
    df["bb_pct"]      = bb.bollinger_pband()
    df["bb_width"]    = (df["bb_high"] - df["bb_low"]) / df["ema20"]

    # ── ATR volatility ──────────────────────────────────────
    df["atr"]         = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_pct"]     = df["atr"] / c

    # ── Volume analysis ─────────────────────────────────────
    vol_ma20           = v.rolling(20).mean()
    vol_ma5            = v.rolling(5).mean()
    df["volume_ratio"] = v / vol_ma20
    df["volume_spike"] = (v > vol_ma20 * 2).astype(int)
    df["obv"]          = ta.volume.on_balance_volume(c, v)
    df["obv_slope"]    = df["obv"].diff(5) / df["obv"].shift(5).abs()

    # ── Price action ────────────────────────────────────────
    df["price_change"]  = c.pct_change()
    df["price_change3"] = c.pct_change(3)
    df["price_change6"] = c.pct_change(6)
    df["high_low_pct"]  = (h - l) / c              # candle range
    df["body_pct"]      = abs(c - df["open"]) / (h - l + 0.0001)
    df["momentum"]      = c - c.shift(10)
    df["volatility"]    = c.rolling(20).std() / c

    # ── Candlestick patterns ─────────────────────────────────
    df["bullish_candle"] = (c > df["open"]).astype(int)
    df["doji"]           = (df["body_pct"] < 0.1).astype(int)
    df["hammer"]         = (
        (df["body_pct"] < 0.3) &
        ((l - c.shift(1).clip(lower=0)) > 2 * abs(c - df["open"]))
    ).astype(int)

    return df


def add_higher_tf_features(df_15m, df_1h):
    """
    Add 1h trend features to 15m data using a bulletproof forward-fill.
    """
    df_1h = df_1h.copy()
    df_1h["ema20_1h"] = ta.trend.ema_indicator(df_1h["close"], window=20)
    df_1h["ema50_1h"] = ta.trend.ema_indicator(df_1h["close"], window=50)
    df_1h["rsi_1h"]   = ta.momentum.rsi(df_1h["close"], window=14)
    df_1h["adx_1h"]   = ta.trend.adx(df_1h["high"], df_1h["low"], df_1h["close"])
    df_1h["trend_1h"] = (df_1h["ema20_1h"] > df_1h["ema50_1h"]).astype(int)

    # Ensure timestamps are actual datetime objects
    df_1h["open_time"] = pd.to_datetime(df_1h["open_time"])
    df_15m["open_time"] = pd.to_datetime(df_15m["open_time"])

    # Sort both dataframes by time (Required for merge_asof)
    df_1h = df_1h.sort_values("open_time")
    df_15m = df_15m.sort_values("open_time")

    # The specific columns we want to bring over
    htf_cols = ["open_time", "ema20_1h", "ema50_1h", "rsi_1h", "adx_1h", "trend_1h"]

    # Perform an 'asof' merge: Matches the closest 1h time that is <= the 15m time
    df_merged = pd.merge_asof(
        df_15m, 
        df_1h[htf_cols], 
        on="open_time", 
        direction="backward"
    )

    return df_merged


def main():
    print("Building professional features...\n")

    df = pd.read_csv(RAW_DATA_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"])
    print(f"  Loaded {len(df):,} rows")
    print(f"  Intervals: {list(df['interval'].unique())}")

    all_groups = []

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol]

        df_15m = sym_df[sym_df["interval"] == "15m"].copy().reset_index(drop=True)
        df_1h  = sym_df[sym_df["interval"] == "1h"].copy().reset_index(drop=True)

        if df_15m.empty:
            continue

        # Add indicators to 15m
        df_15m = add_indicators(df_15m)

        # Merge 1h trend context if available
        if not df_1h.empty:
            df_1h = add_indicators(df_1h)
            try:
                df_15m = add_higher_tf_features(df_15m, df_1h)
            except Exception as e:
                print(f"  Warning: 1h merge failed for {symbol}: {e}")

        df_15m["symbol"] = symbol
        all_groups.append(df_15m)

    result = pd.concat(all_groups, ignore_index=True)
    result.dropna(inplace=True)
    result.to_csv(FEATURES_FILE, index=False)

    # Check features
    missing = [f for f in FEATURES if f not in result.columns]
    if missing:
        print(f"  WARNING — missing features: {missing}")
    else:
        print(f"  All {len(FEATURES)} features present")

    print(f"  Saved {len(result):,} rows to {FEATURES_FILE}")


if __name__ == "__main__":
    main()
