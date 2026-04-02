# feature_engineering.py
# CRITICAL: This file must have add_indicators(df) function.
# trade_executor.py imports this function directly.
# The old version was just a standalone script — this is the fix.

import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV dataframe and returns it with all indicators added.
    Called by trade_executor.py for every symbol during live scanning.
    Uses pure pandas/numpy — no 'ta' library dependency issues.
    """
    df = df.copy()

    # Ensure numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    if len(df) < 20:
        return df

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── EMAs ─────────────────────────────────────────────
    df["ema9"]   = c.ewm(span=9,   adjust=False).mean()
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    df["ema20_slope"] = df["ema20"].diff(3) / df["ema20"].shift(3) * 100
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3) * 100

    df["price_vs_ema20"]  = (c - df["ema20"])  / df["ema20"]  * 100
    df["price_vs_ema50"]  = (c - df["ema50"])  / df["ema50"]  * 100
    df["price_vs_ema200"] = (c - df["ema200"]) / df["ema200"] * 100
    df["ema20_vs_ema50"]  = (df["ema20"] - df["ema50"]) / df["ema50"] * 100

    # ── RSI ───────────────────────────────────────────────
    delta   = c.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)
    avg_g   = gain.ewm(com=13, adjust=False).mean()
    avg_l   = loss.ewm(com=13, adjust=False).mean()
    rs      = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)
    df["rsi_slope"] = df["rsi"].diff(3)

    # Fast RSI (7-period)
    avg_g7  = gain.ewm(com=6, adjust=False).mean()
    avg_l7  = loss.ewm(com=6, adjust=False).mean()
    rs7     = avg_g7 / avg_l7.replace(0, np.nan)
    df["rsi_fast"] = 100 - (100 / (1 + rs7))
    df["rsi_fast"].fillna(50, inplace=True)

    # ── Stochastic ────────────────────────────────────────
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    denom  = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = ((c - low14) / denom * 100).fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(50)

    # ── MACD ─────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_slope"]  = df["macd"].diff(2)

    # ── ADX ───────────────────────────────────────────────
    prev_c  = c.shift(1)
    tr      = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    up_move   = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr14       = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr14.replace(0, np.nan)
    minus_di14 = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr14.replace(0, np.nan)

    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan)
    df["adx"]     = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    df["adx_pos"] = plus_di14.fillna(0)
    df["adx_neg"] = minus_di14.fillna(0)
    df["di_diff"] = df["adx_pos"] - df["adx_neg"]

    # ── ATR ───────────────────────────────────────────────
    df["atr"]     = tr.ewm(alpha=1/14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    # ── Bollinger Bands ───────────────────────────────────
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_high"]  = bb_upper
    df["bb_low"]   = bb_lower
    df["bb_pct"]   = ((c - bb_lower) / bb_range * 100).fillna(50)
    df["bb_width"] = (bb_range / bb_mid * 100).fillna(0)

    # ── Volume ────────────────────────────────────────────
    vol_ma = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = (v / vol_ma).fillna(1)
    df["volume_spike"] = (df["volume_ratio"] > 2).astype(float)
    df["obv_slope"]    = (np.sign(c.diff()) * v).rolling(10).sum().diff(3).fillna(0)

    # ── Price Changes ─────────────────────────────────────
    df["price_change"]  = c.pct_change(1) * 100
    df["price_change3"] = c.pct_change(3) * 100
    df["price_change6"] = c.pct_change(6) * 100

    # ── Candle Properties ─────────────────────────────────
    df["high_low_pct"]  = (h - l) / l * 100
    body                = (c - df["open"]).abs()
    wick_total          = (h - l).replace(0, np.nan)
    df["body_pct"]      = (body / wick_total * 100).fillna(0)
    df["momentum"]      = c - c.shift(10)
    df["volatility"]    = c.rolling(20).std().fillna(0)

    # ── Candle Patterns ───────────────────────────────────
    df["bullish_candle"] = (c > df["open"]).astype(float)
    total_range = (h - l).replace(0, np.nan)
    df["doji"]   = ((c - df["open"]).abs() / total_range < 0.1).fillna(False).astype(float)
    lower_wick   = pd.concat([df["open"], c], axis=1).min(axis=1) - l
    df["hammer"] = ((lower_wick / total_range.replace(0, np.nan) > 0.6)
                    & (body / total_range.replace(0, np.nan) < 0.3)).fillna(False).astype(float)

    # ── Higher timeframe placeholders (filled by trade_executor) ──
    # These get overwritten with actual 1h values in generate_signal()
    if "rsi_1h"   not in df.columns: df["rsi_1h"]   = 50.0
    if "adx_1h"   not in df.columns: df["adx_1h"]   = 0.0
    if "trend_1h" not in df.columns: df["trend_1h"] = 0.0

    # trend column used by higher timeframe check
    df["trend"] = (df["ema20"] > df["ema50"]).astype(float)

    return df.fillna(0)
