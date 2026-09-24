"""
feature_engineering.py — Canonical Feature Engineering & Schema Definition
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Canonical Feature Universe — Must remain strictly identical across train, scan, and promote
ALL_FEATURES = [
    # Momentum & Trend
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx",
    "plus_di",
    "minus_di",
    "trend",
    "ema20_vs_ema50",
    "price_vs_ema200",
    "regime_uptrend",
    # Volatility & Bands
    "atr",
    "atr_pct",
    "bb_width",
    "bb_pos",
    "volatility",
    # Volume Dynamics
    "volume_ratio",
    "volume_spike",
    "obv_slope",
    "vwap_dev",
    "taker_buy_ratio",
    # Time Cycles
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Higher Timeframe Context
    "rsi_1h",
    "adx_1h",
    "trend_1h",
    "rsi_4h",
    "trend_4h",
    # Macro Market Relational (BTC Aligned)
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
]


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Selects a fixed subset of top feature names deterministically."""
    def __init__(self, selected_features: list[str] = None):
        self.selected_features = selected_features or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features].values
        return X


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates canonical indicators on closed OHLCV candles."""
    df = df.copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # 1. EMAs & Trends
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    df["ema20_vs_ema50"] = (ema20 - ema50) / ema50.replace(0, np.nan)
    df["price_vs_ema200"] = (c - ema200) / ema200.replace(0, np.nan)
    df["regime_uptrend"] = (c > ema200).astype(float)
    df["trend"] = np.where(ema20 > ema50, 1.0, -1.0)

    # 2. RSI (14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # 3. MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 4. ATR & Volatility
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    df["atr_pct"] = (df["atr"] / c).fillna(0.0)
    df["volatility"] = c.pct_change().rolling(20, min_periods=5).std().fillna(0.0)

    # 5. Bollinger Bands
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std().fillna(0.0)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    df["bb_width"] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)
    df["bb_pos"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # 6. Directional Movement (ADX)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.rolling(14, min_periods=14).sum()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=14).sum() / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=14).sum() / tr_smooth.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"] = plus_di.fillna(0.0)
    df["minus_di"] = minus_di.fillna(0.0)
    df["adx"] = dx.rolling(14, min_periods=14).mean().fillna(0.0)

    # 7. Volume Dynamics
    vol_sma = v.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = (v / vol_sma.replace(0, np.nan)).fillna(1.0)
    df["volume_spike"] = (v > (vol_sma * 2.0)).astype(float)

    obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
    df["obv_slope"] = obv.diff(5) / (v.rolling(5).sum().replace(0, np.nan))
    df["obv_slope"] = df["obv_slope"].fillna(0.0)

    typical_price = (h + l + c) / 3.0
    vwap = (typical_price * v).cumsum() / v.cumsum().replace(0, np.nan)
    df["vwap_dev"] = (c - vwap) / (df["atr"].replace(0, np.nan))

    if "taker_buy_base_vol" in df.columns:
        df["taker_buy_ratio"] = (df["taker_buy_base_vol"].astype(float) / v.replace(0, np.nan)).fillna(0.5)
    else:
        df["taker_buy_ratio"] = 0.5

    # 8. Cyclical Time Encoding
    if "open_time" in df.columns:
        dt = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        hour = dt.dt.hour + dt.dt.minute / 60.0
        dow = dt.dt.dayofweek
        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    else:
        df["hour_sin"] = df["hour_cos"] = df["dow_sin"] = df["dow_cos"] = 0.0

        # Phase 2D integrity marker:
    # Record whether any currently available canonical model feature was
    # still NaN/invalid BEFORE the final blanket fill. Phase 2D uses this
    # marker to exclude missing-input observations symmetrically instead
    # of silently converting missing inputs into valid-looking zeroes.
    tracked_features = [f for f in ALL_FEATURES if f in df.columns]

    if tracked_features:
        df["_had_missing_inputs"] = (
            df[tracked_features]
            .replace([np.inf, -np.inf], np.nan)
            .isna()
            .any(axis=1)
            .astype(bool)
        )
    else:
        df["_had_missing_inputs"] = False

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
