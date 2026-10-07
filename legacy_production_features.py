"""
Legacy production v2.0 feature compatibility layer.

This module reproduces the exact historical feature formulas used by the
locked production v2.0 model. It is intentionally separate from the
canonical feature_engineering.py schema.

AUTHORITATIVE SOURCE:
    phase2d_prospective_validator.py::_legacy_production_indicators

SOURCE FUNCTION SHA256:
    2b1457ec85607d6be3ed3eea3b220051eeaa1a2d04b4c7aeb1ce746e2a4b9b6d
"""

import numpy as np
import pandas as pd


LOCKED_PRODUCTION_SELECTED_FEATURES = (
    'volume_ratio',
    'volume_spike',
    'obv_slope',
    'bb_width',
    'atr_pct',
    'volatility',
    'vwap_dev',
    'trend_1h',
    'rsi_4h',
    'dow_sin',
    'dow_cos',
    'trend_4h',
    'price_vs_ema200',
    'hour_cos',
    'hour_sin',
    'rsi_1h',
    'regime_transitional',
    'adx',
    'ema20_vs_ema50',
    'adx_1h',
    'ema200',
    'bb_high',
    'ema50_slope',
    'bb_low',
    'vol_regime',
    'ema50',
    'btc_beta_20',
    'ema20',
    'macd_signal',
    'price_vs_ema50',
    'btc_corr_20',
    'ema9',
    'atr',
    'ema20_slope',
    'macd_hist'
)

SOURCE_FUNCTION_SHA256 = "2b1457ec85607d6be3ed3eea3b220051eeaa1a2d04b4c7aeb1ce746e2a4b9b6d"
SOURCE_FUNCTION_PATH = (
    "phase2d_prospective_validator.py::_legacy_production_indicators"
)


def _legacy_production_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the historical feature formulas used by locked production v2.0."""
    if df is None or df.empty:
        return df

    df = df.copy()

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    df["_had_missing_inputs"] = (
        df["_had_missing_inputs"]
        .fillna(False)
        .astype(bool)
    )
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)

    # Legacy EMAs / price relationships
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff(3) / df["ema20"].shift(3) * 100
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3) * 100
    df["price_vs_ema20"] = (c - df["ema20"]) / df["ema20"] * 100
    df["price_vs_ema50"] = (c - df["ema50"]) / df["ema50"] * 100
    df["price_vs_ema200"] = (c - df["ema200"]) / df["ema200"] * 100
    df["ema20_vs_ema50"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100

    # Legacy RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)

    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    df["rsi_slope"] = df["rsi"].diff(3)

    avg_g7 = gain.ewm(com=6, adjust=False).mean()
    avg_l7 = loss.ewm(com=6, adjust=False).mean()
    rs7 = avg_g7 / avg_l7.replace(0, np.nan)

    df["rsi_fast"] = (100 - 100 / (1 + rs7)).fillna(50)

    # Legacy stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()

    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Legacy MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_slope"] = df["macd"].diff(3)

    # Legacy ATR / ADX
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    dm_pos = (h.diff()).clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)

    atr14 = tr.ewm(span=14, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)

    df["adx"] = dx.ewm(span=14, adjust=False).mean()
    df["adx_pos"] = di_pos
    df["adx_neg"] = di_neg
    df["di_diff"] = di_pos - di_neg

    # Legacy Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    bb_high = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    bb_width = bb_high - bb_low

    df["bb_high"] = bb_high
    df["bb_low"] = bb_low
    df["bb_pct"] = (c - bb_low) / (bb_width + 1e-10)
    df["bb_width"] = bb_width / sma20 * 100

    # Legacy volume / VWAP
    vol_ma20 = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma20.replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(5) / (vol_ma20 * 5 + 1e-10)

    df["vwap"] = (c * v).cumsum() / (v.cumsum() + 1e-10)
    df["vwap_dev"] = (c - df["vwap"]) / df["vwap"] * 100

    # Legacy price-action
    df["price_change"] = c.pct_change(1) * 100
    df["price_change3"] = c.pct_change(3) * 100
    df["price_change6"] = c.pct_change(6) * 100
    df["high_low_pct"] = (h - l) / c * 100
    df["body_pct"] = (c - o).abs() / (h - l + 1e-10)
    df["momentum"] = c - c.shift(10)
    df["volatility"] = c.rolling(14).std() / c * 100

    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    df["pivot_dev"] = (c - pivot) / pivot * 100

    # Legacy candlestick patterns
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    rng = h - l + 1e-10

    df["bullish_candle"] = ((c > o) & (body > rng * 0.6)).astype(int)
    df["doji"] = (body < rng * 0.1).astype(int)
    df["hammer"] = ((lower_wick > body * 2) & (upper_wick < body)).astype(int)

    # Legacy trend
    df["trend"] = np.where(
        df["ema20"] > df["ema50"], 1,
        np.where(df["ema20"] < df["ema50"], -1, 0)
    )

    # Legacy volatility regime
    atr_smooth = df["atr_pct"].rolling(5).mean()
    adx_smooth = df["adx"].rolling(3).mean()
    df["vol_regime"] = np.where(
        (atr_smooth > 2.0) & (adx_smooth > 25), 2,
        np.where((atr_smooth < 0.5) | (adx_smooth < 15), 0, 1)
    ).astype(float)

    # Legacy order flow
    if "taker_buy_base_vol" in df.columns:
        vol_safe = v.replace(0, np.nan)
        df["taker_buy_ratio"] = (
            df["taker_buy_base_vol"].astype(float) / vol_safe
        ).fillna(0.5).clip(0, 1)
    else:
        df["taker_buy_ratio"] = 0.5

    # Legacy calendar encoding
    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        hour = ts.dt.hour.fillna(0)
        dow = ts.dt.dayofweek.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0

    if "_had_missing_inputs" not in df.columns:
        df["_had_missing_inputs"] = False

    tracked_features = [
        f
        for f in LOCKED_PRODUCTION_SELECTED_FEATURES
        if f in df.columns
    ]

    if tracked_features:
        current_missing = (
            df[tracked_features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .isna()
            .any(axis=1)
        )

        df["_had_missing_inputs"] = (
            df["_had_missing_inputs"]
            .fillna(False)
            .astype(bool)
            | current_missing
        )

    return (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .ffill()
        .fillna(0.0)
    )


def add_legacy_production_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the locked v2.0-compatible production feature frame."""
    out = _legacy_production_indicators(df)

    # Exact production-builder behavior from the locked validator:
    # regime_transitional = (trend == 0).astype(float)
    if "regime_transitional" not in out.columns:
        out["regime_transitional"] = (
            out["trend"] == 0
        ).astype(float)

    return out
