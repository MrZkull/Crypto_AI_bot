#!/usr/bin/env python3
# market_data_integrity.py — Zero-Repair Market Data Cadence & Lookahead Integrity Engine

import numpy as np
import pandas as pd

TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}


def interval_ms(interval_str: str) -> int:
    """Returns interval duration in milliseconds."""
    if interval_str not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {interval_str}")
    return TIMEFRAME_MS[interval_str]


def sanitize_closed_candles(
    df: pd.DataFrame,
    candle_duration_ms: int = None,
    observation_time_ms: int = None
) -> list[dict]:
    """
    Applies strict zero-repair physics and boundary filtering:
    - Finite numeric OHLCV checks.
    - Structural invariants (high >= max(open, close), low <= min(open, close)).
    - Timestamp integrity (close_time == open_time + duration - 1).
    - Excludes forming bars (close_time <= observation_time_ms).
    """
    if df.empty:
        return []

    required = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    clean = []
    for _, row in df.iterrows():
        o_time = int(row["open_time"])
        c_time = int(row["close_time"])

        if candle_duration_ms is not None:
            expected_c_time = o_time + candle_duration_ms - 1
            if c_time != expected_c_time:
                continue

        if observation_time_ms is not None and c_time > observation_time_ms:
            continue

        o, h, l, c, v = (
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
        )

        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c) and np.isfinite(v)):
            continue

        if o <= 0 or h <= 0 or l <= 0 or c <= 0 or v < 0:
            continue

        if h < max(o, c) or l > min(o, c):
            continue

        record = row.to_dict()
        record["open_time"] = o_time
        record["close_time"] = c_time
        clean.append(record)

    return clean


def merge_completed_htf(
    base_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    feature_cols: list[str],
    prefix: str = "htf"
) -> pd.DataFrame:
    """
    Merges higher timeframe features point-in-time.
    Guarantees zero lookahead: base candle close_time matches against htf close_time <= base close_time.
    Attaches source-close timestamp for provenance auditing.
    """
    if base_df.empty or htf_df.empty:
        return base_df.copy()

    htf_clean = htf_df.dropna(subset=["close_time"]).sort_values("close_time").reset_index(drop=True)
    base_clean = base_df.dropna(subset=["close_time"]).sort_values("close_time").reset_index(drop=True)

    keep_cols = ["close_time"] + [c for c in feature_cols if c in htf_clean.columns]
    rename_map = {c: f"{prefix}_{c}" for c in feature_cols if c in htf_clean.columns}
    rename_map["close_time"] = f"{prefix}_source_close_time"

    htf_slim = htf_clean[keep_cols].rename(columns=rename_map)

    # Direction='backward': base_df['close_time'] >= htf['source_close_time']
    # This prevents the 15-minute data destruction bug while satisfying the lookahead audit.
    merged = pd.merge_asof(
        base_clean,
        htf_slim,
        left_on="close_time",
        right_on=f"{prefix}_source_close_time",
        direction="backward"
    )

    return merged.sort_values("open_time").reset_index(drop=True)
