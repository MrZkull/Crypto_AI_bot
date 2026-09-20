#!/usr/bin/env python3
# market_data_integrity.py — Zero-Repair Market Data Cadence & Lookahead Integrity Engine

import math
import numpy as np
import pandas as pd

TIMEFRAME_MS = {
    "1m":  1 * 60 * 1000,
    "3m":  3 * 60 * 1000,
    "5m":  5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "2h":  2 * 60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "6h":  6 * 60 * 60 * 1000,
    "8h":  8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}


def interval_ms(interval_str: str) -> int:
    """Return interval duration in milliseconds."""
    if interval_str not in TIMEFRAME_MS:
        raise ValueError(
            f"Unsupported timeframe: '{interval_str}'. "
            f"Supported: {list(TIMEFRAME_MS.keys())}"
        )
    return TIMEFRAME_MS[interval_str]


def sanitize_closed_candles(
    raw,
    interval_ms_value: int = None,
    observation_time_ms: int = None,
    candle_duration_ms: int = None,
    **kwargs,
) -> list[dict]:
    """
    Strict zero-repair candle sanitizer.

    Accepted input forms:
      - pandas DataFrame
      - list[dict]
      - exchange-style array rows: [open_time, open, high, low, close, volume, close_time, ...]
      - None

    No values are repaired or interpolated. Invalid rows are rejected.
    """
    if raw is None:
        return []

    duration_ms = (
        interval_ms_value
        if interval_ms_value is not None
        else candle_duration_ms
    )
    if duration_ms is None:
        duration_ms = kwargs.get("duration_ms") or kwargs.get("interval_ms")

    rows = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else raw
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("result", []))

    clean = []
    seen_open_times = set()

    for r in rows or []:
        try:
            if isinstance(r, dict):
                o = int(r["open_time"])
                c = int(r["close_time"])
                vals = {
                    k: float(r[k])
                    for k in ("open", "high", "low", "close", "volume")
                }
                extra = {
                    k: r[k]
                    for k in r
                    if k not in vals and k not in ("open_time", "close_time")
                }
            else:
                o = int(r[0])
                c = int(r[6])
                vals = {
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
                extra = {}
                if len(r) > 9:
                    extra["taker_buy_base_vol"] = float(r[9])

            if c <= o:
                continue

            if duration_ms is not None:
                expected_c_time = o + int(duration_ms) - 1
                if c != expected_c_time:
                    continue

            if observation_time_ms is not None and c > int(observation_time_ms):
                continue

            if not all(math.isfinite(v) for v in vals.values()):
                continue

            if (
                vals["open"] <= 0
                or vals["high"] <= 0
                or vals["low"] <= 0
                or vals["close"] <= 0
                or vals["volume"] < 0
            ):
                continue

            if (
                vals["high"] < max(vals["open"], vals["close"])
                or vals["low"] > min(vals["open"], vals["close"])
            ):
                continue

            if o in seen_open_times:
                continue

            seen_open_times.add(o)
            clean.append({"open_time": o, "close_time": c, **vals, **extra})

        except (TypeError, ValueError, KeyError, OverflowError):
            continue

    clean.sort(key=lambda x: x["open_time"])
    return clean


def sanitize_ohlcv_frame(
    df: pd.DataFrame,
    interval_ms_value: int = None,
    observation_time_ms: int = None,
    candle_duration_ms: int = None,
    **kwargs,
) -> pd.DataFrame:
    """Return a sanitized pandas DataFrame while preserving extra columns."""
    duration_ms = (
        interval_ms_value
        if interval_ms_value is not None
        else candle_duration_ms
    )
    rows = sanitize_closed_candles(
        df,
        interval_ms_value=duration_ms,
        observation_time_ms=observation_time_ms,
        **kwargs,
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if observation_time_ms is not None:
        out = out[out["close_time"] <= int(observation_time_ms)]
    return out.reset_index(drop=True)


def merge_completed_htf(
    base_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    feature_cols: list = None,
    prefix: str = "htf",
    value_cols: list = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Point-in-time HTF alignment.

    The observation timestamp is the completed LTF candle close_time.
    An HTF feature row is usable only when:

        HTF source close_time <= LTF observation close_time

    Equal timestamps are explicitly allowed because the HTF candle is complete
    at that instant.
    """
    if base_df.empty or htf_df.empty:
        return base_df.copy()

    cols_to_merge = feature_cols if feature_cols is not None else value_cols
    if cols_to_merge is None:
        cols_to_merge = kwargs.get("cols", [])

    ltf = base_df.copy()
    htf = htf_df.copy()

    if "close_time" not in ltf.columns or "close_time" not in htf.columns:
        raise ValueError(
            "Both LTF and HTF frames must contain 'close_time' "
            "for point-in-time alignment"
        )

    valid_cols = [c for c in cols_to_merge if c in htf.columns]
    if not valid_cols:
        return ltf

    ltf["_obs_time"] = pd.to_numeric(ltf["close_time"], errors="coerce")
    h = htf[["close_time"] + valid_cols].copy()
    h["close_time"] = pd.to_numeric(h["close_time"], errors="coerce")
    h = h.dropna(subset=["close_time"])
    h = h.sort_values("close_time")

    rename_map = {c: f"{prefix}_{c}" for c in valid_cols}
    rename_map["close_time"] = f"{prefix}_source_close_time"
    h = h.rename(columns=rename_map)

    merged = pd.merge_asof(
        ltf.sort_values("_obs_time"),
        h,
        left_on="_obs_time",
        right_on=f"{prefix}_source_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    return (
        merged.drop(columns=["_obs_time"], errors="ignore")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
