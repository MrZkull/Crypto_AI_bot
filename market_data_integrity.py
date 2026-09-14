"""
market_data_integrity.py — Point-in-Time Market Data Integrity Helpers

Invariants:
- Completed candles only: rejects forming candles via observation_time_ms cutoff.
- Structural physics: prices > 0, volume >= 0, High >= max(Open, Close), Low <= min(Open, Close).
- Unconditional timestamp validity: rejects records where close_time <= open_time.
- Point-in-time HTF alignment: joins on close_time using backward asof matching.
- Provenance preservation: source close timestamps remain auditable per row.
"""

import math
import pandas as pd

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_ms(interval: str) -> int:
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return _INTERVAL_MS[interval]


def sanitize_closed_candles(raw, interval_ms_value=None, observation_time_ms=None):
    """
    Sanitizes raw candle lists or DataFrames.
    Rejects forming candles (close_time > observation_time_ms) and physically impossible bars.
    """
    if raw is None:
        return []

    rows = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else raw
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("result", []))

    out = []
    seen = set()

    for r in rows or []:
        try:
            if isinstance(r, dict):
                o = int(r["open_time"])
                c = int(r["close_time"])
                vals = {k: float(r[k]) for k in ("open", "high", "low", "close", "volume")}
                extra = {k: r[k] for k in r if k not in vals and k not in ("open_time", "close_time")}
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

            # 1. Unconditional timestamp ordering check
            if c <= o:
                continue

            # 2. Numerical finiteness
            if not all(math.isfinite(v) for v in vals.values()):
                continue

            # 3. Price positivity and non-negative volume
            if vals["open"] <= 0 or vals["high"] <= 0 or vals["low"] <= 0 or vals["close"] <= 0 or vals["volume"] < 0:
                continue

            # 4. OHLC structural candle physics
            if vals["high"] < max(vals["open"], vals["close"]) or vals["low"] > min(vals["open"], vals["close"]):
                continue

            # 5. Timestamp deduplication
            if o in seen:
                continue

            # 6. Strict point-in-time observation cutoff
            if observation_time_ms is not None and c > int(observation_time_ms):
                continue

            seen.add(o)
            row = {"open_time": o, "close_time": c, **vals, **extra}
            out.append(row)
        except (TypeError, ValueError, KeyError, OverflowError):
            continue

    out.sort(key=lambda x: x["open_time"])
    return out


def sanitize_ohlcv_frame(df: pd.DataFrame, interval_ms_value=None, observation_time_ms=None) -> pd.DataFrame:
    rows = sanitize_closed_candles(df, interval_ms_value=interval_ms_value, observation_time_ms=observation_time_ms)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if observation_time_ms is not None:
        out = out[out["close_time"] <= int(observation_time_ms)]
    return out.reset_index(drop=True)


def merge_completed_htf(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, value_cols, prefix="htf") -> pd.DataFrame:
    """
    Performs leakage-free point-in-time HTF alignment using backward asof on close_time.
    Every LTF observation receives only HTF data whose source candle was fully completed.
    """
    if df_ltf.empty or df_htf.empty:
        return df_ltf.copy()

    ltf = df_ltf.copy()
    htf = df_htf.copy()

    if "close_time" not in ltf.columns:
        raise ValueError("LTF frame must contain close_time for point-in-time alignment")
    if "close_time" not in htf.columns:
        raise ValueError("HTF frame must contain close_time for point-in-time alignment")

    cols = [c for c in value_cols if c in htf.columns]
    if not cols:
        return ltf

    ltf["_observation_time"] = pd.to_numeric(ltf["close_time"], errors="coerce")
    h = htf[["close_time"] + cols].copy()
    h["close_time"] = pd.to_numeric(h["close_time"], errors="coerce")
    h = h.dropna(subset=["close_time"]).sort_values("close_time")

    rename = {c: f"{prefix}_{c}" for c in cols}
    rename["close_time"] = f"{prefix}_source_close_time"
    h = h.rename(columns=rename)

    out = pd.merge_asof(
        ltf.sort_values("_observation_time"),
        h,
        left_on="_observation_time",
        right_on=f"{prefix}_source_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    return out.drop(columns=["_observation_time"], errors="ignore").reset_index(drop=True)
