#!/usr/bin/env python3
"""
research/audit_archive_cadence.py — Canonical Parquet Archive Integrity & Cadence Auditor

Enforces the Zero-Repair Data Integrity Policy:
1. Schema existence (open_time, close_time, open, high, low, close, volume)
2. Strict numeric type & finiteness across all OHLCV columns
3. Physical candle validity (High >= max(Open, Close), Low <= min(Open, Close), prices > 0, vol >= 0)
4. Monotonic increasing open_time and zero duplicate timestamps
5. Exact inclusive millisecond bar duration (close_time - open_time == interval_ms - 1)
6. Cadence break classification (missing gaps vs sub-cadence overlaps vs chronological inversions)
7. Zero-repair physics check (any candle requiring modification triggers an audit failure)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from config import SYMBOLS
except ImportError:
    SYMBOLS = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
        "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
        "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
        "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT", "FILUSDT"
    ]

from market_data_integrity import interval_ms, sanitize_closed_candles

HISTORICAL_DIR = BASE_DIR / "data" / "historical"
INTERVALS = ["15m", "1h", "4h"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = {"open_time", "close_time", *OHLCV_COLUMNS}


def audit_parquet_file(symbol: str, interval: str) -> dict:
    path = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return {"status": "MISSING_FILE", "errors": [f"File not found: {path.name}"]}

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"status": "READ_ERROR", "errors": [f"Corrupted Parquet file: {e}"]}

    errors = []
    cadence_ms = interval_ms(interval)
    expected_duration_ms = cadence_ms - 1

    # 1. Schema Existence
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return {"status": "SCHEMA_FAIL", "errors": [f"Missing columns: {sorted(missing_cols)}"]}

    # 2. Timestamp Type, Finiteness, and Null Checks
    for col in ["open_time", "close_time"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"status": "TYPE_FAIL", "errors": [f"Timestamp '{col}' must be numeric integer"]}
        if df[col].isna().any() or np.isinf(df[col]).any():
            return {"status": "NULL_TIMESTAMP", "errors": [f"Timestamp '{col}' contains NaN or Inf values"]}

    # 3. OHLCV Type, Finiteness, and Positivity
    for col in OHLCV_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"OHLCV column '{col}' is not a numeric dtype")
            continue
        if df[col].isna().any() or np.isinf(df[col]).any():
            errors.append(f"OHLCV column '{col}' contains NaN or Inf values")
        if col != "volume" and (df[col] <= 0).any():
            errors.append(f"OHLCV column '{col}' contains zero or negative prices")
        elif col == "volume" and (df[col] < 0).any():
            errors.append("Volume contains negative values")

    if errors:
        return {"status": "DATA_CORRUPTION", "errors": errors}

    # 4. Duplicate Open Timestamps
    dup_count = int(df["open_time"].duplicated().sum())
    if dup_count > 0:
        errors.append(f"{dup_count} duplicate open_time timestamps detected")

    # 5. Strict Monotonicity
    if not df["open_time"].is_monotonic_increasing:
        errors.append("open_time is not strictly monotonic increasing")

    # 6. Inclusive Duration: close_time - open_time == interval_ms - 1
    durations = df["close_time"] - df["open_time"]
    invalid_durations = durations != expected_duration_ms
    if invalid_durations.any():
        errors.append(
            f"{int(invalid_durations.sum())} rows fail inclusive duration check "
            f"(expected duration: {expected_duration_ms}ms)"
        )

    # 7. Cadence Anomaly Classification
    diffs = df["open_time"].diff().dropna()
    anomalies = diffs[diffs != cadence_ms]

    if not anomalies.empty:
        missing_candles = anomalies[anomalies > cadence_ms]
        overlaps = anomalies[(anomalies > 0) & (anomalies < cadence_ms)]
        inversions = anomalies[anomalies <= 0]

        if not missing_candles.empty:
            max_gap = missing_candles.max()
            missing_bars = int((missing_candles // cadence_ms - 1).sum())
            errors.append(
                f"CADENCE_GAP: {len(missing_candles)} gap events "
                f"({missing_bars} missing bars total, largest gap: {max_gap / 3_600_000:.2f}h)"
            )
        if not overlaps.empty:
            errors.append(f"TIMESTAMP_OVERLAP: {len(overlaps)} events with sub-cadence spacing")
        if not inversions.empty:
            errors.append(f"CHRONOLOGY_CORRUPTION: {len(inversions)} non-positive timestamp progressions")

    # 8. Zero-Repair Physical Physics Check
    clean_rows = sanitize_closed_candles(df)
    if len(clean_rows) != len(df):
        dropped = len(df) - len(clean_rows)
        errors.append(
            f"Zero-Repair violation: {dropped} rows violate candle physics "
            f"(High < max(Open, Close) or Low > min(Open, Close))"
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(df),
        "errors": errors
    }


def run_cadence_audit():
    print("═" * 68)
    print(" CANONICAL DATA INTEGRITY GATE: PARQUET CADENCE & SCHEMA AUDIT ")
    print("═" * 68 + "\n")

    if not HISTORICAL_DIR.exists():
        print(f"❌ Missing data directory: {HISTORICAL_DIR}")
        print("   Populate data/historical/{SYMBOL}_{15m,1h,4h}.parquet before running audit.")
        sys.exit(1)

    failures = 0
    total_files = 0

    for sym in SYMBOLS:
        for interval in INTERVALS:
            total_files += 1
            res = audit_parquet_file(sym, interval)
            status_icon = "✓" if res["status"] == "PASS" else "❌"
            print(f"[{status_icon}] {sym:<10} | {interval:<3} | Rows: {res.get('rows', 0):>7,} | Status: {res['status']}")
            if res["errors"]:
                failures += 1
                for err in res["errors"]:
                    print(f"    └── {err}")

    print("-" * 68)
    print(f"Audited: {total_files} files | Failed: {failures}")
    if failures > 0:
        print("🚨 AUDIT FAILED: Correct Parquet datasets prior to model retraining.")
        sys.exit(1)
    print("✅ DATA INTEGRITY VERIFIED: All historical files conform to schema and cadence.")


if __name__ == "__main__":
    run_cadence_audit()
