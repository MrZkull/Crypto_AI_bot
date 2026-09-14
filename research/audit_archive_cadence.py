#!/usr/bin/env python3
"""
research/audit_archive_cadence.py — Canonical Parquet Archive Integrity & Cadence Auditor

Enforces the Zero-Repair Data Integrity Policy:
1. Schema existence (open_time, close_time, open, high, low, close, volume)
2. Strict integer timestamps and finite OHLCV values
3. Physical candle validity
4. Monotonic increasing open_time and zero duplicate timestamps
5. Exact inclusive millisecond bar duration
6. Cadence break classification
7. Zero-repair policy: any rejected/corrupted row fails the audit
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
        "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT",
        "FILUSDT",
    ]

from market_data_integrity import interval_ms, sanitize_closed_candles


HISTORICAL_DIR = BASE_DIR / "data" / "historical"
INTERVALS = ["15m", "1h", "4h"]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = {
    "open_time",
    "close_time",
    *OHLCV_COLUMNS,
}


def audit_parquet_file(symbol: str, interval: str) -> dict:
    path = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"

    if not path.exists():
        return {
            "status": "MISSING_FILE",
            "rows": 0,
            "errors": [f"File not found: {path.name}"],
        }

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {
            "status": "READ_ERROR",
            "rows": 0,
            "errors": [f"Corrupted Parquet file: {e}"],
        }

    errors = []

    try:
        cadence_ms = interval_ms(interval)
    except Exception as e:
        return {
            "status": "CONFIG_ERROR",
            "rows": len(df),
            "errors": [str(e)],
        }

    expected_duration_ms = cadence_ms - 1

    # ------------------------------------------------------------------
    # 1. Schema existence
    # ------------------------------------------------------------------
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return {
            "status": "SCHEMA_FAIL",
            "rows": len(df),
            "errors": [
                f"Missing required columns: {sorted(missing_cols)}"
            ],
        }

    # Empty canonical file is invalid for a training archive.
    if df.empty:
        return {
            "status": "EMPTY_FILE",
            "rows": 0,
            "errors": ["Parquet file contains zero rows"],
        }

    # ------------------------------------------------------------------
    # 2. Timestamp type, finiteness, nulls
    # ------------------------------------------------------------------
    for col in ("open_time", "close_time"):
        if not pd.api.types.is_integer_dtype(df[col]):
            errors.append(
                f"Timestamp '{col}' must use an integer dtype"
            )
            continue

        if df[col].isna().any():
            errors.append(
                f"Timestamp '{col}' contains null values"
            )

        if np.isinf(df[col].to_numpy(dtype=np.float64)).any():
            errors.append(
                f"Timestamp '{col}' contains Inf values"
            )

    # ------------------------------------------------------------------
    # 3. OHLCV type, finiteness, positivity
    # ------------------------------------------------------------------
    for col in OHLCV_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"OHLCV column '{col}' is not numeric"
            )
            continue

        values = df[col].to_numpy(dtype=np.float64, copy=False)

        if not np.isfinite(values).all():
            errors.append(
                f"OHLCV column '{col}' contains NaN or Inf values"
            )

        if col == "volume":
            if (values < 0).any():
                errors.append(
                    "Volume contains negative values"
                )
        else:
            if (values <= 0).any():
                errors.append(
                    f"OHLCV column '{col}' contains zero or negative prices"
                )

    if errors:
        return {
            "status": "DATA_CORRUPTION",
            "rows": len(df),
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # 4. Duplicate open timestamps
    # ------------------------------------------------------------------
    dup_count = int(df["open_time"].duplicated().sum())

    if dup_count > 0:
        errors.append(
            f"{dup_count} duplicate open_time timestamps detected"
        )

    # ------------------------------------------------------------------
    # 5. Chronology
    # ------------------------------------------------------------------
    if not df["open_time"].is_monotonic_increasing:
        errors.append(
            "open_time is not monotonically increasing"
        )

    # ------------------------------------------------------------------
    # 6. Exact inclusive candle duration
    #
    # Binance-style inclusive milliseconds:
    # close_time - open_time == interval_ms - 1
    # ------------------------------------------------------------------
    durations = df["close_time"] - df["open_time"]
    invalid_durations = durations != expected_duration_ms

    if invalid_durations.any():
        errors.append(
            f"{int(invalid_durations.sum())} rows fail inclusive "
            f"duration check (expected {expected_duration_ms}ms)"
        )

    # ------------------------------------------------------------------
    # 7. Candle chronology/cadence diagnostics
    #
    # Differences are classified as:
    #   > cadence  -> missing bars
    #   0 < diff < cadence -> sub-cadence overlap/duplicate spacing
    #   <= 0 -> chronology corruption
    # ------------------------------------------------------------------
    diffs = df["open_time"].diff().dropna()
    anomalies = diffs[diffs != cadence_ms]

    if not anomalies.empty:
        missing_candles = anomalies[anomalies > cadence_ms]
        overlaps = anomalies[
            (anomalies > 0) & (anomalies < cadence_ms)
        ]
        inversions = anomalies[anomalies <= 0]

        if not missing_candles.empty:
            max_gap = missing_candles.max()

            # For a gap of N * cadence, N-1 bars are missing.
            missing_bars = int(
                (missing_candles // cadence_ms - 1).sum()
            )

            errors.append(
                f"CADENCE_GAP: {len(missing_candles)} gap events "
                f"({missing_bars} missing bars total, "
                f"largest gap: {max_gap / 3_600_000:.2f}h)"
            )

        if not overlaps.empty:
            errors.append(
                f"TIMESTAMP_OVERLAP: {len(overlaps)} events "
                f"with sub-cadence spacing"
            )

        if not inversions.empty:
            errors.append(
                f"CHRONOLOGY_CORRUPTION: {len(inversions)} "
                f"non-positive timestamp progressions"
            )

    # ------------------------------------------------------------------
    # 8. Zero-repair physical/structural validation
    #
    # The sanitizer is diagnostic only here. We do NOT replace df with
    # cleaned rows. Any rejection means the archive fails.
    # ------------------------------------------------------------------
    clean_rows = sanitize_closed_candles(
        df,
        interval_ms_value=cadence_ms,
    )

    if len(clean_rows) != len(df):
        dropped = len(df) - len(clean_rows)
        errors.append(
            f"ZERO_REPAIR_VIOLATION: sanitizer rejected {dropped} "
            f"rows; archive must require zero correction/rejection"
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(df),
        "errors": errors,
    }


def run_cadence_audit() -> None:
    print("═" * 72)
    print(" CANONICAL DATA INTEGRITY GATE: PARQUET SCHEMA & CADENCE AUDIT ")
    print("═" * 72)
    print()

    if not HISTORICAL_DIR.exists():
        print(f"❌ Missing data directory: {HISTORICAL_DIR}")
        print(
            "   Populate "
            "data/historical/{SYMBOL}_{15m,1h,4h}.parquet "
            "before running the audit."
        )
        sys.exit(1)

    failures = 0
    total_files = 0

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            total_files += 1

            result = audit_parquet_file(symbol, interval)

            status_icon = (
                "✓" if result["status"] == "PASS" else "❌"
            )

            print(
                f"[{status_icon}] "
                f"{symbol:<10} | "
                f"{interval:<3} | "
                f"Rows: {result.get('rows', 0):>8,} | "
                f"Status: {result['status']}"
            )

            if result["errors"]:
                failures += 1

                for error in result["errors"]:
                    print(f"    └── {error}")

    print("-" * 72)
    print(
        f"Audited: {total_files} files | "
        f"Failed: {failures}"
    )

    if failures > 0:
        print()
        print(
            "🚨 AUDIT FAILED: "
            "Correct/rebuild the affected Parquet datasets "
            "before model retraining."
        )
        sys.exit(1)

    print()
    print(
        "✅ DATA INTEGRITY VERIFIED: "
        "All historical files conform to schema, timestamp, "
        "cadence and candle-physics invariants."
    )


if __name__ == "__main__":
    run_cadence_audit()
