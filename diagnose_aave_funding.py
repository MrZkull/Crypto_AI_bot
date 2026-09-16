#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

path = Path("data/historical/funding/AAVEUSDT_funding.parquet")

if not path.exists():
    raise FileNotFoundError(f"FATAL: File not found: {path}")

df = pd.read_parquet(path)

print("=" * 80)
print("AAVEUSDT FUNDING DIAGNOSTIC")
print("=" * 80)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

for col in df.columns:
    print(f"{col}: dtype={df[col].dtype}")

print()

funding_col = None

for candidate in ["fundingRate", "funding_rate"]:
    if candidate in df.columns:
        funding_col = candidate
        break

if funding_col is None:
    raise ValueError(
        "FATAL: No funding rate column found. "
        "Expected 'fundingRate' or 'funding_rate'."
    )

fr = pd.to_numeric(
    df[funding_col],
    errors="coerce"
)

print(f"Funding column: {funding_col}")
print(f"Non-null rows: {fr.notna().sum()}")
print(f"Null rows: {fr.isna().sum()}")
print(f"Unique values: {fr.nunique(dropna=True)}")

if fr.notna().any():
    print(f"Min: {fr.min()}")
    print(f"Max: {fr.max()}")
    print(f"Mean: {fr.mean()}")
    print(f"Std: {fr.std()}")
    print(f"Zero count: {(fr == 0).sum()}")

print()

for ts_col in ["funding_time", "timestamp", "time", "close_time"]:
    if ts_col in df.columns:
        ts = pd.to_numeric(
            df[ts_col],
            errors="coerce"
        )

        print(f"{ts_col}:")
        print(f"  Non-null: {ts.notna().sum()}")
        print(f"  Min: {ts.min()}")
        print(f"  Max: {ts.max()}")
        print()

print("First 10 rows:")
print(df.head(10).to_string())

print()
print("Last 10 rows:")
print(df.tail(10).to_string())

print("=" * 80)
