#!/usr/bin/env python3
"""
download_training_data.py — Canonical Historical Parquet Downloader

Invariants:
1. Bounded backward pagination (prevents infinite loop 429 timeouts in CI)
2. Guarantees BTCUSDT_15m benchmark generation
3. Enforces canonical inclusive timestamps: close_time = open_time + interval_ms - 1
4. Strict OHLCV candle physics validation
5. Outputs to data/historical/{SYMBOL}_{15m,1h,4h}.parquet
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

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

BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"
INTERVALS = ("15m", "1h", "4h")

BINANCE_ENDPOINTS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)

TARGET_CANDLES = 8000
REQUEST_LIMIT = 1000
REQUEST_TIMEOUT = 15
RETRY_COUNT = 3
BATCH_SLEEP_SECONDS = 0.08

REQUIRED_COLUMNS = [
    "open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy_base_vol"
]


def _fetch_klines_batch(session: requests.Session, symbol: str, interval: str, end_time_ms: int = None) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": REQUEST_LIMIT}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    for attempt in range(RETRY_COUNT):
        for endpoint in BINANCE_ENDPOINTS:
            try:
                r = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        time.sleep(1.0 * (attempt + 1))
    return []


def _raw_to_frame(raw: list, interval: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    cadence = interval_ms(interval)
    expected_duration = cadence - 1
    rows = []

    for k in raw:
        if len(k) < 7:
            continue
        try:
            o_time = int(k[0])
            c_time = int(k[6])
            
            # Canonical standard: verify or map to inclusive ms
            computed_c_time = o_time + expected_duration
            if abs(c_time - computed_c_time) > 1:
                c_time = computed_c_time

            taker_vol = float(k[9]) if len(k) > 9 else float(k[5]) * 0.5

            rows.append({
                "open_time": o_time,
                "close_time": c_time,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_base_vol": taker_vol,
            })
        except (ValueError, TypeError, IndexError):
            continue

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def fetch_symbol_history(session: requests.Session, symbol: str, interval: str, target_candles: int = TARGET_CANDLES) -> pd.DataFrame:
    batches = []
    cadence = interval_ms(interval)
    now_ms = int(time.time() * 1000)
    
    # Anchor to the start of current candle to exclude forming bars
    current_open = (now_ms // cadence) * cadence
    end_time_ms = current_open - 1
    total_fetched = 0

    while total_fetched < target_candles:
        raw = _fetch_klines_batch(session, symbol, interval, end_time_ms=end_time_ms)
        if not raw:
            break

        df_batch = _raw_to_frame(raw, interval)
        if df_batch.empty:
            break

        batches.append(df_batch)
        total_fetched += len(df_batch)
        oldest_open = int(df_batch["open_time"].min())

        next_end = oldest_open - 1
        if next_end >= end_time_ms or len(raw) < REQUEST_LIMIT:
            break
        end_time_ms = next_end
        time.sleep(BATCH_SLEEP_SECONDS)

    if not batches:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    combined = pd.concat(batches, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Sanitize closed bars
    clean_rows = sanitize_closed_candles(combined, interval_ms_value=cadence, observation_time_ms=now_ms)
    clean_df = pd.DataFrame(clean_rows)
    return clean_df.reset_index(drop=True)


def build_canonical_archive():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "CryptoBot-AI/Canonical-Archive-Downloader"})

    # Ensure benchmark BTC is downloaded
    download_targets = list(dict.fromkeys(list(SYMBOLS) + ["BTCUSDT"]))
    total_tasks = len(download_targets) * len(INTERVALS)
    completed = 0

    print("=" * 65)
    print(" CRYPTOBOT AI — CANONICAL HISTORICAL ARCHIVE BUILDER ")
    print("=" * 65)
    print(f"Target Universe: {len(download_targets)} symbols | Intervals: {INTERVALS}")
    print(f"Target Directory: {HISTORICAL_DIR}\n")

    failures = 0
    for symbol in download_targets:
        for interval in INTERVALS:
            completed += 1
            out_file = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"
            print(f"[{completed}/{total_tasks}] {symbol:<10} | {interval:<3} ...", end=" ", flush=True)

            try:
                df = fetch_symbol_history(session, symbol, interval, target_candles=TARGET_CANDLES)
                if df.empty or len(df) < 100:
                    print(f"❌ FAILED (Insufficient candles: {len(df)})")
                    failures += 1
                    continue

                df.to_parquet(out_file, index=False)
                print(f"✓ Wrote {len(df):>6,} candles")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                failures += 1

    print("-" * 65)
    if failures > 0:
        print(f"🚨 Archive creation completed with {failures} failures.")
        sys.exit(1)
    print("✅ Canonical historical archive successfully populated.")


if __name__ == "__main__":
    build_canonical_archive()
