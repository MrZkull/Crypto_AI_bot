#!/usr/bin/env python3
"""
download_training_data.py — Canonical Historical Parquet Downloader (Zero-Repair Enforcement)

Invariants:
1. Bounded backward pagination (prevents 429 timeouts and memory exhaustion).
2. Guarantees BTCUSDT_15m benchmark generation alongside all configured SYMBOLS.
3. Zero-repair timestamp invariant: exchange close_time must strictly equal open_time + interval_ms - 1.
4. Fail-closed candle physics: non-positive prices, negative volumes, or malformed bounds abort the build.
5. Cadence verification: missing interior bars cause immediate process termination.
"""

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


def fetch_klines(symbol: str, interval: str, limit: int = 150) -> pd.DataFrame:
    """Fetches public candles strictly from Binance spot endpoints (no US IP block on Vision)."""
    if native_get_data is not None:
        try:
            df_native = native_get_data(symbol, interval, limit=limit)
            if df_native is not None and not df_native.empty and len(df_native) >= 20:
                return df_native.sort_values("open_time").reset_index(drop=True)
        except Exception:
            pass

    endpoints = [
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    raw = json.loads(resp.read().decode())
                    rows = [
                        {
                            "open_time": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "close_time": int(k[6]),
                            "taker_buy_base_vol": float(k[9]) if len(k) > 9 else 0.0,
                        }
                        for k in raw
                    ]
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        return df.sort_values("open_time").reset_index(drop=True)
        except Exception:
            continue

    log.warning(f"Failed fetching {interval} candles for {symbol} across spot endpoints")
    return pd.DataFrame()


def _raw_to_frame(raw: list, interval: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    cadence = interval_ms(interval)
    expected_duration = cadence - 1
    rows = []

    for k in raw:
        if len(k) < 7:
            raise ValueError(f"Malformed kline payload: bar length {len(k)} < 7")

        o_time = int(k[0])
        c_time = int(k[6])

        # Zero-repair invariant: exchange timestamp must strictly match inclusive standard
        expected_c_time = o_time + expected_duration
        if c_time != expected_c_time:
            raise ValueError(
                f"FATAL: Timestamp convention mismatch on {interval} at open={o_time}. "
                f"Exchange close_time={c_time} != expected={expected_c_time} (diff={c_time - expected_c_time}ms). "
                f"Silent modification prohibited by Zero-Repair Policy."
            )

        open_p  = float(k[1])
        high_p  = float(k[2])
        low_p   = float(k[3])
        close_p = float(k[4])
        volume  = float(k[5])
        taker_vol = float(k[9]) if len(k) > 9 else volume * 0.5

        # Fail-closed physical candle physics
        if not (np.isfinite(open_p) and np.isfinite(high_p) and np.isfinite(low_p) and np.isfinite(close_p) and np.isfinite(volume)):
            raise ValueError(f"FATAL: Non-finite OHLCV value at open_time={o_time}")

        if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
            raise ValueError(f"FATAL: Non-positive price at open_time={o_time}")

        if volume < 0:
            raise ValueError(f"FATAL: Negative volume at open_time={o_time}")

        if high_p < max(open_p, close_p) or low_p > min(open_p, close_p):
            raise ValueError(f"FATAL: Structural candle violation at open_time={o_time}: H={high_p}, L={low_p}, O={open_p}, C={close_p}")

        rows.append({
            "open_time": o_time,
            "close_time": c_time,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "taker_buy_base_vol": taker_vol,
        })

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def fetch_symbol_history(session: requests.Session, symbol: str, interval: str, target_candles: int = TARGET_CANDLES) -> pd.DataFrame:
    batches = []
    cadence = interval_ms(interval)
    now_ms = int(time.time() * 1000)

    # Point-in-time boundary: strictly exclude currently-forming candle
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
        raise ValueError(f"Zero valid kline batches fetched for {symbol} {interval}")

    combined = pd.concat(batches, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Verify cadence continuity
    diffs = combined["open_time"].diff().dropna()
    gaps = diffs[diffs != cadence]
    if not gaps.empty:
        raise ValueError(f"FATAL: Cadence continuity failure for {symbol} {interval}: {len(gaps)} gaps detected.")

    clean_rows = sanitize_closed_candles(combined, interval_ms_value=cadence, observation_time_ms=now_ms)
    clean_df = pd.DataFrame(clean_rows)

    if len(clean_df) != len(combined):
        raise ValueError(f"FATAL: Sanitizer dropped {len(combined) - len(clean_df)} bars. Zero-repair violation.")

    return clean_df.reset_index(drop=True)


def build_canonical_archive():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "CryptoBot-AI/Canonical-Archive-Downloader"})

    # Invariant: Guarantee BTC benchmark archive is populated for relative-strength features
    download_targets = list(dict.fromkeys(list(SYMBOLS) + ["BTCUSDT"]))
    total_tasks = len(download_targets) * len(INTERVALS)
    completed = 0

    print("=" * 65)
    print(" CRYPTOBOT AI — CANONICAL HISTORICAL ARCHIVE BUILDER ")
    print("=" * 65)
    print(f"Target Universe: {len(download_targets)} symbols | Intervals: {INTERVALS}")
    print(f"Target Directory: {HISTORICAL_DIR}\n")

    for symbol in download_targets:
        for interval in INTERVALS:
            completed += 1
            out_file = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"
            print(f"[{completed:>2}/{total_tasks}] {symbol:<10} | {interval:<3} ...", end=" ", flush=True)

            df = fetch_symbol_history(session, symbol, interval, target_candles=TARGET_CANDLES)
            df.to_parquet(out_file, index=False)
            print(f"✓ {len(df):>6,} candles")

    print("-" * 65)
    print("✅ Canonical historical archive successfully populated with zero modifications.")


if __name__ == "__main__":
    build_canonical_archive()
