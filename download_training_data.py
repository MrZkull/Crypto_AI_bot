#!/usr/bin/env python3
"""
download_training_data.py — Canonical historical market-data downloader.

Purpose
-------
Populate the canonical research archive used by train_model.py:

    data/historical/{SYMBOL}_{15m,1h,4h}.parquet

The downloader:
1. Fetches Binance klines in bounded batches.
2. Keeps only the canonical 15m / 1h / 4h intervals.
3. Preserves exact inclusive millisecond timestamps:
       close_time = open_time + interval_ms - 1
4. Validates OHLCV structure and completed-candle status.
5. Rejects duplicate/corrupt rows instead of repairing them.
6. Requires continuous expected candle cadence.
7. Writes one immutable research dataset per symbol/timeframe.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import SYMBOLS
from market_data_integrity import interval_ms, sanitize_closed_candles


BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"

INTERVALS = ("15m", "1h", "4h")

BINANCE_ENDPOINTS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)

REQUEST_LIMIT = 1000
REQUEST_TIMEOUT = 20
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 2.0
BATCH_SLEEP_SECONDS = 0.25

REQUIRED_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def _fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    end_time_ms: int | None,
) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": REQUEST_LIMIT,
    }

    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    last_error = None

    for attempt in range(RETRY_COUNT):
        for endpoint in BINANCE_ENDPOINTS:
            try:
                response = session.get(
                    endpoint,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, list):
                    raise ValueError(
                        f"Unexpected Binance response type: {type(payload).__name__}"
                    )

                return payload

            except Exception as exc:
                last_error = exc

        if attempt < RETRY_COUNT - 1:
            time.sleep(RETRY_SLEEP_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"Binance kline download failed for {symbol} {interval}: {last_error}"
    )


def _raw_to_frame(raw: list, interval: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rows = []

    for kline in raw:
        if len(kline) < 7:
            continue

        try:
            open_time = int(kline[0])
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])

            # Binance kline close_time is authoritative for the raw exchange
            # response, but the canonical archive enforces the exact inclusive
            # millisecond convention, so we calculate it from the interval.
            close_time = open_time + interval_ms(interval) - 1

            rows.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
        except (TypeError, ValueError, OverflowError):
            continue

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _validate_archive_frame(df: pd.DataFrame, interval: str) -> None:
    if df.empty:
        raise ValueError(f"{interval}: no valid candles after sanitization")

    cadence = interval_ms(interval)
    expected_duration = cadence - 1

    if not pd.api.types.is_integer_dtype(df["open_time"]):
        raise ValueError(f"{interval}: open_time is not integer dtype")

    if not pd.api.types.is_integer_dtype(df["close_time"]):
        raise ValueError(f"{interval}: close_time is not integer dtype")

    if df["open_time"].duplicated().any():
        raise ValueError(f"{interval}: duplicate open_time timestamps found")

    if not df["open_time"].is_monotonic_increasing:
        raise ValueError(f"{interval}: open_time is not monotonically increasing")

    durations = df["close_time"] - df["open_time"]
    if not (durations == expected_duration).all():
        raise ValueError(
            f"{interval}: inclusive duration violation; "
            f"expected {expected_duration}ms"
        )

    diffs = df["open_time"].diff().dropna()
    if not (diffs == cadence).all():
        anomalies = diffs[diffs != cadence]
        raise ValueError(
            f"{interval}: cadence violation; "
            f"{len(anomalies)} anomalous spacings detected"
        )

    numeric = df[
        ["open", "high", "low", "close", "volume"]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric).all():
        raise ValueError(f"{interval}: non-finite OHLCV values found")

    if (
        (df["open"] <= 0).any()
        or (df["high"] <= 0).any()
        or (df["low"] <= 0).any()
        or (df["close"] <= 0).any()
    ):
        raise ValueError(f"{interval}: non-positive price found")

    if (df["volume"] < 0).any():
        raise ValueError(f"{interval}: negative volume found")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{interval}: high below max(open, close)")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{interval}: low above min(open, close)")


def fetch_full_history(
    session: requests.Session,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    """
    Walk backward through Binance history until no older candles remain.

    The query boundary is moved to one millisecond before the oldest candle
    returned by the previous request, avoiding repeated boundary rows.
    """
    batches = []
    end_time_ms = int(time.time() * 1000)

    while True:
        raw = _fetch_klines(
            session=session,
            symbol=symbol,
            interval=interval,
            end_time_ms=end_time_ms,
        )

        if not raw:
            break

        frame = _raw_to_frame(raw, interval)

        if frame.empty:
            raise ValueError(
                f"{symbol} {interval}: Binance returned only malformed rows"
            )

        batches.append(frame)

        oldest_open = int(frame["open_time"].min())

        print(
            f"      batch={len(batches):>3} "
            f"rows={len(frame):>4} "
            f"oldest={pd.to_datetime(oldest_open, unit='ms', utc=True)}"
        )

        # Move strictly before the oldest returned candle.
        next_end = oldest_open - 1

        if next_end >= end_time_ms:
            raise RuntimeError(
                f"{symbol} {interval}: pagination did not move backward"
            )

        end_time_ms = next_end

        # Once the API returns fewer than the request limit, this is the
        # earliest available page for practical purposes.
        if len(raw) < REQUEST_LIMIT:
            break

        time.sleep(BATCH_SLEEP_SECONDS)

    if not batches:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    combined = pd.concat(batches, ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["open_time"],
        keep="first",
    )

    combined = combined.sort_values("open_time").reset_index(drop=True)

    return combined


def build_canonical_archive() -> None:
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "CryptoBot-AI/Canonical-Archive-Downloader",
            "Accept": "application/json",
        }
    )

    total = len(SYMBOLS) * len(INTERVALS)
    completed = 0

    print("=" * 72)
    print(" CRYPTOBOT AI — CANONICAL HISTORICAL ARCHIVE BUILDER ")
    print("=" * 72)
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print(f"Target files: {total}")
    print(f"Output: {HISTORICAL_DIR}")
    print()

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            completed += 1
            print(
                f"[{completed}/{total}] "
                f"Downloading {symbol} {interval}"
            )

            try:
                raw_df = fetch_full_history(
                    session=session,
                    symbol=symbol,
                    interval=interval,
                )

                if raw_df.empty:
                    raise ValueError("No historical candles returned")

                # Reject anything not fully available at observation time.
                observation_ms = int(time.time() * 1000)

                clean_rows = sanitize_closed_candles(
                    raw_df,
                    interval_ms_value=interval_ms(interval),
                    observation_time_ms=observation_ms,
                )

                clean_df = pd.DataFrame(clean_rows, columns=REQUIRED_COLUMNS)

                # Zero-repair policy: downloader must never silently remove
                # historical rows and still call the file complete.
                if len(clean_df) != len(raw_df):
                    rejected = len(raw_df) - len(clean_df)
                    raise ValueError(
                        f"Sanitizer rejected {rejected} rows; "
                        f"refusing to create canonical archive"
                    )

                _validate_archive_frame(clean_df, interval)

                output = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"

                clean_df.to_parquet(
                    output,
                    index=False,
                )

                print(
                    f"    ✓ Wrote {len(clean_df):,} candles -> "
                    f"{output.relative_to(BASE_DIR)}"
                )

            except Exception as exc:
                print(
                    f"    ❌ FAILED {symbol} {interval}: {exc}"
                )
                raise

    print()
    print("=" * 72)
    print("✅ CANONICAL HISTORICAL ARCHIVE BUILD COMPLETE")
    print("=" * 72)


def main() -> None:
    build_canonical_archive()


if __name__ == "__main__":
    main()
