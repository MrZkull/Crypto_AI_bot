#!/usr/bin/env python3
"""
download_training_data.py — Canonical Historical Archive Builder.

Builds:
    data/historical/{SYMBOL}_{15m,1h,4h}.parquet

The archive is the canonical research input for train_model.py.

Properties:
- 15m / 1h / 4h only
- Binance closed-candle data
- exact inclusive millisecond timestamps
- zero-repair validation
- duplicate/cadence validation
- bounded history per symbol/timeframe
- atomic file replacement
"""

from __future__ import annotations

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

# Keep the research build bounded so GitHub Actions remains practical.
# These are candles PER SYMBOL / TIMEFRAME.
CANDLES_PER_INTERVAL = {
    "15m": 15_000,
    "1h": 10_000,
    "4h": 5_000,
}

# BTC 15m is required separately by train_model.py for BTC context features.
BTC_BENCHMARK_CANDLES = CANDLES_PER_INTERVAL["15m"]

REQUEST_LIMIT = 1000
REQUEST_TIMEOUT = 20
RETRIES = 3
RETRY_SLEEP = 2.0
REQUEST_SLEEP = 0.15

BINANCE_ENDPOINTS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)

REQUIRED_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def fetch_batch(
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

    for attempt in range(1, RETRIES + 1):
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
                        f"Unexpected Binance response type: "
                        f"{type(payload).__name__}"
                    )

                return payload

            except Exception as exc:
                last_error = exc

        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(
        f"Failed fetching {symbol} {interval}: {last_error}"
    )


def raw_to_frame(raw: list, interval: str) -> pd.DataFrame:
    rows = []

    expected_duration = interval_ms(interval) - 1

    for candle in raw:
        if len(candle) < 7:
            continue

        try:
            open_time = int(candle[0])
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])

            # Binance's close_time is retained only indirectly.
            # The archive uses our canonical inclusive-ms invariant.
            close_time = open_time + expected_duration

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


def fetch_history(
    session: requests.Session,
    symbol: str,
    interval: str,
    max_candles: int,
) -> pd.DataFrame:
    batches = []

    # Current UTC time. Binance will not return future candles.
    end_time_ms = int(time.time() * 1000)

    collected = 0

    while collected < max_candles:
        raw = fetch_batch(
            session,
            symbol,
            interval,
            end_time_ms,
        )

        if not raw:
            break

        frame = raw_to_frame(raw, interval)

        if frame.empty:
            raise RuntimeError(
                f"{symbol} {interval}: received only malformed candles"
            )

        batches.append(frame)
        collected += len(frame)

        oldest = int(frame["open_time"].min())

        end_time_ms = oldest - 1

        print(
            f"      batch={len(batches):>2} "
            f"candles={len(frame):>4} "
            f"collected={collected:>6}/{max_candles}"
        )

        if len(raw) < REQUEST_LIMIT:
            break

        time.sleep(REQUEST_SLEEP)

    if not batches:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.concat(
        batches,
        ignore_index=True,
    )

    df = (
        df.drop_duplicates("open_time", keep="first")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # Keep exactly the requested number of newest candles.
    if len(df) > max_candles:
        df = df.iloc[-max_candles:].reset_index(drop=True)

    return df


def validate_archive_frame(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
) -> None:
    if df.empty:
        raise ValueError(
            f"{symbol} {interval}: empty historical dataset"
        )

    cadence = interval_ms(interval)
    expected_duration = cadence - 1

    if not pd.api.types.is_integer_dtype(df["open_time"]):
        raise ValueError(
            f"{symbol} {interval}: open_time must be integer dtype"
        )

    if not pd.api.types.is_integer_dtype(df["close_time"]):
        raise ValueError(
            f"{symbol} {interval}: close_time must be integer dtype"
        )

    if df["open_time"].duplicated().any():
        raise ValueError(
            f"{symbol} {interval}: duplicate open_time values"
        )

    if not df["open_time"].is_monotonic_increasing:
        raise ValueError(
            f"{symbol} {interval}: open_time is not monotonic"
        )

    durations = df["close_time"] - df["open_time"]

    if not (durations == expected_duration).all():
        raise ValueError(
            f"{symbol} {interval}: invalid inclusive candle duration"
        )

    diffs = df["open_time"].diff().dropna()

    if not diffs.empty and not (diffs == cadence).all():
        bad = int((diffs != cadence).sum())
        raise ValueError(
            f"{symbol} {interval}: {bad} cadence anomalies detected"
        )

    numeric = df[
        ["open", "high", "low", "close", "volume"]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{symbol} {interval}: non-finite OHLCV detected"
        )

    if (
        (df["open"] <= 0).any()
        or (df["high"] <= 0).any()
        or (df["low"] <= 0).any()
        or (df["close"] <= 0).any()
    ):
        raise ValueError(
            f"{symbol} {interval}: non-positive price detected"
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            f"{symbol} {interval}: negative volume detected"
        )

    if (
        df["high"] < df[["open", "close"]].max(axis=1)
    ).any():
        raise ValueError(
            f"{symbol} {interval}: high violates OHLC physics"
        )

    if (
        df["low"] > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError(
            f"{symbol} {interval}: low violates OHLC physics"
        )


def build_one(
    session: requests.Session,
    symbol: str,
    interval: str,
    max_candles: int,
) -> Path:
    print(
        f"    Downloading {symbol} {interval} "
        f"({max_candles:,} candles)..."
    )

    raw_df = fetch_history(
        session,
        symbol,
        interval,
        max_candles,
    )

    if raw_df.empty:
        raise RuntimeError(
            f"{symbol} {interval}: no data returned"
        )

    observation_time_ms = int(time.time() * 1000)

    clean_rows = sanitize_closed_candles(
        raw_df,
        interval_ms_value=interval_ms(interval),
        observation_time_ms=observation_time_ms,
    )

    clean_df = pd.DataFrame(
        clean_rows,
        columns=REQUIRED_COLUMNS,
    )

    # Zero-repair rule:
    # the downloader must never silently discard invalid archive rows.
    if len(clean_df) != len(raw_df):
        rejected = len(raw_df) - len(clean_df)
        raise RuntimeError(
            f"{symbol} {interval}: sanitizer rejected "
            f"{rejected} rows; refusing to write archive"
        )

    validate_archive_frame(
        clean_df,
        symbol,
        interval,
    )

    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = HISTORICAL_DIR / f"{symbol}_{interval}.parquet"
    temp = target.with_suffix(".parquet.tmp")

    clean_df.to_parquet(
        temp,
        index=False,
    )

    # Final integrity readback before replacement.
    verify = pd.read_parquet(temp)

    if len(verify) != len(clean_df):
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{symbol} {interval}: Parquet readback row count mismatch"
        )

    validate_archive_frame(
        verify,
        symbol,
        interval,
    )

    temp.replace(target)

    print(
        f"      ✓ {target.relative_to(BASE_DIR)} "
        f"({len(clean_df):,} rows)"
    )

    return target


def main() -> None:
    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "CryptoBot-AI/Canonical-Archive",
            "Accept": "application/json",
        }
    )

    symbols = list(dict.fromkeys(SYMBOLS))

    # BTC benchmark is required even when BTCUSDT is not in SYMBOLS.
    if "BTCUSDT" not in symbols:
        benchmark_targets = [("BTCUSDT", "15m")]
    else:
        benchmark_targets = []

    targets = [
        (symbol, interval)
        for symbol in symbols
        for interval in INTERVALS
    ]

    targets.extend(benchmark_targets)

    print("=" * 72)
    print(" CRYPTOBOT AI — CANONICAL HISTORICAL ARCHIVE BUILDER")
    print("=" * 72)
    print(f"Symbols: {len(symbols)}")
    print("Intervals: 15m, 1h, 4h")
    print(f"Output: {HISTORICAL_DIR}")
    print(f"Files to build: {len(targets)}")
    print()

    completed = 0

    for symbol, interval in targets:
        completed += 1

        print(
            f"[{completed}/{len(targets)}] "
            f"{symbol} {interval}"
        )

        if symbol == "BTCUSDT" and interval == "15m":
            count = BTC_BENCHMARK_CANDLES
        else:
            count = CANDLES_PER_INTERVAL[interval]

        build_one(
            session,
            symbol,
            interval,
            count,
        )

        time.sleep(0.25)

    print()
    print("=" * 72)
    print("✅ CANONICAL HISTORICAL ARCHIVE BUILD COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
