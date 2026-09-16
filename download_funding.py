#!/usr/bin/env python3
"""
download_funding.py — Standalone Binance Futures Funding Downloader
Pulls funding rate history for all target symbols directly into data/historical/funding/.
Fails FATALLY if Binance blocks the request or if variance is zero.
"""

import sys
import time
from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent
FUNDING_DIR = BASE_DIR / "data" / "historical" / "funding"
HISTORICAL_DIR = BASE_DIR / "data" / "historical"

BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
REQUEST_TIMEOUT = 15
BATCH_SLEEP_SECONDS = 0.08


def fetch_binance_funding_strict(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    records = []
    current_start = start_ms

    while current_start <= end_ms:
        params = {"symbol": symbol, "startTime": current_start, "limit": 1000}
        resp = session.get(BINANCE_FUNDING_URL, params=params, timeout=REQUEST_TIMEOUT)

        if resp.status_code != 200:
            raise ValueError(f"FATAL: Binance Futures API returned HTTP {resp.status_code} for {symbol}.")

        data = resp.json()
        if not data or not isinstance(data, list):
            break

        for item in data:
            ft = int(item["fundingTime"])
            if ft <= end_ms:
                records.append({
                    "funding_time": ft,
                    "fundingRate": float(item["fundingRate"])
                })

        if len(data) < 1000:
            break

        current_start = int(data[-1]["fundingTime"]) + 1
        time.sleep(BATCH_SLEEP_SECONDS)

    if not records:
        raise ValueError(f"FATAL: Zero funding records retrieved for {symbol}.")

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["funding_time"])
        .sort_values("funding_time")
    )

    if df["fundingRate"].std() == 0:
        raise ValueError(f"FATAL: Zero variance in funding rate for {symbol}.")

    return df.reset_index(drop=True)


def main():
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    })

    symbols = [s for s in SYMBOLS if s != "BTCUSDT"]
    total = len(symbols)

    print(
        f"Fetching funding rates for {total} symbols "
        f"into {FUNDING_DIR}...\n"
    )

    for idx, symbol in enumerate(symbols, 1):

        # Match time range of existing 15m candle data
        candle_file = HISTORICAL_DIR / f"{symbol}_15m.parquet"

        if not candle_file.exists():
            print(
                f"[{idx:>2}/{total}] {symbol:<10} ... "
                f"✗ Missing {candle_file.name}"
            )
            continue

        candle_df = pd.read_parquet(candle_file)

        # Fetch funding history 24 hours before the first candle
        # so the earliest candle has a prior PIT funding observation.
        FUNDING_LOOKBACK_MS = 24 * 60 * 60 * 1000

        start_ms = max(
            0,
            int(candle_df["open_time"].min()) - FUNDING_LOOKBACK_MS,
        )

        end_ms = int(candle_df["close_time"].max())

        out_file = FUNDING_DIR / f"{symbol}_funding.parquet"

        print(
            f"[{idx:>2}/{total}] {symbol:<10} ...",
            end=" ",
            flush=True
        )

        try:
            funding_df = fetch_binance_funding_strict(
                session,
                symbol,
                start_ms,
                end_ms
            )

            funding_df.to_parquet(
                out_file,
                index=False
            )

            fr_std = funding_df["fundingRate"].std()

            print(
                f"✓ {len(funding_df):>4} events "
                f"(σ={fr_std:.6f})"
            )

        except ValueError as e:
            print(f"\n❌ {e}")
            sys.exit(1)

    print(
        "\n✅ All funding parquet files successfully generated."
    )


if __name__ == "__main__":
    main()
