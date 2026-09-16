#!/usr/bin/env python3
"""
download_funding.py — Standalone Binance Futures Funding Downloader

Pulls funding rate history for all target symbols directly into
data/historical/funding/.

Fails FATALLY if Binance blocks the request or if variance is zero.

Important:
- Uses a 24-hour lookback before the first candle.
- This ensures the earliest candle can receive a prior
  point-in-time funding observation.
- Does NOT fabricate or zero-fill missing funding values.
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
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "NEARUSDT",
        "LTCUSDT",
        "UNIUSDT",
        "BCHUSDT",
        "DOTUSDT",
        "ALGOUSDT",
        "ENAUSDT",
        "DOGEUSDT",
        "TRUMPUSDT",
        "PUMPUSDT",
        "AAVEUSDT",
        "LINKUSDT",
        "SUIUSDT",
        "AVAXUSDT",
        "ADAUSDT",
        "TRXUSDT",
        "XLMUSDT",
        "ZECUSDT",
        "HBARUSDT",
        "CRVUSDT",
        "FILUSDT",
    ]


BASE_DIR = Path(__file__).resolve().parent

FUNDING_DIR = (
    BASE_DIR
    / "data"
    / "historical"
    / "funding"
)

HISTORICAL_DIR = (
    BASE_DIR
    / "data"
    / "historical"
)

BINANCE_FUNDING_URL = (
    "https://fapi.binance.com/fapi/v1/fundingRate"
)

REQUEST_TIMEOUT = 15

BATCH_SLEEP_SECONDS = 0.08

# Safety lookback for point-in-time funding coverage.
# This does not create data; it simply asks Binance for
# funding observations before the first candle.
FUNDING_LOOKBACK_MS = (
    24 * 60 * 60 * 1000
)


def fetch_binance_funding_strict(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Download Binance Futures funding history strictly.

    Fails closed on:
    - HTTP errors
    - completely empty responses
    - zero funding records
    - zero funding variance
    """

    records = []

    current_start = start_ms

    while current_start <= end_ms:

        params = {
            "symbol": symbol,
            "startTime": current_start,
            "limit": 1000,
        }

        resp = session.get(
            BINANCE_FUNDING_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            raise ValueError(
                f"FATAL: Binance Futures API returned "
                f"HTTP {resp.status_code} for {symbol}."
            )

        data = resp.json()

        if not data or not isinstance(data, list):
            break

        for item in data:

            ft = int(item["fundingTime"])

            if ft <= end_ms:
                records.append(
                    {
                        "funding_time": ft,
                        "fundingRate": float(
                            item["fundingRate"]
                        ),
                    }
                )

        if len(data) < 1000:
            break

        last_funding_time = int(
            data[-1]["fundingTime"]
        )

        # Prevent accidental infinite loops.
        next_start = last_funding_time + 1

        if next_start <= current_start:
            raise ValueError(
                f"FATAL: Funding pagination did not advance "
                f"for {symbol}."
            )

        current_start = next_start

        time.sleep(
            BATCH_SLEEP_SECONDS
        )

    if not records:
        raise ValueError(
            f"FATAL: Zero funding records retrieved "
            f"for {symbol}."
        )

    df = (
        pd.DataFrame(records)
        .drop_duplicates(
            subset=["funding_time"]
        )
        .sort_values(
            "funding_time"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"FATAL: Funding dataframe is empty "
            f"after processing {symbol}."
        )

    funding_values = pd.to_numeric(
        df["fundingRate"],
        errors="coerce",
    )

    if funding_values.isna().any():
        raise ValueError(
            f"FATAL: Non-numeric/NaN funding rate "
            f"detected for {symbol}."
        )

    if not (
        funding_values
        .replace([float("inf"), float("-inf")], pd.NA)
        .notna()
        .all()
    ):
        raise ValueError(
            f"FATAL: Non-finite funding rate "
            f"detected for {symbol}."
        )

    if funding_values.nunique() < 2:
        raise ValueError(
            f"FATAL: Fewer than 2 unique funding values "
            f"for {symbol}."
        )

    if funding_values.std() == 0:
        raise ValueError(
            f"FATAL: Zero variance in funding rate "
            f"for {symbol}."
        )

    return df


def main():
    FUNDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),
            "Accept": "application/json",
        }
    )

    symbols = [
        s
        for s in SYMBOLS
        if s != "BTCUSDT"
    ]

    total = len(symbols)

    print(
        f"Fetching funding rates for {total} symbols "
        f"into {FUNDING_DIR}...\n"
    )

    for idx, symbol in enumerate(
        symbols,
        1,
    ):

        # -------------------------------------------------------------
        # Load existing canonical 15m candle data.
        # -------------------------------------------------------------

        candle_file = (
            HISTORICAL_DIR
            / f"{symbol}_15m.parquet"
        )

        if not candle_file.exists():

            print(
                f"[{idx:>2}/{total}] "
                f"{symbol:<10} ... "
                f"✗ Missing {candle_file.name}"
            )

            continue

        try:
            candle_df = pd.read_parquet(
                candle_file
            )
        except Exception as exc:
            print(
                f"\n❌ FATAL: Could not read "
                f"{candle_file}: {exc}"
            )
            sys.exit(1)

        required_columns = {
            "open_time",
            "close_time",
        }

        missing_columns = (
            required_columns
            - set(candle_df.columns)
        )

        if missing_columns:

            print(
                f"\n❌ FATAL: {symbol} candle file "
                f"is missing required columns: "
                f"{sorted(missing_columns)}"
            )

            sys.exit(1)

        if candle_df.empty:

            print(
                f"\n❌ FATAL: {symbol} 15m candle "
                f"file is empty."
            )

            sys.exit(1)

        try:
            first_open = int(
                candle_df["open_time"].min()
            )

            last_close = int(
                candle_df["close_time"].max()
            )

        except Exception as exc:

            print(
                f"\n❌ FATAL: Invalid candle timestamps "
                f"for {symbol}: {exc}"
            )

            sys.exit(1)

        if last_close <= first_open:

            print(
                f"\n❌ FATAL: Invalid candle time range "
                f"for {symbol}: "
                f"start={first_open}, "
                f"end={last_close}"
            )

            sys.exit(1)

        # -------------------------------------------------------------
        # IMPORTANT FIX
        #
        # Start funding download 24 hours BEFORE the first
        # canonical candle.
        #
        # This gives merge_asof(direction="backward") a prior
        # funding observation for the earliest candles.
        # -------------------------------------------------------------

        start_ms = max(
            0,
            first_open
            - FUNDING_LOOKBACK_MS,
        )

        end_ms = last_close

        out_file = (
            FUNDING_DIR
            / f"{symbol}_funding.parquet"
        )

        print(
            f"[{idx:>2}/{total}] "
            f"{symbol:<10} ... ",
            end="",
            flush=True,
        )

        try:

            funding_df = (
                fetch_binance_funding_strict(
                    session=session,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

            funding_df.to_parquet(
                out_file,
                index=False,
            )

            fr_std = funding_df[
                "fundingRate"
            ].std()

            print(
                f"✓ {len(funding_df):>4} events "
                f"(σ={fr_std:.6f})"
            )

        except ValueError as exc:

            print(
                f"\n❌ {exc}"
            )

            sys.exit(1)

        except Exception as exc:

            print(
                f"\n❌ FATAL: Unexpected error "
                f"for {symbol}: {exc}"
            )

            sys.exit(1)

    print(
        "\n✅ All funding parquet files "
        "successfully generated."
    )


if __name__ == "__main__":
    main()
