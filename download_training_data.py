# download_training_data.py - Downloads maximum training data

import requests
import pandas as pd
import time
from config import SYMBOLS, RAW_DATA_FILE

# More timeframes = better patterns learned
INTERVALS = ["3m", "5m", "15m", "1h"]

# Maximum allowed by Binance free API = 1000 candles per request
# We make multiple requests to get more history
LIMIT = 1000
REQUESTS_PER_SYMBOL = 3  # 3 x 1000 = 3000 candles per timeframe


def fetch_klines(symbol, interval, limit=1000, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    except Exception as e:
        print(f"  Error {symbol} {interval}: {e}")
        return pd.DataFrame()


def fetch_extended(symbol, interval):
    """Fetch multiple batches to get more history"""
    all_data = []
    end_time = None

    for batch in range(REQUESTS_PER_SYMBOL):
        df = fetch_klines(symbol, interval, limit=1000, end_time=end_time)
        if df.empty:
            break
        all_data.append(df)
        # Next batch ends where this one started
        end_time = int(df["open_time"].iloc[0].timestamp() * 1000)
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data).drop_duplicates("open_time")
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined


def main():
    all_frames = []
    total = len(SYMBOLS) * len(INTERVALS)
    count = 0

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            count += 1
            print(f"  [{count}/{total}] Downloading {symbol} {interval}...")
            df = fetch_extended(symbol, interval)
            if not df.empty:
                df["symbol"] = symbol
                df["interval"] = interval
                all_frames.append(df)
                print(f"    Got {len(df):,} candles")
            time.sleep(0.5)

    if not all_frames:
        print("No data downloaded!")
        return

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset.to_csv(RAW_DATA_FILE, index=False)

    print(f"\nTotal rows: {len(dataset):,}")
    print(f"Symbols: {dataset['symbol'].nunique()}")
    print(f"Intervals: {list(dataset['interval'].unique())}")
    print(f"Saved to {RAW_DATA_FILE}")


if __name__ == "__main__":
    print("Downloading extended training data...\n")
    main()