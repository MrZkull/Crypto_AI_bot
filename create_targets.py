# create_targets.py
# Professional ATR-based target labeling
# Labels BUY only if TP is hit before SL — clean real signals

import pandas as pd
import numpy as np
import ta
from config import FEATURES_FILE, DATASET_FILE, RANDOM_STATE

# How many candles ahead to check for TP/SL hit
LOOKAHEAD     = 12    # 12 x 15min = 3 hours
ATR_SL_MULT   = 1.0   # stop loss  = 1x ATR
ATR_TP_MULT   = 1.5   # take profit = 1.5x ATR (1:1.5 RR)
MIN_ADX       = 18    # only label trending markets


def label_row(i, df, atr_col="atr", adx_col="adx"):
    """
    For each candle check if price hits TP or SL first
    in the next LOOKAHEAD candles.
    Returns BUY, SELL, or NO_TRADE.
    """
    if i + LOOKAHEAD >= len(df):
        return "NO_TRADE"

    row  = df.iloc[i]
    adx  = row.get(adx_col, 0)
    atr  = row.get(atr_col, 0)

    # Skip weak trends — not worth trading
    if adx < MIN_ADX or atr <= 0:
        return "NO_TRADE"

    entry = row["close"]

    buy_tp  = entry + atr * ATR_TP_MULT
    buy_sl  = entry - atr * ATR_SL_MULT
    sell_tp = entry - atr * ATR_TP_MULT
    sell_sl = entry + atr * ATR_SL_MULT

    buy_hit  = False
    sell_hit = False

    for j in range(1, LOOKAHEAD + 1):
        future_high = df.iloc[i + j]["high"]
        future_low  = df.iloc[i + j]["low"]

        # Check BUY scenario
        if not buy_hit:
            if future_high >= buy_tp:
                buy_hit = True    # TP hit first = valid BUY
                break
            if future_low <= buy_sl:
                buy_hit = False   # SL hit first = not a BUY
                break

        # Check SELL scenario
        if not sell_hit:
            if future_low <= sell_tp:
                sell_hit = True   # TP hit first = valid SELL
                break
            if future_high >= sell_sl:
                sell_hit = False  # SL hit first = not a SELL
                break

    # Check both directions cleanly
    buy_result  = False
    sell_result = False

    for j in range(1, LOOKAHEAD + 1):
        fh = df.iloc[i + j]["high"]
        fl = df.iloc[i + j]["low"]

        if not buy_result:
            if fh >= buy_tp:
                buy_result = True
                break
            if fl <= buy_sl:
                break

    for j in range(1, LOOKAHEAD + 1):
        fh = df.iloc[i + j]["high"]
        fl = df.iloc[i + j]["low"]

        if not sell_result:
            if fl <= sell_tp:
                sell_result = True
                break
            if fh >= sell_sl:
                break

    if buy_result and not sell_result:
        return "BUY"
    elif sell_result and not buy_result:
        return "SELL"
    else:
        return "NO_TRADE"


def label_dataframe(df):
    """Label entire dataframe using ATR-based method."""
    df = df.copy().reset_index(drop=True)

    # Add ATR and ADX if not present
    if "atr" not in df.columns:
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14)
    if "adx" not in df.columns:
        df["adx"] = ta.trend.adx(
            df["high"], df["low"], df["close"], window=14)

    targets = []
    for i in range(len(df)):
        targets.append(label_row(i, df))

    df["target"] = targets
    return df


def main():
    print("Creating professional ATR-based targets...\n")
    print(f"  Settings:")
    print(f"    Lookahead:  {LOOKAHEAD} candles (15m x {LOOKAHEAD} = {LOOKAHEAD*15} min)")
    print(f"    Stop Loss:  {ATR_SL_MULT}x ATR")
    print(f"    Take Profit:{ATR_TP_MULT}x ATR")
    print(f"    Min ADX:    {MIN_ADX} (only trend markets)\n")

    df = pd.read_csv(FEATURES_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"])
    print(f"  Loaded {len(df):,} rows")

    # Process per symbol to avoid lookahead between coins
    groups = []
    symbols = df["symbol"].unique() if "symbol" in df.columns else ["ALL"]

    for symbol in symbols:
        print(f"  Labeling {symbol}...")
        if "symbol" in df.columns:
            group = df[df["symbol"] == symbol].copy()
        else:
            group = df.copy()

        group = group.sort_values("open_time").reset_index(drop=True)

        # Only use 15m data for targets
        if "interval" in group.columns:
            group = group[group["interval"] == "15m"].copy().reset_index(drop=True)

        if len(group) < 100:
            continue

        group = label_dataframe(group)
        groups.append(group)

        dist = group["target"].value_counts()
        total = len(group)
        buy_pct  = dist.get("BUY", 0) / total * 100
        sell_pct = dist.get("SELL", 0) / total * 100
        nt_pct   = dist.get("NO_TRADE", 0) / total * 100
        print(f"    BUY:{buy_pct:.0f}%  SELL:{sell_pct:.0f}%  NO_TRADE:{nt_pct:.0f}%")

    result = pd.concat(groups, ignore_index=True)

    # Remove rows without target
    result = result[result["target"].notna()]
    result.to_csv(DATASET_FILE, index=False)

    print(f"\n  Final distribution:")
    dist  = result["target"].value_counts()
    total = len(result)
    for label, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:10s}: {count:6,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Total rows: {len(result):,}")
    print(f"  Saved to {DATASET_FILE}")
    print(f"\n  Now run: python train_model.py")


if __name__ == "__main__":
    main()