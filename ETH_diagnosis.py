import pandas as pd
from pathlib import Path

fund = pd.read_parquet("data/historical/funding/ETHUSDT_funding.parquet")

print("=== ETH FUNDING ===")
print("rows:", len(fund))
print("columns:", fund.columns.tolist())
print("min funding_time:", fund["funding_time"].min())
print("max funding_time:", fund["funding_time"].max())
print("unique funding_time:", fund["funding_time"].nunique())
print("zero count:", (fund["fundingRate"] == 0).sum())
print("min fundingRate:", fund["fundingRate"].min())
print("max fundingRate:", fund["fundingRate"].max())

c = pd.read_parquet("data/historical/ETHUSDT_15m.parquet")

print("\n=== ETH 15M ===")
print("rows:", len(c))
print("columns:", c.columns.tolist())
print("min close_time:", c["close_time"].min())
print("max close_time:", c["close_time"].max())

f = fund.copy()
f["funding_time"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True)

base = c.copy()
base["close_time"] = pd.to_datetime(base["close_time"], unit="ms", utc=True)

m = pd.merge_asof(
    base[["close_time"]],
    f[["funding_time", "fundingRate"]],
    left_on="close_time",
    right_on="funding_time",
    direction="backward",
)

missing = m["fundingRate"].isna()

print("\n=== MERGE ===")
print("missing:", missing.sum())

if missing.any():
    print("\nMissing candle times:")
    print(base.loc[missing, "close_time"].to_string(index=False))
