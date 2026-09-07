"""
capital_tier_screener.py
Extends coin_screener.py's live-data approach to answer the actual question:
"which coins fit my risk budget at $10, and which additional coins unlock as
I scale to $20/$30/$40/$50/$100?" — one run, one table, no re-running per tier.

Usage:
    python capital_tier_screener.py
    python capital_tier_screener.py --risk-pct 3.0 --tiers 10,20,30,40,50,100
"""

import argparse
import requests


DERIBIT_INSTRUMENTS_URL = "https://www.deribit.com/api/v2/public/get_instruments"
BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_24H_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"


def get_deribit_usdc_perpetuals():
    resp = requests.get(
        DERIBIT_INSTRUMENTS_URL,
        params={"currency": "USDC", "kind": "future", "expired": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    instruments = resp.json().get("result", [])
    return [
        i for i in instruments
        if i.get("kind") == "future" and i.get("settlement_period") == "perpetual"
    ]


def get_binance_usdt_perp_symbols():
    resp = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        s["symbol"]: s for s in data.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }


def get_binance_24h_volumes():
    resp = requests.get(BINANCE_24H_TICKER_URL, timeout=15)
    resp.raise_for_status()
    return {row["symbol"]: float(row.get("quoteVolume", 0)) for row in resp.json()}


def screen(risk_pct: float, tiers: list, min_binance_24h_vol_usd: float):
    deribit_perps = get_deribit_usdc_perpetuals()
    binance_symbols = get_binance_usdt_perp_symbols()
    binance_vols = get_binance_24h_volumes()

    results = []
    for inst in deribit_perps:
        base = inst.get("base_currency", "")
        deribit_name = inst.get("instrument_name")
        min_trade_amount = inst.get("min_trade_amount")
        mark_price = inst.get("mark_price") or inst.get("last_price")

        binance_symbol = f"{base}USDT"
        if binance_symbol not in binance_symbols:
            continue

        vol_24h = binance_vols.get(binance_symbol, 0)
        if vol_24h < min_binance_24h_vol_usd:
            continue
        if not mark_price or not min_trade_amount:
            continue

        min_lot_notional_usd = min_trade_amount * mark_price

        # For each capital tier, a coin "fits" if a single min-lot's notional
        # risk (worst case: you're forced to take the whole lot, so your
        # real risk is roughly the lot's price distance times its size) does
        # not blow past that tier's risk-per-trade budget by more than a
        # generous headroom multiple. This mirrors validate_and_normalize
        # order_size()'s own logic, just checked in advance across tiers.
        tier_fit = {}
        for cap in tiers:
            risk_budget = cap * (risk_pct / 100.0)
            tier_fit[cap] = min_lot_notional_usd <= (risk_budget * 20)  # ~ same generous headroom as before

        results.append({
            "base": base,
            "deribit_instrument": deribit_name,
            "binance_symbol": binance_symbol,
            "binance_24h_volume_usd": round(vol_24h, 0),
            "min_trade_amount": min_trade_amount,
            "min_lot_notional_usd": round(min_lot_notional_usd, 4),
            "tier_fit": tier_fit,
        })

    results.sort(key=lambda r: -r["binance_24h_volume_usd"])
    return results


def print_table(results, tiers, risk_pct):
    header = f"{'BASE':<8}{'24H VOL($)':<15}{'LOT $':<10}"
    for t in tiers:
        header += f"${t:<7}"
    print("=" * len(header))
    print(f"Risk per trade: {risk_pct}% of capital  |  Tiers: {tiers}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['base']:<8}{r['binance_24h_volume_usd']:<15,.0f}{r['min_lot_notional_usd']:<10.4f}"
        for t in tiers:
            row += f"{'YES':<8}" if r['tier_fit'][t] else f"{'no':<8}"
        print(row)
    print("=" * len(header))

    # Summary: at what tier does each currently-unfit coin start fitting?
    print("\nTier at which each coin FIRST becomes viable (lowest tier where fit=YES):")
    for r in results:
        first_fit = next((t for t in tiers if r["tier_fit"][t]), None)
        if first_fit and first_fit != tiers[0]:
            print(f"  {r['base']:<8} unlocks at ${first_fit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-pct", type=float, default=3.0, help="Risk per trade as %% of capital (matches RISK_PER_TRADE in config.py)")
    parser.add_argument("--tiers", type=str, default="10,20,30,40,50,100", help="Comma-separated capital tiers in USD")
    parser.add_argument("--min-binance-24h-vol-usd", type=float, default=2_000_000)
    args = parser.parse_args()

    tiers = [float(x) for x in args.tiers.split(",")]
    rows = screen(args.risk_pct, tiers, args.min_binance_24h_vol_usd)
    print_table(rows, tiers, args.risk_pct)
