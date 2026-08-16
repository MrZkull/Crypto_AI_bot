"""
coin_screener.py — Cloud-Resilient Cross-Exchange Screener

Queries Deribit Linear USDC Perpetuals and cross-references against 
Binance Vision Public Market Data (bypassing US datacenter 451 geoblocks).

Usage:
    python coin_screener.py --max-risk-inr 15 --inr-usd 0.0116 --min-binance-24h-vol-usd 5000000
"""

import argparse
import requests
import sys

DERIBIT_INSTRUMENTS_URL = "https://www.deribit.com/api/v2/public/get_instruments"
BINANCE_24H_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"


def get_deribit_usdc_perpetuals():
    """Return active Linear USDC perpetual instruments from Deribit."""
    try:
        resp = requests.get(
            DERIBIT_INSTRUMENTS_URL,
            params={"currency": "USDC", "kind": "future", "expired": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        instruments = resp.json().get("result", [])
        return [
            i for i in instruments
            if "PERPETUAL" in i.get("instrument_name", "").upper()
            or i.get("settlement_period") == "perpetual"
        ]
    except Exception as e:
        print(f"Error fetching Deribit instruments: {e}")
        return []


def get_binance_market_data():
    """Fetch live prices and 24h quote volumes from Binance Vision."""
    try:
        resp = requests.get(BINANCE_24H_TICKER_URL, timeout=15)
        resp.raise_for_status()
        data = {}
        for row in resp.json():
            sym = row.get("symbol", "")
            if sym.endswith("USDT"):
                data[sym] = {
                    "price": float(row.get("lastPrice", 0) or 0),
                    "volume": float(row.get("quoteVolume", 0) or 0)
                }
        return data
    except Exception as e:
        print(f"Error fetching Binance 24h tickers: {e}")
        return {}


def screen(max_risk_inr: float, inr_to_usd: float, min_binance_24h_vol_usd: float):
    risk_usd = max_risk_inr * inr_to_usd
    deribit_perps = get_deribit_usdc_perpetuals()
    binance_data = get_binance_market_data()

    if not deribit_perps:
        print("No active Deribit perpetuals found.")
        return []

    results = []
    seen_symbols = set()

    for inst in deribit_perps:
        deribit_name = inst.get("instrument_name", "")
        base = inst.get("base_currency") or deribit_name.split("_")[0].split("-")[0]
        base = base.upper()

        binance_symbol = f"{base}USDT"
        if binance_symbol not in binance_data or binance_symbol in seen_symbols:
            continue

        b_info = binance_data[binance_symbol]
        vol_24h = b_info["volume"]
        live_price = b_info["price"]

        if vol_24h < min_binance_24h_vol_usd or live_price <= 0:
            continue

        min_trade_amount = float(inst.get("min_trade_amount") or 0)
        if min_trade_amount <= 0:
            continue

        min_lot_notional_usd = min_trade_amount * live_price
        
        # Sizing check: 1 minimum lot notional should be compatible with ₹15 risk (headroom check)
        risk_compatible = min_lot_notional_usd <= (risk_usd * 20)

        results.append({
            "base": base,
            "deribit_instrument": deribit_name,
            "binance_symbol": binance_symbol,
            "binance_24h_volume_usd": round(vol_24h, 0),
            "deribit_min_trade_amount": min_trade_amount,
            "min_lot_notional_usd": round(min_lot_notional_usd, 4),
            "risk_compatible_estimate": risk_compatible,
        })
        seen_symbols.add(binance_symbol)

    # Sort descending by 24h trading volume
    results.sort(key=lambda r: -r["binance_24h_volume_usd"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-risk-inr", type=float, default=15.0)
    parser.add_argument("--inr-usd", type=float, default=0.0116)
    parser.add_argument("--min-binance-24h-vol-usd", type=float, default=5_000_000)
    args = parser.parse_args()

    rows = screen(args.max_risk_inr, args.inr_usd, args.min_binance_24h_vol_usd)
    
    print("\n" + "=" * 94)
    print(f"{'BASE':<8}{'DERIBIT PERP':<24}{'BINANCE PAIR':<14}{'24H VOL ($)':<16}{'MIN LOT':<12}{'LOT ($)':<10}{'FIT?'}")
    print("=" * 94)
    
    for r in rows:
        fit_label = "YES" if r["risk_compatible_estimate"] else "no"
        print(
            f"{r['base']:<8}{r['deribit_instrument']:<24}{r['binance_symbol']:<14}"
            f"{r['binance_24h_volume_usd']:<16,.0f}{r['deribit_min_trade_amount']:<12}"
            f"{r['min_lot_notional_usd']:<10.2f}{fit_label}"
        )
    print("=" * 94 + "\n")
