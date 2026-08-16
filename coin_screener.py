"""
coin_screener.py
Auto-discovers coins that satisfy Task 4's three conditions RIGHT NOW,
instead of relying on a hardcoded list that goes stale every time Deribit
adds/removes an instrument or changes lot sizes (as just happened on
18 Aug 2026).

Run this periodically (e.g. weekly) rather than trusting a static table.
Requires: requests

Usage:
    python coin_screener.py --max-risk-inr 15 --inr-usd 0.0116 --min-binance-24h-vol-usd 5000000
"""

import argparse
import requests

DERIBIT_INSTRUMENTS_URL = "https://www.deribit.com/api/v2/public/get_instruments"
BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_24H_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"


def get_deribit_usdc_perpetuals():
    """Return active Linear USDC perpetual instruments from Deribit."""
    resp = requests.get(
        DERIBIT_INSTRUMENTS_URL,
        params={"currency": "USDC", "kind": "future", "expired": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    instruments = resp.json().get("result", [])
    return [
        i for i in instruments
        if i.get("instrument_name", "").endswith("PERPETUAL") or "PERPETUAL" in i.get("instrument_name", "")
        or i.get("kind") == "future" and i.get("settlement_period") == "perpetual"
    ]


def get_binance_usdt_perp_symbols():
    """Return set of active Binance USDT-margined perpetual symbols."""
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


def screen(max_risk_inr: float, inr_to_usd: float, min_binance_24h_vol_usd: float):
    risk_usd = max_risk_inr * inr_to_usd
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
            continue  # not dual-listed

        vol_24h = binance_vols.get(binance_symbol, 0)
        if vol_24h < min_binance_24h_vol_usd:
            continue  # not liquid enough by our threshold

        if not mark_price or not min_trade_amount:
            continue

        min_lot_notional_usd = min_trade_amount * mark_price
        # A trade is workable at this risk budget if a SINGLE lot's notional
        # is not itself larger than what a full-size risk-based position
        # would need -- rough compatibility check, not a precise ATR-based one.
        risk_compatible = min_lot_notional_usd <= (risk_usd * 20)  # generous headroom

        results.append({
            "base": base,
            "deribit_instrument": deribit_name,
            "binance_symbol": binance_symbol,
            "binance_24h_volume_usd": round(vol_24h, 0),
            "deribit_min_trade_amount": min_trade_amount,
            "min_lot_notional_usd": round(min_lot_notional_usd, 4),
            "risk_compatible_estimate": risk_compatible,
        })

    results.sort(key=lambda r: -r["binance_24h_volume_usd"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-risk-inr", type=float, default=15.0)
    parser.add_argument("--inr-usd", type=float, default=0.0116)
    parser.add_argument("--min-binance-24h-vol-usd", type=float, default=5_000_000)
    args = parser.parse_args()

    rows = screen(args.max_risk_inr, args.inr_usd, args.min_binance_24h_vol_usd)
    print(f"{'BASE':<8}{'DERIBIT':<20}{'BINANCE':<14}{'24H VOL($)':<16}{'MIN LOT':<12}{'LOT $':<10}{'FIT?'}")
    for r in rows:
        print(
            f"{r['base']:<8}{r['deribit_instrument']:<20}{r['binance_symbol']:<14}"
            f"{r['binance_24h_volume_usd']:<16,.0f}{r['deribit_min_trade_amount']:<12}"
            f"{r['min_lot_notional_usd']:<10}{'YES' if r['risk_compatible_estimate'] else 'no'}"
        )
