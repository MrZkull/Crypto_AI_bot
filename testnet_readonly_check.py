#!/usr/bin/env python3
"""Run a Deribit Testnet read-only connectivity and instrument preflight.

This program intentionally contains no order placement, cancellation, leverage,
transfer, or account mutation calls. It writes a credential-free evidence file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MAX_CLOCK_SKEW_MS = 2000.0
CLOCK_SYNC_SAMPLES = 3


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _select_symbols(configured_symbols, max_symbols):
    if (
        isinstance(max_symbols, bool)
        or not isinstance(max_symbols, int)
        or not 1 <= max_symbols <= 25
    ):
        raise ValueError("max_symbols_invalid")

    if (
        isinstance(configured_symbols, (str, bytes))
        or not isinstance(configured_symbols, (list, tuple))
    ):
        raise ValueError("configured_symbol_list_invalid")

    selected = []
    seen = set()

    for raw_symbol in configured_symbols:
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError("configured_symbol_list_contains_invalid_entry")

        symbol = raw_symbol.strip()
        key = symbol.casefold()

        if key not in seen:
            seen.add(key)
            selected.append(symbol)

    selected = selected[:max_symbols]

    if not selected:
        raise ValueError("configured_symbol_list_empty")

    return selected


def _validate_account_snapshot(account):
    if not isinstance(account, dict):
        raise ValueError("account_snapshot_not_an_object")

    assets = account.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("account_snapshot_assets_missing_or_empty")

    seen = set()

    for row in assets:
        if not isinstance(row, dict):
            raise ValueError("account_snapshot_asset_row_malformed")

        asset = row.get("asset")
        unit = row.get("unit")

        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("account_snapshot_asset_name_invalid")

        asset_key = asset.strip().upper()

        if not isinstance(unit, str) or unit.strip().upper() != asset_key:
            raise ValueError("account_snapshot_asset_unit_mismatch")

        if asset_key in seen:
            raise ValueError("account_snapshot_duplicate_asset")
        seen.add(asset_key)

        if _number(row.get("equity")) is None:
            raise ValueError("account_snapshot_asset_equity_invalid")

        available = row.get("available")
        if available is not None and _number(available) is None:
            raise ValueError("account_snapshot_asset_available_invalid")

    return len(assets)


def _validate_instrument_contract_result(instrument, contract):
    if not isinstance(instrument, str) or not instrument.strip():
        raise ValueError("instrument_name_invalid")

    # The production verifier returns validated exchange metadata.
    # Its result is not a {"verified": true} wrapper.
    if (
        not isinstance(contract, dict)
        or contract.get("instrument_name") != instrument
    ):
        raise ValueError("instrument_contract_unverified")

    return True




def _measure_server_clock(client, *, max_skew_ms: float = MAX_CLOCK_SKEW_MS, samples: int = CLOCK_SYNC_SAMPLES) -> dict:
    """Measure server-clock offset using the lowest-RTT of several public reads.

    The half-RTT is retained as uncertainty. The conservative skew bound is
    abs(offset) + uncertainty; an uncertain measurement cannot pass merely
    because its midpoint offset is small. This routine never changes the host
    clock and never uses a trading mutation endpoint.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 5:
        raise ValueError("clock_sync_sample_count_invalid")
    limit = _number(max_skew_ms)
    if limit is None or limit <= 0:
        raise ValueError("clock_sync_limit_invalid")

    measurements = []
    for _ in range(samples):
        before_ms = time.time_ns() / 1_000_000.0
        server_ms = client.get_server_time_ms()
        after_ms = time.time_ns() / 1_000_000.0
        if isinstance(server_ms, bool) or not isinstance(server_ms, int) or server_ms <= 0:
            raise ValueError("server_time_response_invalid")
        rtt_ms = after_ms - before_ms
        if not math.isfinite(rtt_ms) or rtt_ms < 0:
            raise ValueError("clock_sync_round_trip_invalid")
        midpoint_ms = (before_ms + after_ms) / 2.0
        offset_ms = float(server_ms) - midpoint_ms
        uncertainty_ms = rtt_ms / 2.0
        conservative_bound_ms = abs(offset_ms) + uncertainty_ms
        measurements.append({
            "round_trip_ms": rtt_ms,
            "offset_ms": offset_ms,
            "uncertainty_ms": uncertainty_ms,
            "conservative_skew_bound_ms": conservative_bound_ms,
        })

    best = min(measurements, key=lambda row: row["round_trip_ms"])
    if best["conservative_skew_bound_ms"] > limit:
        raise ValueError("server_clock_skew_exceeds_limit")
    return {
        **best,
        "samples": len(measurements),
        "max_skew_limit_ms": limit,
        "clock_adjustment_performed": False,
    }


def _validate_order_book(book, *, max_spread_fraction: float) -> dict:
    """Validate a live top-of-book snapshot without repairing or guessing values.

    DeribitClient.spread_pct is historically a fractional mid-price spread, not
    percentage points. The returned fields make the unit explicit.
    """
    if not isinstance(book, dict):
        raise ValueError("order_book_not_an_object")
    bid = _number(book.get("best_bid"))
    ask = _number(book.get("best_ask"))
    reported = _number(book.get("spread_pct"))
    if bid is None or ask is None or reported is None:
        raise ValueError("order_book_has_missing_or_non_finite_values")
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise ValueError("order_book_top_of_book_not_positive_and_ordered")
    midpoint = (bid + ask) / 2.0
    if not math.isfinite(midpoint) or midpoint <= 0:
        raise ValueError("order_book_midpoint_invalid")
    computed = (ask - bid) / midpoint
    if not math.isfinite(computed) or computed < 0:
        raise ValueError("order_book_spread_invalid")
    if not math.isclose(reported, computed, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError("order_book_spread_field_mismatch")
    if computed > max_spread_fraction:
        raise ValueError("order_book_spread_exceeds_pre_registered_limit")
    if book.get("is_wide") is True:
        raise ValueError("order_book_client_marks_spread_wide")
    return {
        "best_bid": bid,
        "best_ask": ask,
        "spread_fraction": computed,
        "spread_bps": round(computed * 10000.0, 4),
        "spread_limit_fraction": max_spread_fraction,
        "spread_limit_bps": round(max_spread_fraction * 10000.0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="testnet_readonly_evidence.json")
    parser.add_argument("--max-symbols", type=int, default=25)
    args = parser.parse_args()

    client_id = os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_ID", "").strip()
    client_secret = os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print("FAIL: read-only Testnet credentials are not configured", file=sys.stderr)
        return 2
    if args.max_symbols < 1 or args.max_symbols > 25:
        print("FAIL: --max-symbols must be between 1 and 25", file=sys.stderr)
        return 2

    started_monotonic = time.monotonic()
    checked_at = datetime.now(timezone.utc).isoformat()

    from config import SYMBOLS
    from deribit_client import DeribitClient, TESTNET_BASE, WIDE_SPREAD_WARN_PCT

    # Reject an empty or malformed configured universe before exchange access.
    symbols = _select_symbols(SYMBOLS, args.max_symbols)

    client = DeribitClient(
        client_id,
        client_secret,
        testnet=True,
        auth_scope="trade:read account:read",
        allow_mutations=False,
    )
    expected_testnet_base = "https://test.deribit.com/api/v2"
    if (
        TESTNET_BASE != expected_testnet_base
        or getattr(client, "base", None) != expected_testnet_base
    ):
        raise RuntimeError("Client host is not exactly pinned to Deribit Testnet")

    # One strict private account snapshot avoids duplicate reads and keeps
    # each native balance explicitly denominated in its own currency.
    clock_sync = _measure_server_clock(client)
    account = client.get_risk_capital_snapshot()
    _validate_account_snapshot(account)
    positions = client.get_positions()  # rejects partial/malformed snapshots
    if not isinstance(positions, list):
        raise RuntimeError("Position snapshot is malformed")

    instruments = []
    errors = []

    for symbol in symbols:
        try:
            if not client.is_supported(symbol):
                raise RuntimeError("not_supported")
            instrument = client.get_instrument_name(symbol)
            contract = client.verify_instrument_contract(symbol)
            _validate_instrument_contract_result(instrument, contract)
            book = client.get_order_book_spread(symbol)
            book_evidence = _validate_order_book(
                book,
                max_spread_fraction=float(WIDE_SPREAD_WARN_PCT),
            )
            instruments.append({
                "symbol": symbol,
                "instrument_name": instrument,
                "contract_verified": True,
                **book_evidence,
            })
        except Exception as exc:
            errors.append({"symbol": symbol, "error": type(exc).__name__})

    # The private account endpoints are queried to prove authentication and
    # complete snapshots, but account amounts, currencies, position sizes and
    # PnL are deliberately excluded from uploaded evidence artifacts.
    evidence = {
        "schema_version": 2,
        "environment": "DERIBIT_TESTNET",
        "verified_host": "test.deribit.com",
        "authenticated_private_api": True,
        "server_clock_sync": clock_sync,
        "read_only_preflight": True,
        "started_at": checked_at,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID") or None,
        "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT") or None,
        "commit_sha": os.getenv("GITHUB_SHA") or None,
        "account_snapshot_complete": True,
        "balance_asset_count": len(account["assets"]),
        "position_snapshot_complete": True,
        "position_count": len(positions),
        "position_instrument_count": len({
            str(row.get("instrument_name", ""))
            for row in positions
            if isinstance(row, dict) and row.get("instrument_name")
        }),
        "selected_symbol_count": len(symbols),
        "verified_instrument_count": len(instruments),
        "instrument_results": instruments,
        "instrument_errors": errors,
        "order_submission_performed": False,
        "order_cancellation_performed": False,
        "leverage_change_performed": False,
        "account_values_recorded": False,
        "position_values_recorded": False,
        "credential_values_recorded": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "environment": evidence["environment"],
        "verified_host": evidence["verified_host"],
        "authenticated_private_api": True,
        "server_clock_offset_ms": round(clock_sync["offset_ms"], 3),
        "server_clock_skew_bound_ms": round(clock_sync["conservative_skew_bound_ms"], 3),
        "complete_account_snapshot": evidence["account_snapshot_complete"],
        "complete_position_snapshot": evidence["position_snapshot_complete"],
        "position_count": evidence["position_count"],
        "verified_instruments": len(instruments),
        "selected_instruments": len(symbols),
        "instrument_errors": len(errors),
        "evidence_file": str(output),
        "orders_submitted": False,
    }, indent=2))
    return 0 if not errors and len(instruments) == len(symbols) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not print request exception text because it may expose sensitive data.
        print(f"FAIL: Testnet read-only preflight failed ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1)
