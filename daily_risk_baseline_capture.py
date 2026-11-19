"""Read-only Deribit Testnet daily USDC baseline capture.

This process intentionally does not construct the execution gateway and has no
exchange mutation scope. Its PostgreSQL role can update only risk_capture_leases
and insert/read account_risk_snapshots; it cannot access trading run_leases or
order-intent tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC_WINDOW_SECONDS = 20 * 60
LEASE_TTL_SECONDS = 180.0
EXPECTED_TESTNET_BASE = "https://test.deribit.com/api/v2"


class BaselineCaptureError(RuntimeError):
    pass


def _finite_positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if value is None or isinstance(value, bool):
        raise BaselineCaptureError(f"{field} is missing or invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise BaselineCaptureError(f"{field} is invalid") from None
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise BaselineCaptureError(f"{field} is outside valid bounds")
    return number


def _server_time_checked(client, *, max_clock_skew_ms: int) -> float:
    before = time.time()
    server_ms = client.get_server_time_ms()
    after = time.time()
    if not isinstance(server_ms, int) or isinstance(server_ms, bool) or server_ms <= 0:
        raise BaselineCaptureError("Deribit server clock response is invalid")
    midpoint_ms = ((before + after) / 2.0) * 1000.0
    if abs(float(server_ms) - midpoint_ms) > max_clock_skew_ms:
        raise BaselineCaptureError("Deribit server clock skew exceeds the configured limit")
    return float(server_ms) / 1000.0


def _usdc_equity_from_summaries(response: Any) -> float:
    if not isinstance(response, dict) or not isinstance(response.get("summaries"), list):
        raise BaselineCaptureError("Deribit account-summary response is malformed")
    rows = response["summaries"]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise BaselineCaptureError("Deribit account summaries are empty or malformed")
    margin_models = {str(row.get("margin_model") or "").strip().lower() for row in rows}
    if len(margin_models) != 1 or next(iter(margin_models)) not in {"segregated_sm", "segregated_pm"}:
        raise BaselineCaptureError("Daily baseline requires a verified segregated account")
    usdc_rows = [row for row in rows if str(row.get("currency") or "").strip().upper() == "USDC"]
    if len(usdc_rows) != 1:
        raise BaselineCaptureError("Exactly one USDC account summary is required")
    # Available funds may legitimately be zero when collateral is fully used.
    # The daily baseline measures equity, not whether a new order is possible.
    equity = _finite_positive(usdc_rows[0].get("equity"), "USDC equity")
    available = usdc_rows[0].get("available_funds")
    if available is not None:
        _finite_positive(available, "USDC available funds", allow_zero=True)
    return equity


def _validate_opening_window(server_epoch: float) -> tuple[str, float, float]:
    moment = datetime.fromtimestamp(server_epoch, timezone.utc)
    start = datetime(moment.year, moment.month, moment.day, tzinfo=timezone.utc).timestamp()
    end = start + UTC_WINDOW_SECONDS
    if server_epoch < start or server_epoch > end:
        raise BaselineCaptureError("Daily baseline capture is outside the first 20-minute UTC window")
    return moment.date().isoformat(), start, end


def capture_daily_baseline(
    *, client_id: str,
    client_secret: str,
    account_id: str,
    database_dsn: str,
    sslmode: str = "require",
    max_clock_skew_ms: int = 5000,
    client_factory=None,
    ledger_factory=None,
) -> dict:
    """Capture one immutable UTC-day opening checkpoint using independent fences."""
    if not all(str(value or "").strip() for value in (client_id, client_secret, account_id, database_dsn)):
        raise BaselineCaptureError("Read-only credentials, account ID and risk-capture DSN are required")
    if int(max_clock_skew_ms) < 100 or int(max_clock_skew_ms) > 30000:
        raise BaselineCaptureError("MAX_DERIBIT_CLOCK_SKEW_MS must be between 100 and 30000")

    if client_factory is None:
        from deribit_client import DeribitClient
        client_factory = DeribitClient
    if ledger_factory is None:
        from postgres_execution_ledger import PostgresExecutionLedger
        ledger_factory = PostgresExecutionLedger

    client = None
    ledger = None
    lease_key = f"DERIBIT:TESTNET:{account_id}:DAILY_RISK_BASELINE"
    owner_id = f"{os.getenv('GITHUB_RUN_ID', 'manual')}:{os.getenv('GITHUB_RUN_ATTEMPT', '1')}:{uuid.uuid4().hex}"
    fencing_token = None
    try:
        # Constructor enforces exactly trade:read account:read and pins public
        # initialization to Testnet; allow_mutations remains false.
        client = client_factory(
            client_id, client_secret, testnet=True,
            auth_scope="trade:read account:read", allow_mutations=False,
        )
        if getattr(client, "base", None) != EXPECTED_TESTNET_BASE:
            raise BaselineCaptureError("Deribit client is not pinned to the exact Testnet host")
        if getattr(client, "allow_mutations", True) is not False:
            raise BaselineCaptureError("Baseline service refuses a mutation-enabled Deribit client")
        if getattr(client, "require_execution_gateway", False):
            raise BaselineCaptureError("Baseline service must not instantiate the execution gateway")

        ledger = ledger_factory(database_dsn, sslmode=sslmode, require_fenced_writes=True)
        acquire = getattr(ledger, "acquire_fenced_risk_capture_lease", None)
        if not callable(acquire):
            raise BaselineCaptureError("Ledger lacks the isolated risk-capture lease API")
        fencing_token = acquire(lease_key, owner_id, ttl_seconds=LEASE_TTL_SECONDS)
        if fencing_token is None:
            raise BaselineCaptureError("Another daily baseline capture owns the fenced lease")

        # Check for an already-captured opening baseline before spending another
        # private API request. Existing immutable checkpoints are never replaced.
        initial_server_time = _server_time_checked(client, max_clock_skew_ms=max_clock_skew_ms)
        utc_day, day_start, window_end = _validate_opening_window(initial_server_time)
        find_baseline = getattr(ledger, "first_account_risk_snapshot_between", None)
        if not callable(find_baseline):
            raise BaselineCaptureError("Ledger lacks durable daily-baseline lookup")
        existing = find_baseline(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            currency="USDC", start=day_start, end=window_end,
        )
        if isinstance(existing, dict):
            return {
                "status": "ALREADY_CAPTURED", "utc_day": utc_day,
                "observed_at": float(existing["observed_at"]),
                "equity_usdc": str(existing["equity"]),
            }

        summaries = client.get_account_summaries()
        equity = _usdc_equity_from_summaries(summaries)
        # Timestamp the account-equity observation immediately after the account
        # summary read; position verification must not shift the equity baseline.
        observed_at = _server_time_checked(client, max_clock_skew_ms=max_clock_skew_ms)
        utc_day_after_summary, _, _ = _validate_opening_window(observed_at)
        if utc_day_after_summary != utc_day:
            raise BaselineCaptureError("UTC day changed during account-summary capture")
        positions = client.get_positions()  # strict method rejects partial snapshots
        if not isinstance(positions, list) or any(not isinstance(item, dict) for item in positions):
            raise BaselineCaptureError("Complete verified position snapshot is required")
        verified_at = _server_time_checked(client, max_clock_skew_ms=max_clock_skew_ms)
        utc_day_after_positions, _, _ = _validate_opening_window(verified_at)
        if utc_day_after_positions != utc_day:
            raise BaselineCaptureError("UTC day changed during position verification")
        renew = getattr(ledger, "renew_fenced_risk_capture_lease", None)
        if not callable(renew) or not renew(
            lease_key, owner_id, int(fencing_token), ttl_seconds=LEASE_TTL_SECONDS,
        ):
            raise BaselineCaptureError("Risk-capture lease was lost before checkpoint persistence")

        recorder = getattr(ledger, "record_account_risk_snapshot_with_capture_lease", None)
        if not callable(recorder):
            raise BaselineCaptureError("Ledger lacks isolated risk-capture snapshot persistence")
        row = recorder(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            currency="USDC", equity=str(equity), observed_at=observed_at,
            source=f"readonly_daily_baseline_capture:positions={len(positions)}",
            risk_capture_lease_key=lease_key,
            risk_capture_lease_owner_id=owner_id,
            risk_capture_fencing_token=int(fencing_token),
        )
        return {
            "status": "CAPTURED", "utc_day": utc_day,
            "observed_at": float(row["observed_at"]),
            "equity_usdc": str(row["equity"]),
            "positions_verified": len(positions),
        }
    finally:
        if ledger is not None and fencing_token is not None:
            try:
                ledger.release_fenced_risk_capture_lease(lease_key, owner_id, int(fencing_token))
            except Exception:
                pass
        if ledger is not None:
            try:
                ledger.close()
            except Exception:
                pass
        if client is not None:
            try:
                client.session.close()
            except Exception:
                pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="daily_risk_baseline_evidence.json")
    args = parser.parse_args(argv)
    result: dict
    try:
        result = capture_daily_baseline(
            client_id=os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_ID", ""),
            client_secret=os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_SECRET", ""),
            account_id=os.getenv("DERIBIT_EXECUTION_ACCOUNT_ID", ""),
            database_dsn=os.getenv("EXECUTION_LEDGER_RISK_CAPTURE_DSN", ""),
            sslmode=os.getenv("EXECUTION_LEDGER_RISK_CAPTURE_SSLMODE", "require"),
            max_clock_skew_ms=int(os.getenv("MAX_DERIBIT_CLOCK_SKEW_MS", "5000")),
        )
        result["ok"] = True
        print(f"DAILY_RISK_BASELINE_{result['status']}: day={result['utc_day']} equity_usdc={result['equity_usdc']}")
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:240]}
        print(f"DAILY_RISK_BASELINE_FAILED: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
    try:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Could not write baseline evidence: {type(exc).__name__}", file=sys.stderr)
        return 3
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
