from __future__ import annotations

import os
import uuid

import pytest

from postgres_execution_ledger import PostgresExecutionLedger

RISK_CAPTURE_DSN = os.getenv("EXECUTION_LEDGER_RISK_CAPTURE_TEST_DSN", "").strip()
RISK_CAPTURE_SSLMODE = os.getenv("EXECUTION_LEDGER_RISK_CAPTURE_TEST_SSLMODE", "disable").strip().lower()


@pytest.mark.skipif(not RISK_CAPTURE_DSN, reason="Set EXECUTION_LEDGER_RISK_CAPTURE_TEST_DSN for least-privilege integration")
def test_risk_capture_writer_has_only_its_fenced_lease_and_snapshot_permissions():
    import psycopg
    from psycopg import errors

    ledger = PostgresExecutionLedger(
        RISK_CAPTURE_DSN, sslmode=RISK_CAPTURE_SSLMODE, require_fenced_writes=True,
    )
    account_id = "risk-capture-it-" + uuid.uuid4().hex[:16]
    lease_key = f"DERIBIT:TESTNET:{account_id}:DAILY_RISK_BASELINE"
    owner = "risk-capture-test-" + uuid.uuid4().hex[:16]
    token = None
    try:
        token = ledger.acquire_fenced_risk_capture_lease(lease_key, owner, ttl_seconds=60)
        assert token is not None and token > 0
        snapshot_time = 1_800_000_000.0 + (int(uuid.uuid4().hex[:6], 16) / 1_000_000)
        inserted = ledger.record_account_risk_snapshot_with_capture_lease(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            currency="USDC", equity="1000.25", observed_at=snapshot_time,
            source="risk_capture_least_privilege_integration",
            risk_capture_lease_key=lease_key,
            risk_capture_lease_owner_id=owner,
            risk_capture_fencing_token=token,
        )
        assert inserted["equity"] == "1000.25"
        baseline = ledger.first_account_risk_snapshot_between(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            currency="USDC", start=snapshot_time - 1, end=snapshot_time + 1,
        )
        assert baseline is not None
        assert baseline["equity"] == "1000.25"
        assert ledger.renew_fenced_risk_capture_lease(lease_key, owner, token, ttl_seconds=60)

        for table in (
            "run_leases", "order_intents", "order_events", "execution_mutations",
            "execution_mutation_events", "execution_protection_links", "daily_risk_halts",
        ):
            with pytest.raises(errors.InsufficientPrivilege):
                with psycopg.connect(RISK_CAPTURE_DSN, sslmode=RISK_CAPTURE_SSLMODE) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT * FROM {table} LIMIT 1")

        assert ledger.release_fenced_risk_capture_lease(lease_key, owner, token)
        old_token = token
        old_owner = owner
        token = None

        # A new owner must receive a higher generation after the prior lease
        # is released; the previous owner cannot append a later checkpoint.
        new_owner = "risk-capture-restart-" + uuid.uuid4().hex[:16]
        new_token = ledger.acquire_fenced_risk_capture_lease(lease_key, new_owner, ttl_seconds=60)
        assert new_token is not None and new_token > old_token
        with pytest.raises(Exception, match="expired, replaced or fenced"):
            ledger.record_account_risk_snapshot_with_capture_lease(
                exchange="DERIBIT", environment="testnet", account_id=account_id,
                currency="USDC", equity="999.00", observed_at=snapshot_time + 2,
                source="stale-risk-capture-owner",
                risk_capture_lease_key=lease_key,
                risk_capture_lease_owner_id=old_owner,
                risk_capture_fencing_token=old_token,
            )
        accepted = ledger.record_account_risk_snapshot_with_capture_lease(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            currency="USDC", equity="1001.00", observed_at=snapshot_time + 3,
            source="new-risk-capture-owner",
            risk_capture_lease_key=lease_key,
            risk_capture_lease_owner_id=new_owner,
            risk_capture_fencing_token=new_token,
        )
        assert accepted["equity"] == "1001.00"
        assert ledger.release_fenced_risk_capture_lease(lease_key, new_owner, new_token)
    finally:
        if token is not None:
            ledger.release_fenced_risk_capture_lease(lease_key, owner, token)
        ledger.close()
