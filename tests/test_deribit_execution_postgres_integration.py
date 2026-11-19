from __future__ import annotations

import os
import uuid

import pytest

from deribit_execution_gateway import (
    DeribitExecutionGateway,
    ExecutionGatewayBlocked,
    ExecutionGatewayOutcomeUnknown,
)
from execution_coordinator import ExecutionCoordinator
from exchange_reliability import CircuitBreaker, OrderState
from postgres_execution_ledger import PostgresExecutionLedger


POSTGRES_DSN = os.getenv("EXECUTION_LEDGER_TEST_DSN", "").strip()
POSTGRES_SSLMODE = os.getenv("EXECUTION_LEDGER_TEST_SSLMODE", "require").strip().lower()
POSTGRES_MIGRATION_DSN = os.getenv("EXECUTION_LEDGER_TEST_MIGRATION_DSN", "").strip()


@pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="Set EXECUTION_LEDGER_TEST_DSN for real PostgreSQL gateway integration",
)
def test_postgres_gateway_persists_order_and_unknown_control_mutation_across_restart():
    account_id = "pg-gateway-" + uuid.uuid4().hex[:16]
    lease_key = "pg-gateway-lease-" + uuid.uuid4().hex
    owner1 = "pg-worker-1-" + uuid.uuid4().hex[:10]
    owner2 = "pg-worker-2-" + uuid.uuid4().hex[:10]
    label = "pg-ci-" + uuid.uuid4().hex[:24]
    order_id = "pg-order-" + uuid.uuid4().hex[:16]

    ledger1 = PostgresExecutionLedger(
        POSTGRES_DSN,
        sslmode=POSTGRES_SSLMODE,
        require_fenced_writes=True,
    )
    gateway1 = None
    ledger2 = None
    gateway2 = None

    try:
        token1 = ledger1.acquire_fenced_run_lease(
            lease_key, owner1, ttl_seconds=60
        )
        assert token1 is not None and token1 > 0

        coordinator1 = ExecutionCoordinator(
            ledger1,
            CircuitBreaker(),
            exchange="DERIBIT",
            environment="testnet",
            account_id=account_id,
            lease_key=lease_key,
            lease_owner_id=owner1,
            fencing_token=token1,
        )

        gateway1 = DeribitExecutionGateway(
            coordinator1,
            lease_renew=lambda: ledger1.renew_fenced_run_lease(
                lease_key, owner1, token1, ttl_seconds=60
            ),
            lease_release=lambda: ledger1.release_fenced_run_lease(
                lease_key, owner1, token1
            ),
        )

        sent = []
        body = {
            "instrument_name": "ETH_USDC-PERPETUAL",
            "amount": "1",
            "type": "limit",
            "label": label,
            "reduce_only": False,
        }

        # The callback is a local stub. No Deribit request is made.
        response = gateway1.submit_order(
            "/private/buy",
            body,
            lambda: sent.append("order") or {
                "order": {
                    "order_id": order_id,
                    "order_state": "open",
                    "filled_amount": "0",
                }
            },
        )

        assert sent == ["order"]
        assert response["order"]["order_id"] == order_id

        intent = ledger1.get_intent_by_client_order_id(
            exchange="DERIBIT",
            environment="testnet",
            account_id=account_id,
            client_order_id=label,
        )
        assert intent is not None
        assert intent["state"] == OrderState.ACKNOWLEDGED.value
        assert intent["exchange_order_id"] == order_id

        # Force an ambiguous control mutation and verify durable UNKNOWN.
        def ambiguous_timeout():
            raise TimeoutError("simulated transport ambiguity")

        with pytest.raises(ExecutionGatewayOutcomeUnknown):
            gateway1.execute_control_mutation(
                "/private/edit",
                {"order_id": order_id, "price": "1999"},
                ambiguous_timeout,
            )

        unresolved = ledger1.unresolved_mutations(
            exchange="DERIBIT",
            environment="testnet",
            account_id=account_id,
        )
        assert len(unresolved) == 1
        assert unresolved[0]["state"] == "UNKNOWN"

        mutation_id = unresolved[0]["mutation_id"]

        # Close the first gateway, releasing its fenced lease.
        gateway1.close()
        gateway1 = None
        ledger1.close()

        # Simulate a new process acquiring a higher fencing generation.
        ledger2 = PostgresExecutionLedger(
            POSTGRES_DSN,
            sslmode=POSTGRES_SSLMODE,
            require_fenced_writes=True,
        )
        token2 = ledger2.acquire_fenced_run_lease(
            lease_key, owner2, ttl_seconds=60
        )
        assert token2 is not None and token2 > token1

        coordinator2 = ExecutionCoordinator(
            ledger2,
            CircuitBreaker(),
            exchange="DERIBIT",
            environment="testnet",
            account_id=account_id,
            lease_key=lease_key,
            lease_owner_id=owner2,
            fencing_token=token2,
        )
        gateway2 = DeribitExecutionGateway(
            coordinator2,
            lease_renew=lambda: ledger2.renew_fenced_run_lease(
                lease_key, owner2, token2, ttl_seconds=60
            ),
            lease_release=lambda: ledger2.release_fenced_run_lease(
                lease_key, owner2, token2
            ),
        )

        # An unresolved UNKNOWN mutation must block a second exchange call.
        second_call = []
        with pytest.raises(ExecutionGatewayBlocked):
            gateway2.execute_control_mutation(
                "/private/cancel",
                {"order_id": order_id},
                lambda: second_call.append("must-not-run"),
            )

        assert second_call == []
        assert ledger2.get_intent_by_client_order_id(
            exchange="DERIBIT",
            environment="testnet",
            account_id=account_id,
            client_order_id=label,
        )["state"] == OrderState.ACKNOWLEDGED.value

        persisted = ledger2.get_mutation(mutation_id)
        assert persisted["state"] == "UNKNOWN"
        assert len(ledger2.mutation_events(mutation_id)) >= 2

    finally:
        if gateway2 is not None:
            gateway2.close()
        if ledger2 is not None:
            ledger2.close()
        if gateway1 is not None:
            gateway1.close()
        ledger1.close()