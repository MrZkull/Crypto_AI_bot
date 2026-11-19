from __future__ import annotations

from decimal import Decimal

import pytest

from deribit_execution_gateway import (
    DeribitExecutionGateway,
    ExecutionGatewayBlocked,
    ExecutionGatewayOutcomeUnknown,
)
from exchange_reliability import CircuitBreaker, ExecutionLedger, OrderState
from execution_coordinator import ExecutionCoordinator


def make_gateway(tmp_path, *, account_id="gateway-test-account", breaker=None):
    ledger = ExecutionLedger(tmp_path / "gateway.sqlite3")
    breaker = breaker or CircuitBreaker(failure_threshold=1)
    coordinator = ExecutionCoordinator(
        ledger, breaker, exchange="DERIBIT", environment="testnet", account_id=account_id,
    )
    return ledger, breaker, DeribitExecutionGateway(coordinator)


def order_body(label="gw-label-0001", *, reduce_only=False, order_type="limit", side_path="/private/buy"):
    return {
        "instrument_name": "ETH_USDC-PERPETUAL",
        "amount": "1",
        "type": order_type,
        "label": label,
        "reduce_only": reduce_only,
        "trigger": "mark_price" if order_type == "stop_market" else None,
        "trigger_price": "1900" if order_type == "stop_market" else None,
    }


def test_gateway_persists_entry_before_dispatch_and_records_ack(tmp_path):
    ledger, breaker, gateway = make_gateway(tmp_path)
    sent = []
    response = {"order": {
        "order_id": "ex-ack-1", "order_state": "open", "filled_amount": "0",
    }}

    result = gateway.submit_order(
        "/private/buy", order_body(), lambda: sent.append("send") or response,
    )

    assert result is response
    assert sent == ["send"]
    intent = ledger.get_intent_by_client_order_id(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
        client_order_id="gw-label-0001",
    )
    assert intent["state"] == OrderState.ACKNOWLEDGED.value
    assert intent["exchange_order_id"] == "ex-ack-1"
    assert intent["payload"]["order_type"] == "limit"
    gateway.close()
    ledger.close()


def test_gateway_rejects_non_boolean_reduce_only_without_dispatch(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    sent = []
    body = order_body()
    body["reduce_only"] = "sometimes"

    with pytest.raises(ExecutionGatewayBlocked, match="strict boolean"):
        gateway.submit_order("/private/buy", body, lambda: sent.append(True))

    assert sent == []
    gateway.close()
    ledger.close()


def test_ambiguous_order_submission_is_durable_and_blocks_new_entry_after_restart(tmp_path):
    db = tmp_path / "gateway.sqlite3"
    ledger, breaker, gateway = make_gateway(tmp_path)

    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.submit_order("/private/buy", order_body(), lambda: (_ for _ in ()).throw(TimeoutError()))

    first = ledger.get_intent_by_client_order_id(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
        client_order_id="gw-label-0001",
    )
    assert first["state"] == OrderState.UNKNOWN.value
    gateway.close()
    ledger.close()

    reopened = ExecutionLedger(db)
    new_coordinator = ExecutionCoordinator(
        reopened, CircuitBreaker(), exchange="DERIBIT", environment="testnet",
        account_id="gateway-test-account",
    )
    resumed = DeribitExecutionGateway(new_coordinator)
    sent = []
    with pytest.raises(ExecutionGatewayBlocked):
        resumed.submit_order(
            "/private/buy", order_body("gw-label-0002"), lambda: sent.append(True),
        )
    assert sent == []
    resumed.close()
    reopened.close()


def test_ambiguous_control_mutation_blocks_replay_after_restart(tmp_path):
    db = tmp_path / "gateway.sqlite3"
    ledger, _, gateway = make_gateway(tmp_path)
    body = {"order_id": "cancel-me"}

    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.execute_control_mutation(
            "/private/cancel", body, lambda: (_ for _ in ()).throw(TimeoutError()),
        )
    unresolved = ledger.unresolved_mutations(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
    )
    assert len(unresolved) == 1
    assert unresolved[0]["state"] == "UNKNOWN"
    gateway.close()
    ledger.close()

    reopened = ExecutionLedger(db)
    resumed_coordinator = ExecutionCoordinator(
        reopened, CircuitBreaker(), exchange="DERIBIT", environment="testnet",
        account_id="gateway-test-account",
    )
    resumed = DeribitExecutionGateway(resumed_coordinator)
    sent = []
    with pytest.raises(ExecutionGatewayBlocked, match="journal"):
        resumed.execute_control_mutation(
            "/private/cancel", body, lambda: sent.append(True),
        )
    assert sent == []
    resumed.close()
    reopened.close()


def test_control_mutation_reopens_after_explicit_durable_resolution(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    body = {"order_id": "cancel-me"}
    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.execute_control_mutation(
            "/private/cancel", body, lambda: (_ for _ in ()).throw(TimeoutError()),
        )
    unresolved = ledger.unresolved_mutations(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
    )
    assert len(unresolved) == 1
    gateway.resolve_control_mutation(
        unresolved[0]["mutation_id"], state="ACKNOWLEDGED",
        resolved_by="unit-test-operator",
        evidence={
            "verified": True,
            "verification_method": "read_only_order_lookup",
            "observed_status": "cancelled",
            "observed_at": "2026-11-18T12:00:00Z",
            "resource_id": "cancel-me",
        },
    )
    assert ledger.unresolved_mutations(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
    ) == []
    events = ledger.mutation_events(unresolved[0]["mutation_id"])
    assert [event["event_type"] for event in events] == [
        "SUBMISSION_STARTED", "EXCHANGE_RESPONSE_RECORDED", "READ_ONLY_RECONCILIATION_CONFIRMED",
    ]
    gateway.coordinator.breaker.manual_reset(acknowledged_by="unit-test-operator")

    sent = []
    result = gateway.execute_control_mutation(
        "/private/cancel", body, lambda: sent.append(True) or {"order": {"order_id": "cancel-me"}},
    )
    assert sent == [True]
    assert result["order"]["order_id"] == "cancel-me"
    gateway.close()
    ledger.close()


def test_order_filled_state_must_match_requested_quantity(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    bad = {"order": {
        "order_id": "ex-bad-fill", "order_state": "filled",
        "filled_amount": "0.4", "average_price": "2000",
    }}
    with pytest.raises(ExecutionGatewayOutcomeUnknown, match="conflicts"):
        gateway.submit_order("/private/buy", order_body(), lambda: bad)
    intent = ledger.get_intent_by_client_order_id(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
        client_order_id="gw-label-0001",
    )
    assert intent["state"] == OrderState.UNKNOWN.value
    gateway.close()
    ledger.close()


def test_confirm_protection_requires_resting_opposite_stop_market(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    entry = gateway.coordinator.create_intent(
        client_order_id="entry-label", instrument_id="ETH_USDC-PERPETUAL",
        side="BUY", quantity="0.5", settlement_currency="USDC", reduce_only=False,
        payload={"rpc_method": "/private/buy", "order_type": "limit"},
    )
    gateway.coordinator.claim_submission(entry["intent_id"])
    gateway.coordinator.record_fill(
        entry["intent_id"], cumulative_filled_quantity="0.5",
        average_fill_price="2000", exchange_order_id="entry-exchange-id",
    )
    stop = gateway.coordinator.create_intent(
        client_order_id="stop-label", instrument_id="ETH_USDC-PERPETUAL",
        side="SELL", quantity="0.5", settlement_currency="USDC", reduce_only=True,
        payload={"rpc_method": "/private/sell", "order_type": "stop_market", "trigger": "mark_price"},
    )
    gateway.coordinator.claim_submission(stop["intent_id"])
    gateway.coordinator.record_acknowledged(stop["intent_id"], exchange_order_id="stop-exchange-id")

    verified = gateway.confirm_protection(
        entry_client_order_id="entry-label", stop_client_order_id="stop-label",
        protective_order_id="stop-exchange-id", protected_quantity="0.5",
    )
    assert verified["protection_state"] == "PROTECTED"
    assert Decimal(verified["protected_quantity"]) == Decimal("0.5")
    gateway.close()
    ledger.close()


def test_confirm_protection_rejects_same_side_stop(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    entry = gateway.coordinator.create_intent(
        client_order_id="entry-label", instrument_id="ETH_USDC-PERPETUAL",
        side="BUY", quantity="0.5", settlement_currency="USDC",
    )
    gateway.coordinator.claim_submission(entry["intent_id"])
    gateway.coordinator.record_fill(
        entry["intent_id"], cumulative_filled_quantity="0.5",
        average_fill_price="2000", exchange_order_id="entry-exchange-id",
    )
    stop = gateway.coordinator.create_intent(
        client_order_id="stop-label", instrument_id="ETH_USDC-PERPETUAL",
        side="BUY", quantity="0.5", settlement_currency="USDC", reduce_only=True,
        payload={"order_type": "stop_market", "trigger": "mark_price"},
    )
    gateway.coordinator.claim_submission(stop["intent_id"])
    gateway.coordinator.record_acknowledged(stop["intent_id"], exchange_order_id="stop-exchange-id")

    with pytest.raises(ExecutionGatewayBlocked, match="oppose"):
        gateway.confirm_protection(
            entry_client_order_id="entry-label", stop_client_order_id="stop-label",
            protective_order_id="stop-exchange-id", protected_quantity="0.5",
        )
    gateway.close()
    ledger.close()


def test_exact_label_read_only_match_reconciles_unknown_order_after_restart(tmp_path):
    db = tmp_path / "gateway.sqlite3"
    ledger, _, gateway = make_gateway(tmp_path)
    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.submit_order("/private/buy", order_body(), lambda: (_ for _ in ()).throw(TimeoutError()))
    gateway.close()
    ledger.close()

    reopened = ExecutionLedger(db)
    coordinator = ExecutionCoordinator(
        reopened, CircuitBreaker(), exchange="DERIBIT", environment="testnet",
        account_id="gateway-test-account",
    )
    resumed = DeribitExecutionGateway(coordinator)
    result = resumed.reconcile_order_label(
        "gw-label-0001",
        [{
            "order_id": "ex-label-open", "label": "gw-label-0001",
            "instrument_name": "ETH_USDC-PERPETUAL", "direction": "buy",
            "amount": "1", "filled_amount": "0", "order_state": "open",
        }],
        queries_complete=True,
    )
    assert result["state"] == OrderState.ACKNOWLEDGED.value
    assert result["exchange_order_id"] == "ex-label-open"
    resumed.close()
    reopened.close()


def test_empty_or_incomplete_label_window_never_reconciles_to_rejected(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.submit_order("/private/buy", order_body(), lambda: (_ for _ in ()).throw(TimeoutError()))

    result = gateway.reconcile_order_label("gw-label-0001", [], queries_complete=True)
    assert result["state"] == OrderState.UNKNOWN.value
    assert result["exchange_order_id"] is None
    assert result["state"] != OrderState.REJECTED.value
    gateway.close()
    ledger.close()


def test_multiple_exact_label_matches_remain_unknown(tmp_path):
    ledger, _, gateway = make_gateway(tmp_path)
    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.submit_order("/private/buy", order_body(), lambda: (_ for _ in ()).throw(TimeoutError()))
    rows = [
        {"order_id": "ex-1", "label": "gw-label-0001", "instrument_name": "ETH_USDC-PERPETUAL", "direction": "buy", "amount": "1", "filled_amount": "0", "order_state": "open"},
        {"order_id": "ex-2", "label": "gw-label-0001", "instrument_name": "ETH_USDC-PERPETUAL", "direction": "buy", "amount": "1", "filled_amount": "0", "order_state": "open"},
    ]
    with pytest.raises(ExecutionGatewayOutcomeUnknown, match="Multiple exchange orders"):
        gateway.reconcile_order_label("gw-label-0001", rows, queries_complete=True)
    intent = ledger.get_intent_by_client_order_id(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
        client_order_id="gw-label-0001",
    )
    assert intent["state"] == OrderState.UNKNOWN.value
    gateway.close()
    ledger.close()


def test_mismatched_exact_label_evidence_does_not_break_unknown_recovery_path(tmp_path):
    ledger, breaker, gateway = make_gateway(tmp_path)
    with pytest.raises(ExecutionGatewayOutcomeUnknown):
        gateway.submit_order("/private/buy", order_body(), lambda: (_ for _ in ()).throw(TimeoutError()))

    with pytest.raises(ExecutionGatewayOutcomeUnknown, match="instrument"):
        gateway.reconcile_order_label(
            "gw-label-0001",
            [{
                "order_id": "wrong-instrument", "label": "gw-label-0001",
                "instrument_name": "BTC_USDC-PERPETUAL", "direction": "buy",
                "amount": "1", "filled_amount": "0", "order_state": "open",
            }],
            queries_complete=True,
        )
    intent = ledger.get_intent_by_client_order_id(
        exchange="DERIBIT", environment="testnet", account_id="gateway-test-account",
        client_order_id="gw-label-0001",
    )
    assert intent["state"] == OrderState.UNKNOWN.value
    assert breaker.manual_ack_required is True
    gateway.close()
    ledger.close()
