import pytest

from exchange_reliability import (
    BreakerState,
    CircuitBreaker,
    ExecutionLedger,
    OrderState,
)
from execution_coordinator import CoordinatorError, ExecutionCoordinator


def make_coordinator(tmp_path, breaker=None):
    ledger = ExecutionLedger(tmp_path / "coordinator.sqlite")
    breaker = breaker or CircuitBreaker(failure_threshold=1)
    coordinator = ExecutionCoordinator(
        ledger,
        breaker,
        exchange="DERIBIT",
        environment="testnet",
        account_id="demo-account",
    )
    return ledger, breaker, coordinator


def create(coordinator, *, reduce_only=False, client_order_id="client-1"):
    return coordinator.create_intent(
        client_order_id=client_order_id,
        instrument_id="ETH_USDC-PERPETUAL",
        side="SELL" if reduce_only else "BUY",
        quantity="1",
        settlement_currency="USDC",
        reduce_only=reduce_only,
    )


def test_degraded_breaker_blocks_new_entries_but_allows_reduce_only(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    with pytest.raises(CoordinatorError):
        create(coordinator, client_order_id="entry")

    intent = create(coordinator, reduce_only=True, client_order_id="protect")
    submitting = coordinator.mark_submitting(intent["intent_id"])
    assert submitting["state"] == OrderState.SUBMITTING.value
    ledger.close()


def test_halted_order_breaker_blocks_even_reduce_only_until_acknowledged(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    breaker.record_failure(order_path=True)
    intent = create(coordinator, reduce_only=True, client_order_id="protect")

    with pytest.raises(CoordinatorError):
        coordinator.mark_submitting(intent["intent_id"])

    breaker.manual_reset(acknowledged_by="operator-1")
    assert coordinator.mark_submitting(intent["intent_id"])["state"] == OrderState.SUBMITTING.value
    ledger.close()


def test_unknown_open_order_without_exchange_id_remains_unknown(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator)
    coordinator.mark_submitting(intent["intent_id"])
    coordinator.mark_unknown(intent["intent_id"], reason="submit timeout")

    result = coordinator.reconcile_unknown(
        intent["intent_id"], lambda _: {"status": "open"}
    )

    assert result["state"] == OrderState.UNKNOWN.value
    assert result["exchange_order_id"] is None
    ledger.close()


def test_unknown_open_order_with_id_reconciles_to_acknowledged(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator)
    coordinator.mark_submitting(intent["intent_id"])
    coordinator.mark_unknown(intent["intent_id"], reason="submit timeout")

    result = coordinator.reconcile_unknown(
        intent["intent_id"],
        lambda _: {"status": "open", "exchange_order_id": "ex-123"},
    )

    assert result["state"] == OrderState.ACKNOWLEDGED.value
    assert result["exchange_order_id"] == "ex-123"
    ledger.close()


def test_unknown_fill_without_exchange_id_remains_unknown(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator)
    coordinator.mark_submitting(intent["intent_id"])
    coordinator.mark_unknown(intent["intent_id"], reason="submit timeout")

    result = coordinator.reconcile_unknown(
        intent["intent_id"],
        lambda _: {
            "status": "partially_filled",
            "cumulative_filled_quantity": "0.25",
            "average_fill_price": "2000",
        },
    )

    assert result["state"] == OrderState.UNKNOWN.value
    assert result["filled_quantity"] == "0"
    ledger.close()


def test_restart_gate_blocks_new_entries_until_prior_order_is_resolved(tmp_path):
    db = tmp_path / "persistent-gate.sqlite"
    ledger = ExecutionLedger(db)
    breaker = CircuitBreaker()
    coordinator = ExecutionCoordinator(
        ledger, breaker, exchange="DERIBIT",
        environment="testnet", account_id="demo-account",
    )
    intent = create(coordinator, client_order_id="pending-after-restart")
    coordinator.mark_submitting(intent["intent_id"])
    coordinator.record_acknowledged(intent["intent_id"], exchange_order_id="exchange-1")
    ledger.close()

    reopened = ExecutionLedger(db)
    resumed = ExecutionCoordinator(
        reopened, CircuitBreaker(), exchange="DERIBIT",
        environment="testnet", account_id="demo-account",
    )
    with pytest.raises(CoordinatorError, match="unresolved"):
        create(resumed, client_order_id="must-be-blocked")

    # A reduce-only intent can still be recorded for risk-reducing handling.
    protective = create(resumed, reduce_only=True, client_order_id="protective-action")
    assert protective["reduce_only"]
    reopened.transition(intent["intent_id"], OrderState.CANCELLED, event_type="TEST_RESOLVED")
    reopened.transition(protective["intent_id"], OrderState.CANCELLED, event_type="TEST_RESOLVED")
    allowed = create(resumed, client_order_id="after-resolution")
    assert allowed["state"] == OrderState.INTENT_CREATED.value
    reopened.close()

def test_open_reconciliation_preserves_confirmed_partial_fill(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator, client_order_id="partial-open")
    coordinator.mark_submitting(intent["intent_id"])
    ledger.record_fill(
        intent["intent_id"], cumulative_filled_quantity="0.4",
        average_fill_price="2000", exchange_order_id="ex-partial",
    )
    coordinator.mark_unknown(intent["intent_id"], reason="timeout after partial fill")
    result = coordinator.reconcile_unknown(
        intent["intent_id"], lambda _: {
            "status": "open", "exchange_order_id": "ex-partial",
            "cumulative_filled_quantity": "0.4", "average_fill_price": "2000",
        }
    )
    assert result["state"] == OrderState.PARTIALLY_FILLED.value
    assert result["filled_quantity"] == "0.4"
    ledger.close()

def test_cancelled_partial_fill_remains_protectable_and_gates_until_protected(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator, client_order_id="partial-cancelled")
    coordinator.mark_submitting(intent["intent_id"])
    coordinator.mark_unknown(intent["intent_id"], reason="submit timeout")
    result = coordinator.reconcile_unknown(
        intent["intent_id"], lambda _: {
            "status": "cancelled", "exchange_order_id": "ex-cancelled",
            "cumulative_filled_quantity": "0.4", "average_fill_price": "2000",
        }
    )
    assert result["state"] == OrderState.CANCELLED.value
    assert result["filled_quantity"] == "0.4"
    assert result["protection_state"] == "REQUIRED"
    breaker.manual_reset(acknowledged_by="unit-test-operator")
    with pytest.raises(CoordinatorError, match="unresolved"):
        create(coordinator, client_order_id="blocked-before-protection")
    protected = coordinator.verify_protection(
        intent["intent_id"], protected_quantity="0.4"
    )
    assert protected["protection_state"] == "PROTECTED"
    breaker.manual_reset(acknowledged_by="unit-test-operator")
    allowed = create(coordinator, client_order_id="allowed-after-protection")
    assert allowed["state"] == OrderState.INTENT_CREATED.value
    ledger.close()


def test_rejected_reconciliation_does_not_erase_prior_fill(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    intent = create(coordinator, client_order_id="rejected-with-fill")
    coordinator.mark_submitting(intent["intent_id"])
    ledger.record_fill(
        intent["intent_id"], cumulative_filled_quantity="0.25",
        average_fill_price="2000", exchange_order_id="ex-uncertain",
    )
    coordinator.mark_unknown(intent["intent_id"], reason="late response conflict")
    result = coordinator.reconcile_unknown(
        intent["intent_id"], lambda _: {
            "status": "rejected", "exchange_order_id": "ex-uncertain",
        }
    )
    assert result["state"] == OrderState.UNKNOWN.value
    assert result["filled_quantity"] == "0.25"
    ledger.close()



class _FencedSpyLedger:
    require_fenced_writes = True

    def __init__(self):
        self.calls = []

    def acquire_fenced_run_lease(self, *args, **kwargs):
        return 9

    def create_intent(self, **kwargs):
        self.calls.append(("create_intent", kwargs))
        return {"intent_id": "spy-intent", "state": OrderState.INTENT_CREATED.value}

    def get_intent(self, intent_id):
        return {"intent_id": intent_id, "reduce_only": False, "state": OrderState.INTENT_CREATED.value}

    def transition(self, intent_id, state, **kwargs):
        self.calls.append(("transition", kwargs))
        return {"intent_id": intent_id, "state": OrderState(state).value}


def test_coordinator_requires_and_propagates_fencing_context_for_shared_ledger():
    ledger = _FencedSpyLedger()
    breaker = CircuitBreaker()
    with pytest.raises(ValueError, match="Fenced PostgreSQL ledger requires"):
        ExecutionCoordinator(
            ledger, breaker, exchange="DERIBIT", environment="testnet", account_id="acct"
        )

    coordinator = ExecutionCoordinator(
        ledger, breaker, exchange="DERIBIT", environment="testnet", account_id="acct",
        lease_key="execution", lease_owner_id="run-123", fencing_token=9,
    )
    created = coordinator.create_intent(
        client_order_id="client", instrument_id="ETH_USDC-PERPETUAL",
        side="BUY", quantity="1", settlement_currency="USDC",
    )
    assert created["intent_id"] == "spy-intent"
    assert ledger.calls[-1][1]["fencing_token"] == 9
    coordinator.mark_submitting("spy-intent")
    assert ledger.calls[-1][1] == {
        "event_type": "SUBMISSION_STARTED",
        "lease_key": "execution", "lease_owner_id": "run-123", "fencing_token": 9,
    }


def test_protected_entry_allows_new_entry_with_resting_reduce_only_stop_and_tp(tmp_path):
    ledger, breaker, coordinator = make_coordinator(tmp_path)
    entry = create(coordinator, client_order_id="protected-entry")
    coordinator.mark_submitting(entry["intent_id"])
    coordinator.record_fill(
        entry["intent_id"], cumulative_filled_quantity="1",
        average_fill_price="2000", exchange_order_id="entry-ex-1",
    )
    coordinator.verify_protection(entry["intent_id"], protected_quantity="1")

    stop = create(coordinator, reduce_only=True, client_order_id="resting-stop")
    coordinator.mark_submitting(stop["intent_id"])
    coordinator.record_acknowledged(stop["intent_id"], exchange_order_id="stop-ex-1")
    tp = create(coordinator, reduce_only=True, client_order_id="resting-tp")
    coordinator.mark_submitting(tp["intent_id"])
    coordinator.record_acknowledged(tp["intent_id"], exchange_order_id="tp-ex-1")

    # Resting reduce-only orders cannot add exposure; they must not serialize
    # every unrelated entry once the existing position is fully protected.
    next_entry = create(coordinator, client_order_id="second-protected-entry")
    assert next_entry["state"] == OrderState.INTENT_CREATED.value
    ledger.close()
