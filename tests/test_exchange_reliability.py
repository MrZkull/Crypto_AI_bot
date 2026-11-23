from decimal import Decimal
from pathlib import Path

import pytest

from exchange_reliability import (
    BreakerState,
    CircuitBreaker,
    ExecutionLedger,
    FailureClass,
    FailurePolicy,
    IdempotencyConflict,
    InstrumentSpec,
    InvalidTransition,
    OrderState,
    ProtectionInvariantError,
    RiskCapital,
    RiskLimits,
    calculate_position_size,
    policy_for,
    retry_read,
    round_price_to_tick,
)



def test_available_funds_caps_notional_without_assuming_leverage():
    capital = RiskCapital(
        equity=Decimal("1000"), currency="USDC",
        available_funds=Decimal("1"), margin_mode="segregated", source="test",
    )
    spec = InstrumentSpec(
        exchange="DERIBIT", symbol="ETHUSDT", instrument_id="ETH_USDC-PERPETUAL",
        settlement_currency="USDC", quantity_unit="BASE",
        contract_value_base=Decimal("1"), qty_step=Decimal("0.01"),
        min_qty=Decimal("0.01"), max_qty=Decimal("100"), tick_size=Decimal("0.05"),
    )
    result = calculate_position_size(
        capital, spec, entry_price=Decimal("100"), stop_price=Decimal("99"),
        limits=RiskLimits(risk_fraction=Decimal("0.05"), max_notional_fraction=Decimal("0.50")),
    )
    assert result.accepted
    assert result.notional <= Decimal("1")


def test_failure_policy_catalogue_is_explicit():
    assert policy_for(FailureClass.READ_TIMEOUT) is FailurePolicy.RETRY
    assert policy_for(FailureClass.AUTHENTICATION) is FailurePolicy.HALT
    assert policy_for(FailureClass.ORDER_OUTCOME_UNKNOWN) is FailurePolicy.DEGRADE
    assert policy_for(FailureClass.POSITION_SNAPSHOT_FAILED) is FailurePolicy.DEGRADE
    assert policy_for(FailureClass.PROTECTION_MISSING) is FailurePolicy.FLATTEN
    assert len(FailureClass) == len({item.value for item in FailureClass})


def test_retry_read_retries_then_returns_value():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("temporary")
        return "snapshot"

    result = retry_read(
        operation,
        attempts=3,
        deadline_seconds=5,
        sleeper=lambda _: None,
        random_value=lambda: 0.0,
    )
    assert result == "snapshot"
    assert calls["n"] == 3


def test_retry_read_does_not_retry_non_transient_error():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise PermissionError("not retryable")

    with pytest.raises(PermissionError):
        retry_read(
            operation,
            attempts=3,
            retry_if=lambda exc: isinstance(exc, TimeoutError),
            sleeper=lambda _: None,
        )
    assert calls["n"] == 1


def test_circuit_breaker_opens_and_read_probe_recovers():
    breaker = CircuitBreaker(failure_threshold=1, reset_after_seconds=0.01)
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow_new_entries()
    import time
    time.sleep(0.02)
    assert breaker.allow_read()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow_new_entries()


def test_order_path_failure_requires_manual_acknowledgement():
    breaker = CircuitBreaker(failure_threshold=1, reset_after_seconds=0.001)
    breaker.record_failure(order_path=True)
    import time
    time.sleep(0.01)
    assert breaker.state is BreakerState.HALTED
    assert breaker.manual_ack_required
    assert not breaker.allow_new_entries()
    with pytest.raises(ValueError):
        breaker.manual_reset(acknowledged_by=" ")
    breaker.manual_reset(acknowledged_by="operator-1")
    assert breaker.allow_new_entries()


def _spec(**overrides):
    values = {
        "exchange": "DERIBIT",
        "symbol": "ETHUSDT",
        "instrument_id": "ETH_USDC-PERPETUAL",
        "settlement_currency": "USDC",
        "quantity_unit": "BASE",
        "contract_value_base": Decimal("1"),
        "qty_step": Decimal("0.001"),
        "min_qty": Decimal("0.001"),
        "max_qty": Decimal("100"),
        "tick_size": Decimal("0.05"),
        "min_notional": Decimal("0"),
        "linear": True,
    }
    values.update(overrides)
    return InstrumentSpec(**values)


def _capital(**overrides):
    values = {
        "equity": Decimal("1000"),
        "currency": "USDC",
        "available_funds": Decimal("800"),
        "margin_mode": "segregated",
        "source": "unit-test",
    }
    values.update(overrides)
    return RiskCapital(**values)


def test_position_sizing_is_currency_explicit_and_respects_lot():
    result = calculate_position_size(
        _capital(),
        _spec(),
        entry_price="2000",
        stop_price="1980",
        limits=RiskLimits(
            risk_fraction=Decimal("0.01"),
            max_notional_fraction=Decimal("0.05"),
            taker_fee_rate=Decimal("0.0005"),
        ),
    )
    assert result.accepted
    assert result.risk_currency == "USDC"
    assert result.quantity > 0
    assert result.total_risk <= Decimal("10")


def test_position_sizing_rejects_currency_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        calculate_position_size(
            _capital(currency="USD"),
            _spec(),
            entry_price=2000,
            stop_price=1980,
            limits=RiskLimits(),
        )


def test_position_sizing_never_rounds_up_to_minimum_lot():
    spec = _spec(min_qty=Decimal("1"), qty_step=Decimal("1"))
    result = calculate_position_size(
        _capital(equity=Decimal("5"), available_funds=Decimal("5")),
        spec,
        entry_price=2000,
        stop_price=1900,
        limits=RiskLimits(
            risk_fraction=Decimal("0.005"),
            max_notional_fraction=Decimal("0.05"),
        ),
    )
    assert not result.accepted
    assert result.quantity == 0
    assert result.reason == "BELOW_MINIMUM_WITHIN_RISK"


def test_inverse_contracts_are_rejected_until_dedicated_sizing_exists():
    with pytest.raises(ValueError, match="Inverse/non-linear"):
        _spec(linear=False)


def _create_intent(ledger, client_order_id="client-1"):
    return ledger.create_intent(
        exchange="DERIBIT",
        environment="testnet",
        account_id="account-A",
        client_order_id=client_order_id,
        instrument_id="ETH_USDC-PERPETUAL",
        side="BUY",
        quantity="2.0",
        settlement_currency="USDC",
        payload={"signal_id": "sig-1"},
    )


def test_ledger_persists_intent_and_is_idempotent(tmp_path):
    db = tmp_path / "execution.sqlite"
    ledger = ExecutionLedger(db)
    first = _create_intent(ledger)
    again = _create_intent(ledger)
    assert first["intent_id"] == again["intent_id"]
    assert len(ledger.events(first["intent_id"])) == 1
    ledger.close()

    reopened = ExecutionLedger(db)
    persisted = reopened.get_intent(first["intent_id"])
    assert persisted["state"] == OrderState.INTENT_CREATED.value
    assert persisted["payload"]["signal_id"] == "sig-1"
    reopened.close()


def test_ledger_rejects_client_order_id_reuse_for_different_order(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    _create_intent(ledger)
    with pytest.raises(IdempotencyConflict):
        ledger.create_intent(
            exchange="DERIBIT",
            environment="testnet",
            account_id="account-A",
            client_order_id="client-1",
            instrument_id="BTC_USDC-PERPETUAL",
            side="BUY",
            quantity="2.0",
            settlement_currency="USDC",
        )
    ledger.close()


def test_ledger_rejects_invalid_lifecycle_transition(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    intent = _create_intent(ledger)
    with pytest.raises(InvalidTransition):
        ledger.transition(intent["intent_id"], OrderState.FILLED)
    ledger.close()


def test_ledger_records_unknown_outcome_without_retrying(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    intent = _create_intent(ledger)
    ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
    ledger.transition(intent["intent_id"], OrderState.UNKNOWN, payload={"reason": "timeout"})
    assert len(ledger.pending_reconciliation(exchange="DERIBIT", environment="testnet")) == 1
    with pytest.raises(InvalidTransition):
        ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
    ledger.transition(intent["intent_id"], OrderState.RECONCILING)
    ledger.transition(intent["intent_id"], OrderState.ACKNOWLEDGED, exchange_order_id="ex-123")
    assert not ledger.pending_reconciliation()
    ledger.close()


def test_ledger_allows_immediate_partial_fill_from_submitting(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    intent = _create_intent(ledger)
    ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
    row = ledger.record_fill(
        intent["intent_id"],
        cumulative_filled_quantity="0.75",
        average_fill_price="2000",
        exchange_order_id="ex-123",
        fee="0.2",
        fee_currency="USDC",
    )
    assert row["state"] == OrderState.PARTIALLY_FILLED.value
    assert row["filled_quantity"] == "0.75"
    assert row["protection_state"] == "REQUIRED"
    ledger.close()


def test_ledger_rejects_non_monotonic_or_oversized_fill(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    intent = _create_intent(ledger)
    ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
    ledger.record_fill(
        intent["intent_id"], cumulative_filled_quantity="1", average_fill_price="2000"
    )
    with pytest.raises(ValueError, match="monotonic"):
        ledger.record_fill(
            intent["intent_id"], cumulative_filled_quantity="0.5", average_fill_price="2000"
        )
    with pytest.raises(ValueError, match="monotonic"):
        ledger.record_fill(
            intent["intent_id"], cumulative_filled_quantity="3", average_fill_price="2000"
        )
    ledger.close()


def test_ledger_requires_protection_to_cover_confirmed_fills(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    intent = _create_intent(ledger)
    ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
    ledger.record_fill(
        intent["intent_id"], cumulative_filled_quantity="1.25", average_fill_price="2000"
    )
    with pytest.raises(ProtectionInvariantError):
        ledger.mark_protected(intent["intent_id"], "1.24")
    protected = ledger.mark_protected(intent["intent_id"], "1.25")
    assert protected["protection_state"] == "PROTECTED"
    assert protected["protected_quantity"] == "1.25"
    ledger.close()


def test_ledger_separates_exchange_environment_and_account(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    a = _create_intent(ledger)
    b = ledger.create_intent(
        exchange="DELTA",
        environment="testnet",
        account_id="account-A",
        client_order_id="client-1",
        instrument_id="12345",
        side="BUY",
        quantity="2.0",
        settlement_currency="USD",
    )
    c = ledger.create_intent(
        exchange="DERIBIT",
        environment="testnet",
        account_id="account-B",
        client_order_id="client-1",
        instrument_id="ETH_USDC-PERPETUAL",
        side="BUY",
        quantity="2.0",
        settlement_currency="USDC",
    )
    assert len({a["intent_id"], b["intent_id"], c["intent_id"]}) == 3
    ledger.close()


def test_sqlite_run_lease_prevents_overlapping_local_runs(tmp_path):
    ledger = ExecutionLedger(tmp_path / "leases.sqlite")
    assert ledger.acquire_run_lease("DERIBIT:testnet:scanner", "run-A", ttl_seconds=60)
    assert not ledger.acquire_run_lease("DERIBIT:testnet:scanner", "run-B", ttl_seconds=60)
    assert ledger.renew_run_lease("DERIBIT:testnet:scanner", "run-A", ttl_seconds=60)
    assert not ledger.release_run_lease("DERIBIT:testnet:scanner", "run-B")
    assert ledger.release_run_lease("DERIBIT:testnet:scanner", "run-A")
    assert ledger.acquire_run_lease("DERIBIT:testnet:scanner", "run-B", ttl_seconds=60)
    ledger.close()


def test_sqlite_run_lease_expires_and_can_be_reclaimed(tmp_path):
    ledger = ExecutionLedger(tmp_path / "leases.sqlite")
    assert ledger.acquire_run_lease("scanner", "run-A", ttl_seconds=0.01)
    import time
    time.sleep(0.02)
    assert ledger.acquire_run_lease("scanner", "run-B", ttl_seconds=60)
    ledger.close()


def test_retry_read_honors_retry_after_within_total_deadline():
    calls = {"n": 0}
    now = {"value": 0.0}
    waits = []

    class RateLimitedError(RuntimeError):
        retry_after_seconds = 0.6

    def operation():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitedError("429")
        return "recovered"

    def sleeper(delay):
        waits.append(delay)
        now["value"] += delay

    result = retry_read(
        operation, attempts=3, deadline_seconds=2.0,
        base_delay_seconds=0.1, random_value=lambda: 0.0,
        sleeper=sleeper, clock=lambda: now["value"],
    )

    assert result == "recovered"
    assert calls["n"] == 2
    assert waits == [0.6]


def test_retry_read_does_not_shorten_retry_after_beyond_deadline():
    calls = {"n": 0}
    waits = []

    class RateLimitedError(RuntimeError):
        retry_after_seconds = 5.0

    def operation():
        calls["n"] += 1
        raise RateLimitedError("429")

    with pytest.raises(RateLimitedError):
        retry_read(
            operation, attempts=3, deadline_seconds=1.0,
            base_delay_seconds=0.1, random_value=lambda: 0.0,
            sleeper=waits.append, clock=lambda: 0.0,
        )

    assert calls["n"] == 1
    assert waits == []


def test_durable_risk_snapshots_are_normalized_and_select_first_daily_window(tmp_path):
    ledger = ExecutionLedger(tmp_path / "daily-risk.sqlite")
    try:
        ledger.record_account_risk_snapshot(
            exchange="deribit", environment="TESTNET", account_id="acct-A",
            currency="usdc", equity="1000", observed_at=100.0,
        )
        ledger.record_account_risk_snapshot(
            exchange="DERIBIT", environment="testnet", account_id="acct-A",
            currency="USDC", equity="1010", observed_at=110.0,
        )
        row = ledger.first_account_risk_snapshot_between(
            exchange="DERIBIT", environment="testnet", account_id="acct-A",
            currency="USDC", start=90.0, end=120.0,
        )
        assert row["equity"] == "1000"
        assert row["observed_at"] == 100.0
    finally:
        ledger.close()


def test_daily_risk_halt_is_sticky_in_sqlite_ledger(tmp_path):
    ledger = ExecutionLedger(tmp_path / "daily-risk-halt.sqlite")
    try:
        first = ledger.latch_daily_risk_halt(
            exchange="DERIBIT", environment="testnet", account_id="acct-A",
            utc_day="2026-11-18", reason="MAX_DAILY_LOSS_LIMIT",
            baseline_equity="1000", current_equity="980", loss_fraction="0.02",
            observed_at=100.0,
        )
        second = ledger.latch_daily_risk_halt(
            exchange="DERIBIT", environment="testnet", account_id="acct-A",
            utc_day="2026-11-18", reason="SHOULD_NOT_OVERWRITE",
            baseline_equity="1000", current_equity="900", loss_fraction="0.10",
            observed_at=200.0,
        )
        assert first["reason"] == "MAX_DAILY_LOSS_LIMIT"
        assert second["reason"] == "MAX_DAILY_LOSS_LIMIT"
        assert ledger.get_daily_risk_halt(
            exchange="DERIBIT", environment="testnet", account_id="acct-A",
            utc_day="2026-11-18",
        )["reason"] == "MAX_DAILY_LOSS_LIMIT"
    finally:
        ledger.close()


def test_tick_rounding_is_directional_and_decimal_exact():
    assert round_price_to_tick("100.123", "0.05", rounding="DOWN") == Decimal("100.10")
    assert round_price_to_tick("100.123", "0.05", rounding="UP") == Decimal("100.15")


def test_short_position_fees_use_the_higher_stop_exit_notional():
    result = calculate_position_size(
        _capital(), _spec(), entry_price="100", stop_price="110",
        limits=RiskLimits(
            risk_fraction=Decimal("0.01"), max_notional_fraction=Decimal("0.05"),
            taker_fee_rate=Decimal("0.001"),
        ),
        side="SELL",
    )
    assert result.accepted
    expected = result.quantity * (Decimal("100") + Decimal("110")) * Decimal("0.001")
    assert result.estimated_round_trip_fees == expected


def test_available_funds_reserve_haircut_reduces_maximum_notional():
    no_reserve = calculate_position_size(
        _capital(), _spec(), entry_price="100", stop_price="99",
        limits=RiskLimits(
            risk_fraction=Decimal("0.05"), max_notional_fraction=Decimal("1"),
            available_funds_reserve_fraction=Decimal("0"),
        ),
    )
    with_reserve = calculate_position_size(
        _capital(), _spec(), entry_price="100", stop_price="99",
        limits=RiskLimits(
            risk_fraction=Decimal("0.05"), max_notional_fraction=Decimal("1"),
            available_funds_reserve_fraction=Decimal("0.20"),
        ),
    )
    assert no_reserve.accepted and with_reserve.accepted
    assert with_reserve.notional <= no_reserve.notional
    assert with_reserve.notional <= Decimal("640")


def test_tier_equity_cap_limits_risk_budget_even_with_higher_live_equity():
    result = calculate_position_size(
        _capital(equity=Decimal("10000")), _spec(),
        entry_price="100", stop_price="90",
        limits=RiskLimits(
            risk_fraction=Decimal("0.01"), max_notional_fraction=Decimal("0.05"),
            tier_cap=Decimal("1000"),
        ),
    )
    assert result.accepted
    assert result.total_risk <= Decimal("10")
