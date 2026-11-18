"""Exchange-independent reliability primitives for CryptoBot AI.

This module deliberately contains no order-submission code. It provides the
contracts, policy decisions, persistent order-intent ledger, and risk sizing
primitives that exchange-specific adapters and the coordinator can share.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence


class ReliabilityError(RuntimeError):
    """Base class for explicit reliability failures."""


class SnapshotUnavailable(ReliabilityError):
    """An exchange read failed or was incomplete; it must not be treated as empty."""


class IdempotencyConflict(ReliabilityError):
    """A reused idempotency key refers to a different order intent."""


class InvalidTransition(ReliabilityError):
    """An order lifecycle transition is not allowed."""


class ProtectionInvariantError(ReliabilityError):
    """Verified protective quantity does not fully cover confirmed filled quantity."""


class UnresolvedExecutionState(ReliabilityError):
    """A prior order or filled position remains unresolved in the durable ledger."""


class FailurePolicy(str, Enum):
    RETRY = "RETRY"
    HOLD = "HOLD"
    DEGRADE = "DEGRADE"
    HALT = "HALT"
    FLATTEN = "FLATTEN"


class FailureClass(str, Enum):
    READ_TIMEOUT = "READ_TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    AUTHENTICATION = "AUTHENTICATION"
    CANDLE_GAP = "CANDLE_GAP"
    FORMING_CANDLE = "FORMING_CANDLE"
    DUPLICATE_OR_OUT_OF_ORDER_CANDLES = "DUPLICATE_OR_OUT_OF_ORDER_CANDLES"
    PRICE_SOURCE_DISAGREEMENT = "PRICE_SOURCE_DISAGREEMENT"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    INVALID_MARKET_DATA = "INVALID_MARKET_DATA"
    FEATURE_MISSING = "FEATURE_MISSING"
    MODEL_VERSION_DRIFT = "MODEL_VERSION_DRIFT"
    MODEL_CONTRACT = "MODEL_CONTRACT"
    INVALID_PROBABILITY = "INVALID_PROBABILITY"
    STALE_SIGNAL = "STALE_SIGNAL"
    CONCURRENT_RUN = "CONCURRENT_RUN"
    ORDER_OUTCOME_UNKNOWN = "ORDER_OUTCOME_UNKNOWN"
    ORDER_REJECTED = "ORDER_REJECTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    PROTECTION_MISSING = "PROTECTION_MISSING"
    STOP_VERIFICATION_FAILED = "STOP_VERIFICATION_FAILED"
    POSITION_SNAPSHOT_FAILED = "POSITION_SNAPSHOT_FAILED"
    STATE_CORRUPT = "STATE_CORRUPT"
    STATE_WRITE_CONFLICT = "STATE_WRITE_CONFLICT"
    SCHEDULE_MISSED = "SCHEDULE_MISSED"
    RUNNER_INTERRUPTED = "RUNNER_INTERRUPTED"
    DEPOSIT_WITHDRAWAL_DETECTED = "DEPOSIT_WITHDRAWAL_DETECTED"
    FEE_CURRENCY_UNKNOWN = "FEE_CURRENCY_UNKNOWN"
    FUNDING_RECONCILIATION_FAILED = "FUNDING_RECONCILIATION_FAILED"
    MARGIN_EMERGENCY = "MARGIN_EMERGENCY"
    FAILURE_BUDGET_EXCEEDED = "FAILURE_BUDGET_EXCEEDED"
    CLOCK_SKEW = "CLOCK_SKEW"


_FAILURE_POLICIES = {
    FailureClass.READ_TIMEOUT: FailurePolicy.RETRY,
    FailureClass.RATE_LIMIT: FailurePolicy.RETRY,
    FailureClass.AUTHENTICATION: FailurePolicy.HALT,
    FailureClass.CANDLE_GAP: FailurePolicy.HOLD,
    FailureClass.FORMING_CANDLE: FailurePolicy.HOLD,
    FailureClass.DUPLICATE_OR_OUT_OF_ORDER_CANDLES: FailurePolicy.HOLD,
    FailureClass.PRICE_SOURCE_DISAGREEMENT: FailurePolicy.HOLD,
    FailureClass.STALE_MARKET_DATA: FailurePolicy.HOLD,
    FailureClass.INVALID_MARKET_DATA: FailurePolicy.HOLD,
    FailureClass.FEATURE_MISSING: FailurePolicy.HALT,
    FailureClass.MODEL_VERSION_DRIFT: FailurePolicy.HALT,
    FailureClass.MODEL_CONTRACT: FailurePolicy.HALT,
    FailureClass.INVALID_PROBABILITY: FailurePolicy.HALT,
    FailureClass.STALE_SIGNAL: FailurePolicy.HOLD,
    FailureClass.CONCURRENT_RUN: FailurePolicy.HOLD,
    FailureClass.ORDER_OUTCOME_UNKNOWN: FailurePolicy.DEGRADE,
    FailureClass.ORDER_REJECTED: FailurePolicy.HOLD,
    FailureClass.PARTIAL_FILL: FailurePolicy.DEGRADE,
    FailureClass.PROTECTION_MISSING: FailurePolicy.FLATTEN,
    FailureClass.STOP_VERIFICATION_FAILED: FailurePolicy.FLATTEN,
    FailureClass.POSITION_SNAPSHOT_FAILED: FailurePolicy.DEGRADE,
    FailureClass.STATE_CORRUPT: FailurePolicy.HALT,
    FailureClass.STATE_WRITE_CONFLICT: FailurePolicy.HOLD,
    FailureClass.SCHEDULE_MISSED: FailurePolicy.HOLD,
    FailureClass.RUNNER_INTERRUPTED: FailurePolicy.HOLD,
    FailureClass.DEPOSIT_WITHDRAWAL_DETECTED: FailurePolicy.HOLD,
    FailureClass.FEE_CURRENCY_UNKNOWN: FailurePolicy.HALT,
    FailureClass.FUNDING_RECONCILIATION_FAILED: FailurePolicy.HOLD,
    FailureClass.MARGIN_EMERGENCY: FailurePolicy.FLATTEN,
    FailureClass.FAILURE_BUDGET_EXCEEDED: FailurePolicy.DEGRADE,
    FailureClass.CLOCK_SKEW: FailurePolicy.HALT,
}


def policy_for(failure: FailureClass) -> FailurePolicy:
    return _FAILURE_POLICIES[failure]


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    HALTED = "HALTED"


class CircuitBreaker:
    """Small circuit breaker with an irreversible-until-acknowledged order latch."""

    def __init__(self, failure_threshold: int = 3, reset_after_seconds: float = 30.0):
        if failure_threshold < 1 or reset_after_seconds <= 0:
            raise ValueError("Invalid circuit-breaker configuration")
        self.failure_threshold = int(failure_threshold)
        self.reset_after_seconds = float(reset_after_seconds)
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._state = BreakerState.CLOSED
        self._manual_ack_required = False
        self._lock = threading.RLock()

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._advance_time()
            return self._state

    @property
    def manual_ack_required(self) -> bool:
        with self._lock:
            return self._manual_ack_required

    def _advance_time(self) -> None:
        if (
            self._state == BreakerState.OPEN
            and not self._manual_ack_required
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_after_seconds
        ):
            self._state = BreakerState.HALF_OPEN

    def allow_read(self) -> bool:
        with self._lock:
            self._advance_time()
            return self._state in {BreakerState.CLOSED, BreakerState.HALF_OPEN}

    def allow_new_entries(self) -> bool:
        with self._lock:
            self._advance_time()
            return self._state == BreakerState.CLOSED and not self._manual_ack_required

    def allow_protective_actions(self) -> bool:
        """Allow verified reduce-only protection/closure unless manually HALTED.

        OPEN/HALF_OPEN breakers stop new exposure but should not prevent safe
        risk-reducing actions. An order-path failure latches HALTED and blocks
        all further mutations until an operator acknowledges and resets it.
        """
        with self._lock:
            self._advance_time()
            return self._state != BreakerState.HALTED and not self._manual_ack_required

    def record_success(self) -> None:
        with self._lock:
            if self._manual_ack_required or self._state == BreakerState.HALTED:
                return
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.CLOSED
                self._failures = 0
                self._opened_at = None
            elif self._state == BreakerState.CLOSED:
                self._failures = 0

    def record_failure(self, *, order_path: bool = False) -> None:
        with self._lock:
            if order_path:
                self._manual_ack_required = True
                self._state = BreakerState.HALTED
                self._opened_at = time.monotonic()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()

    def manual_reset(self, *, acknowledged_by: str) -> None:
        if not str(acknowledged_by or "").strip():
            raise ValueError("Manual reset requires an acknowledging operator identity")
        with self._lock:
            self._manual_ack_required = False
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._opened_at = None


def retry_read(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    deadline_seconds: float = 8.0,
    base_delay_seconds: float = 0.15,
    max_delay_seconds: float = 1.5,
    retry_if: Optional[Callable[[BaseException], bool]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Retry a read-only callable within a total deadline, then raise.

    Do not wrap an order placement or any side-effecting operation with this.
    The caller owns classification and must pass a retry predicate for errors
    that are not transient.
    """
    if attempts < 1 or deadline_seconds <= 0:
        raise ValueError("attempts and deadline_seconds must be positive")
    deadline = clock() + deadline_seconds
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if retry_if is not None and not retry_if(exc):
                raise
            if attempt + 1 >= attempts:
                raise
            remaining = deadline - clock()
            if remaining <= 0:
                raise
            exponential = min(max_delay_seconds, base_delay_seconds * (2 ** attempt))
            # Respect a server-provided Retry-After value when available. The
            # wait is still bounded by the caller's total deadline; a server
            # request longer than the remaining budget ends retries instead
            # of being shortened into an early retry.
            retry_after = getattr(exc, "retry_after_seconds", None)
            try:
                retry_after_value = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError, OverflowError):
                retry_after_value = None
            if retry_after_value is not None and math.isfinite(retry_after_value):
                requested_delay = max(exponential * (0.5 + 0.5 * random_value()), retry_after_value)
                if requested_delay > remaining:
                    raise last_error
                delay = requested_delay
            else:
                # Full-ish jitter, bounded by the remaining deadline.
                delay = min(remaining, exponential * (0.5 + 0.5 * random_value()))
            if delay > 0:
                sleeper(delay)
            if clock() >= deadline:
                raise last_error
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class RiskCapital:
    equity: Decimal
    currency: str
    available_funds: Decimal
    margin_mode: str
    source: str

    def __post_init__(self) -> None:
        equity = _decimal(self.equity, "equity")
        available = _decimal(self.available_funds, "available_funds")
        currency = str(self.currency or "").upper()
        mode = str(self.margin_mode or "").lower()
        if equity <= 0 or available < 0:
            raise ValueError("Risk equity must be positive and available funds non-negative")
        if not currency or len(currency) > 12:
            raise ValueError("A valid risk currency is required")
        if mode not in {"segregated", "cross_collateral", "portfolio_margin"}:
            raise ValueError("Unsupported margin mode")
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "available_funds", available)
        object.__setattr__(self, "currency", currency)


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} is missing or invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name} is invalid") from None
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class InstrumentSpec:
    exchange: str
    symbol: str
    instrument_id: str
    settlement_currency: str
    quantity_unit: str  # BASE or CONTRACTS
    contract_value_base: Decimal
    qty_step: Decimal
    min_qty: Decimal
    max_qty: Optional[Decimal]
    tick_size: Decimal
    min_notional: Decimal = Decimal("0")
    linear: bool = True

    def __post_init__(self) -> None:
        for key in ("exchange", "symbol", "instrument_id", "settlement_currency"):
            if not str(getattr(self, key) or "").strip():
                raise ValueError(f"{key} is required")
        unit = str(self.quantity_unit).upper()
        if unit not in {"BASE", "CONTRACTS"}:
            raise ValueError("quantity_unit must be BASE or CONTRACTS")
        object.__setattr__(self, "quantity_unit", unit)
        for key in ("contract_value_base", "qty_step", "min_qty", "tick_size", "min_notional"):
            value = _decimal(getattr(self, key), key)
            if key != "min_notional" and value <= 0:
                raise ValueError(f"{key} must be positive")
            if key == "min_notional" and value < 0:
                raise ValueError("min_notional cannot be negative")
            object.__setattr__(self, key, value)
        if self.max_qty is not None:
            max_qty = _decimal(self.max_qty, "max_qty")
            if max_qty < self.min_qty:
                raise ValueError("max_qty must be >= min_qty")
            object.__setattr__(self, "max_qty", max_qty)
        if not self.linear:
            raise ValueError("Inverse/non-linear contracts require a separate sizing model")


@dataclass(frozen=True)
class RiskLimits:
    risk_fraction: Decimal = Decimal("0.01")
    max_notional_fraction: Decimal = Decimal("0.05")
    tier_cap: Optional[Decimal] = None
    taker_fee_rate: Decimal = Decimal("0.0005")

    def __post_init__(self) -> None:
        for key in ("risk_fraction", "max_notional_fraction", "taker_fee_rate"):
            value = _decimal(getattr(self, key), key)
            if value < 0:
                raise ValueError(f"{key} cannot be negative")
            object.__setattr__(self, key, value)
        if not (Decimal("0") < self.risk_fraction <= Decimal("0.05")):
            raise ValueError("risk_fraction must be in (0, 0.05]")
        if not (Decimal("0") < self.max_notional_fraction <= Decimal("1")):
            raise ValueError("max_notional_fraction must be in (0, 1]")
        if self.tier_cap is not None:
            cap = _decimal(self.tier_cap, "tier_cap")
            if cap <= 0:
                raise ValueError("tier_cap must be positive")
            object.__setattr__(self, "tier_cap", cap)


@dataclass(frozen=True)
class SizeResult:
    accepted: bool
    quantity: Decimal
    notional: Decimal
    stop_risk: Decimal
    estimated_round_trip_fees: Decimal
    total_risk: Decimal
    risk_currency: str
    reason: str


def calculate_position_size(
    capital: RiskCapital,
    instrument: InstrumentSpec,
    *,
    entry_price: Any,
    stop_price: Any,
    limits: RiskLimits,
    risk_multiplier: Any = Decimal("1"),
) -> SizeResult:
    """Calculate linear-contract quantity, rejecting minimum-lot risk overruns.

    Risk and notional are expressed in the instrument settlement currency.
    No FX parity is assumed and no quantity is rounded upward to meet a lot.
    """
    if capital.currency != instrument.settlement_currency.upper():
        raise ValueError(
            f"Risk currency {capital.currency} does not match instrument settlement "
            f"{instrument.settlement_currency.upper()}"
        )
    entry = _decimal(entry_price, "entry_price")
    stop = _decimal(stop_price, "stop_price")
    multiplier = _decimal(risk_multiplier, "risk_multiplier")
    if entry <= 0 or stop <= 0 or multiplier <= 0 or entry == stop:
        raise ValueError("Entry, stop and risk multiplier must define valid geometry")
    if capital.available_funds <= 0:
        return SizeResult(False, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), capital.currency, "NO_AVAILABLE_MARGIN")

    sizing_equity = min(capital.equity, limits.tier_cap) if limits.tier_cap else capital.equity
    risk_budget = sizing_equity * limits.risk_fraction * multiplier
    # Do not assume leverage: bound notional by both the strategy ceiling and
    # exchange-reported available collateral.
    max_notional = min(
        sizing_equity * limits.max_notional_fraction,
        capital.available_funds,
    )
    move = abs(entry - stop)
    contract_value = instrument.contract_value_base
    fee_rate = limits.taker_fee_rate

    # Worst-case stop movement plus conservative entry+exit taker fees per unit.
    risk_per_unit = (move * contract_value) + (Decimal("2") * entry * contract_value * fee_rate)
    if risk_per_unit <= 0 or risk_budget <= 0:
        return SizeResult(False, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), capital.currency, "ZERO_RISK_BUDGET")

    qty_by_risk = risk_budget / risk_per_unit
    qty_by_notional = max_notional / (entry * contract_value)
    raw_qty = min(qty_by_risk, qty_by_notional)
    if instrument.max_qty is not None:
        raw_qty = min(raw_qty, instrument.max_qty)

    steps = (raw_qty / instrument.qty_step).to_integral_value(rounding=ROUND_DOWN)
    qty = steps * instrument.qty_step
    if qty < instrument.min_qty:
        return SizeResult(False, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), capital.currency, "BELOW_MINIMUM_WITHIN_RISK")

    notional = qty * contract_value * entry
    stop_risk = qty * contract_value * move
    fees = qty * contract_value * entry * fee_rate * Decimal("2")
    total_risk = stop_risk + fees

    if instrument.min_notional and notional < instrument.min_notional:
        return SizeResult(False, Decimal("0"), notional, stop_risk, fees, total_risk, capital.currency, "BELOW_MINIMUM_NOTIONAL")
    if total_risk > risk_budget or notional > max_notional:
        return SizeResult(False, Decimal("0"), notional, stop_risk, fees, total_risk, capital.currency, "RISK_OR_NOTIONAL_CAP")

    return SizeResult(True, qty, notional, stop_risk, fees, total_risk, capital.currency, "OK")


class OrderState(str, Enum):
    INTENT_CREATED = "INTENT_CREATED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
    RECONCILING = "RECONCILING"


_UNRESOLVED_ORDER_STATES = (
    "INTENT_CREATED", "SUBMITTING", "ACKNOWLEDGED",
    "PARTIALLY_FILLED", "UNKNOWN", "RECONCILING",
)


_ALLOWED_TRANSITIONS = {
    # UNKNOWN is allowed when exact exchange evidence proves an unexpected
    # same-label order exists even though the local claim record is incomplete.
    OrderState.INTENT_CREATED: {OrderState.SUBMITTING, OrderState.CANCELLED, OrderState.UNKNOWN},
    OrderState.SUBMITTING: {
        OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.REJECTED, OrderState.UNKNOWN,
    },
    OrderState.ACKNOWLEDGED: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    OrderState.UNKNOWN: {OrderState.RECONCILING},
    OrderState.RECONCILING: {
        OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.CANCELLED, OrderState.REJECTED, OrderState.UNKNOWN,
    },
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
}


class ExecutionLedger:
    """SQLite-backed per-exchange order-intent/event ledger.

    It records intent before submission. It deliberately does not submit orders.
    Use one database per durable deployment/environment and protect the file
    with a single-writer process lock or transactional database in multi-host use.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=10000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    exchange TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                    quantity TEXT NOT NULL,
                    settlement_currency TEXT NOT NULL,
                    reduce_only INTEGER NOT NULL CHECK(reduce_only IN (0,1)),
                    state TEXT NOT NULL,
                    exchange_order_id TEXT,
                    filled_quantity TEXT NOT NULL DEFAULT '0',
                    average_fill_price TEXT,
                    protected_quantity TEXT NOT NULL DEFAULT '0',
                    protection_state TEXT NOT NULL DEFAULT 'NOT_APPLICABLE',
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(exchange, environment, account_id, client_order_id)
                );
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    state TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_order_events_intent
                    ON order_events(intent_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_order_intents_recovery
                    ON order_intents(exchange, environment, state, updated_at);
                CREATE TABLE IF NOT EXISTS execution_mutations (
                    mutation_id TEXT PRIMARY KEY,
                    exchange TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_execution_mutations_scope
                    ON execution_mutations(exchange, environment, account_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_execution_mutations_unresolved
                    ON execution_mutations(exchange, environment, account_id, state)
                    WHERE state IN ('SUBMITTING','UNKNOWN');
                CREATE TABLE IF NOT EXISTS execution_mutation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mutation_id TEXT NOT NULL REFERENCES execution_mutations(mutation_id),
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_execution_mutation_events_mutation
                    ON execution_mutation_events(mutation_id, event_id);
                CREATE TABLE IF NOT EXISTS execution_protection_links (
                    parent_intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    protective_intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(parent_intent_id, protective_intent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_execution_protection_active
                    ON execution_protection_links(protective_intent_id, active);
                CREATE TABLE IF NOT EXISTS run_leases (
                    lease_key TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()

    @staticmethod
    def _json(payload: Optional[Mapping[str, Any]]) -> str:
        return json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":"), default=str)

    def create_intent(
        self,
        *,
        exchange: str,
        environment: str,
        account_id: str,
        client_order_id: str,
        instrument_id: str,
        side: str,
        quantity: Any,
        settlement_currency: str,
        reduce_only: bool = False,
        payload: Optional[Mapping[str, Any]] = None,
        enforce_unresolved_gate: bool = False,
    ) -> dict:
        values = {
            "exchange": str(exchange).upper(),
            "environment": str(environment).lower(),
            "account_id": str(account_id),
            "client_order_id": str(client_order_id),
            "instrument_id": str(instrument_id),
            "side": str(side).upper(),
            "quantity": str(_decimal(quantity, "quantity")),
            "settlement_currency": str(settlement_currency).upper(),
            "reduce_only": bool(reduce_only),
        }
        if values["side"] not in {"BUY", "SELL"} or _decimal(values["quantity"], "quantity") <= 0:
            raise ValueError("Intent side or quantity is invalid")
        if not all(values[k] for k in ("exchange", "environment", "account_id", "client_order_id", "instrument_id", "settlement_currency")):
            raise ValueError("Intent identity fields must be non-empty")
        intent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(values[k] for k in ("exchange", "environment", "account_id", "client_order_id"))))
        payload_json = self._json(payload)
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                existing = cur.execute(
                    "SELECT intent_id FROM order_intents WHERE exchange=? AND environment=? AND account_id=? AND client_order_id=?",
                    tuple(values[key] for key in ("exchange", "environment", "account_id", "client_order_id")),
                ).fetchone()
                if enforce_unresolved_gate and not values["reduce_only"] and existing is None:
                    unresolved = cur.execute(
                        """SELECT intent_id FROM order_intents
                        WHERE exchange=? AND environment=? AND account_id=? AND reduce_only=0
                          AND (state IN ('INTENT_CREATED','SUBMITTING','ACKNOWLEDGED','PARTIALLY_FILLED','UNKNOWN','RECONCILING')
                               OR (CAST(filled_quantity AS NUMERIC) > 0
                                   AND (protection_state != 'PROTECTED'
                                        OR CAST(protected_quantity AS NUMERIC) < CAST(filled_quantity AS NUMERIC))))
                        ORDER BY created_at LIMIT 1""",
                        (values["exchange"], values["environment"], values["account_id"]),
                    ).fetchone()
                    if unresolved is not None:
                        raise UnresolvedExecutionState(
                            "New entry blocked by unresolved order/position intent " + str(unresolved["intent_id"])
                        )
                cur.execute(
                    """INSERT OR IGNORE INTO order_intents
                    (intent_id,exchange,environment,account_id,client_order_id,
                     instrument_id,side,quantity,settlement_currency,reduce_only,
                     state,payload_json,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (intent_id, values["exchange"], values["environment"], values["account_id"],
                     values["client_order_id"], values["instrument_id"], values["side"],
                     values["quantity"], values["settlement_currency"], int(values["reduce_only"]),
                     OrderState.INTENT_CREATED.value, payload_json, now, now),
                )
                inserted = cur.rowcount == 1
                row = cur.execute("SELECT * FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    raise ReliabilityError("Intent insert failed")
                for key, column in (("instrument_id", "instrument_id"), ("side", "side"), ("quantity", "quantity"), ("settlement_currency", "settlement_currency"), ("reduce_only", "reduce_only")):
                    expected = int(values[key]) if key == "reduce_only" else values[key]
                    if row[column] != expected:
                        raise IdempotencyConflict(f"Client order id reused with different {key}")
                if inserted:
                    cur.execute(
                        "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                        (intent_id, OrderState.INTENT_CREATED.value, "INTENT_CREATED", payload_json, now),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_intent(intent_id)

    def get_intent_by_client_order_id(
        self, *, exchange: str, environment: str, account_id: str, client_order_id: str
    ) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM order_intents WHERE exchange=? AND environment=?
                AND account_id=? AND client_order_id=?""",
                (str(exchange).upper(), str(environment).lower(), str(account_id), str(client_order_id)),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["reduce_only"] = bool(result["reduce_only"])
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def get_intent_by_exchange_order_id(self, exchange_order_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM order_intents WHERE exchange_order_id=? ORDER BY updated_at DESC LIMIT 1",
                (str(exchange_order_id),),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["reduce_only"] = bool(result["reduce_only"])
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def claim_submission(self, intent_id: str) -> bool:
        """Atomically claim the only right to send a previously created intent."""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute(
                    "UPDATE order_intents SET state=?, updated_at=? WHERE intent_id=? AND state=?",
                    (OrderState.SUBMITTING.value, now, str(intent_id), OrderState.INTENT_CREATED.value),
                )
                claimed = cur.rowcount == 1
                if claimed:
                    cur.execute(
                        "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                        (str(intent_id), OrderState.SUBMITTING.value, "SUBMISSION_CLAIMED", "{}", now),
                    )
                self._conn.commit()
                return claimed
            except Exception:
                self._conn.rollback()
                raise

    def link_protection_order(
        self, parent_intent_id: str, protective_intent_id: str,
        *, payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            parent = self._conn.execute("SELECT * FROM order_intents WHERE intent_id=?", (str(parent_intent_id),)).fetchone()
            protective = self._conn.execute("SELECT * FROM order_intents WHERE intent_id=?", (str(protective_intent_id),)).fetchone()
            if parent is None or protective is None:
                raise KeyError("parent or protective intent missing")
            if not bool(protective["reduce_only"]) or parent["instrument_id"] != protective["instrument_id"]:
                raise ReliabilityError("Protective link must reference a same-instrument reduce-only order")
            self._conn.execute(
                """INSERT INTO execution_protection_links
                (parent_intent_id,protective_intent_id,active,payload_json,created_at,updated_at)
                VALUES(?,?,1,?,?,?) ON CONFLICT(parent_intent_id,protective_intent_id)
                DO UPDATE SET active=1,payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (str(parent_intent_id), str(protective_intent_id), self._json(payload), now, now),
            )
            self._conn.commit()

    def active_protection_parent_ids(self, protective_intent_id: Optional[str] = None) -> list[str]:
        sql = "SELECT DISTINCT parent_intent_id FROM execution_protection_links WHERE active=1"
        params: tuple = ()
        if protective_intent_id is not None:
            sql += " AND protective_intent_id=?"
            params = (str(protective_intent_id),)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [str(row["parent_intent_id"]) for row in rows]

    def deactivate_protection_links(self, protective_intent_id: Optional[str] = None) -> int:
        now = time.time()
        with self._lock:
            if protective_intent_id is None:
                cur = self._conn.execute(
                    "UPDATE execution_protection_links SET active=0,updated_at=? WHERE active=1", (now,)
                )
            else:
                cur = self._conn.execute(
                    "UPDATE execution_protection_links SET active=0,updated_at=? WHERE active=1 AND protective_intent_id=?",
                    (now, str(protective_intent_id)),
                )
            self._conn.commit()
            return int(cur.rowcount)

    def begin_mutation(
        self, *, mutation_id: str, exchange: str, environment: str, account_id: str,
        operation: str, resource_id: str, request_hash: str, payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        now = time.time()
        values = (str(mutation_id), str(exchange).upper(), str(environment).lower(), str(account_id),
                  str(operation), str(resource_id), str(request_hash), self._json(payload), "SUBMITTING", now, now)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                unresolved = cur.execute(
                    """SELECT mutation_id,state,operation,resource_id FROM execution_mutations
                    WHERE exchange=? AND environment=? AND account_id=?
                      AND state IN ('SUBMITTING','UNKNOWN')
                    ORDER BY created_at LIMIT 1""",
                    (str(exchange).upper(), str(environment).lower(), str(account_id)),
                ).fetchone()
                if unresolved is not None:
                    raise UnresolvedExecutionState(
                        "Prior exchange control mutation remains unresolved: "
                        + str(unresolved['mutation_id']) + " (" + str(unresolved['state']) + ")"
                    )
                cur.execute(
                    """INSERT INTO execution_mutations
                    (mutation_id,exchange,environment,account_id,operation,resource_id,request_hash,
                     payload_json,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (str(mutation_id), "SUBMISSION_STARTED", "SUBMITTING", self._json(payload), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_mutation(mutation_id)

    def finish_mutation(
        self, mutation_id: str, *, state: str, response_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        state = str(state).upper()
        if state not in {"ACKNOWLEDGED", "REJECTED", "UNKNOWN"}:
            raise ValueError("Invalid execution mutation terminal state")
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute(
                    "SELECT state,payload_json FROM execution_mutations WHERE mutation_id=?",
                    (str(mutation_id),),
                ).fetchone()
                if row is None:
                    raise KeyError(mutation_id)
                if row["state"] != "SUBMITTING":
                    raise ReliabilityError("Execution mutation is no longer submitting")
                cur.execute(
                    "UPDATE execution_mutations SET state=?,response_id=?,payload_json=?,updated_at=? WHERE mutation_id=?",
                    (state, response_id, self._json(payload), now, str(mutation_id)),
                )
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (str(mutation_id), "EXCHANGE_RESPONSE_RECORDED", state, self._json(payload), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_mutation(mutation_id)

    def get_mutation(self, mutation_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_mutations WHERE mutation_id=?", (str(mutation_id),)
            ).fetchone()
        if row is None:
            raise KeyError(mutation_id)
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def resolve_mutation(
        self, mutation_id: str, *, state: str, evidence: Mapping[str, Any], resolved_by: str,
    ) -> dict:
        state = str(state).upper()
        if state not in {"ACKNOWLEDGED", "REJECTED"}:
            raise ValueError("A reconciled mutation must resolve to ACKNOWLEDGED or REJECTED")
        if not isinstance(evidence, Mapping) or evidence.get("verified") is not True:
            raise ValueError("Read-only reconciliation evidence must explicitly set verified=True")
        required = ("verification_method", "observed_status", "observed_at", "resource_id")
        if any(not str(evidence.get(key) or "").strip() for key in required):
            raise ValueError("Reconciliation evidence requires method, status, timestamp and resource identity")
        operator = str(resolved_by or "").strip()
        if not operator:
            raise ValueError("resolved_by operator identity is required")
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute(
                    "SELECT state,payload_json,resource_id FROM execution_mutations WHERE mutation_id=?",
                    (str(mutation_id),),
                ).fetchone()
                if row is None:
                    raise KeyError(mutation_id)
                if row["state"] != "UNKNOWN":
                    raise ReliabilityError("Only UNKNOWN control mutations require explicit reconciliation")
                if str(evidence.get("resource_id")) != str(row["resource_id"]):
                    raise ReliabilityError("Reconciliation evidence resource does not match the mutation")
                payload = json.loads(row["payload_json"])
                payload["reconciliation"] = dict(evidence)
                payload["reconciled_by"] = operator
                cur.execute(
                    "UPDATE execution_mutations SET state=?,payload_json=?,updated_at=? WHERE mutation_id=? AND state='UNKNOWN'",
                    (state, self._json(payload), now, str(mutation_id)),
                )
                if cur.rowcount != 1:
                    raise ReliabilityError("Control mutation reconciliation lost its state race")
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (str(mutation_id), "READ_ONLY_RECONCILIATION_CONFIRMED", state,
                     self._json({"reconciliation": dict(evidence), "resolved_by": operator}), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_mutation(mutation_id)

    def mutation_events(self, mutation_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM execution_mutation_events WHERE mutation_id=? ORDER BY event_id",
                (str(mutation_id),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def unresolved_mutations(
        self, *, exchange: Optional[str] = None, environment: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> list[dict]:
        clauses = ["state IN ('SUBMITTING','UNKNOWN')"]
        params = []
        for column, value, normalizer in (
            ("exchange", exchange, str.upper),
            ("environment", environment, str.lower),
            ("account_id", account_id, str),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(normalizer(str(value)))
        sql = "SELECT * FROM execution_mutations WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def get_intent(self, intent_id: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT * FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
            if row is None:
                raise KeyError(intent_id)
            result = dict(row)
            result["payload"] = json.loads(result.pop("payload_json"))
            return result

    def transition(
        self,
        intent_id: str,
        new_state: OrderState,
        *,
        event_type: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        if not isinstance(new_state, OrderState):
            new_state = OrderState(str(new_state))
        now = time.time()
        payload_json = self._json(payload)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute("SELECT state FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                current = OrderState(row["state"])
                if new_state == current:
                    self._conn.commit()
                    return self.get_intent(intent_id)
                if new_state not in _ALLOWED_TRANSITIONS[current]:
                    raise InvalidTransition(f"{current.value} -> {new_state.value} is not allowed")
                cur.execute(
                    "UPDATE order_intents SET state=?, exchange_order_id=COALESCE(?,exchange_order_id), updated_at=? WHERE intent_id=?",
                    (new_state.value, exchange_order_id, now, intent_id),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (intent_id, new_state.value, event_type or new_state.value, payload_json, now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_intent(intent_id)

    def record_fill(
        self,
        intent_id: str,
        *,
        cumulative_filled_quantity: Any,
        average_fill_price: Any,
        exchange_order_id: Optional[str] = None,
        fee: Optional[Any] = None,
        fee_currency: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        filled = _decimal(cumulative_filled_quantity, "cumulative_filled_quantity")
        price = _decimal(average_fill_price, "average_fill_price")
        if filled < 0 or (filled > 0 and price <= 0):
            raise ValueError("Invalid fill quantity or price")
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute("SELECT * FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                requested = Decimal(row["quantity"])
                previous = Decimal(row["filled_quantity"])
                state = OrderState(row["state"])
                if filled < previous or filled > requested:
                    raise ValueError("Cumulative filled quantity must be monotonic and <= requested quantity")
                if state not in {OrderState.SUBMITTING, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.UNKNOWN, OrderState.RECONCILING}:
                    raise InvalidTransition(f"Cannot record a fill while order state is {state.value}")
                new_state = OrderState.FILLED if filled == requested else OrderState.PARTIALLY_FILLED
                if filled == 0:
                    raise ValueError("A fill event must have positive cumulative filled quantity")
                if new_state not in _ALLOWED_TRANSITIONS[state] and new_state != state:
                    raise InvalidTransition(f"{state.value} -> {new_state.value} is not allowed")
                if new_state == state and filled == previous:
                    self._conn.commit()
                    return self.get_intent(intent_id)
                merged = dict(payload or {})
                if fee is not None:
                    if not str(fee_currency or "").strip():
                        raise ValueError("fee_currency is required when a fee is recorded")
                    merged["fee"] = str(_decimal(fee, "fee"))
                    merged["fee_currency"] = str(fee_currency).upper()
                cur.execute(
                    """UPDATE order_intents SET state=?, exchange_order_id=COALESCE(?,exchange_order_id),
                    filled_quantity=?, average_fill_price=?, protection_state='REQUIRED', updated_at=?
                    WHERE intent_id=?""",
                    (new_state.value, exchange_order_id, str(filled), str(price), now, intent_id),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (intent_id, new_state.value, "FILL_UPDATE", self._json(merged), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_intent(intent_id)

    def mark_protected(self, intent_id: str, protected_quantity: Any, *, payload: Optional[Mapping[str, Any]] = None) -> dict:
        protected = _decimal(protected_quantity, "protected_quantity")
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute("SELECT filled_quantity,state FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                filled = Decimal(row["filled_quantity"])
                if filled <= 0 or protected < filled:
                    raise ProtectionInvariantError(
                        f"Protective quantity {protected} does not cover filled quantity {filled}"
                    )
                if OrderState(row["state"]) not in {
                    OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED
                }:
                    raise InvalidTransition("Protection cannot be confirmed before an exchange fill")
                cur.execute(
                    "UPDATE order_intents SET protected_quantity=?, protection_state='PROTECTED', updated_at=? WHERE intent_id=?",
                    (str(protected), now, intent_id),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (intent_id, row["state"], "PROTECTION_VERIFIED", self._json(payload), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_intent(intent_id)

    def mark_unprotected(self, intent_id: str, *, payload: Optional[Mapping[str, Any]] = None) -> dict:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT filled_quantity FROM order_intents WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                if Decimal(row["filled_quantity"]) <= 0:
                    raise ProtectionInvariantError("No confirmed filled quantity to protect")
                self._conn.execute(
                    "UPDATE order_intents SET protection_state='UNPROTECTED', updated_at=? WHERE intent_id=?",
                    (now, intent_id),
                )
                self._conn.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (intent_id, "UNPROTECTED", "PROTECTION_FAILURE", self._json(payload), now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_intent(intent_id)

    def unresolved_intents(
        self, *, exchange: str, environment: str, account_id: str
    ) -> list[dict]:
        """Return order intents or filled positions that still block new entries."""
        query = """SELECT * FROM order_intents
            WHERE exchange=? AND environment=? AND account_id=?
              AND (state IN ('INTENT_CREATED','SUBMITTING','ACKNOWLEDGED','PARTIALLY_FILLED','UNKNOWN','RECONCILING')
                   OR (CAST(filled_quantity AS NUMERIC) > 0
                       AND (protection_state != 'PROTECTED'
                            OR CAST(protected_quantity AS NUMERIC) < CAST(filled_quantity AS NUMERIC))))
            ORDER BY created_at"""
        with self._lock:
            rows = self._conn.execute(
                query, (str(exchange).upper(), str(environment).lower(), str(account_id))
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            results.append(item)
        return results

    def has_unresolved_intents(
        self, *, exchange: str, environment: str, account_id: str
    ) -> bool:
        return bool(self.unresolved_intents(
            exchange=exchange, environment=environment, account_id=account_id
        ))

    def pending_reconciliation(self, *, exchange: Optional[str] = None, environment: Optional[str] = None) -> list[dict]:
        clauses = ["state IN ('UNKNOWN','RECONCILING')"]
        params: list[Any] = []
        if exchange:
            clauses.append("exchange=?")
            params.append(exchange.upper())
        if environment:
            clauses.append("environment=?")
            params.append(environment.lower())
        sql = "SELECT * FROM order_intents WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def acquire_run_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 120.0
    ) -> bool:
        """Acquire an atomic same-database lease; expires after ttl_seconds.

        This coordinates processes sharing the same durable SQLite database.
        It is not a cross-host/GitHub-runner distributed lock.
        """
        lease_key = str(lease_key or "").strip()
        owner_id = str(owner_id or "").strip()
        if not lease_key or not owner_id or ttl_seconds <= 0:
            raise ValueError("lease_key, owner_id and positive ttl_seconds are required")
        now = time.time()
        expires = now + float(ttl_seconds)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute(
                    "SELECT owner_id, expires_at FROM run_leases WHERE lease_key=?",
                    (lease_key,),
                ).fetchone()
                if row is not None and float(row["expires_at"]) > now and row["owner_id"] != owner_id:
                    self._conn.rollback()
                    return False
                cur.execute(
                    "INSERT INTO run_leases(lease_key,owner_id,expires_at,updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(lease_key) DO UPDATE SET owner_id=excluded.owner_id, "
                    "expires_at=excluded.expires_at, updated_at=excluded.updated_at",
                    (lease_key, owner_id, expires, now),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def renew_run_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 120.0
    ) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE run_leases SET expires_at=?, updated_at=? "
                "WHERE lease_key=? AND owner_id=? AND expires_at>?",
                (now + float(ttl_seconds), now, str(lease_key), str(owner_id), now),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def release_run_lease(self, lease_key: str, owner_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM run_leases WHERE lease_key=? AND owner_id=?",
                (str(lease_key), str(owner_id)),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def events(self, intent_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM order_events WHERE intent_id=? ORDER BY event_id", (intent_id,)
            ).fetchall()
        return [dict(row) for row in rows]


class ReadOnlyExchangeAdapter(Protocol):
    exchange: str
    environment: str

    def read_account(self) -> dict: ...
    def read_positions(self) -> list[dict]: ...
    def read_open_orders(self, instrument_id: Optional[str] = None) -> list[dict]: ...
    def read_recent_fills(self, instrument_id: str, *, limit: int = 50) -> list[dict]: ...
