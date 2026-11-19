"""Persistent exchange-order lifecycle coordinator (no order submission).

This module intentionally exposes only intent recording, acknowledgement,
fill/protection recording, and read-only reconciliation. Live submit/cancel
capabilities are not implemented here; exchange adapters remain read-only.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional

from exchange_reliability import (
    BreakerState,
    CircuitBreaker,
    ExecutionLedger,
    InvalidTransition,
    OrderState,
    ProtectionInvariantError,
    UnresolvedExecutionState,
)


class CoordinatorError(RuntimeError):
    pass


class ExecutionCoordinator:
    """Coordinate durable lifecycle events for a single venue/account scope."""

    def __init__(
        self,
        ledger: ExecutionLedger,
        breaker: CircuitBreaker,
        *,
        exchange: str,
        environment: str,
        account_id: str,
        lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ):
        if not exchange or not environment or not account_id:
            raise ValueError("Exchange, environment and account identity are required")
        self.ledger = ledger
        self.breaker = breaker
        self.exchange = str(exchange).upper()
        self.environment = str(environment).lower()
        self.account_id = str(account_id)
        self.lease_key = str(lease_key or "").strip() or None
        self.lease_owner_id = str(lease_owner_id or "").strip() or None
        self.fencing_token = int(fencing_token) if fencing_token is not None else None
        if getattr(ledger, "require_fenced_writes", False):
            if not self.lease_key or not self.lease_owner_id or not self.fencing_token or self.fencing_token < 1:
                raise ValueError("Fenced PostgreSQL ledger requires lease_key, lease_owner_id and fencing_token")

    def _lease_kwargs(self) -> dict:
        if not hasattr(self.ledger, "acquire_fenced_run_lease"):
            return {}
        if not self.lease_key or not self.lease_owner_id or not self.fencing_token:
            return {}
        return {
            "lease_key": self.lease_key,
            "lease_owner_id": self.lease_owner_id,
            "fencing_token": self.fencing_token,
        }

    def create_intent(
        self,
        *,
        client_order_id: str,
        instrument_id: str,
        side: str,
        quantity: Any,
        settlement_currency: str,
        reduce_only: bool = False,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        if not self.breaker.allow_new_entries() and not reduce_only:
            raise CoordinatorError("Circuit breaker blocks new order intents")
        try:
            return self.ledger.create_intent(
                exchange=self.exchange,
                environment=self.environment,
                account_id=self.account_id,
                client_order_id=client_order_id,
                instrument_id=instrument_id,
                side=side,
                quantity=quantity,
                settlement_currency=settlement_currency,
                reduce_only=reduce_only,
                payload=payload,
                enforce_unresolved_gate=not reduce_only,
                **self._lease_kwargs(),
            )
        except UnresolvedExecutionState as exc:
            raise CoordinatorError(str(exc)) from exc

    def mark_submitting(self, intent_id: str) -> dict:
        intent = self.ledger.get_intent(intent_id)
        is_reduce_only = bool(intent.get("reduce_only"))
        allowed = (
            self.breaker.allow_protective_actions()
            if is_reduce_only
            else self.breaker.allow_new_entries()
        )
        if not allowed:
            raise CoordinatorError(
                "Circuit breaker blocks this submission lifecycle"
            )
        return self.ledger.transition(
            intent_id, OrderState.SUBMITTING, event_type="SUBMISSION_STARTED",
            **self._lease_kwargs(),
        )

    def claim_submission(self, intent_id: str) -> bool:
        """CAS-claim an intent so concurrent callers cannot submit it twice."""
        intent = self.ledger.get_intent(intent_id)
        is_reduce_only = bool(intent.get("reduce_only"))
        allowed = self.breaker.allow_protective_actions() if is_reduce_only else self.breaker.allow_new_entries()
        if not allowed:
            raise CoordinatorError("Circuit breaker blocks this submission lifecycle")
        claim = getattr(self.ledger, "claim_submission", None)
        if not callable(claim):
            raise CoordinatorError("Execution ledger does not support atomic submission claims")
        return bool(claim(intent_id, **self._lease_kwargs()))

    def record_rejected(self, intent_id: str, *, reason: str, exchange_order_id: Optional[str] = None) -> dict:
        return self.ledger.transition(
            intent_id, OrderState.REJECTED, event_type="EXCHANGE_REJECTED",
            exchange_order_id=exchange_order_id, payload={"reason": str(reason)},
            **self._lease_kwargs(),
        )

    def record_acknowledged(
        self, intent_id: str, *, exchange_order_id: str, payload: Optional[Mapping[str, Any]] = None
    ) -> dict:
        if not exchange_order_id:
            raise ValueError("Exchange order ID is required for acknowledgement")
        result = self.ledger.transition(
            intent_id,
            OrderState.ACKNOWLEDGED,
            exchange_order_id=str(exchange_order_id),
            event_type="EXCHANGE_ACKNOWLEDGED",
            payload=payload,
            **self._lease_kwargs(),
        )
        self.breaker.record_success()
        return result

    def record_fill(
        self,
        intent_id: str,
        *,
        cumulative_filled_quantity: Any,
        average_fill_price: Any,
        exchange_order_id: Optional[str] = None,
        fee: Any = None,
        fee_currency: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        return self.ledger.record_fill(
            intent_id,
            cumulative_filled_quantity=cumulative_filled_quantity,
            average_fill_price=average_fill_price,
            exchange_order_id=exchange_order_id,
            fee=fee,
            fee_currency=fee_currency,
            payload=payload,
            **self._lease_kwargs(),
        )

    def mark_unknown(self, intent_id: str, *, reason: str) -> dict:
        result = self.ledger.transition(
            intent_id,
            OrderState.UNKNOWN,
            event_type="ORDER_OUTCOME_UNKNOWN",
            payload={"reason": str(reason)},
            **self._lease_kwargs(),
        )
        self.breaker.record_failure(order_path=True)
        return result

    def reconcile_unknown(
        self,
        intent_id: str,
        lookup: Callable[[dict], Optional[Mapping[str, Any]]],
    ) -> dict:
        """Resolve UNKNOWN via injected read-only lookup; never resubmit.

        Expected lookup payload: status=open|partially_filled|filled|cancelled|
        rejected, plus exchange_order_id and, for a fill, cumulative_filled_quantity
        and average_fill_price. None/exception/unknown state remains UNKNOWN.
        """
        current = self.ledger.get_intent(intent_id)
        if current["state"] != OrderState.UNKNOWN.value:
            raise InvalidTransition("Only UNKNOWN orders can be reconciled")
        self.ledger.transition(
            intent_id, OrderState.RECONCILING, event_type="READ_ONLY_RECONCILIATION_STARTED",
            **self._lease_kwargs(),
        )
        try:
            observed = lookup(current)
        except Exception as exc:
            observed = None
            error = type(exc).__name__
        else:
            error = "NO_MATCH_OR_AMBIGUOUS"

        if not isinstance(observed, Mapping):
            return self.ledger.transition(
                intent_id,
                OrderState.UNKNOWN,
                event_type="RECONCILIATION_UNRESOLVED",
                payload={"reason": error},
                **self._lease_kwargs(),
            )

        status = str(observed.get("status", "")).lower()
        exchange_order_id = observed.get("exchange_order_id")
        if status in {"open", "pending", "new", "acknowledged"}:
            if not exchange_order_id:
                return self.ledger.transition(
                    intent_id,
                    OrderState.UNKNOWN,
                    event_type="RECONCILIATION_MISSING_EXCHANGE_ORDER_ID",
                    **self._lease_kwargs(),
                )
            # Preserve a previously confirmed partial fill; never demote it to
            # a plain ACKNOWLEDGED state after an ambiguous submission.
            try:
                prior_filled = Decimal(str(current.get("filled_quantity", "0")))
            except (InvalidOperation, TypeError, ValueError):
                prior_filled = Decimal("-1")
            if prior_filled > 0:
                observed_filled = observed.get("cumulative_filled_quantity", current.get("filled_quantity"))
                observed_price = observed.get("average_fill_price", current.get("average_fill_price"))
                if observed_price is None:
                    return self.ledger.transition(
                        intent_id, OrderState.UNKNOWN,
                        event_type="RECONCILIATION_PRIOR_FILL_PRICE_MISSING",
                        **self._lease_kwargs(),
                    )
                return self.ledger.record_fill(
                    intent_id,
                    cumulative_filled_quantity=observed_filled,
                    average_fill_price=observed_price,
                    exchange_order_id=str(exchange_order_id),
                    payload={"reconciled_read_only": True, "order_still_open": True},
                    **self._lease_kwargs(),
                )
            return self.ledger.transition(
                intent_id,
                OrderState.ACKNOWLEDGED,
                exchange_order_id=str(exchange_order_id),
                event_type="RECONCILED_OPEN_ORDER",
                **self._lease_kwargs(),
            )
        if status in {"partially_filled", "partial", "filled"}:
            filled = observed.get("cumulative_filled_quantity")
            price = observed.get("average_fill_price")
            if not exchange_order_id or filled is None or price is None:
                return self.ledger.transition(
                    intent_id, OrderState.UNKNOWN, event_type="RECONCILIATION_MISSING_FILL_DATA",
                    **self._lease_kwargs(),
                )
            if status == "filled":
                try:
                    is_full = Decimal(str(filled)) == Decimal(str(current["quantity"]))
                except (InvalidOperation, TypeError, ValueError):
                    is_full = False
                if not is_full:
                    return self.ledger.transition(
                        intent_id, OrderState.UNKNOWN,
                        event_type="RECONCILIATION_FILLED_STATUS_QUANTITY_MISMATCH",
                        payload={"reported_cumulative_filled_quantity": str(filled)},
                        **self._lease_kwargs(),
                    )
            return self.ledger.record_fill(
                intent_id,
                cumulative_filled_quantity=filled,
                average_fill_price=price,
                exchange_order_id=str(exchange_order_id),
                payload={"reconciled_read_only": True},
                **self._lease_kwargs(),
            )
        if status in {"cancelled", "canceled", "expired"}:
            order_id = exchange_order_id or current.get("exchange_order_id")
            filled = observed.get("cumulative_filled_quantity", current.get("filled_quantity", "0"))
            price = observed.get("average_fill_price", current.get("average_fill_price"))
            try:
                filled_value = Decimal(str(filled))
            except (InvalidOperation, TypeError, ValueError):
                filled_value = Decimal("-1")
            if filled_value > 0:
                if not order_id or price is None:
                    return self.ledger.transition(
                        intent_id, OrderState.UNKNOWN,
                        event_type="RECONCILIATION_CANCELLED_PARTIAL_FILL_DATA_MISSING",
                        **self._lease_kwargs(),
                    )
                updated = self.ledger.record_fill(
                    intent_id,
                    cumulative_filled_quantity=filled,
                    average_fill_price=price,
                    exchange_order_id=str(order_id),
                    payload={"reconciled_read_only": True, "cancelled_with_fill": True},
                    **self._lease_kwargs(),
                )
                if updated["state"] == OrderState.PARTIALLY_FILLED.value:
                    return self.ledger.transition(
                        intent_id, OrderState.CANCELLED,
                        exchange_order_id=str(order_id),
                        event_type="RECONCILED_CANCELLED_WITH_PARTIAL_FILL",
                        **self._lease_kwargs(),
                    )
                return updated
            return self.ledger.transition(
                intent_id,
                OrderState.CANCELLED,
                exchange_order_id=str(order_id) if order_id else None,
                event_type="RECONCILED_CANCELLED",
                **self._lease_kwargs(),
            )
        if status in {"rejected", "failed"}:
            try:
                prior_filled = Decimal(str(current.get("filled_quantity", "0")))
            except (InvalidOperation, TypeError, ValueError):
                prior_filled = Decimal("-1")
            if prior_filled > 0:
                return self.ledger.transition(
                    intent_id, OrderState.UNKNOWN,
                    exchange_order_id=str(exchange_order_id) if exchange_order_id else None,
                    event_type="RECONCILIATION_REJECTED_WITH_PRIOR_FILL_CONFLICT",
                    payload={"prior_filled_quantity": str(prior_filled)},
                    **self._lease_kwargs(),
                )
            return self.ledger.transition(
                intent_id,
                OrderState.REJECTED,
                exchange_order_id=str(exchange_order_id) if exchange_order_id else None,
                event_type="RECONCILED_REJECTED",
                **self._lease_kwargs(),
            )
        return self.ledger.transition(
            intent_id,
            OrderState.UNKNOWN,
            event_type="RECONCILIATION_UNKNOWN_STATE",
            payload={"observed_status": status or None},
            **self._lease_kwargs(),
        )

    def verify_protection(
        self, intent_id: str, *, protected_quantity: Any, payload: Optional[Mapping[str, Any]] = None
    ) -> dict:
        try:
            return self.ledger.mark_protected(
                intent_id, protected_quantity, payload=payload,
                **self._lease_kwargs(),
            )
        except ProtectionInvariantError:
            self.ledger.mark_unprotected(
                intent_id,
                payload={"required_protection_quantity": self.ledger.get_intent(intent_id)["filled_quantity"]},
                **self._lease_kwargs(),
            )
            self.breaker.record_failure(order_path=True)
            raise

    @property
    def new_entries_allowed(self) -> bool:
        return self.breaker.allow_new_entries()

    @property
    def state(self) -> BreakerState:
        return self.breaker.state

    # Deliberately no submit_order(), place_order(), cancel_order(),
    # set_leverage() or change_margin_mode() methods in this coordinator.
