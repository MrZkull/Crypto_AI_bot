"""Persist-first, single-owner Deribit Testnet mutation gateway.

All private mutation RPCs are routed through this boundary when required by
DeribitClient. It does not switch the execution gate on and does not retry
unknown outcomes.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import uuid
import threading
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional

from execution_coordinator import ExecutionCoordinator
from exchange_reliability import CircuitBreaker, OrderState


class ExecutionGatewayError(RuntimeError):
    pass


class ExecutionGatewayBlocked(ExecutionGatewayError):
    """Mutation blocked before the exchange call."""


class ExecutionGatewayOutcomeUnknown(ExecutionGatewayError):
    """Exchange may have processed the mutation; reconcile and never retry blindly."""


class DuplicateExecutionIntent(ExecutionGatewayBlocked):
    pass


_ORDER_MUTATIONS = {"/private/buy", "/private/sell"}
_CONTROL_MUTATIONS = {"/private/cancel", "/private/cancel_all", "/private/edit", "/private/set_leverage"}
_CURRENCY_RE = re.compile(r"_([A-Z]{3,5})-PERPETUAL$")


def _decimal(value: Any, name: str, *, zero_ok: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} is missing or invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"{name} is invalid") from None
    if not result.is_finite() or result < 0 or (not zero_ok and result <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if zero_ok else 'positive'}")
    return result


def _currency(instrument: str) -> str:
    m = _CURRENCY_RE.search(str(instrument or "").strip().upper())
    if not m or m.group(1) != "USDC":
        raise ExecutionGatewayBlocked("Gateway currently permits only explicitly classified USDC perpetuals")
    return m.group(1)


def _avg_price(order: Mapping[str, Any], response: Mapping[str, Any]) -> Optional[Decimal]:
    for key in ("average_price", "average_fill_price"):
        if order.get(key) is not None:
            try:
                return _decimal(order[key], key)
            except ValueError:
                return None
    trades = response.get("trades")
    if isinstance(trades, list) and trades:
        total_qty = Decimal("0")
        total_value = Decimal("0")
        for row in trades:
            if not isinstance(row, Mapping):
                return None
            try:
                qty = _decimal(row.get("amount"), "trade.amount")
                price = _decimal(row.get("price"), "trade.price")
            except ValueError:
                return None
            total_qty += qty
            total_value += qty * price
        return total_value / total_qty if total_qty > 0 else None
    return None


def _response_id(response: Any) -> Optional[str]:
    if isinstance(response, Mapping):
        order = response.get("order", response)
        if isinstance(order, Mapping):
            value = order.get("order_id")
            return str(value).strip() if value is not None and str(value).strip() else None
    return None


class DeribitExecutionGateway:
    def __init__(
        self,
        coordinator: ExecutionCoordinator,
        *,
        lease_renew: Optional[Callable[[], bool]] = None,
        lease_release: Optional[Callable[[], bool]] = None,
        ledger_close: Optional[Callable[[], None]] = None,
        register_atexit: bool = False,
    ) -> None:
        if not isinstance(coordinator, ExecutionCoordinator):
            raise TypeError("A configured ExecutionCoordinator is required")
        self.coordinator = coordinator
        self.ledger = coordinator.ledger
        self.exchange = coordinator.exchange
        self.environment = coordinator.environment
        self.account_id = coordinator.account_id
        self._renew_cb = lease_renew
        self._release_cb = lease_release
        self._close_cb = ledger_close
        self._closed = False
        self._mutation_lock = threading.RLock()
        if str(self.exchange).upper() != "DERIBIT" or str(self.environment).lower() != "testnet":
            raise ExecutionGatewayBlocked("Execution gateway is restricted to Deribit Testnet")
        if register_atexit:
            atexit.register(self.close)

    def _lease_kwargs(self) -> dict:
        return self.coordinator._lease_kwargs()

    def _renew(self) -> None:
        if self._closed:
            raise ExecutionGatewayBlocked("Execution gateway is closed")
        if self._renew_cb is not None:
            try:
                valid = bool(self._renew_cb())
            except Exception:
                valid = False
            if not valid:
                raise ExecutionGatewayBlocked("Execution ownership lease is lost or expired")

    @staticmethod
    def _label(value: Any) -> str:
        label = str(value or "").strip()
        if not label or len(label) > 64 or any(ord(c) < 33 or ord(c) > 126 for c in label):
            raise ExecutionGatewayBlocked("A stable valid Deribit client label is required")
        return label

    def submit_order(self, path: str, body: Mapping[str, Any], submit: Callable[[], Any]) -> Any:
        if path not in _ORDER_MUTATIONS or not isinstance(body, Mapping):
            raise ExecutionGatewayBlocked("Invalid order mutation request")
        try:
            label = self._label(body.get("label"))
            instrument = str(body.get("instrument_name") or "").strip()
            settlement = _currency(instrument)
            side = "BUY" if path == "/private/buy" else "SELL"
            quantity = _decimal(body.get("amount"), "amount")
            reduce_only = body.get("reduce_only", False)
            if isinstance(reduce_only, str):
                token = reduce_only.strip().lower()
                if token not in {"true", "false", "1", "0"}:
                    raise ValueError("reduce_only must be a strict boolean")
                reduce_only = token in {"true", "1"}
            elif isinstance(reduce_only, int) and not isinstance(reduce_only, bool):
                if reduce_only not in {0, 1}:
                    raise ValueError("reduce_only must be a strict boolean")
                reduce_only = bool(reduce_only)
            elif not isinstance(reduce_only, bool):
                raise ValueError("reduce_only must be a strict boolean")
        except (TypeError, ValueError) as exc:
            raise ExecutionGatewayBlocked(str(exc)) from None

        self._renew()
        existing = self.ledger.get_intent_by_client_order_id(
            exchange=self.exchange, environment=self.environment,
            account_id=self.account_id, client_order_id=label,
        )
        if existing is not None:
            raise DuplicateExecutionIntent("Client label already exists in execution ledger; reconcile before another submit")

        try:
            intent = self.coordinator.create_intent(
                client_order_id=label,
                instrument_id=instrument,
                side=side,
                quantity=quantity,
                settlement_currency=settlement,
                reduce_only=reduce_only,
                payload={
                    "rpc_method": path,
                    "order_type": str(body.get("type") or ""),
                    "price": body.get("price"),
                    "trigger_price": body.get("trigger_price"),
                    "trigger": body.get("trigger"),
                    "time_in_force": body.get("time_in_force"),
                    "label": label,
                },
            )
            self._renew()
            if not self.coordinator.claim_submission(intent["intent_id"]):
                raise DuplicateExecutionIntent("Order intent was already claimed; no second request sent")
        except ExecutionGatewayError:
            raise
        except Exception as exc:
            raise ExecutionGatewayBlocked(
                f"Intent persistence/admission failed ({type(exc).__name__}); no exchange request sent"
            ) from None

        try:
            response = submit()
        except Exception as exc:
            if type(exc).__name__ == "DeribitRPCError":
                try:
                    self.coordinator.record_rejected(
                        intent["intent_id"], reason=f"DeribitRPCError:{getattr(exc, 'code', 'unknown')}"
                    )
                except Exception:
                    self.coordinator.breaker.record_failure(order_path=True)
                    raise ExecutionGatewayOutcomeUnknown("Known rejection could not be durably recorded") from None
                raise
            try:
                self.coordinator.mark_unknown(
                    intent["intent_id"], reason=f"{type(exc).__name__}:transport_or_response_ambiguous"
                )
            except Exception:
                self.coordinator.breaker.record_failure(order_path=True)
            raise ExecutionGatewayOutcomeUnknown(
                "Order outcome unknown; reconcile exact label, instrument, orders, trades and positions; never retry blindly"
            ) from None

        try:
            self._renew()
            self._record_order_response(intent["intent_id"], response, quantity)
        except ExecutionGatewayOutcomeUnknown:
            raise
        except Exception as exc:
            try:
                current = self.ledger.get_intent(intent["intent_id"])
                if current.get("state") not in {OrderState.UNKNOWN.value, OrderState.REJECTED.value}:
                    self.coordinator.mark_unknown(
                        intent["intent_id"], reason=f"acknowledgement_commit_failed:{type(exc).__name__}"
                    )
            except Exception:
                self.coordinator.breaker.record_failure(order_path=True)
            raise ExecutionGatewayOutcomeUnknown(
                "Exchange response received but lifecycle record could not be committed; reconcile before another mutation"
            ) from None
        return response

    def _record_order_response(self, intent_id: str, response: Any, requested: Decimal) -> None:
        if not isinstance(response, Mapping):
            self.coordinator.mark_unknown(intent_id, reason="malformed_response")
            raise ExecutionGatewayOutcomeUnknown("Malformed order response")
        order = response.get("order", response)
        if not isinstance(order, Mapping):
            self.coordinator.mark_unknown(intent_id, reason="missing_order_object")
            raise ExecutionGatewayOutcomeUnknown("Order response omitted order object")
        order_id = str(order.get("order_id") or "").strip()
        if not order_id:
            self.coordinator.mark_unknown(intent_id, reason="missing_order_id")
            raise ExecutionGatewayOutcomeUnknown("Order response omitted exchange order ID")
        state = str(order.get("order_state") or "").strip().lower()
        valid_states = {
            "open", "new", "untriggered", "partially_filled", "filled",
            "cancelled", "canceled", "expired", "closed", "rejected", "failed",
        }
        if state not in valid_states:
            self.coordinator.mark_unknown(intent_id, reason=f"unrecognized_exchange_order_state:{state or 'missing'}")
            raise ExecutionGatewayOutcomeUnknown("Order response has an unrecognized or missing state")
        try:
            filled = _decimal(order.get("filled_amount", "0"), "filled_amount", zero_ok=True)
        except ValueError:
            self.coordinator.mark_unknown(intent_id, reason="invalid_filled_amount")
            raise ExecutionGatewayOutcomeUnknown("Invalid filled_amount") from None
        if filled > requested:
            self.coordinator.mark_unknown(intent_id, reason="filled_amount_exceeds_requested")
            raise ExecutionGatewayOutcomeUnknown("Fill exceeds requested amount")
        if state in {"rejected", "failed"} and filled > 0:
            self.coordinator.mark_unknown(intent_id, reason="rejected_state_conflicts_with_fill")
            raise ExecutionGatewayOutcomeUnknown("Rejected order response conflicts with a reported fill")
        if state == "filled" and filled != requested:
            self.coordinator.mark_unknown(intent_id, reason="filled_status_quantity_mismatch")
            raise ExecutionGatewayOutcomeUnknown("Filled state conflicts with cumulative fill")
        if state == "partially_filled" and not (Decimal("0") < filled < requested):
            self.coordinator.mark_unknown(intent_id, reason="partial_status_quantity_mismatch")
            raise ExecutionGatewayOutcomeUnknown("Partially-filled state conflicts with cumulative fill")
        if state in {"open", "new", "untriggered"} and filled == requested:
            self.coordinator.mark_unknown(intent_id, reason="open_state_conflicts_with_full_fill")
            raise ExecutionGatewayOutcomeUnknown("Open order state conflicts with full cumulative fill")
        if state in {"rejected", "failed"}:
            self.coordinator.record_rejected(intent_id, reason=f"exchange_state:{state}", exchange_order_id=order_id)
            return
        if filled > 0:
            average = _avg_price(order, response)
            if average is None:
                self.coordinator.mark_unknown(intent_id, reason="fill_price_missing_or_invalid")
                raise ExecutionGatewayOutcomeUnknown("Fill has no independently usable average price")
            self.coordinator.record_fill(
                intent_id, cumulative_filled_quantity=filled,
                average_fill_price=average, exchange_order_id=order_id,
                payload={"exchange_order_state": state},
            )
        else:
            self.coordinator.record_acknowledged(
                intent_id, exchange_order_id=order_id,
                payload={"exchange_order_state": state},
            )
        if state in {"cancelled", "canceled", "expired", "closed"}:
            self.ledger.transition(
                intent_id, OrderState.CANCELLED, exchange_order_id=order_id,
                event_type="TERMINAL_WITH_PARTIAL_FILL" if filled > 0 else "TERMINAL_ZERO_FILL",
                payload={"exchange_order_state": state}, **self._lease_kwargs(),
            )

    def _invalidate_protection_before_mutation(self, path: str, body: Mapping[str, Any]) -> list[str]:
        protective_intent = None
        if path in {"/private/cancel", "/private/edit"}:
            order_id = str(body.get("order_id") or "").strip()
            if order_id:
                protective_intent = self.ledger.get_intent_by_exchange_order_id(order_id)
        if protective_intent is not None:
            parents = self.ledger.active_protection_parent_ids(
                protective_intent["intent_id"], **self._lease_kwargs()
            )
        elif path == "/private/cancel_all":
            parents = self.ledger.active_protection_parent_ids(**self._lease_kwargs())
        else:
            parents = []
        for parent_id in parents:
            self.ledger.mark_unprotected(
                parent_id,
                payload={"reason": "protective_order_mutation_requested", "rpc_method": path},
                **self._lease_kwargs(),
            )
        return parents

    def observe_order_state(self, order: Mapping[str, Any]) -> None:
        """Merge a verified read-only order snapshot into its durable intent."""
        if not isinstance(order, Mapping):
            return
        order_id = str(order.get("order_id") or "").strip()
        if not order_id:
            return
        self._renew()
        intent = self.ledger.get_intent_by_exchange_order_id(order_id)
        if intent is None:
            return
        state = str(order.get("order_state") or "").strip().lower()
        state_map = {
            "open": "open", "new": "open", "untriggered": "open",
            "partially_filled": "partially_filled", "filled": "filled",
            "cancelled": "cancelled", "canceled": "cancelled",
            "expired": "cancelled", "closed": "cancelled",
            "rejected": "rejected", "failed": "rejected",
        }
        status = state_map.get(state)
        if status is None:
            if intent["state"] not in {OrderState.UNKNOWN.value, OrderState.RECONCILING.value}:
                self.coordinator.mark_unknown(intent["intent_id"], reason="unrecognized_read_only_order_state")
            raise ExecutionGatewayOutcomeUnknown("Read-only order snapshot has an unrecognized state")
        filled = _decimal(order.get("filled_amount", intent.get("filled_quantity", "0")), "filled_amount", zero_ok=True)
        requested = _decimal(intent.get("quantity"), "requested_quantity")
        if filled > requested:
            self.coordinator.mark_unknown(intent["intent_id"], reason="observed_fill_exceeds_requested")
            raise ExecutionGatewayOutcomeUnknown("Observed fill exceeds requested quantity")
        if status == "filled" and filled != requested:
            self.coordinator.mark_unknown(intent["intent_id"], reason="observed_filled_quantity_mismatch")
            raise ExecutionGatewayOutcomeUnknown("Observed filled state conflicts with cumulative quantity")
        if status == "partially_filled" and not (Decimal("0") < filled < requested):
            self.coordinator.mark_unknown(intent["intent_id"], reason="observed_partial_quantity_mismatch")
            raise ExecutionGatewayOutcomeUnknown("Observed partial state conflicts with cumulative quantity")
        average = _avg_price(order, {"trades": order.get("trades", [])}) if filled > 0 else None
        current_state = intent["state"]
        if current_state == OrderState.UNKNOWN.value:
            observed = {
                "status": status,
                "exchange_order_id": order_id,
                "cumulative_filled_quantity": str(filled),
            }
            if average is not None:
                observed["average_fill_price"] = str(average)
            elif filled > 0:
                observed["average_fill_price"] = order.get("average_price") or order.get("average_fill_price")
            self.coordinator.reconcile_unknown(intent["intent_id"], lambda _: observed)
            intent = self.ledger.get_intent(intent["intent_id"])
        else:
            prior = _decimal(intent.get("filled_quantity", "0"), "prior_filled", zero_ok=True)
            if filled < prior:
                raise ExecutionGatewayBlocked("Exchange order snapshot regressed cumulative filled quantity")
            if filled > prior:
                if average is None:
                    self.coordinator.mark_unknown(intent["intent_id"], reason="observed_fill_missing_average_price")
                    raise ExecutionGatewayOutcomeUnknown("Observed fill has no verifiable average price")
                self.coordinator.record_fill(
                    intent["intent_id"], cumulative_filled_quantity=filled,
                    average_fill_price=average, exchange_order_id=order_id,
                    payload={"observed_read_only_order_state": state},
                )
                intent = self.ledger.get_intent(intent["intent_id"])
            if status == "rejected" and intent["state"] not in {OrderState.REJECTED.value}:
                if filled > 0:
                    self.coordinator.mark_unknown(intent["intent_id"], reason="observed_rejected_state_conflicts_with_fill")
                    raise ExecutionGatewayOutcomeUnknown("Rejected snapshot conflicts with previously observed fill")
                self.coordinator.record_rejected(intent["intent_id"], reason="read_only_exchange_state_rejected", exchange_order_id=order_id)
                intent = self.ledger.get_intent(intent["intent_id"])
            if status == "cancelled" and intent["state"] in {
                OrderState.ACKNOWLEDGED.value, OrderState.PARTIALLY_FILLED.value,
            }:
                self.ledger.transition(
                    intent["intent_id"], OrderState.CANCELLED,
                    exchange_order_id=order_id, event_type="READ_ONLY_ORDER_STATE_TERMINAL",
                    payload={"exchange_order_state": state}, **self._lease_kwargs(),
                )
                intent = self.ledger.get_intent(intent["intent_id"])
        if status in {"filled", "cancelled", "rejected"}:
            parents = self.ledger.active_protection_parent_ids(
                intent["intent_id"], **self._lease_kwargs()
            )
            for parent_id in parents:
                self.ledger.mark_unprotected(
                    parent_id,
                    payload={"reason": "protective_order_no_longer_resting", "protective_order_id": order_id, "exchange_order_state": state},
                    **self._lease_kwargs(),
                )
                self.ledger.deactivate_protection_links(intent["intent_id"], **self._lease_kwargs())

    def reconcile_order_label(
        self, client_order_id: str, rows: list[Mapping[str, Any]], *, queries_complete: bool,
    ) -> Optional[dict]:
        """Reconcile a claimed order using exact-label read-only exchange results.

        Empty or incomplete label windows never prove rejection. A single
        complete, identity-matching row can resolve the durable order intent;
        multiple or contradictory matches remain UNKNOWN.
        """
        self._renew()
        label = self._label(client_order_id)
        intent = self.ledger.get_intent_by_client_order_id(
            exchange=self.exchange, environment=self.environment,
            account_id=self.account_id, client_order_id=label,
        )
        if intent is None:
            return None

        def latch_unknown(reason: str) -> None:
            current = self.ledger.get_intent(intent["intent_id"])
            state_now = current.get("state")
            if state_now in {
                OrderState.INTENT_CREATED.value, OrderState.SUBMITTING.value,
                OrderState.ACKNOWLEDGED.value, OrderState.PARTIALLY_FILLED.value,
                OrderState.RECONCILING.value,
            }:
                self.coordinator.mark_unknown(intent["intent_id"], reason=reason)
            else:
                # UNKNOWN can be encountered after a process restart; terminal
                # conflicts cannot be reopened by an ordinary transition.
                self.coordinator.breaker.record_failure(order_path=True)
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            if intent["state"] in {OrderState.INTENT_CREATED.value, OrderState.SUBMITTING.value}:
                latch_unknown("malformed_label_reconciliation_snapshot")
            raise ExecutionGatewayOutcomeUnknown("Malformed order-label reconciliation evidence")
        if not queries_complete:
            if intent["state"] == OrderState.SUBMITTING.value:
                latch_unknown("label_reconciliation_incomplete")
            return self.ledger.get_intent(intent["intent_id"])

        candidates = []
        seen = {}
        for row in rows:
            if str(row.get("label") or "") != label:
                continue
            oid = str(row.get("order_id") or "").strip()
            if not oid:
                continue
            # Every field used to connect the exchange row to an intent is
            # independently checked; a same-label row on another instrument
            # or side is a conflict, not a match to ignore.
            if str(row.get("instrument_name") or "") != str(intent["instrument_id"]):
                latch_unknown("label_match_instrument_mismatch")
                raise ExecutionGatewayOutcomeUnknown("Label match instrument differs from durable intent")
            if str(row.get("direction") or "").upper() != str(intent["side"]).upper():
                latch_unknown("label_match_side_mismatch")
                raise ExecutionGatewayOutcomeUnknown("Label match side differs from durable intent")
            try:
                amount = _decimal(row.get("amount"), "observed_amount")
                expected = _decimal(intent.get("quantity"), "requested_quantity")
            except ValueError:
                latch_unknown("label_match_amount_invalid")
                raise ExecutionGatewayOutcomeUnknown("Label match has invalid quantity") from None
            if amount != expected:
                latch_unknown("label_match_amount_mismatch")
                raise ExecutionGatewayOutcomeUnknown("Label match quantity differs from durable intent")
            normalized = dict(row)
            if oid in seen:
                previous = seen[oid]
                for key in ("instrument_name", "direction", "amount", "filled_amount", "order_state", "average_price"):
                    if str(previous.get(key)) != str(normalized.get(key)):
                        latch_unknown("contradictory_duplicate_label_rows")
                        raise ExecutionGatewayOutcomeUnknown("Duplicate label rows contain conflicting exchange state")
                continue
            seen[oid] = normalized
            candidates.append(normalized)
        if len(candidates) > 1:
            latch_unknown("multiple_exchange_orders_for_client_label")
            raise ExecutionGatewayOutcomeUnknown("Multiple exchange orders match one durable client label")
        if not candidates:
            if intent["state"] == OrderState.SUBMITTING.value:
                latch_unknown("label_lookup_no_match_or_history_window_incomplete")
            return self.ledger.get_intent(intent["intent_id"])

        row = candidates[0]
        exchange_state = str(row.get("order_state") or "").strip().lower()
        status_map = {
            "open": "open", "new": "open", "untriggered": "open",
            "partially_filled": "partially_filled", "filled": "filled",
            "cancelled": "cancelled", "canceled": "cancelled",
            "expired": "cancelled", "closed": "cancelled",
            "rejected": "rejected", "failed": "rejected",
        }
        status = status_map.get(exchange_state)
        if status is None:
            latch_unknown("unrecognized_label_order_state")
            raise ExecutionGatewayOutcomeUnknown("Label match has an unrecognized order state")
        if intent["state"] not in {OrderState.UNKNOWN.value, OrderState.RECONCILING.value}:
            self.coordinator.mark_unknown(intent["intent_id"], reason="reconciling_order_from_exact_label")
        observed = {
            "status": status,
            "exchange_order_id": str(row["order_id"]),
            "cumulative_filled_quantity": row.get("filled_amount", "0"),
        }
        avg = row.get("average_price", row.get("average_fill_price"))
        if avg is not None:
            observed["average_fill_price"] = avg
        return self.coordinator.reconcile_unknown(intent["intent_id"], lambda _: observed)

    def execute_control_mutation(self, path: str, body: Mapping[str, Any], submit: Callable[[], Any]) -> Any:
        if path not in _CONTROL_MUTATIONS or not isinstance(body, Mapping):
            raise ExecutionGatewayBlocked("Unclassified or malformed control mutation")
        self._renew()
        safe_body = {
            str(k): v for k, v in body.items()
            if str(k).lower() not in {"client_secret", "access_token", "refresh_token", "authorization"}
        }
        canonical = json.dumps(safe_body, sort_keys=True, separators=(",", ":"), default=str)
        mutation_id = str(uuid.uuid4())
        resource_id = str(body.get("order_id") or body.get("instrument_name") or "ACCOUNT")
        try:
            self.ledger.begin_mutation(
                mutation_id=mutation_id, exchange=self.exchange,
                environment=self.environment, account_id=self.account_id,
                operation=path.rsplit("/", 1)[-1].upper(), resource_id=resource_id,
                request_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                payload={"request": safe_body}, **self._lease_kwargs(),
            )
        except Exception as exc:
            raise ExecutionGatewayBlocked(
                f"Could not journal {path} ({type(exc).__name__}); no exchange request sent"
            ) from None
        try:
            affected_protection_parents = self._invalidate_protection_before_mutation(path, body)
        except Exception as exc:
            raise ExecutionGatewayBlocked(
                f"Could not safely invalidate linked protection before {path} ({type(exc).__name__}); no exchange request sent"
            ) from None
        try:
            response = submit()
        except Exception as exc:
            status = "REJECTED" if type(exc).__name__ == "DeribitRPCError" else "UNKNOWN"
            try:
                self._renew()
                self.ledger.finish_mutation(
                    mutation_id, state=status, payload={"error_type": type(exc).__name__},
                    **self._lease_kwargs(),
                )
            except Exception:
                self.coordinator.breaker.record_failure(order_path=True)
                raise ExecutionGatewayOutcomeUnknown(f"{path} outcome and journal state are unknown") from None
            if status == "UNKNOWN":
                self.coordinator.breaker.record_failure(order_path=True)
                raise ExecutionGatewayOutcomeUnknown(f"{path} outcome unknown; reconcile before repeating") from None
            raise
        try:
            self._renew()
            response_id = _response_id(response)
            self.ledger.finish_mutation(
                mutation_id, state="ACKNOWLEDGED", response_id=response_id,
                payload={"response_type": type(response).__name__}, **self._lease_kwargs(),
            )
            if path == "/private/cancel" and body.get("order_id"):
                protective = self.ledger.get_intent_by_exchange_order_id(str(body["order_id"]))
                if protective is not None:
                    self.ledger.deactivate_protection_links(
                        protective["intent_id"], **self._lease_kwargs()
                    )
            elif path == "/private/cancel_all":
                self.ledger.deactivate_protection_links(**self._lease_kwargs())
        except Exception:
            self.coordinator.breaker.record_failure(order_path=True)
            raise ExecutionGatewayOutcomeUnknown(
                f"{path} response received but journal commit failed; reconcile before repeating"
            ) from None
        return response

    def confirm_protection(
        self, *, entry_client_order_id: str, stop_client_order_id: str,
        protective_order_id: str, protected_quantity: Any,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        self._renew()
        entry = self.ledger.get_intent_by_client_order_id(
            exchange=self.exchange, environment=self.environment,
            account_id=self.account_id, client_order_id=entry_client_order_id,
        )
        stop = self.ledger.get_intent_by_client_order_id(
            exchange=self.exchange, environment=self.environment,
            account_id=self.account_id, client_order_id=stop_client_order_id,
        )
        if not entry or not stop:
            raise ExecutionGatewayBlocked("Entry or stop intent is missing")
        if str(stop.get("exchange_order_id") or "") != str(protective_order_id):
            raise ExecutionGatewayBlocked("Stop order ID does not match the durable stop intent")
        if not stop.get("reduce_only") or stop.get("instrument_id") != entry.get("instrument_id"):
            raise ExecutionGatewayBlocked("Stop must be reduce-only on the entry instrument")
        if stop.get("state") != OrderState.ACKNOWLEDGED.value:
            raise ExecutionGatewayBlocked("Protective stop must be acknowledged and resting")
        stop_payload = stop.get("payload") if isinstance(stop.get("payload"), Mapping) else {}
        if str(stop_payload.get("order_type") or "").lower() != "stop_market":
            raise ExecutionGatewayBlocked("Protective order must be a stop_market intent")
        if str(stop_payload.get("trigger") or "").lower() != "mark_price":
            raise ExecutionGatewayBlocked("Protective stop must use the verified mark_price trigger")
        if str(stop.get("side") or "").upper() == str(entry.get("side") or "").upper():
            raise ExecutionGatewayBlocked("Protective stop side must oppose the entry side")
        try:
            protected = _decimal(protected_quantity, "protected_quantity")
            stop_quantity = _decimal(stop.get("quantity"), "stop_quantity")
            entry_filled = _decimal(entry.get("filled_quantity", "0"), "entry_filled", zero_ok=True)
        except ValueError as exc:
            raise ExecutionGatewayBlocked(str(exc)) from None
        if protected != stop_quantity or protected <= 0:
            raise ExecutionGatewayBlocked("Protected quantity must exactly match the durable stop order quantity")
        if protected < entry_filled:
            raise ExecutionGatewayBlocked("Protective stop quantity does not cover the confirmed entry fill")
        link_payload = {
            "protective_order_id": str(protective_order_id),
            "protective_client_order_id": str(stop_client_order_id),
            **dict(evidence or {}),
        }
        self.ledger.link_protection_order(
            entry["intent_id"], stop["intent_id"], payload=link_payload,
            **self._lease_kwargs(),
        )
        return self.coordinator.verify_protection(
            entry["intent_id"], protected_quantity=protected_quantity,
            payload={"protective_intent_id": stop["intent_id"], **link_payload},
        )

    def resolve_control_mutation(
        self, mutation_id: str, *, state: str, evidence: Mapping[str, Any], resolved_by: str,
    ) -> dict:
        """Resolve an UNKNOWN control call only from explicit read-only evidence.

        The order-path circuit breaker is deliberately not reset here; callers
        must separately acknowledge the incident after examining the evidence.
        """
        self._renew()
        resolver = getattr(self.ledger, "resolve_mutation", None)
        if not callable(resolver):
            raise ExecutionGatewayBlocked("Execution ledger does not support audited mutation reconciliation")
        return resolver(
            mutation_id, state=state, evidence=evidence, resolved_by=resolved_by,
            **self._lease_kwargs(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._release_cb is not None:
            try:
                self._release_cb()
            except Exception:
                pass
        if self._close_cb is not None:
            try:
                self._close_cb()
            except Exception:
                pass


def create_deribit_execution_gateway(client_id: str, breaker: CircuitBreaker) -> DeribitExecutionGateway:
    """Build a lease-owning gateway. GitHub Actions must use fenced PostgreSQL."""
    from execution_storage import create_execution_ledger

    is_github = str(os.getenv("GITHUB_ACTIONS", "false")).strip().lower() == "true"
    backend = os.getenv("EXECUTION_LEDGER_BACKEND", "postgresql" if is_github else "sqlite")
    ledger = create_execution_ledger(
        backend=backend, shared_required=is_github, require_fenced_writes=is_github,
    )
    account_id = str(os.getenv("DERIBIT_EXECUTION_ACCOUNT_ID", "")).strip()
    if not account_id:
        if is_github:
            ledger.close()
            raise ExecutionGatewayBlocked("DERIBIT_EXECUTION_ACCOUNT_ID is required in GitHub Actions")
        account_id = "client-" + hashlib.sha256(str(client_id).encode("utf-8")).hexdigest()[:24]
    owner_id = f"{os.getenv('GITHUB_RUN_ID', 'local')}:{os.getenv('GITHUB_RUN_ATTEMPT', '1')}:{uuid.uuid4().hex}"
    lease_key = f"DERIBIT:TESTNET:{account_id}"
    ttl = 180.0
    token = None
    try:
        if hasattr(ledger, "acquire_fenced_run_lease"):
            token = ledger.acquire_fenced_run_lease(lease_key, owner_id, ttl_seconds=ttl)
            if token is None:
                raise ExecutionGatewayBlocked("Another Testnet execution owner holds the lease")
            renew = lambda: ledger.renew_fenced_run_lease(lease_key, owner_id, token, ttl_seconds=ttl)
            release = lambda: ledger.release_fenced_run_lease(lease_key, owner_id, token)
        else:
            if not ledger.acquire_run_lease(lease_key, owner_id, ttl_seconds=ttl):
                raise ExecutionGatewayBlocked("Another local Testnet execution owner holds the lease")
            renew = lambda: ledger.renew_run_lease(lease_key, owner_id, ttl_seconds=ttl)
            release = lambda: ledger.release_run_lease(lease_key, owner_id)
        coordinator = ExecutionCoordinator(
            ledger, breaker, exchange="DERIBIT", environment="testnet", account_id=account_id,
            lease_key=lease_key, lease_owner_id=owner_id, fencing_token=token,
        )
        return DeribitExecutionGateway(
            coordinator, lease_renew=renew, lease_release=release,
            ledger_close=ledger.close, register_atexit=True,
        )
    except Exception:
        try:
            ledger.close()
        except Exception:
            pass
        raise
