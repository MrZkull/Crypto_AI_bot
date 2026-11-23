"""Strict classification and admission validation for tiny bot-created residuals.

A dust disposition is not a general orphan exemption. It requires a terminal,
filled bot entry, two stable same-direction exchange position reads, two empty
open-order snapshots, and a current mark notional below the explicit policy cap.
The portfolio reserves the configured per-position ceiling, not just the
observed notional, while the residual remains open.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

DUST_POLICY_VERSION = "BOT_CREATED_SUBMINIMUM_RESIDUAL_V1"
DEFAULT_MAX_DUST_POSITION_NOTIONAL_USDC = Decimal("0.10")
DEFAULT_MAX_TOTAL_DUST_RESERVE_USDC = Decimal("0.25")
DEFAULT_DUST_RESERVE_PER_POSITION_USDC = Decimal("0.10")
_POSITION_READ_REL_TOLERANCE = Decimal("0.000001")


class DustResidualError(ValueError):
    """Residual evidence is incomplete, unstable, or outside the dust policy."""


def _d(value: Any, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise DustResidualError(f"{field} is missing or invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        raise DustResidualError(f"{field} is invalid") from None
    if not result.is_finite():
        raise DustResidualError(f"{field} is non-finite")
    return result


def _position_tolerance(minimum: Decimal) -> Decimal:
    return max(minimum * _POSITION_READ_REL_TOLERANCE, Decimal("1e-12"))


def classify_verified_dust_residual(
    *,
    symbol: str,
    signal: str,
    first_signed_position: Any,
    second_signed_position: Any,
    min_trade_amount: Any,
    mark_price: Any,
    first_open_orders: Sequence[Mapping[str, Any]],
    second_open_orders: Sequence[Mapping[str, Any]],
    entry_order_id: str,
    entry_order_state: str,
    entry_filled_quantity: Any,
    prediction_id: str,
    max_position_notional_usdc: Any = DEFAULT_MAX_DUST_POSITION_NOTIONAL_USDC,
    observed_at: datetime | None = None,
) -> dict:
    """Create a durable dust disposition only from complete exchange evidence."""
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_signal = str(signal or "").strip().upper()
    expected_direction = "buy" if normalized_signal == "BUY" else "sell" if normalized_signal == "SELL" else ""
    if not normalized_symbol or not expected_direction:
        raise DustResidualError("Dust symbol or signal identity is invalid")

    first = _d(first_signed_position, "first_signed_position")
    second = _d(second_signed_position, "second_signed_position")
    minimum = _d(min_trade_amount, "min_trade_amount")
    mark = _d(mark_price, "mark_price")
    cap = _d(max_position_notional_usdc, "max_position_notional_usdc")
    filled = _d(entry_filled_quantity, "entry_filled_quantity")
    if minimum <= 0 or mark <= 0 or cap <= 0 or filled <= 0:
        raise DustResidualError("Dust quantity, mark, cap and entry fill must be positive")
    expected_sign = Decimal("1") if expected_direction == "buy" else Decimal("-1")
    if first == 0 or second == 0 or first * expected_sign <= 0 or second * expected_sign <= 0:
        raise DustResidualError("Both position reads must be non-zero and match the bot entry direction")
    if abs(first - second) > _position_tolerance(minimum):
        raise DustResidualError("Position quantity was not stable across two exchange reads")

    quantity = max(abs(first), abs(second))
    if quantity >= minimum:
        raise DustResidualError("Residual is not below the exchange minimum trade amount")
    if filled + _position_tolerance(minimum) < quantity:
        raise DustResidualError("Verified entry fill is smaller than the residual position")
    if not isinstance(first_open_orders, (list, tuple)) or not isinstance(second_open_orders, (list, tuple)):
        raise DustResidualError("Both open-order snapshots must be complete lists")
    if first_open_orders or second_open_orders:
        raise DustResidualError("Dust classification requires two empty open-order snapshots")

    order_id = str(entry_order_id or "").strip()
    prediction = str(prediction_id or "").strip()
    state = str(entry_order_state or "").strip().lower()
    if not order_id or not prediction:
        raise DustResidualError("Bot entry order and prediction provenance are required")
    if state not in {"filled", "cancelled"}:
        raise DustResidualError("Bot entry order must have a verified terminal state")

    notional = quantity * mark
    if notional > cap:
        raise DustResidualError("Residual mark notional exceeds the per-position dust ceiling")
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise DustResidualError("Dust verification timestamp must be timezone-aware")

    return {
        "status": "VERIFIED",
        "policy": DUST_POLICY_VERSION,
        "symbol": normalized_symbol,
        "quantity": str(quantity),
        "first_position_signed": str(first),
        "second_position_signed": str(second),
        "minimum_trade_amount": str(minimum),
        "direction": expected_direction,
        "mark_price": str(mark),
        "mark_notional_usdc": str(notional),
        "entry_order_id": order_id,
        "entry_order_state": state,
        "entry_filled_quantity": str(filled),
        "prediction_id": prediction,
        "position_read_count": 2,
        "empty_open_order_read_count": 2,
        "verified_at": timestamp.astimezone(timezone.utc).isoformat(),
    }


def validate_verified_dust_residual(
    trade_record: Mapping[str, Any],
    *,
    signed_position: Any,
    min_trade_amount: Any,
    mark_price: Any,
    open_orders: Sequence[Mapping[str, Any]],
    max_position_notional_usdc: Any = DEFAULT_MAX_DUST_POSITION_NOTIONAL_USDC,
    reserve_per_position_usdc: Any = DEFAULT_DUST_RESERVE_PER_POSITION_USDC,
) -> dict:
    """Re-verify a prior dust disposition against current live exchange state."""
    if not isinstance(trade_record, Mapping) or trade_record.get("closed") is not True:
        raise DustResidualError("Dust disposition is not a terminal local trade record")
    for flag in ("emergency_recovery_required", "manual_reconciliation_required", "protection_reconciliation_required"):
        if trade_record.get(flag):
            raise DustResidualError("Dust record still carries unresolved recovery flags")
    dust = trade_record.get("dust_residual")
    if not isinstance(dust, Mapping):
        raise DustResidualError("Verified dust evidence is missing")
    if dust.get("status") != "VERIFIED" or dust.get("policy") != DUST_POLICY_VERSION:
        raise DustResidualError("Dust disposition status or policy identity is invalid")
    if int(dust.get("position_read_count", 0)) < 2 or int(dust.get("empty_open_order_read_count", 0)) < 2:
        raise DustResidualError("Dust disposition lacks dual-read evidence")
    if not str(dust.get("entry_order_id") or "").strip() or not str(dust.get("prediction_id") or "").strip():
        raise DustResidualError("Dust provenance is missing")
    record_orders = trade_record.get("order_ids")
    if not isinstance(record_orders, Mapping) or str(record_orders.get("entry") or "") != str(dust.get("entry_order_id")):
        raise DustResidualError("Dust entry order does not match the persisted trade record")
    if str(trade_record.get("pred_id") or "") != str(dust.get("prediction_id")):
        raise DustResidualError("Dust prediction does not match the persisted trade record")

    quantity = _d(dust.get("quantity"), "recorded_dust_quantity")
    entry_filled = _d(dust.get("entry_filled_quantity"), "recorded_entry_filled_quantity")
    entry_state = str(dust.get("entry_order_state") or "").strip().lower()
    if entry_state not in {"filled", "cancelled"} or entry_filled <= 0:
        raise DustResidualError("Dust entry order lacks a verified terminal positive fill")
    live_signed = _d(signed_position, "live_signed_position")
    minimum = _d(min_trade_amount, "min_trade_amount")
    mark = _d(mark_price, "mark_price")
    cap = _d(max_position_notional_usdc, "max_position_notional_usdc")
    reserve = _d(reserve_per_position_usdc, "reserve_per_position_usdc")
    direction = str(dust.get("direction") or "").lower()
    expected_sign = Decimal("1") if direction == "buy" else Decimal("-1") if direction == "sell" else Decimal("0")
    if expected_sign == 0 or minimum <= 0 or mark <= 0 or cap <= 0 or reserve <= 0:
        raise DustResidualError("Live dust validation parameters are invalid")
    if live_signed == 0 or live_signed * expected_sign <= 0:
        raise DustResidualError("Live position is flat or has changed direction")
    if abs(live_signed) >= minimum:
        raise DustResidualError("Live residual is no longer below the exchange minimum")
    if entry_filled + _position_tolerance(minimum) < abs(live_signed):
        raise DustResidualError("Live dust exceeds the verified entry fill quantity")
    if abs(abs(live_signed) - quantity) > _position_tolerance(minimum):
        raise DustResidualError("Live residual quantity differs from its verified disposition")
    notional = abs(live_signed) * mark
    if notional > cap:
        raise DustResidualError("Live dust mark notional exceeds its configured ceiling")
    if not isinstance(open_orders, (list, tuple)) or open_orders:
        raise DustResidualError("Verified dust symbol has an open or unverifiable order")
    if str(dust.get("symbol") or "").upper() != str(trade_record.get("symbol") or "").upper():
        raise DustResidualError("Dust symbol identity does not match the local record")
    return {
        "quantity": abs(live_signed),
        "mark_notional_usdc": notional,
        "reserve_usdc": reserve,
    }
