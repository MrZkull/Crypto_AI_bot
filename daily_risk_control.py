"""Fail-closed UTC-day account-equity drawdown control for Deribit Testnet.

The daily limit is based on exchange-reported USDC equity (therefore including
mark-to-market), adjusted only for independently identified external cash flows.
Daily baselines and sticky halts live in the execution ledger, not an ephemeral
runner-local JSON file.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


MAX_DAY_START_SNAPSHOT_DELAY_SECONDS = 20 * 60
MAX_TRANSACTION_LOG_PAGES = 20
TRANSACTION_LOG_PAGE_SIZE = 1000
TRANSACTION_LOG_MIN_INTERVAL_SECONDS = 1.05
_TRANSACTION_LOG_RATE_LOCK = threading.Lock()
_LAST_TRANSACTION_LOG_REQUEST_MONOTONIC: float | None = None

# These transaction types are account-level cash flows, not strategy PnL.
# Their signed `change` is added to the day-start baseline so a deposit or
# withdrawal is not interpreted as a trading gain or loss.
_EXTERNAL_FLOW_TYPES = {"deposit", "withdrawal", "transfer"}

# Known transaction types whose economics remain part of account-equity PnL.
# Position moves and exceptional balance corrections are deliberately not in
# this set: their impact cannot safely be classified as ordinary strategy PnL.
_EQUITY_PNL_TYPES = {
    "trade", "trade_allocation", "settlement", "delivery", "expiry",
    "assignment", "exercise", "swap", "block_trade", "combo_trade",
    "commission", "negative_balance_fee", "creation", "summary",
    "options_settlement_summary", "custody_settlement_fee",
    "sweep_isolated_position", "structured_product_spread", "socialize_loses",
    "affiliate", "partner", "mm_lsp_payment", "stake_reward", "usdc_reward",
    "usdc_reward_received", "usdc_platform_reward", "usdc_reward_payment",
    "ethena_reward", "buidl_reward", "reward",
}

_REQUIRES_RECONCILIATION_TYPES = {
    "position", "insurance", "insurance_transfer", "correction",
    "margin_loan", "corporate_action_adjustment", "stock_split",
    "rights_offering", "spin_off", "m_and_a", "rename", "liquidation",
    "fee_deposit", "fee_balance_transfer", "lock_deposit_balance",
    "unlock_deposit_balance", "transfer_fee", "lsp_transfer", "adl_transfer",
    "liquidation_transfer",
}


class DailyRiskControlError(RuntimeError):
    """Daily equity risk cannot be calculated from complete verified evidence."""


@dataclass(frozen=True)
class DailyRiskResult:
    allowed: bool
    reason: str
    utc_day: str
    baseline_equity: Decimal | None = None
    adjusted_baseline_equity: Decimal | None = None
    current_equity: Decimal | None = None
    external_flows: Decimal | None = None
    daily_pnl: Decimal | None = None
    loss_fraction: Decimal | None = None


def _decimal(value: Any, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise DailyRiskControlError(f"{field} is missing or invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        raise DailyRiskControlError(f"{field} is invalid") from None
    if not result.is_finite():
        raise DailyRiskControlError(f"{field} is non-finite")
    return result


def calculate_daily_equity_drawdown(
    *, baseline_equity: Any,
    current_equity: Any,
    net_external_flows: Any,
    max_daily_loss_fraction: Any,
    max_daily_loss_amount: Any | None = None,
) -> dict:
    """Calculate net daily equity PnL without counting account transfers as PnL."""
    baseline = _decimal(baseline_equity, "baseline_equity")
    current = _decimal(current_equity, "current_equity")
    flows = _decimal(net_external_flows, "net_external_flows")
    limit = _decimal(max_daily_loss_fraction, "max_daily_loss_fraction")
    if baseline <= 0 or current < 0:
        raise DailyRiskControlError("Daily equity values are outside valid bounds")
    if limit <= 0 or limit > Decimal("0.10"):
        raise DailyRiskControlError("Daily loss limit must be in (0, 0.10]")

    adjusted_baseline = baseline + flows
    if adjusted_baseline <= 0:
        raise DailyRiskControlError("External flows make the adjusted daily baseline invalid")

    daily_pnl = current - adjusted_baseline
    loss_fraction = max(Decimal("0"), -daily_pnl / adjusted_baseline)
    percentage_limit_amount = adjusted_baseline * limit
    if max_daily_loss_amount is None:
        effective_limit_amount = percentage_limit_amount
    else:
        absolute_limit = _decimal(max_daily_loss_amount, "max_daily_loss_amount")
        if absolute_limit <= 0:
            raise DailyRiskControlError("Absolute daily loss limit must be positive")
        effective_limit_amount = min(percentage_limit_amount, absolute_limit)
    loss_amount = max(Decimal("0"), -daily_pnl)
    return {
        "baseline_equity": baseline,
        "adjusted_baseline_equity": adjusted_baseline,
        "current_equity": current,
        "external_flows": flows,
        "daily_pnl": daily_pnl,
        "loss_fraction": loss_fraction,
        "effective_limit_amount": effective_limit_amount,
        "loss_amount": loss_amount,
        "limit_exceeded": loss_amount >= effective_limit_amount,
    }


def extract_external_flows(rows: Iterable[dict], *, currency: str = "USDC") -> Decimal:
    """Sum signed external balance changes; reject unknown or malformed events."""
    expected_currency = str(currency or "").strip().upper()
    if not expected_currency:
        raise DailyRiskControlError("Transaction-log currency is required")
    total = Decimal("0")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DailyRiskControlError(f"Transaction-log row {index} is malformed")
        kind = str(row.get("type") or "").strip().lower()
        if not kind:
            raise DailyRiskControlError(f"Transaction-log row {index} has no type")
        row_currency = row.get("currency")
        if row_currency is None or str(row_currency).strip().upper() != expected_currency:
            raise DailyRiskControlError("Transaction log currency is missing or mismatched")

        role = str(row.get("role") or "").strip().lower()
        if kind in {"lsp", "adl"} and role in {"source", "destination"}:
            raise DailyRiskControlError(f"Position transfer requires manual reconciliation: {kind}")
        if kind in _EXTERNAL_FLOW_TYPES:
            change = _decimal(row.get("change"), f"transaction[{index}].change")
            if kind == "deposit" and change < 0:
                raise DailyRiskControlError("Deposit has a negative balance change")
            if kind == "withdrawal" and change > 0:
                raise DailyRiskControlError("Withdrawal has a positive balance change")
            total += change
        elif kind in _EQUITY_PNL_TYPES:
            # Trading PnL, settlement, commissions and funding are already
            # reflected in current exchange equity. Do not add them twice.
            continue
        elif kind in _REQUIRES_RECONCILIATION_TYPES:
            raise DailyRiskControlError(f"Transaction type requires manual reconciliation: {kind}")
        else:
            raise DailyRiskControlError(f"Unclassified transaction type: {kind}")
    return total


def _gateway_scope(deribit) -> tuple[Any, Any, dict]:
    gateway = getattr(deribit, "execution_gateway", None)
    if gateway is None or not getattr(deribit, "require_execution_gateway", False):
        raise DailyRiskControlError("Fenced execution gateway is required for daily risk state")
    coordinator = getattr(gateway, "coordinator", None)
    ledger = getattr(gateway, "ledger", None)
    if coordinator is None or ledger is None:
        raise DailyRiskControlError("Execution ledger context is unavailable")
    if not all(getattr(coordinator, key, None) for key in ("exchange", "environment", "account_id")):
        raise DailyRiskControlError("Execution account identity is incomplete")
    fence_factory = getattr(coordinator, "_lease_kwargs", None)
    fence = fence_factory() if callable(fence_factory) else {}
    return coordinator, ledger, fence


def _call_transaction_log_rate_limited(reader, *args, **kwargs):
    """Serialize every transaction-log request to the documented 1 req/sec rate."""
    global _LAST_TRANSACTION_LOG_REQUEST_MONOTONIC
    with _TRANSACTION_LOG_RATE_LOCK:
        now = time.monotonic()
        if _LAST_TRANSACTION_LOG_REQUEST_MONOTONIC is not None:
            elapsed = now - _LAST_TRANSACTION_LOG_REQUEST_MONOTONIC
            remaining = TRANSACTION_LOG_MIN_INTERVAL_SECONDS - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        _LAST_TRANSACTION_LOG_REQUEST_MONOTONIC = now
        return reader(*args, **kwargs)


def _read_transaction_logs(deribit, *, start_epoch: float, end_epoch: float) -> list[dict]:
    reader = getattr(deribit, "get_transaction_log", None)
    if not callable(reader):
        raise DailyRiskControlError("Deribit transaction-log reader is unavailable")
    start_ms = int(start_epoch * 1000)
    end_ms = int(end_epoch * 1000)
    continuation = None
    seen_continuations: set[int] = set()
    rows: list[dict] = []

    for page in range(MAX_TRANSACTION_LOG_PAGES):
        response = _call_transaction_log_rate_limited(
            reader, "USDC", start_timestamp=start_ms, end_timestamp=end_ms,
            count=TRANSACTION_LOG_PAGE_SIZE, continuation=continuation,
        )
        if not isinstance(response, dict) or not isinstance(response.get("logs"), list):
            raise DailyRiskControlError("Deribit transaction-log response is malformed")
        page_rows = response["logs"]
        if any(not isinstance(row, dict) for row in page_rows):
            raise DailyRiskControlError("Deribit transaction-log contains a malformed row")
        rows.extend(page_rows)
        next_token = response.get("continuation")
        if next_token is None:
            return rows
        try:
            next_token = int(next_token)
        except (TypeError, ValueError, OverflowError):
            raise DailyRiskControlError("Transaction-log continuation token is invalid") from None
        if next_token <= 0 or next_token in seen_continuations:
            raise DailyRiskControlError("Transaction-log continuation token repeated or invalid")
        seen_continuations.add(next_token)
        continuation = next_token

    raise DailyRiskControlError("Transaction log exceeded the supported page limit")


def evaluate_daily_risk_guard(
    deribit,
    *,
    current_equity: Any,
    max_daily_loss_fraction: Any = Decimal("0.02"),
    max_daily_loss_amount: Any | None = None,
    now: datetime | None = None,
) -> DailyRiskResult:
    """Evaluate the sticky UTC-day loss gate from ledger + exchange evidence.

    The baseline is the first durable exchange equity checkpoint captured in
    the first 20 minutes after 00:00 UTC. It becomes the effective start of the
    daily measurement window. Using a pre-midnight mark-equity snapshot would
    misclassify mark-to-market changes around midnight. If the runner misses
    this opening window or the transaction log is incomplete, entries are
    blocked for the day rather than inventing a baseline.
    """
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current_time = current_time.astimezone(timezone.utc)
    utc_day = current_time.date().isoformat()
    day_start = datetime(
        current_time.year, current_time.month, current_time.day, tzinfo=timezone.utc
    ).timestamp()

    try:
        coordinator, ledger, fence = _gateway_scope(deribit)
        required_methods = (
            "first_account_risk_snapshot_between", "get_daily_risk_halt", "latch_daily_risk_halt",
        )
        if any(not callable(getattr(ledger, name, None)) for name in required_methods):
            raise DailyRiskControlError("Execution ledger does not implement durable daily risk state")

        existing_halt = ledger.get_daily_risk_halt(
            exchange=coordinator.exchange, environment=coordinator.environment,
            account_id=coordinator.account_id, utc_day=utc_day,
        )
        if existing_halt:
            return DailyRiskResult(False, str(existing_halt.get("reason") or "DAILY_RISK_HALTED"), utc_day)

        baseline_row = ledger.first_account_risk_snapshot_between(
            exchange=coordinator.exchange, environment=coordinator.environment,
            account_id=coordinator.account_id, currency="USDC",
            start=day_start, end=day_start + MAX_DAY_START_SNAPSHOT_DELAY_SECONDS,
        )
        if not isinstance(baseline_row, dict):
            return DailyRiskResult(False, "DAY_START_BASELINE_MISSING_OR_LATE", utc_day)

        baseline_time = float(baseline_row.get("observed_at"))
        if (
            not math.isfinite(baseline_time)
            or baseline_time < day_start
            or baseline_time > day_start + MAX_DAY_START_SNAPSHOT_DELAY_SECONDS
        ):
            return DailyRiskResult(False, "DAY_START_BASELINE_OUTSIDE_WINDOW", utc_day)

        baseline_equity = _decimal(baseline_row.get("equity"), "baseline_equity")
        current_value = _decimal(current_equity, "current_equity")
        if baseline_equity <= 0 or current_value <= 0:
            raise DailyRiskControlError("Daily account equity must be positive")

        log_rows = _read_transaction_logs(
            deribit, start_epoch=baseline_time, end_epoch=current_time.timestamp(),
        )
        external_flows = extract_external_flows(log_rows, currency="USDC")
        measured = calculate_daily_equity_drawdown(
            baseline_equity=baseline_equity,
            current_equity=current_value,
            net_external_flows=external_flows,
            max_daily_loss_fraction=max_daily_loss_fraction,
            max_daily_loss_amount=max_daily_loss_amount,
        )

        if measured["limit_exceeded"]:
            try:
                halt = ledger.latch_daily_risk_halt(
                    exchange=coordinator.exchange,
                    environment=coordinator.environment,
                    account_id=coordinator.account_id,
                    utc_day=utc_day,
                    reason="MAX_DAILY_LOSS_LIMIT",
                    baseline_equity=measured["adjusted_baseline_equity"],
                    current_equity=current_value,
                    loss_fraction=measured["loss_fraction"],
                    observed_at=current_time.timestamp(),
                    **fence,
                )
            except Exception as exc:
                raise DailyRiskControlError(
                    f"Daily loss breached but halt persistence failed ({type(exc).__name__})"
                ) from None
            return DailyRiskResult(
                False, str(halt.get("reason") or "MAX_DAILY_LOSS_LIMIT"), utc_day,
                baseline_equity=measured["baseline_equity"],
                adjusted_baseline_equity=measured["adjusted_baseline_equity"],
                current_equity=measured["current_equity"],
                external_flows=measured["external_flows"],
                daily_pnl=measured["daily_pnl"],
                loss_fraction=measured["loss_fraction"],
            )

        return DailyRiskResult(
            True, "DAILY_RISK_WITHIN_LIMIT", utc_day,
            baseline_equity=measured["baseline_equity"],
            adjusted_baseline_equity=measured["adjusted_baseline_equity"],
            current_equity=measured["current_equity"],
            external_flows=measured["external_flows"],
            daily_pnl=measured["daily_pnl"],
            loss_fraction=measured["loss_fraction"],
        )
    except Exception as exc:
        # No unknown or unavailable daily-risk state can authorize a new entry.
        return DailyRiskResult(False, f"DAILY_RISK_UNAVAILABLE:{type(exc).__name__}", utc_day)
