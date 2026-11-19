from datetime import datetime, timezone
from types import SimpleNamespace
from decimal import Decimal

import pytest

import daily_risk_control as daily_control
from daily_risk_control import (
    DailyRiskControlError, calculate_daily_equity_drawdown, extract_external_flows,
    evaluate_daily_risk_guard,
)


def test_drawdown_counts_realized_and_unrealized_equity_change_and_hits_boundary():
    result = calculate_daily_equity_drawdown(
        baseline_equity="1000", current_equity="980", net_external_flows="0",
        max_daily_loss_fraction="0.02",
    )
    assert result["daily_pnl"] == Decimal("-20")
    assert result["loss_fraction"] == Decimal("0.02")
    assert result["limit_exceeded"] is True


@pytest.mark.parametrize(("current", "flow"), [("150", "50"), ("80", "-20")])
def test_deposits_withdrawals_and_internal_transfers_are_rebased(current, flow):
    result = calculate_daily_equity_drawdown(
        baseline_equity="100", current_equity=current, net_external_flows=flow,
        max_daily_loss_fraction="0.02",
    )
    assert result["daily_pnl"] == Decimal("0")
    assert result["limit_exceeded"] is False


def test_external_flows_only_include_account_funding_movements():
    rows = [
        {"type": "deposit", "currency": "USDC", "change": "50"},
        {"type": "withdrawal", "currency": "USDC", "change": "-10"},
        {"type": "transfer", "currency": "USDC", "change": "-5"},
        {"type": "trade", "currency": "USDC", "change": "-0.5"},
        {"type": "settlement", "currency": "USDC", "change": "2"},
        {"type": "stake_reward", "currency": "USDC", "change": "1"},
    ]
    assert extract_external_flows(rows) == Decimal("35")


def test_unknown_currency_type_or_position_transfer_fails_closed():
    with pytest.raises(DailyRiskControlError, match="currency"):
        extract_external_flows([{"type": "deposit", "change": "1"}])
    with pytest.raises(DailyRiskControlError, match="Unclassified"):
        extract_external_flows([{"type": "new_unseen_type", "currency": "USDC", "change": "1"}])
    with pytest.raises(DailyRiskControlError, match="manual reconciliation"):
        extract_external_flows([{"type": "lsp", "currency": "USDC", "role": "source"}])
    with pytest.raises(DailyRiskControlError, match="manual reconciliation"):
        extract_external_flows([{"type": "liquidation", "currency": "USDC"}])


class _FakeLedger:
    def __init__(self, baseline, halt=None):
        self.baseline = baseline
        self.halt = halt
        self.latched = []

    def first_account_risk_snapshot_between(self, **kwargs):
        return self.baseline

    def get_daily_risk_halt(self, **kwargs):
        return self.halt

    def latch_daily_risk_halt(self, **kwargs):
        self.latched.append(kwargs)
        self.halt = {"reason": kwargs["reason"]}
        return self.halt


class _FakeDeribit:
    require_execution_gateway = True

    def __init__(self, ledger, rows=None):
        self.execution_gateway = SimpleNamespace(
            coordinator=SimpleNamespace(
                exchange="DERIBIT", environment="testnet", account_id="acct-test",
                _lease_kwargs=lambda: {"lease_key": "k", "lease_owner_id": "o", "fencing_token": 1},
            ),
            ledger=ledger,
        )
        self.rows = rows or []
        self.calls = 0

    def get_transaction_log(self, currency, **kwargs):
        self.calls += 1
        return {"logs": self.rows, "continuation": None}


def _fixture(now):
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
    baseline = {"observed_at": day_start + 60, "equity": "1000"}
    return day_start, baseline


def test_daily_guard_latches_sticky_halt_at_threshold():
    now = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)
    _, baseline = _fixture(now)
    ledger = _FakeLedger(baseline)
    client = _FakeDeribit(ledger)
    result = evaluate_daily_risk_guard(client, current_equity="980", now=now)
    assert result.allowed is False
    assert result.reason == "MAX_DAILY_LOSS_LIMIT"
    assert len(ledger.latched) == 1
    assert ledger.latched[0]["loss_fraction"] == Decimal("0.02")


def test_daily_guard_allows_loss_inside_limit():
    now = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)
    _, baseline = _fixture(now)
    ledger = _FakeLedger(baseline)
    result = evaluate_daily_risk_guard(_FakeDeribit(ledger), current_equity="981", now=now)
    assert result.allowed is True
    assert result.reason == "DAILY_RISK_WITHIN_LIMIT"


def test_daily_guard_blocks_when_opening_baseline_missing_or_late():
    now = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)
    for baseline in (None, {"observed_at": datetime(2026, 11, 18, 0, 21, tzinfo=timezone.utc).timestamp(), "equity": "1000"}):
        result = evaluate_daily_risk_guard(_FakeDeribit(_FakeLedger(baseline)), current_equity="1000", now=now)
        assert result.allowed is False
        assert "BASELINE" in result.reason


def test_existing_daily_halt_blocks_without_requerying_transaction_log():
    now = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)
    client = _FakeDeribit(_FakeLedger(None, halt={"reason": "MAX_DAILY_LOSS_LIMIT"}))
    result = evaluate_daily_risk_guard(client, current_equity="1000", now=now)
    assert result.allowed is False
    assert result.reason == "MAX_DAILY_LOSS_LIMIT"
    assert client.calls == 0


def test_unknown_transaction_type_fails_closed():
    now = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)
    _, baseline = _fixture(now)
    client = _FakeDeribit(_FakeLedger(baseline), rows=[{"type": "future_new_type", "currency": "USDC"}])
    result = evaluate_daily_risk_guard(client, current_equity="1000", now=now)
    assert result.allowed is False
    assert result.reason.startswith("DAILY_RISK_UNAVAILABLE:")


def test_transaction_log_requests_are_rate_limited_across_independent_checks(monkeypatch):
    clock = [10.0]
    sleeps = []
    calls = []
    monkeypatch.setattr(daily_control, "_LAST_TRANSACTION_LOG_REQUEST_MONOTONIC", None)
    monkeypatch.setattr(daily_control.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(daily_control.time, "sleep", fake_sleep)

    def reader(*args, **kwargs):
        calls.append(clock[0])
        return {"logs": [], "continuation": None}

    daily_control._call_transaction_log_rate_limited(reader, "USDC")
    daily_control._call_transaction_log_rate_limited(reader, "USDC")

    assert len(calls) == 2
    assert calls[1] - calls[0] >= daily_control.TRANSACTION_LOG_MIN_INTERVAL_SECONDS
    assert len(sleeps) == 1


def test_absolute_daily_loss_cap_takes_precedence_over_looser_percentage_cap():
    result = calculate_daily_equity_drawdown(
        baseline_equity="1000", current_equity="994", net_external_flows="0",
        max_daily_loss_fraction="0.02", max_daily_loss_amount="5",
    )
    assert result["loss_amount"] == Decimal("6")
    assert result["effective_limit_amount"] == Decimal("5")
    assert result["limit_exceeded"] is True


def test_absolute_daily_loss_cap_does_not_loosen_percentage_cap():
    result = calculate_daily_equity_drawdown(
        baseline_equity="1000", current_equity="980", net_external_flows="0",
        max_daily_loss_fraction="0.02", max_daily_loss_amount="50",
    )
    assert result["effective_limit_amount"] == Decimal("20.00")
    assert result["limit_exceeded"] is True
