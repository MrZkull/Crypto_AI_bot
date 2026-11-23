from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import daily_risk_baseline_capture as capture


class FakeClient:
    base = capture.EXPECTED_TESTNET_BASE
    allow_mutations = False
    require_execution_gateway = False

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.session = SimpleNamespace(close=lambda: None)
        self.account_reads = 0
        self.position_reads = 0

    def get_server_time_ms(self):
        return int(FIXED_EPOCH * 1000)

    def get_account_summaries(self):
        self.account_reads += 1
        return {"summaries": [
            {"currency": "USDC", "equity": "1000", "available_funds": "0", "margin_model": "segregated_sm"},
            {"currency": "BTC", "equity": "0.01", "available_funds": "0", "margin_model": "segregated_sm"},
        ]}

    def get_positions(self):
        self.position_reads += 1
        return []


class FakeLedger:
    def __init__(self, *args, **kwargs):
        self.snapshots = []
        self.released = False
        self.renewed = False

    def acquire_fenced_risk_capture_lease(self, *args, **kwargs):
        self.lease_args = (args, kwargs)
        return 9

    def first_account_risk_snapshot_between(self, **kwargs):
        self.baseline_query = kwargs
        return self.snapshots[0] if self.snapshots else None

    def renew_fenced_risk_capture_lease(self, *args, **kwargs):
        self.renewed = True
        return True

    def record_account_risk_snapshot_with_capture_lease(self, **kwargs):
        self.record = kwargs
        row = {"observed_at": kwargs["observed_at"], "equity": kwargs["equity"]}
        self.snapshots.append(row)
        return row

    def release_fenced_risk_capture_lease(self, *args, **kwargs):
        self.released = True
        return True

    def close(self):
        pass


FIXED_EPOCH = datetime(2026, 11, 19, 0, 5, tzinfo=timezone.utc).timestamp()


def run_capture(monkeypatch, client=None, ledger=None):
    client = client or FakeClient()
    ledger = ledger or FakeLedger()
    monkeypatch.setattr(capture.time, "time", lambda: FIXED_EPOCH)
    def client_factory(*args, **kwargs):
        client.kwargs = kwargs
        return client

    result = capture.capture_daily_baseline(
        client_id="read-only-id", client_secret="read-only-secret",
        account_id="test-account", database_dsn="postgresql://dedicated-risk-writer",
        client_factory=client_factory,
        ledger_factory=lambda *a, **k: ledger,
    )
    return result, client, ledger


def test_capture_uses_readonly_client_and_dedicated_capture_lease(monkeypatch):
    result, client, ledger = run_capture(monkeypatch)
    assert result["status"] == "CAPTURED"
    assert result["equity_usdc"] == "1000.0"
    assert client.kwargs["auth_scope"] == "trade:read account:read"
    assert client.kwargs["allow_mutations"] is False
    assert ledger.record["source"].startswith("readonly_daily_baseline_capture")
    assert ledger.record["risk_capture_lease_key"].endswith(":DAILY_RISK_BASELINE")
    assert ledger.renewed and ledger.released


def test_zero_available_funds_does_not_prevent_equity_baseline(monkeypatch):
    result, _, _ = run_capture(monkeypatch)
    assert result["status"] == "CAPTURED"


def test_existing_baseline_is_immutable_and_skips_private_account_reads(monkeypatch):
    client = FakeClient()
    ledger = FakeLedger()
    ledger.snapshots.append({"observed_at": FIXED_EPOCH, "equity": "999"})
    result, client, ledger = run_capture(monkeypatch, client, ledger)
    assert result["status"] == "ALREADY_CAPTURED"
    assert client.account_reads == 0
    assert client.position_reads == 0
    assert ledger.released


def test_capture_fails_outside_opening_window_and_still_releases_lease(monkeypatch):
    client = FakeClient()
    ledger = FakeLedger()
    late = datetime(2026, 11, 19, 0, 21, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(capture.time, "time", lambda: late)
    client.get_server_time_ms = lambda: int(late * 1000)
    with pytest.raises(capture.BaselineCaptureError, match="first 20-minute"):
        capture.capture_daily_baseline(
            client_id="id", client_secret="secret", account_id="acct",
            database_dsn="postgresql://dedicated-risk-writer",
            client_factory=lambda *a, **k: client,
            ledger_factory=lambda *a, **k: ledger,
        )
    assert ledger.released


def test_capture_refuses_mutation_enabled_client(monkeypatch):
    class UnsafeClient(FakeClient):
        allow_mutations = True
    client = UnsafeClient()
    ledger = FakeLedger()
    monkeypatch.setattr(capture.time, "time", lambda: FIXED_EPOCH)
    with pytest.raises(capture.BaselineCaptureError, match="mutation-enabled"):
        capture.capture_daily_baseline(
            client_id="id", client_secret="secret", account_id="acct",
            database_dsn="postgresql://dedicated-risk-writer",
            client_factory=lambda *a, **k: client,
            ledger_factory=lambda *a, **k: ledger,
        )
    assert not ledger.released  # No lease was ever acquired.


def test_capture_rejects_cross_collateral_or_missing_usdc_summary():
    with pytest.raises(capture.BaselineCaptureError, match="segregated"):
        capture._usdc_equity_from_summaries({"summaries": [
            {"currency": "USDC", "equity": "1000", "margin_model": "cross_pm"},
        ]})
    with pytest.raises(capture.BaselineCaptureError, match="Exactly one USDC"):
        capture._usdc_equity_from_summaries({"summaries": [
            {"currency": "BTC", "equity": "1", "margin_model": "segregated_sm"},
        ]})
