import math
import pytest

import trade_executor


class FakeDeribit:
    def __init__(self, snapshot, positions=None):
        self.snapshot = snapshot
        self.positions = positions if positions is not None else []
        self.summary_reads = 0

    def get_risk_capital_snapshot(self):
        self.summary_reads += 1
        if isinstance(self.snapshot, Exception):
            raise self.snapshot
        return self.snapshot

    def get_positions(self):
        return list(self.positions)


def segregated_snapshot():
    return {
        "entry_enabled": True,
        "risk_equity": 1500.0,
        "risk_currency": "USDC",
        "available_funds": 900.0,
        "margin_mode": "segregated",
        "account_equity_usd": None,
        "assets": [
            {"asset": "USDC", "equity": 1500.0, "available": 900.0, "unit": "USDC"}
        ],
        "source": "private/get_account_summaries",
        "risk_block_reason": None,
    }


def test_save_balance_uses_one_snapshot_and_never_adds_upnl(monkeypatch):
    captured = {}
    fake = FakeDeribit(segregated_snapshot(), [{"floating_profit_loss_usd": 42.5}])
    monkeypatch.setattr(trade_executor, "save_json", lambda path, payload: captured.update(payload))

    result = trade_executor.save_balance(fake)

    assert result == 1500.0
    assert fake.summary_reads == 1
    assert captured["risk_equity"] == 1500.0
    assert captured["risk_equity_currency"] == "USDC"
    assert captured["usdt"] == 0.0
    assert captured["equity"] == 0.0
    assert captured["unrealised_usd"] == 42.5
    assert fake._last_risk_capital_snapshot["entry_enabled"] is True


def test_save_balance_cross_collateral_disables_entries(monkeypatch):
    captured = {}
    snapshot = segregated_snapshot()
    snapshot.update({
        "entry_enabled": False,
        "risk_equity": None,
        "available_funds": None,
        "margin_mode": "cross_collateral",
        "account_equity_usd": 2200.0,
        "risk_block_reason": "CROSS_COLLATERAL_USDC_ALLOCATION_NOT_VERIFIED",
    })
    fake = FakeDeribit(snapshot)
    monkeypatch.setattr(trade_executor, "save_json", lambda path, payload: captured.update(payload))

    assert trade_executor.save_balance(fake) == 0.0
    assert captured["account_equity_usd"] == 2200.0
    assert captured["risk_equity"] == 0.0
    assert captured["risk_entry_enabled"] is False


def test_save_balance_failure_invalidates_previous_risk_capital(monkeypatch):
    captured = {}
    fake = FakeDeribit(TimeoutError("snapshot failed"))
    monkeypatch.setattr(trade_executor, "save_json", lambda path, payload: captured.update(payload))

    assert trade_executor.save_balance(fake) == 0.0
    assert captured["risk_equity"] == 0.0
    assert captured["risk_entry_enabled"] is False
    assert captured["source"] == "unavailable"


def test_execute_trade_blocks_without_validated_usdc_snapshot(monkeypatch):
    monkeypatch.setattr(trade_executor, "_execution_allows_new_entries", lambda: True)
    monkeypatch.setattr(trade_executor, "_deribit_testnet_mutations_enabled", lambda: True)
    fake = FakeDeribit(segregated_snapshot())
    fake._last_risk_capital_snapshot = {
        "entry_enabled": False,
        "risk_currency": "USDC",
        "risk_equity": None,
    }

    result = trade_executor.execute_trade(
        fake, {}, 1.0, 1500.0, base_min_conf=55.0
    )

    assert result is False


def test_execute_trade_blocks_usd_balance_for_usdc_settled_strategy(monkeypatch):
    monkeypatch.setattr(trade_executor, "_execution_allows_new_entries", lambda: True)
    monkeypatch.setattr(trade_executor, "_deribit_testnet_mutations_enabled", lambda: True)
    fake = FakeDeribit(segregated_snapshot())
    fake._last_risk_capital_snapshot = {
        "entry_enabled": True,
        "risk_currency": "USD",
        "risk_equity": 1500.0,
        "available_funds": 1000.0,
    }

    result = trade_executor.execute_trade(
        fake, {"symbol": "ETHUSDT"}, 1.0, 1500.0, base_min_conf=55.0
    )

    assert result is False



def test_portfolio_risk_blocks_untracked_subminimum_position():
    from deribit_client import SYMBOL_MAP
    symbol = "ETHUSDT"
    instrument = SYMBOL_MAP[symbol]["instrument"]

    class DustExchange:
        def get_min_trade_amount(self, _symbol):
            return 0.001

        def get_position_size(self, _symbol):
            return 0.0001

        def get_mark_price(self, _symbol):
            return 2500.0

        def get_open_orders(self, _symbol):
            return []

    with pytest.raises(ValueError, match="Untracked non-zero"):
        trade_executor._verified_open_portfolio_risk_usdc(
            DustExchange(), {}, [{"instrument_name": instrument, "size_currency": 0.0001, "direction": "buy"}],
        )


def test_portfolio_risk_counts_verified_dust_reserve_for_unrelated_symbol():
    from deribit_client import SYMBOL_MAP
    from dust_residual_policy import classify_verified_dust_residual
    from datetime import datetime, timezone
    symbol = "ETHUSDT"
    instrument = SYMBOL_MAP[symbol]["instrument"]
    dust = classify_verified_dust_residual(
        symbol=symbol, signal="BUY", first_signed_position="0.00001",
        second_signed_position="0.00001", min_trade_amount="0.0001",
        mark_price="2500", first_open_orders=[], second_open_orders=[],
        entry_order_id="entry-1", entry_order_state="filled",
        entry_filled_quantity="0.00002", prediction_id="pred-1",
        observed_at=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    trades = {symbol: {
        "symbol": symbol, "closed": True, "pred_id": "pred-1",
        "order_ids": {"entry": "entry-1"}, "dust_residual": dust,
        "emergency_recovery_required": False,
        "manual_reconciliation_required": False,
        "protection_reconciliation_required": False,
    }}

    class DustExchange:
        def get_min_trade_amount(self, _symbol):
            return 0.0001

        def get_position_size(self, _symbol):
            return 0.00001

        def get_mark_price(self, _symbol):
            return 2500.0

        def get_open_orders(self, _symbol):
            return []

    result = trade_executor._verified_open_portfolio_risk_usdc(
        DustExchange(), trades,
        [{"instrument_name": instrument, "size_currency": 0.00001, "direction": "buy"}],
    )
    assert result == 0.10


def test_emergency_close_does_not_treat_tiny_nonzero_residual_as_flat(monkeypatch):
    observed = {}

    class TinyDustExchange:
        def get_position_size(self, _symbol):
            return 1e-8

        def get_min_trade_amount(self, _symbol):
            return 0.0001

    exchange = TinyDustExchange()
    original_verify = trade_executor._verify_actually_closed

    def verify_with_captured_tolerance(client, symbol, tolerance=1e-8):
        observed["tolerance"] = tolerance
        return original_verify(client, symbol, tolerance=tolerance)

    def record_dust(_client, _trades, _sig, _symbol, _signal, _entry, _orders, first, _reason):
        observed["dust_signed"] = first
        return True

    monkeypatch.setattr(trade_executor, "_verify_actually_closed", verify_with_captured_tolerance)
    monkeypatch.setattr(trade_executor, "_cancel_all_open_orders_for_symbol", lambda *_a, **_k: True)
    monkeypatch.setattr(trade_executor, "_record_verified_dust_residual", record_dust)

    result = trade_executor._emergency_close_and_verify(
        exchange, {}, {}, "ETHUSDT", "BUY", 2500.0, 0.0001,
        {"entry": "entry-tiny-dust"}, "post-fill invariant test",
    )

    assert result is False  # Non-zero residual was not reported as flat.
    assert observed["tolerance"] == 1e-12
    assert observed["dust_signed"] == 1e-8
