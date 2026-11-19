from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import testnet_readonly_check


class FakeReadOnlyClient:
    base = "https://test.deribit.com/api/v2"

    def __init__(self, client_id, client_secret, **kwargs):
        assert client_id == "readonly-client"
        assert client_secret == "readonly-secret"
        assert kwargs == {
            "testnet": True,
            "auth_scope": "trade:read account:read",
            "allow_mutations": False,
        }

    def get_server_time_ms(self):
        import time
        return int(time.time() * 1000)

    def get_risk_capital_snapshot(self):
        return {
            "assets": [{"asset": "USDC", "unit": "USDC", "equity": 123456.789, "available": 123450.0}],
            "account_equity_usd": 123456.789,
            "risk_equity": 123456.789,
            "risk_currency": "USDC",
            "entry_enabled": True,
            "risk_block_reason": "private-risk-marker",
            "margin_mode": "private-margin-marker",
        }

    def get_positions(self):
        return [{
            "instrument_name": "ETH_USDC-PERPETUAL",
            "direction": "buy",
            "size": 1.234567,
            "size_currency": 0.1234567,
            "average_price": 2000.0,
            "mark_price": 2010.0,
            "floating_profit_loss_usd": 99.99,
        }]

    def is_supported(self, symbol):
        return True

    def get_instrument_name(self, symbol):
        return "ETH_USDC-PERPETUAL"

    def verify_instrument_contract(self, symbol):
        return {"instrument_name": "ETH_USDC-PERPETUAL"}

    def get_order_book_spread(self, symbol):
        return {"best_bid": 2000.0, "best_ask": 2001.0, "spread_pct": (2001.0 - 2000.0) / ((2001.0 + 2000.0) / 2.0), "is_wide": False}


def test_readonly_evidence_redacts_account_and_position_values(monkeypatch, tmp_path):
    config = types.ModuleType("config")
    config.SYMBOLS = ["ETHUSDT"]
    deribit = types.ModuleType("deribit_client")
    deribit.DeribitClient = FakeReadOnlyClient
    deribit.TESTNET_BASE = "https://test.deribit.com/api/v2"
    deribit.WIDE_SPREAD_WARN_PCT = 0.003
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.setitem(sys.modules, "deribit_client", deribit)
    monkeypatch.setenv("DERIBIT_TESTNET_READONLY_CLIENT_ID", "readonly-client")
    monkeypatch.setenv("DERIBIT_TESTNET_READONLY_CLIENT_SECRET", "readonly-secret")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "abcdef123456")
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["testnet_readonly_check.py", "--output", str(output)])

    assert testnet_readonly_check.main() == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))

    assert evidence["environment"] == "DERIBIT_TESTNET"
    assert evidence["server_clock_sync"]["samples"] == 3
    assert evidence["server_clock_sync"]["clock_adjustment_performed"] is False
    assert evidence["server_clock_sync"]["conservative_skew_bound_ms"] <= 2000.0
    assert evidence["account_snapshot_complete"] is True
    assert evidence["balance_asset_count"] == 1
    assert evidence["position_snapshot_complete"] is True
    assert evidence["position_count"] == 1
    assert evidence["position_instrument_count"] == 1
    assert evidence["workflow_run_id"] == "12345"
    assert evidence["workflow_run_attempt"] == "2"
    assert evidence["commit_sha"] == "abcdef123456"
    assert evidence["order_submission_performed"] is False
    assert evidence["order_cancellation_performed"] is False
    assert evidence["account_values_recorded"] is False
    assert evidence["position_values_recorded"] is False

    serialized = output.read_text(encoding="utf-8")
    for forbidden in (
        "123456.789", "123450.0", "1.234567", "0.1234567", "99.99",
        "account_equity_usd", "risk_equity", "risk_currency", "balances",
        '"positions"', "private-risk-marker", "private-margin-marker",
        "readonly-secret",
        '"position_instruments"', '"balances"',
    ):
        assert forbidden not in serialized


import pytest


@pytest.mark.parametrize(
    "book, expected_error",
    [
        ({"best_bid": 2001.0, "best_ask": 2000.0, "spread_pct": 0.0005, "is_wide": False}, "not_positive_and_ordered"),
        ({"best_bid": 2000.0, "best_ask": 2020.0, "spread_pct": (2020.0 - 2000.0) / ((2020.0 + 2000.0) / 2.0), "is_wide": True}, "exceeds_pre_registered_limit"),
        ({"best_bid": 2000.0, "best_ask": 2001.0, "spread_pct": 0.05, "is_wide": False}, "spread_field_mismatch"),
        ({"best_bid": 2000.0, "best_ask": float("nan"), "spread_pct": 0.0, "is_wide": False}, "missing_or_non_finite"),
    ],
)
def test_readonly_preflight_rejects_crossed_or_wide_books(book, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        testnet_readonly_check._validate_order_book(book, max_spread_fraction=0.003)


def test_readonly_preflight_accepts_ordered_finite_book_below_limit():
    result = testnet_readonly_check._validate_order_book(
        {
            "best_bid": 2000.0,
            "best_ask": 2001.0,
            "spread_pct": (2001.0 - 2000.0) / ((2001.0 + 2000.0) / 2.0),
            "is_wide": False,
        },
        max_spread_fraction=0.003,
    )
    assert result["spread_bps"] > 0
    assert result["spread_limit_bps"] == 30.0


def test_readonly_preflight_rejects_server_clock_skew():
    class SkewedClock:
        def get_server_time_ms(self):
            import time
            return int(time.time() * 1000) + 5000

    with pytest.raises(ValueError, match="server_clock_skew_exceeds_limit"):
        testnet_readonly_check._measure_server_clock(SkewedClock(), max_skew_ms=2000.0, samples=2)


def test_readonly_preflight_rejects_invalid_server_clock_response():
    class InvalidClock:
        def get_server_time_ms(self):
            return 0

    with pytest.raises(ValueError, match="server_time_response_invalid"):
        testnet_readonly_check._measure_server_clock(InvalidClock(), samples=1)

def test_readonly_preflight_symbol_selection_rejects_empty_configuration():
    with pytest.raises(ValueError, match="configured_symbol_list_empty"):
        testnet_readonly_check._select_symbols([], 25)


@pytest.mark.parametrize(
    "configured_symbols",
    [None, "ETHUSDT", ["ETHUSDT", ""]],
)
def test_readonly_preflight_symbol_selection_rejects_malformed_configuration(configured_symbols):
    with pytest.raises(ValueError):
        testnet_readonly_check._select_symbols(configured_symbols, 25)


def test_readonly_preflight_symbol_selection_deduplicates_and_respects_limit():
    result = testnet_readonly_check._select_symbols(
        ["ETHUSDT", "ETHUSDT", "BTCUSDT"],
        2,
    )
    assert result == ["ETHUSDT", "BTCUSDT"]


def test_readonly_preflight_accepts_complete_account_asset_snapshot():
    account = {
        "assets": [
            {
                "asset": "USDC",
                "unit": "USDC",
                "equity": 100.0,
                "available": None,
            }
        ]
    }
    assert testnet_readonly_check._validate_account_snapshot(account) == 1


@pytest.mark.parametrize(
    "account",
    [
        {"assets": []},
        {"assets": [None]},
        {"assets": [{"asset": "USDC", "unit": "BTC", "equity": 1.0, "available": None}]},
        {"assets": [{"asset": "USDC", "unit": "USDC", "equity": float("nan"), "available": None}]},
        {"assets": [{"asset": "USDC", "unit": "USDC", "equity": 1.0, "available": float("inf")}]},
        {
            "assets": [
                {"asset": "USDC", "unit": "USDC", "equity": 1.0, "available": None},
                {"asset": "usdc", "unit": "USDC", "equity": 2.0, "available": None},
            ]
        },
    ],
)
def test_readonly_preflight_rejects_incomplete_or_malformed_account_assets(account):
    with pytest.raises(ValueError):
        testnet_readonly_check._validate_account_snapshot(account)


def test_readonly_preflight_accepts_matching_contract_metadata():
    assert testnet_readonly_check._validate_instrument_contract_result(
        "ETH_USDC-PERPETUAL",
        {"instrument_name": "ETH_USDC-PERPETUAL"},
    ) is True


@pytest.mark.parametrize(
    "instrument, contract",
    [
        ("", {"instrument_name": ""}),
        ("ETH_USDC-PERPETUAL", None),
        ("ETH_USDC-PERPETUAL", {}),
        ("ETH_USDC-PERPETUAL", {"instrument_name": "BTC_USDC-PERPETUAL"}),
    ],
)
def test_readonly_preflight_rejects_missing_or_mismatched_contract_metadata(instrument, contract):
    with pytest.raises(ValueError):
        testnet_readonly_check._validate_instrument_contract_result(
            instrument,
            contract,
        )
