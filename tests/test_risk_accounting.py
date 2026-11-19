import pytest

from risk_accounting import (
    RiskAccountingError,
    build_balance_payload,
    unavailable_balance_payload,
)


def segregated_snapshot():
    return {
        "entry_enabled": True,
        "risk_equity": 1500.0,
        "risk_currency": "USDC",
        "available_funds": 900.0,
        "margin_mode": "segregated",
        "account_equity_usd": None,
        "assets": [
            {"asset": "USDC", "equity": 1500.0, "available": 900.0, "unit": "USDC"},
            {"asset": "BTC", "equity": 0.02, "available": 0.01, "unit": "BTC"},
        ],
        "source": "private/get_account_summaries",
    }


def test_segregated_balance_keeps_native_units_and_returns_usdc_risk():
    payload, risk = build_balance_payload(
        segregated_snapshot(),
        [{"floating_profit_loss_usd": 15.0}],
        updated_at="2026-11-13 10:00 UTC",
    )
    assert risk == 1500.0
    assert payload["risk_equity_currency"] == "USDC"
    assert payload["risk_equity"] == 1500.0
    assert payload["usdt"] == 0.0
    assert payload["equity_currency"] is None
    assert payload["assets"][1]["total"] == "0.02"
    assert payload["assets"][1]["unit"] == "BTC"
    assert payload["unrealised_usd"] == 15.0


def test_unrealized_pnl_is_not_added_to_equity():
    payload, risk = build_balance_payload(
        segregated_snapshot(),
        [{"floating_profit_loss_usd": 25.0}],
    )
    assert risk == 1500.0
    assert payload["risk_equity"] == 1500.0
    assert payload["unrealised_usd"] == 25.0


def test_cross_collateral_balance_is_reporting_only_and_entry_disabled():
    snapshot = segregated_snapshot()
    snapshot.update({
        "entry_enabled": False,
        "risk_equity": None,
        "available_funds": None,
        "margin_mode": "cross_collateral",
        "account_equity_usd": 2400.0,
        "risk_block_reason": "CROSS_COLLATERAL_USDC_ALLOCATION_NOT_VERIFIED",
    })
    payload, risk = build_balance_payload(snapshot, [])
    assert risk == 0.0
    assert payload["risk_entry_enabled"] is False
    assert payload["usdt"] == 2400.0
    assert payload["equity_currency"] == "USD"


def test_segregated_account_cannot_claim_usd_equity():
    snapshot = segregated_snapshot()
    snapshot["account_equity_usd"] = 1500.0
    with pytest.raises(RiskAccountingError, match="must not relabel"):
        build_balance_payload(snapshot, [])


def test_native_assets_require_correct_units_and_unique_currency():
    snapshot = segregated_snapshot()
    snapshot["assets"].append({"asset": "BTC", "equity": 0.01, "available": 0.01, "unit": "BTC"})
    with pytest.raises(RiskAccountingError, match="duplicated"):
        build_balance_payload(snapshot, [])


def test_account_pnl_field_is_informational_only_and_never_assumed():
    payload, _ = build_balance_payload(
        segregated_snapshot(), [{"floating_profit_loss": 25.0}]
    )
    assert payload["unrealised_usd"] is None
    assert payload["risk_equity"] == 1500.0


def test_unavailable_payload_zeroes_stale_risk_capital():
    payload = unavailable_balance_payload()
    assert payload["risk_equity"] == 0.0
    assert payload["risk_entry_enabled"] is False
    assert payload["usdt"] == 0.0
