import pytest

from deribit_client import DeribitClient


def make_client(monkeypatch, rows):
    client = DeribitClient.__new__(DeribitClient)
    monkeypatch.setattr(client, "get_account_summaries", lambda: {"summaries": rows})
    return client


def row(currency, equity, available, cross=False, total_usd=None, margin_model=None):
    result = {
        "currency": currency,
        "equity": equity,
        "available_funds": available,
        # cross_collateral_enabled may indicate feature availability; the
        # active margin_model is the authoritative account classification.
        "cross_collateral_enabled": cross,
        "margin_model": margin_model or ("cross_sm" if cross else "segregated_sm"),
    }
    if total_usd is not None:
        result["total_equity_usd"] = total_usd
    return result


def test_segregated_risk_capital_uses_usdc_only(monkeypatch):
    client = make_client(monkeypatch, [
        row("BTC", 0.25, 0.20),
        row("USDC", 1500, 900),
        row("ETH", 2.0, 1.5),
    ])
    snapshot = client.get_risk_capital_snapshot()
    assert snapshot["entry_enabled"] is True
    assert snapshot["risk_equity"] == 1500.0
    assert snapshot["risk_currency"] == "USDC"
    assert snapshot["margin_mode"] == "segregated"


def test_cross_collateral_has_usd_reporting_but_disables_new_entries(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 400, True, 2400),
        row("BTC", 0.02, 0.01, True, 2400),
    ])
    snapshot = client.get_risk_capital_snapshot()
    assert snapshot["entry_enabled"] is False
    assert snapshot["risk_equity"] is None
    assert snapshot["account_equity_usd"] == 2400.0
    assert snapshot["risk_currency"] == "USDC"


def test_cross_collateral_requires_consistent_usd_totals(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 400, True, 2400),
        row("BTC", 0.02, 0.01, True, 2300),
    ])
    with pytest.raises(ValueError, match="Inconsistent"):
        client.get_risk_capital_snapshot()


def test_account_summary_flags_must_agree(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 400, False),
        row("BTC", 0.02, 0.01, True, 2400),
    ])
    with pytest.raises(ValueError, match="disagree"):
        client.get_risk_capital_snapshot()


def test_missing_margin_model_fails_closed(monkeypatch):
    client = make_client(monkeypatch, [{"currency": "USDC", "equity": 1000, "available_funds": 800}])
    with pytest.raises(ValueError, match="margin_model"):
        client.get_risk_capital_snapshot()


def test_segregated_account_requires_usdc_and_available_funds(monkeypatch):
    only_btc = make_client(monkeypatch, [row("BTC", 0.25, 0.20)])
    with pytest.raises(ValueError, match="no USDC"):
        only_btc.get_risk_capital_snapshot()

    no_available = make_client(monkeypatch, [row("USDC", 1000, 0)])
    with pytest.raises(ValueError, match="not positive"):
        no_available.get_risk_capital_snapshot()


def test_legacy_balance_methods_keep_native_units_and_refuse_false_usd(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 800),
        row("BTC", 0.02, 0.01),
    ])
    balances = client.get_all_balances()
    assert balances["USDC"]["equity"] == 1000
    assert balances["BTC"]["equity"] == 0.02
    assert balances["BTC"]["unit"] == "BTC"
    with pytest.raises(RuntimeError, match="must not be relabelled USD"):
        client.get_total_equity_usd()


def test_total_equity_usd_returns_only_explicit_cross_account_value(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 800, True, 2400),
        row("BTC", 0.02, 0.01, True, 2400),
    ])
    assert client.get_total_equity_usd() == 2400.0


def test_margin_model_overrides_cross_collateral_feature_flag(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 800, True, 2400, margin_model="segregated_sm"),
        row("BTC", 0.02, 0.01, True, 2400, margin_model="segregated_sm"),
    ])
    snapshot = client.get_risk_capital_snapshot()
    assert snapshot["entry_enabled"] is True
    assert snapshot["risk_equity"] == 1000
    assert snapshot["margin_mode"] == "segregated"
    assert snapshot["margin_model"] == "segregated_sm"
    assert snapshot["account_equity_usd"] is None


def test_cross_margin_model_is_authoritative_when_feature_flag_false(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 400, False, 2400, margin_model="cross_sm"),
        row("BTC", 0.02, 0.01, False, 2400, margin_model="cross_sm"),
    ])
    snapshot = client.get_risk_capital_snapshot()
    assert snapshot["entry_enabled"] is False
    assert snapshot["risk_equity"] is None
    assert snapshot["margin_mode"] == "cross_collateral"
    assert snapshot["margin_model"] == "cross_sm"
    assert snapshot["account_equity_usd"] == 2400.0


def test_unknown_margin_model_fails_closed(monkeypatch):
    client = make_client(monkeypatch, [
        row("USDC", 1000, 800, margin_model="unknown_model"),
    ])
    with pytest.raises(ValueError, match="Unknown or missing margin_model"):
        client.get_risk_capital_snapshot()
