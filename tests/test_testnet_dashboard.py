from __future__ import annotations

import base64
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import dashboard
from deribit_client import TESTNET_BASE

AUTH = {
    "DASHBOARD_AUTH_USERNAME": "cryptobot-admin",
    "DASHBOARD_AUTH_PASSWORD": "test-only-dashboard-passphrase-long-enough",
    "DASHBOARD_PUBLIC_ORIGIN": "https://cryptobot-test.example",
}


def auth_headers():
    token = base64.b64encode(b"cryptobot-admin:test-only-dashboard-passphrase-long-enough").decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "Origin": AUTH["DASHBOARD_PUBLIC_ORIGIN"],
    }


class FakeTestnetClient:
    base = TESTNET_BASE

    def __init__(self, *, positions=None, orders=None, fills=None, fail_positions=False, fail_orders=False):
        self.positions = positions if positions is not None else []
        self.orders = orders if orders is not None else []
        self.fills = fills if fills is not None else []
        self.fail_positions = fail_positions
        self.fail_orders = fail_orders
        self.open_order_instruments = []
        self.fill_instruments = []

    def get_account_summary(self, currency):
        assert currency == "USDC"
        return {"equity": 100.0, "currency": "USDC"}

    def get_account_summaries(self):
        return {"summaries": [{
            "currency": "USDC", "equity": 100.0, "available_funds": 90.0,
            "cross_collateral_enabled": True, "margin_model": "segregated_sm",
            "total_equity_usd": 100.0,
        }]}

    def get_all_balances(self):
        raise AssertionError("dashboard must use account summaries, not legacy per-currency USD sums")

    def get_positions(self):
        if self.fail_positions:
            raise RuntimeError("private position endpoint unavailable")
        return self.positions

    def get_open_orders_by_instrument(self, instrument_name):
        self.open_order_instruments.append(instrument_name)
        if self.fail_orders:
            raise RuntimeError("open orders unavailable")
        return self.orders

    def get_open_orders(self, symbol):
        if self.fail_orders:
            raise RuntimeError("open orders unavailable")
        return self.orders

    def get_trade_history_by_instrument_name(self, instrument_name, count=10, *, strict=False):
        assert strict is True
        self.fill_instruments.append(instrument_name)
        return self.fills

    def get_trade_history_for_instrument(self, symbol, count=10, *, strict=False):
        assert strict is True
        return self.fills


def _position():
    return {
        "instrument_name": "ETH_USDC-PERPETUAL",
        "size": 2.0,
        "size_currency": 0.2,
        "direction": "buy",
        "average_price": 3000.0,
        "mark_price": 3010.0,
        "floating_profit_loss_usd": 2.0,
    }


def test_testnet_overview_returns_host_pinned_read_only_snapshot():
    client = FakeTestnetClient(
        positions=[_position()],
        orders=[{
            "order_type": "stop_market", "trigger": "mark_price", "reduce_only": True,
            "order_state": "open", "trigger_price": 2900.0,
        }],
        fills=[{
            "trade_id": "trade-1", "order_id": "order-1", "direction": "buy",
            "amount": 2.0, "price": 3000.0, "fee": 0.01, "fee_currency": "USDC",
            "timestamp": 1794000000000,
        }],
    )
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client), \
         patch.object(dashboard, "get", return_value=[]):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["environment"] == "TESTNET"
    assert payload["verified_host"] == "test.deribit.com"
    assert payload["read_only"] is True
    assert payload["account_equity_usd"] is None
    assert payload["account_equity_native"] == 100.0
    assert payload["account_equity_currency"] == "USDC"
    assert client.open_order_instruments == ["ETH_USDC-PERPETUAL"]
    assert client.fill_instruments == ["ETH_USDC-PERPETUAL"]
    assert payload["positions"][0]["stop_status"] == "PRESENT"
    assert payload["fills"][0]["order_id"] == "order-1"
    assert payload["bot_journal"]["source"] == "GitHub trade_history.json"


def test_missing_readonly_credentials_fail_closed_without_local_balance_fallback():
    with patch.dict(os.environ, {**AUTH, "DERIBIT_TESTNET_READONLY_CLIENT_ID": "", "DERIBIT_TESTNET_READONLY_CLIENT_SECRET": ""}), \
         patch.object(dashboard, "deribit_client", return_value=None), \
         patch.object(dashboard, "get") as get_file:
        response = dashboard.app.test_client().get("/api/balance", headers=auth_headers())

    assert response.status_code == 503
    assert response.get_json()["ok"] is False
    get_file.assert_not_called()


def test_open_positions_failure_is_not_returned_as_flat_account():
    client = FakeTestnetClient(fail_positions=True)
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())

    assert response.status_code == 503
    assert response.get_json()["connection"] == "UNVERIFIED"


def test_open_orders_failure_is_reported_as_unknown_stop_state():
    client = FakeTestnetClient(positions=[_position()], fail_orders=True)
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client), \
         patch.object(dashboard, "get", return_value=[]):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["positions"][0]["stop_status"] == "UNKNOWN"
    assert payload["order_query_errors"][0]["instrument_name"] == "ETH_USDC-PERPETUAL"


def test_wrong_deribit_host_fails_closed_before_private_reads():
    client = FakeTestnetClient()
    client.base = "https://www.deribit.com/api/v2"
    client.get_account_summary = Mock(side_effect=AssertionError("must not query production"))
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())

    assert response.status_code == 503
    client.get_account_summary.assert_not_called()


def test_read_only_deployment_rejects_mutating_request_before_handler():
    client_factory = Mock(side_effect=AssertionError("mutating handler must not reach exchange"))
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "true"}), \
         patch.object(dashboard, "deribit_client", client_factory):
        response = dashboard.app.test_client().post(
            "/api/close_trade",
            headers={**auth_headers(), "Origin": AUTH["DASHBOARD_PUBLIC_ORIGIN"]},
            json={"symbol": "ETHUSDT"},
        )

    assert response.status_code == 403
    assert response.get_json()["error"] == "dashboard_read_only"
    client_factory.assert_not_called()


def test_execution_metrics_do_not_invent_fee_volume_or_slippage():
    journal = [{
        "symbol": "ETHUSDT", "signal": "BUY", "closed": True,
        "opened_at": "2026-11-01T10:00:00+00:00", "closed_at": "2026-11-01T12:00:00+00:00",
        "pnl": 1.5, "pnl_currency": "USDT",
    }]
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "get", return_value=journal):
        response = dashboard.app.test_client().get("/api/execution", headers=auth_headers())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["gross_pnl"] == 1.5
    assert payload["pnl_currency"] == "USDT"
    assert payload["avg_duration_h"] == 2.0
    assert payload["total_fees"] is None
    assert payload["net_pnl"] is None
    assert payload["total_volume_traded_usd"] is None
    assert payload["avg_slippage_pct"] is None
    assert payload["fee_coverage_complete"] is False


def test_model_health_uses_performance_artifact_and_not_hardcoded_accuracy():
    performance = {"test_accuracy": "50.4%"}
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "get_model_metadata", return_value={
        "model_name": "Ensemble", "rec_buy_conf": 44.0, "rec_sell_conf": 52.0,
    }), patch.object(dashboard, "get", return_value=performance):
        response = dashboard.app.test_client().get("/api/model_health", headers=auth_headers())

    assert response.status_code == 200
    assert response.get_json()["test_accuracy"] == "50.4%"
    assert response.get_json()["calibration"] is None


def test_open_trade_metadata_falls_back_when_root_ledger_is_empty():
    client = FakeTestnetClient(positions=[_position()])
    def fake_get(path, default):
        if path == "trades.json":
            return {}
        if path == "data/trades.json":
            return {"ETHUSDT": {"stop": 2900.0, "tp1": 3200.0, "tp2": 3400.0}}
        return default

    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client), \
         patch.object(dashboard, "get", side_effect=fake_get):
        response = dashboard.app.test_client().get("/api/trades/open", headers=auth_headers())

    assert response.status_code == 200
    assert response.get_json()[0]["stop"] == 2900.0
    assert response.get_json()[0]["tp1"] == 3200.0



def test_native_currency_equity_is_not_mislabeled_as_usd():
    client = FakeTestnetClient()
    client.get_account_summaries = lambda: {"summaries": [
        {"currency": "USDC", "equity": 75.0, "available_funds": 70.0,
         "cross_collateral_enabled": False, "margin_model": "segregated_sm"},
        {"currency": "BTC", "equity": 0.01, "available_funds": 0.009,
         "cross_collateral_enabled": False, "margin_model": "segregated_sm"},
    ]}
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["account_equity_usd"] is None
    assert payload["account_equity_native"] == 75.0
    assert payload["account_equity_currency"] == "USDC"
    assert {item["asset"] for item in payload["balances"]} == {"USDC", "BTC"}


def test_exact_unknown_instrument_is_never_rewritten_as_usdt_symbol():
    position = {
        "instrument_name": "ETH-PERPETUAL", "size": 2.0, "size_currency": 0.1,
        "direction": "buy", "average_price": 3000.0, "mark_price": 3010.0,
        "floating_profit_loss_usd": 1.0,
    }
    client = FakeTestnetClient(positions=[position])
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client), \
         patch.object(dashboard, "get", return_value=[]):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["positions"][0]["symbol"] == "ETH-PERPETUAL"
    assert payload["positions"][0]["instrument_name"] == "ETH-PERPETUAL"
    assert client.open_order_instruments == ["ETH-PERPETUAL"]
    assert client.fill_instruments == ["ETH-PERPETUAL"]


def test_dashboard_mode_is_independent_of_exchange_credentials():
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "true"}):
        response = dashboard.app.test_client().get("/api/dashboard/mode", headers=auth_headers())
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "read_only": True,
        "execution_controls_enabled": False,
        "control_plane": "separate_approved_execution_service",
    }


def test_read_only_mode_allows_in_memory_pdf_export_only():
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "true"}), \
         patch.object(dashboard, "HAS_REPORTLAB", True), \
         patch.object(dashboard, "generate_pdf_bytes", return_value=b"%PDF-test"):
        response = dashboard.app.test_client().post(
            "/api/download_report_pdf",
            headers={**auth_headers(), "Origin": AUTH["DASHBOARD_PUBLIC_ORIGIN"]},
            json={"scope": "test", "summary": {}, "trades": []},
        )
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"


def test_read_only_mode_allows_in_memory_paper_pdf_export_only():
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "true"}), \
         patch.object(dashboard, "HAS_REPORTLAB", True), \
         patch.object(dashboard, "generate_paper_pdf_bytes", return_value=b"%PDF-paper-test"):
        response = dashboard.app.test_client().post(
            "/api/download_paper_report_pdf",
            headers={**auth_headers(), "Origin": AUTH["DASHBOARD_PUBLIC_ORIGIN"]},
            json={"paper_only": True, "events": [], "portfolio": {}},
        )
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"


def test_read_only_mode_rejects_dashboard_admin_and_notification_actions():
    paths = (
        "/api/probation/bench",
        "/api/probation/release",
        "/api/cooldown/release",
        "/api/adaptation_proposals/proposal-1/approve",
        "/api/adaptation_proposals/proposal-1/reject",
        "/api/send_report",
    )
    headers = {**auth_headers(), "Origin": AUTH["DASHBOARD_PUBLIC_ORIGIN"]}
    with patch.dict(os.environ, AUTH):
        for path in paths:
            response = dashboard.app.test_client().post(path, headers=headers, json={})
            assert response.status_code == 403, path
            assert response.get_json()["error"] == "dashboard_read_only", path


def test_balance_endpoint_reports_currency_units_without_fabricating_usd_total():
    client = FakeTestnetClient()
    client.get_account_summaries = lambda: {"summaries": [
        {"currency": "USDC", "equity": 75.0, "available_funds": 70.0,
         "cross_collateral_enabled": False, "margin_model": "segregated_sm"},
        {"currency": "BTC", "equity": 0.01, "available_funds": 0.009,
         "cross_collateral_enabled": False, "margin_model": "segregated_sm"},
    ]}
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/balance", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["usdt"] is None
    assert payload["equity"] is None
    assert payload["native_equity"] == 75.0
    assert payload["native_equity_currency"] == "USDC"
    assert {item["unit"] for item in payload["assets"]} == {"USDC", "BTC"}


def test_open_trades_preserves_unmapped_exchange_instrument_identity():
    position = {
        "instrument_name": "ETH-PERPETUAL", "size": 2.0, "size_currency": 0.1,
        "direction": "buy", "average_price": 3000.0, "mark_price": 3010.0,
        "floating_profit_loss_usd": 1.0,
    }
    client = FakeTestnetClient(positions=[position])
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client),          patch.object(dashboard, "get", return_value={"ETHUSDT": {"stop": 2900.0}}):
        response = dashboard.app.test_client().get("/api/trades/open", headers=auth_headers())
    assert response.status_code == 200
    trade = response.get_json()[0]
    assert trade["symbol"] == "ETH-PERPETUAL"
    assert trade["stop"] == 0.0


def test_deribit_exact_instrument_order_and_trade_reads_do_not_remap_symbols():
    from deribit_client import DeribitClient
    client = DeribitClient.__new__(DeribitClient)
    client._get = Mock(return_value=[{"order_id": "order-1"}])
    assert client.get_open_orders_by_instrument("ETH-PERPETUAL")[0]["order_id"] == "order-1"
    client._get.assert_called_once_with(
        "/private/get_open_orders_by_instrument", {"instrument_name": "ETH-PERPETUAL"}
    )
    client._get.reset_mock(return_value=True)
    client._get.return_value = [{"trade_id": "trade-1"}]
    assert client.get_trade_history_by_instrument_name("ETH-PERPETUAL", strict=True)[0]["trade_id"] == "trade-1"
    client._get.assert_called_once_with(
        "/private/get_user_trades_by_instrument",
        {"instrument_name": "ETH-PERPETUAL", "count": 10, "sorting": "desc"},
    )


def test_deribit_strict_account_summary_propagates_private_api_failure():
    from deribit_client import DeribitClient

    client = DeribitClient.__new__(DeribitClient)
    client._get = Mock(side_effect=RuntimeError("private account endpoint failed"))
    with pytest.raises(RuntimeError):
        client.get_account_summary("USDC")
    client._get.assert_called_once_with(
        "/private/get_account_summary", {"currency": "USDC", "extended": "true"}
    )


def test_deribit_account_summary_rejects_unknown_currency_and_malformed_payload():
    from deribit_client import DeribitClient

    client = DeribitClient.__new__(DeribitClient)
    client._get = Mock(return_value=[])
    with pytest.raises(ValueError):
        client.get_account_summary("DOGE")
    client._get.assert_not_called()
    with pytest.raises(ValueError):
        client.get_account_summary("USDC")


def test_currency_safe_pnl_refuses_unlabelled_or_mixed_currency_records():
    assert dashboard._currency_safe_pnl([
        {"pnl": 1.25, "pnl_currency": "USDC"},
        {"pnl": -0.25, "pnl_currency": "USDC"},
    ]) == (1.0, "USDC")
    assert dashboard._currency_safe_pnl([
        {"pnl": 1.0, "pnl_currency": "USDC"},
        {"pnl": 2.0, "pnl_currency": "USDT"},
    ]) == (None, None)
    assert dashboard._currency_safe_pnl([{"pnl": 1.0}]) == (None, None)


def test_currency_safe_pnl_empty_history_has_zero_configured_settlement_pnl():
    assert dashboard._currency_safe_pnl([]) == (0.0, "USDC")


def test_currency_safe_pnl_rejects_non_finite_or_malformed_values():
    assert dashboard._currency_safe_pnl([
        {"pnl": float("nan"), "pnl_currency": "USDC"}
    ]) == (None, None)
    assert dashboard._currency_safe_pnl([
        {"pnl": "not-a-number", "pnl_currency": "USDC"}
    ]) == (None, None)


def test_dashboard_uses_active_margin_model_not_cross_feature_flag():
    client = FakeTestnetClient()
    client.get_account_summaries = lambda: {"summaries": [
        {"currency": "USDC", "equity": 75.0, "available_funds": 70.0,
         "cross_collateral_enabled": True, "margin_model": "segregated_sm",
         "total_equity_usd": 75.0},
    ]}
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["margin_model"] == "segregated_sm"
    assert payload["account_equity_usd"] is None
    assert payload["account_equity_native"] == 75.0


def test_dashboard_cross_model_uses_account_wide_usd_total():
    client = FakeTestnetClient()
    client.get_account_summaries = lambda: {"summaries": [
        {"currency": "USDC", "equity": 75.0, "available_funds": 40.0,
         "cross_collateral_enabled": False, "margin_model": "cross_sm",
         "total_equity_usd": 120.0},
    ]}
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().get("/api/testnet/overview", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["margin_model"] == "cross_sm"
    assert payload["account_equity_usd"] == 120.0


def _heal_route_client(*, tp_timeout=False, stop_timeout=False):
    client = Mock()
    client.get_position_size.return_value = 0.25
    client.get_live_price.return_value = 2000.0
    client.round_amount.side_effect = lambda symbol, amount: amount
    client.round_price.side_effect = lambda symbol, price: price
    client.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    client.place_stop_loss.return_value = {"order": {"order_id": "sl-1"}}
    client.place_limit_order.return_value = {"order": {"order_id": "tp-1"}}
    client.get_order.return_value = {
        "order_id": "tp-1", "instrument_name": "ETH_USDC-PERPETUAL",
        "order_type": "limit", "direction": "sell", "reduce_only": True,
        "amount": 0.25, "price": 2030.0, "order_state": "open",
    }
    if tp_timeout:
        client.place_limit_order.side_effect = TimeoutError("TP response timeout")
    if stop_timeout:
        client.place_stop_loss.side_effect = TimeoutError("stop response timeout")
    return client


def _post_dashboard_heal(tmp_path, client, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dashboard, "get", lambda key, default=None: {} if key == "trades.json" else default)
    monkeypatch.setattr(dashboard, "get_live_config", lambda: {
        "atr_stop_mult": 1.0, "atr_target1_mult": 1.0,
    })
    monkeypatch.setattr(dashboard, "gh_push", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "bust", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard.requests, "get", lambda *args, **kwargs: Mock(ok=False))
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().post(
            "/api/trades/heal", headers=auth_headers(), json={"symbol": "ETHUSDT"}
        )
    return response


def test_dashboard_heal_is_disabled_in_monitoring_release(tmp_path, monkeypatch):
    client = _heal_route_client()
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "false"}), \
         patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().post(
            "/api/trades/heal", headers=auth_headers(), json={"symbol": "ETHUSDT"}
        )

    assert response.status_code == 403
    assert response.get_json()["error"] == "dashboard_read_only"
    client.place_stop_loss.assert_not_called()
    client.place_limit_order.assert_not_called()
    assert not (tmp_path / "trades.json").exists()


def test_dashboard_close_and_kill_switch_are_disabled_in_monitoring_release():
    client = Mock()
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "false"}), \
         patch.object(dashboard, "deribit_client", return_value=client):
        test_client = dashboard.app.test_client()
        close_response = test_client.post(
            "/api/close_trade", headers=auth_headers(), json={"symbol": "ETHUSDT"}
        )
        kill_response = test_client.post("/api/kill_switch", headers=auth_headers())

    assert close_response.status_code == 403
    assert kill_response.status_code == 403
    assert close_response.get_json()["error"] == "dashboard_read_only"
    assert kill_response.get_json()["error"] == "dashboard_read_only"
    client.assert_not_called()
    assert not client.method_calls


def test_dashboard_scan_dispatch_is_disabled_in_monitoring_release():
    with patch.dict(os.environ, {**AUTH, "DASHBOARD_READ_ONLY_MODE": "false"}), \
         patch.object(dashboard.requests, "post") as dispatch:
        response = dashboard.app.test_client().post("/api/scan", headers=auth_headers())

    assert response.status_code == 403
    assert response.get_json()["error"] == "dashboard_read_only"
    dispatch.assert_not_called()


def test_dashboard_unresolved_heal_intent_cannot_be_changed_from_monitoring_view(tmp_path, monkeypatch):
    client = _heal_route_client()
    monkeypatch.chdir(tmp_path)
    pending = {
        "ETHUSDT": {
            "closed": False,
            "manual_reconciliation_required": True,
            "stop_loss_submission_state": "UNKNOWN",
            "order_ids": {"stop_loss_label": "DASHSLpersisted"},
        }
    }
    monkeypatch.setattr(dashboard, "get", lambda key, default=None: pending if key == "trades.json" else default)
    with patch.dict(os.environ, AUTH), patch.object(dashboard, "deribit_client", return_value=client):
        response = dashboard.app.test_client().post(
            "/api/trades/heal", headers=auth_headers(), json={"symbol": "ETHUSDT"}
        )

    assert response.status_code == 403
    assert response.get_json()["error"] == "dashboard_read_only"
    client.place_stop_loss.assert_not_called()
    client.place_limit_order.assert_not_called()


def test_dashboard_html_displays_monitoring_only_without_exchange_mutation_controls():
    html = (Path(dashboard.__file__).resolve().parent / "dashboard_static" / "index.html").read_text(encoding="utf-8")
    assert "MONITORING ONLY" in html
    assert 'onclick="triggerScan()"' not in html
    assert 'onclick="openKillModal()"' not in html
    assert 'onclick="closeTrade(' not in html
    assert "healTradeBracket(" not in html
    assert "EMERGENCY KILL SWITCH MODAL" not in html
