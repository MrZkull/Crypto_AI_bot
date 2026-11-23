from decimal import Decimal
from unittest.mock import MagicMock

import pytest
import requests

import trade_executor
from deribit_client import (
    DeribitClient,
    DeribitRPCError,
    DeribitTransportError,
    OrderOutcomeUnknown,
)


class FakeDeribitSizing:
    def __init__(self, metadata=None):
        self.metadata = metadata or {
            "instrument_name": "ETH_USDC-PERPETUAL",
            "is_active": True,
            "state": "open",
            "instrument_type": "linear",
            "kind": "future",
            "settlement_period": "perpetual",
            "settlement_currency": "USDC",
            "base_currency": "ETH",
            "contract_size": 0.0001,
            "min_trade_amount": 0.001,
            "tick_size": 0.05,
            "taker_commission": 0.0005,
        }

    def get_instrument_info(self, symbol):
        return self.metadata

    def get_instrument_name(self, symbol):
        return "ETH_USDC-PERPETUAL"

    def get_tick_size(self, symbol):
        return 0.05


def _risk_snapshot(**overrides):
    payload = {
        "entry_enabled": True,
        "risk_currency": "USDC",
        "risk_equity": 1000,
        "available_funds": 800,
        "margin_mode": "segregated",
        "source": "unit-test",
    }
    payload.update(overrides)
    return payload


def test_execute_trade_decimal_sizer_uses_exchange_minimum_step_and_base_amount(monkeypatch):
    monkeypatch.setattr(trade_executor, "RISK_PER_TRADE", 0.03)
    result, spec, fee, effective_fraction, sizing_equity = trade_executor._calculate_deribit_decimal_size(
        FakeDeribitSizing(),
        "ETHUSDT",
        _risk_snapshot(),
        entry_price="2000",
        stop_price="1980",
        risk_multiplier="1",
        side="BUY",
    )

    assert result.accepted is True
    assert result.quantity == Decimal("0.025")
    assert result.notional == Decimal("50.000")
    assert result.total_risk == Decimal("0.5497500")
    assert spec.qty_step == Decimal("0.001")
    assert spec.contract_value_base == Decimal("1")
    assert fee == Decimal("0.0005")
    assert effective_fraction == Decimal("0.03")
    assert sizing_equity == Decimal("1000")


def test_execute_trade_decimal_sizer_rejects_minimum_lot_over_risk_budget(monkeypatch):
    monkeypatch.setattr(trade_executor, "RISK_PER_TRADE", 0.005)
    metadata = dict(FakeDeribitSizing().metadata)
    metadata["min_trade_amount"] = 1.0
    result, _, _, _, _ = trade_executor._calculate_deribit_decimal_size(
        FakeDeribitSizing(metadata), "ETHUSDT", _risk_snapshot(),
        entry_price="2000", stop_price="1980", risk_multiplier=1, side="BUY",
    )
    assert result.accepted is False
    assert result.quantity == 0
    assert result.reason == "BELOW_MINIMUM_WITHIN_RISK"


def test_execute_trade_decimal_sizer_rejects_missing_exchange_fee_or_spec(monkeypatch):
    metadata = dict(FakeDeribitSizing().metadata)
    metadata.pop("taker_commission")
    with pytest.raises(ValueError, match="fee specification"):
        trade_executor._calculate_deribit_decimal_size(
            FakeDeribitSizing(metadata), "ETHUSDT", _risk_snapshot(),
            entry_price=2000, stop_price=1980, risk_multiplier=1, side="BUY",
        )


def test_execute_trade_decimal_sizer_rejects_effective_risk_over_five_percent(monkeypatch):
    monkeypatch.setattr(trade_executor, "RISK_PER_TRADE", 0.03)
    with pytest.raises(ValueError, match="5% hard ceiling"):
        trade_executor._calculate_deribit_decimal_size(
            FakeDeribitSizing(), "ETHUSDT", _risk_snapshot(),
            entry_price=2000, stop_price=1980, risk_multiplier=2, side="BUY",
        )


def test_entry_submission_intent_records_label_before_any_fill_without_claiming_quantity():
    intent = trade_executor._build_entry_submission_intent(
        {"pred_id": "pred-123", "snapshot": {"source": "unit-test"}},
        "ETHUSDT", "BUY", 2000.0, 2001.0, 0.25, "cb-entry-123",
    )

    assert intent["order_submission_states"] == {"entry": "SUBMITTING"}
    assert intent["order_client_labels"] == {"entry": "cb-entry-123"}
    assert intent["order_ids"] == {"entry_label": "cb-entry-123"}
    assert intent["requested_qty"] == 0.25
    assert intent["qty"] == 0.0
    assert intent["filled_qty"] == 0.0
    assert intent["manual_reconciliation_required"] is True
    assert intent["closed"] is False


def test_terminal_expired_or_cancelled_partial_entry_remains_protectable():
    assert trade_executor._entry_terminal_has_confirmed_fill({
        "order_state": "expired", "filled_amount": 0.2,
    }) is True
    assert trade_executor._entry_terminal_has_confirmed_fill({
        "order_state": "cancelled", "filled_amount": 0.1,
    }) is True


def test_terminal_zero_fill_entry_is_not_an_exposure():
    assert trade_executor._entry_terminal_has_confirmed_fill({
        "order_state": "expired", "filled_amount": 0,
    }) is False
    assert trade_executor._entry_terminal_has_confirmed_fill({
        "order_state": "cancelled", "filled_amount": 0,
    }) is False


def test_rejected_entry_with_reported_fill_is_unknown_not_safe_to_ignore():
    with pytest.raises(OrderOutcomeUnknown, match="rejected entry unexpectedly"):
        trade_executor._entry_terminal_has_confirmed_fill({
            "order_state": "rejected", "filled_amount": 0.1,
        })


def test_confirm_terminal_entry_order_cancels_open_ioc_once_and_uses_terminal_fill():
    deribit = MagicMock()
    deribit.get_order.side_effect = [
        {"order_id": "ord-1", "order_state": "open", "filled_amount": 0.2},
        {"order_id": "ord-1", "order_state": "cancelled", "filled_amount": 0.2, "average_price": 2001},
    ]
    deribit.cancel_order.return_value = {"result": "cancelled"}

    terminal = trade_executor._confirm_terminal_entry_order(
        deribit,
        {"order": {"order_id": "ord-1", "order_state": "open"}},
        requested_qty=1.0,
    )

    assert terminal["order_state"] == "cancelled"
    assert terminal["filled_amount"] == 0.2
    assert deribit.get_order.call_count == 2
    deribit.cancel_order.assert_called_once_with("ord-1")


def test_confirm_terminal_entry_order_refuses_missing_order_id():
    deribit = MagicMock()
    with pytest.raises(OrderOutcomeUnknown, match="no exchange order ID"):
        trade_executor._confirm_terminal_entry_order(
            deribit, {"order": {"order_state": "filled"}}, requested_qty=1,
        )
    deribit.get_order.assert_not_called()
    deribit.cancel_order.assert_not_called()


def test_confirm_terminal_entry_order_never_retries_nonterminal_after_one_cancel():
    deribit = MagicMock()
    deribit.get_order.side_effect = [
        {"order_id": "ord-1", "order_state": "open", "filled_amount": 0},
        {"order_id": "ord-1", "order_state": "open", "filled_amount": 0},
    ]
    with pytest.raises(OrderOutcomeUnknown, match="remains non-terminal"):
        trade_executor._confirm_terminal_entry_order(
            deribit,
            {"order": {"order_id": "ord-1"}},
            requested_qty=1,
        )
    deribit.cancel_order.assert_called_once_with("ord-1")
    assert deribit.get_order.call_count == 2


def test_wait_for_position_requires_expected_direction(monkeypatch):
    monkeypatch.setattr(trade_executor.time, "sleep", lambda _: None)
    monkeypatch.setattr(trade_executor.time, "time", lambda: 100.0)
    deribit = MagicMock()
    deribit.get_position_size.return_value = -0.25
    assert trade_executor._wait_for_position(deribit, "ETHUSDT", "BUY", "ord-1", timeout=0.1) is False


def test_emergency_close_does_not_use_fallback_quantity_when_position_read_fails(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.side_effect = TimeoutError("position read unavailable")
    retained = {}
    monkeypatch.setattr(
        trade_executor,
        "_retain_emergency_recovery_record",
        lambda *args, **kwargs: retained.update({"args": args, "kwargs": kwargs}),
    )

    result = trade_executor._emergency_close_and_verify(
        deribit, {}, {"pred_id": "p1"}, "ETHUSDT", "BUY", 2000,
        0.5, {}, "protection failure",
    )

    assert result is False
    deribit.place_reduce_only_market_order.assert_not_called()
    deribit.cancel_order.assert_not_called()
    assert retained["args"][5] == 0.0
    assert retained["kwargs"]["exchange_quantity_verified"] is False


def test_emergency_close_sizes_single_market_order_from_fresh_verified_position(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.side_effect = [0.5, 0.5]
    deribit.get_min_trade_amount.return_value = 0.1
    deribit.round_amount.side_effect = lambda symbol, qty: qty
    deribit.place_reduce_only_market_order.return_value = {
        "order": {"order_id": "flat-1", "order_state": "filled", "filled_amount": 0.5}
    }
    monkeypatch.setattr(trade_executor, "_cancel_all_open_orders_for_symbol", lambda *_: True)
    monkeypatch.setattr(trade_executor, "_verify_actually_closed", lambda *args, **kwargs: True)
    monkeypatch.setattr(trade_executor, "_send", lambda *_: None)

    result = trade_executor._emergency_close_and_verify(
        deribit, {}, {"pred_id": "p1"}, "ETHUSDT", "BUY", 2000,
        9.0, {}, "stop placement failed",
    )

    assert result is True
    deribit.place_reduce_only_market_order.assert_called_once()
    args, kwargs = deribit.place_reduce_only_market_order.call_args
    assert args == ("ETHUSDT", "SELL", 0.5)
    assert kwargs["client_label"].startswith("cb_")


class FakeHTTPResponse:
    def __init__(self, payload, status=200, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _http_client(write=False):
    client = DeribitClient.__new__(DeribitClient)
    client.client_id = "unit-id"
    client.client_secret = "unit-secret"
    client.base = "https://test.deribit.com/api/v2"
    client.auth_scope = "trade:read_write account:read" if write else "trade:read account:read"
    client.allow_mutations = write
    client.session = requests.Session()
    client._auth_lock = __import__("threading").RLock()
    client._access_token = "cached-token" if write or True else None
    client._refresh_token = None
    client._token_expires_at = __import__("time").time() + 3600
    client._rpc_id = 0
    client._read_breaker = __import__("exchange_reliability").CircuitBreaker(failure_threshold=3)
    client._order_breaker = __import__("exchange_reliability").CircuitBreaker(failure_threshold=3)
    return client


def test_deribit_read_retries_transient_transport_then_returns_snapshot():
    client = _http_client()
    client.session.send = MagicMock(side_effect=[
        requests.Timeout("temporary"),
        FakeHTTPResponse({"jsonrpc": "2.0", "id": 1, "result": {"time": 1}}),
    ])
    result = client._get("/public/get_time")
    assert result == {"time": 1}
    assert client.session.send.call_count == 2


def test_deribit_read_honours_retry_after_header_within_deadline(monkeypatch):
    import deribit_client as deribit_module
    from exchange_reliability import retry_read as real_retry_read

    client = _http_client()
    delays = []

    def captured_retry(operation, **kwargs):
        kwargs["sleeper"] = lambda delay: delays.append(delay)
        kwargs["random_value"] = lambda: 0.0
        return real_retry_read(operation, **kwargs)

    monkeypatch.setattr(deribit_module, "retry_read", captured_retry)
    client.session.send = MagicMock(side_effect=[
        FakeHTTPResponse({"error": {"code": 10028, "message": "rate limited"}}, status=429, headers={"Retry-After": "0.25"}),
        FakeHTTPResponse({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}),
    ])
    assert client._get("/public/get_time") == {"ok": True}
    assert client.session.send.call_count == 2
    assert delays and delays[0] >= 0.25


def test_deribit_read_does_not_retry_structured_rpc_error():
    client = _http_client()
    client.session.send = MagicMock(return_value=FakeHTTPResponse({
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": 13009, "message": "unauthorized"},
    }))
    with pytest.raises(DeribitRPCError):
        client._get("/public/get_time")
    assert client.session.send.call_count == 1


def test_deribit_mutation_transport_timeout_is_unknown_and_latches_order_path():
    client = _http_client(write=True)
    client.session.send = MagicMock(side_effect=requests.Timeout("gateway timeout"))
    with pytest.raises(OrderOutcomeUnknown):
        client._post("/private/buy", {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 1})
    assert client.session.send.call_count == 1
    assert client._order_breaker.manual_ack_required is True
    with pytest.raises(PermissionError, match="HALTED"):
        client._post("/private/sell", {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 1, "reduce_only": True})
    # Authenticated read-only reconciliation remains allowed while writes are halted.
    client.session.send = MagicMock(return_value=FakeHTTPResponse({"jsonrpc": "2.0", "id": 1, "result": {"time": 2}}))
    assert client._get("/public/get_time") == {"time": 2}


def test_deribit_explicit_rpc_rejection_is_not_retried_or_classified_unknown():
    client = _http_client(write=True)
    client.session.send = MagicMock(return_value=FakeHTTPResponse({
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": 10009, "message": "invalid amount"},
    }))
    with pytest.raises(DeribitRPCError):
        client._post("/private/buy", {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 1})
    assert client.session.send.call_count == 1
    assert client._order_breaker.manual_ack_required is False


def test_market_order_does_not_fallback_or_retry_after_ioc_failure():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client.get_order_book_spread = MagicMock(return_value={"best_bid": 1999, "best_ask": 2000, "spread_pct": 0.0005})
    client.round_price = MagicMock(return_value=2001)
    client._post = MagicMock(side_effect=OrderOutcomeUnknown("unknown"))

    with pytest.raises(OrderOutcomeUnknown):
        client.place_market_order("ETHUSDT", "BUY", 0.01, reduce_only=True)

    client._post.assert_called_once()
    assert client._post.call_args.args[0] == "/private/buy"
    assert client._post.call_args.args[1]["type"] == "limit"
    assert client._post.call_args.args[1]["time_in_force"] == "immediate_or_cancel"


def test_reduce_only_market_method_is_single_shot_and_verified():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client._post = MagicMock(return_value={
        "order": {"order_id": "o1", "order_state": "filled", "filled_amount": 0.01}
    })

    result = client.place_reduce_only_market_order("ETHUSDT", "SELL", 0.01)

    assert result["order"]["order_id"] == "o1"
    client._post.assert_called_once()
    assert client._post.call_args.args[1]["type"] == "market"
    assert client._post.call_args.args[1]["reduce_only"] is True


def test_reduce_only_market_method_latches_malformed_response():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client._post = MagicMock(return_value={"order": {"order_id": "o1", "order_state": "open"}})
    with pytest.raises(OrderOutcomeUnknown):
        client.place_reduce_only_market_order("ETHUSDT", "SELL", 0.01)
    assert client._get_order_breaker().manual_ack_required is True


def test_order_state_by_label_is_read_only_allowlisted_and_strict():
    client = DeribitClient.__new__(DeribitClient)
    client._get = MagicMock(return_value=[{
        "order_id": "ETH-1",
        "label": "cb_test_label",
        "instrument_name": "ETH_USDC-PERPETUAL",
        "order_state": "filled",
    }])

    rows = client.get_order_state_by_label("ETH", "cb_test_label")

    assert rows[0]["order_id"] == "ETH-1"
    client._get.assert_called_once_with(
        "/private/get_order_state_by_label",
        {"currency": "ETH", "label": "cb_test_label"},
    )


def test_order_state_by_label_rejects_mismatched_label_and_malformed_result():
    client = DeribitClient.__new__(DeribitClient)
    client._get = MagicMock(return_value=[{"order_id": "ETH-1", "label": "other"}])
    with pytest.raises(ValueError, match="label mismatch"):
        client.get_order_state_by_label("ETH", "expected")

    client._get.return_value = {"orders": []}
    with pytest.raises(ValueError, match="Malformed"):
        client.get_order_state_by_label("ETH", "expected")


def test_historical_order_lookup_is_read_only_pinned_to_exact_instrument():
    client = DeribitClient.__new__(DeribitClient)
    client._get = MagicMock(return_value=[{
        "order_id": "ETH-hist-1", "label": "cb_entry_hist",
        "instrument_name": "ETH_USDC-PERPETUAL", "order_state": "filled",
    }])
    rows = client.get_order_history_by_instrument_name(
        "ETH_USDC-PERPETUAL", count=1000, historical=True,
        include_old=True, include_unfilled=True, offset=0, strict=True,
    )
    assert rows[0]["order_id"] == "ETH-hist-1"
    client._get.assert_called_once_with(
        "/private/get_order_history_by_instrument",
        {
            "instrument_name": "ETH_USDC-PERPETUAL", "count": 1000,
            "offset": 0, "historical": True, "include_old": True,
            "include_unfilled": True,
        },
    )


def test_historical_trades_by_order_is_read_only_and_exact_order_scoped():
    client = DeribitClient.__new__(DeribitClient)
    client._get = MagicMock(return_value=[{
        "order_id": "ETH-hist-1", "trade_id": "trade-hist-1",
        "instrument_name": "ETH_USDC-PERPETUAL",
    }])
    rows = client.get_user_trades_by_order("ETH-hist-1", historical=True)
    assert rows[0]["trade_id"] == "trade-hist-1"
    client._get.assert_called_once_with(
        "/private/get_user_trades_by_order",
        {"order_id": "ETH-hist-1", "historical": True, "sorting": "desc"},
    )


def test_stop_loss_is_one_shot_and_verifies_reduce_only_mark_price_market_stop():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client.round_price = MagicMock(return_value=1900.0)
    client._post = MagicMock(return_value={"order": {"order_id": "sl-1"}})
    client.get_order = MagicMock(return_value={
        "order_id": "sl-1",
        "instrument_name": "ETH_USDC-PERPETUAL",
        "order_type": "stop_market",
        "direction": "sell",
        "trigger": "mark_price",
        "trigger_price": 1900.0,
        "reduce_only": True,
        "amount": 0.25,
        "order_state": "untriggered",
    })

    result = client.place_stop_loss(
        "ETHUSDT", "SELL", 0.25, 1900.0, client_label="cb_stop_1"
    )

    assert result["order"]["order_id"] == "sl-1"
    client._post.assert_called_once()
    args = client._post.call_args.args
    body = args[1]
    assert args[0] == "/private/sell"
    assert body["type"] == "stop_market"
    assert body["trigger"] == "mark_price"
    assert body["reduce_only"] is True
    assert body["label"] == "cb_stop_1"
    client.get_order.assert_called_once_with("sl-1")


def test_stop_loss_verification_failure_does_not_submit_a_second_stop():
    from deribit_client import ProtectiveOrderVerificationError

    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client.round_price = MagicMock(return_value=1900.0)
    client._post = MagicMock(return_value={"order": {"order_id": "sl-1"}})
    client.get_order = MagicMock(return_value={
        "order_id": "sl-1",
        "instrument_name": "ETH_USDC-PERPETUAL",
        "order_type": "stop_limit",  # Wrong type: cannot be accepted as a verified stop-market.
        "direction": "sell",
        "trigger": "mark_price",
        "trigger_price": 1900.0,
        "reduce_only": True,
        "amount": 0.25,
        "order_state": "untriggered",
    })

    with pytest.raises(ProtectiveOrderVerificationError, match="stop_market"):
        client.place_stop_loss("ETHUSDT", "SELL", 0.25, 1900.0, client_label="cb_stop_1")

    client._post.assert_called_once()


def test_stop_loss_missing_ack_id_latches_unknown_and_never_retries():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client.round_price = MagicMock(return_value=1900.0)
    client._post = MagicMock(return_value={"order": {"order_state": "open"}})
    client._latch_unknown_order_outcome = MagicMock()
    client.get_order = MagicMock()

    with pytest.raises(OrderOutcomeUnknown, match="no exchange order ID"):
        client.place_stop_loss("ETHUSDT", "SELL", 0.25, 1900.0, client_label="cb_stop_1")

    client._post.assert_called_once()
    client._latch_unknown_order_outcome.assert_called_once()
    client.get_order.assert_not_called()


def test_deterministic_order_labels_are_stable_unique_and_within_deribit_limit():
    first = trade_executor._deterministic_order_label(
        {"pred_id": "pred-20261115-ETH-001"}, "ETHUSDT", "entry"
    )
    repeated = trade_executor._deterministic_order_label(
        {"pred_id": "pred-20261115-ETH-001"}, "ETHUSDT", "entry"
    )
    other_slot = trade_executor._deterministic_order_label(
        {"pred_id": "pred-20261115-ETH-001"}, "ETHUSDT", "stop"
    )
    other_prediction = trade_executor._deterministic_order_label(
        {"pred_id": "pred-20261115-ETH-002"}, "ETHUSDT", "entry"
    )
    assert first == repeated
    assert first != other_slot
    assert first != other_prediction
    assert 1 <= len(first) <= 64
    assert first.isascii() and first.isprintable()


class FakeLabelReconciliation:
    def get_instrument_name(self, symbol):
        return "ETH_USDC-PERPETUAL"

    def get_instrument_info(self, symbol):
        return {"base_currency": "ETH"}

    def get_order_state_by_label(self, currency, label):
        return [{
            "order_id": "ETH-1", "label": label,
            "instrument_name": "ETH_USDC-PERPETUAL", "order_state": "filled",
            "filled_amount": 0.25, "average_price": 2000.0,
            "direction": "buy",
        }]

    def get_open_orders_by_label(self, currency, label):
        return []

    def get_position_size(self, symbol):
        return 0.25

    def get_trade_history_by_instrument_name(self, instrument, count, strict=True):
        assert count == 1000 and strict is True
        return [{
            "order_id": "ETH-1", "trade_id": "tr-1",
            "instrument_name": "ETH_USDC-PERPETUAL", "amount": 0.25,
        }]

    def get_order_history_by_instrument_name(
        self, instrument, count=1000, historical=False,
        include_old=True, include_unfilled=True, offset=0, strict=True,
    ):
        assert instrument == "ETH_USDC-PERPETUAL"
        assert count == 1000 and historical is True and strict is True
        return []

    def get_user_trades_by_order(self, order_id, historical=False, strict=True):
        assert historical is True and strict is True
        return []


def test_unknown_entry_label_reconciliation_is_read_only_and_collects_position_trades():
    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        FakeLabelReconciliation(), "ETHUSDT", "cb_entry_1"
    )

    assert evidence["read_only"] is True
    assert evidence["matched_order_ids"] == ["ETH-1"]
    assert evidence["position_snapshot"] == {"verified": True, "signed_quantity": 0.25}
    assert evidence["recent_trade_snapshot"]["matching_trade_count"] == 1
    assert evidence["recent_trade_snapshot"]["history_complete"] is True
    assert evidence["read_errors"] == []


def test_unknown_entry_no_label_match_is_not_interpreted_as_rejection():
    class NoMatch(FakeLabelReconciliation):
        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        NoMatch(), "ETHUSDT", "cb_missing_1"
    )

    assert evidence["matched_order_ids"] == []
    assert evidence["position_snapshot"]["verified"] is True
    assert evidence["recent_trade_snapshot"]["verified"] is True
    assert "not proof" in evidence["recent_trade_snapshot"]["note"] if "note" in evidence["recent_trade_snapshot"] else evidence["label_lookup_window"]



def test_unknown_entry_recent_trade_label_recovers_order_when_label_state_window_is_empty():
    class RecentTradeByLabel(FakeLabelReconciliation):
        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(self, instrument, count, strict=True):
            assert count == 1000 and strict is True
            return [{
                "order_id": "ETH-older-1",
                "trade_id": "tr-older-1",
                "instrument_name": "ETH_USDC-PERPETUAL",
                "label": "cb_aged_label_1",
                "amount": 0.25,
            }]

    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        RecentTradeByLabel(), "ETHUSDT", "cb_aged_label_1"
    )

    assert evidence["order_state_matches"] == []
    assert evidence["open_order_matches"] == []
    assert evidence["matched_order_ids"] == ["ETH-older-1"]
    assert evidence["recent_trade_snapshot"]["matching_trade_count"] == 1
    assert evidence["recent_trade_snapshot"]["matched_by_exact_label_count"] == 1
    assert evidence["recent_trade_snapshot"]["history_complete"] is True
    assert "not proof of no fill" in evidence["recent_trade_snapshot"]["note"]


def test_unknown_entry_uses_historical_order_and_order_trade_lookup_after_recent_windows():
    class HistoricalOnly(FakeLabelReconciliation):
        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(self, instrument, count, strict=True):
            assert count == 1000 and strict is True
            return []

        def get_order_history_by_instrument_name(
            self, instrument, count=1000, historical=False,
            include_old=True, include_unfilled=True, offset=0, strict=True,
        ):
            assert historical is True and include_old is True and include_unfilled is True
            return [{
                "order_id": "ETH-historical-99",
                "label": "cb_old_unknown_99",
                "instrument_name": "ETH_USDC-PERPETUAL",
                "order_state": "filled",
                "filled_amount": 0.25,
                "average_price": 2025.0,
                "direction": "buy",
            }]

        def get_user_trades_by_order(self, order_id, historical=False, strict=True):
            assert order_id == "ETH-historical-99"
            assert historical is True and strict is True
            return [{
                "order_id": order_id,
                "trade_id": "trade-historical-99",
                "label": "cb_old_unknown_99",
                "instrument_name": "ETH_USDC-PERPETUAL",
                "amount": 0.25,
            }]

    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        HistoricalOnly(), "ETHUSDT", "cb_old_unknown_99"
    )
    assert evidence["order_state_matches"] == []
    assert evidence["open_order_matches"] == []
    assert evidence["matched_order_ids"] == ["ETH-historical-99"]
    assert evidence["historical_order_matches_count"] == 1
    assert evidence["historical_trade_matches_count"] == 1
    assert evidence["historical_trade_snapshot"]["matching_trade_ids"] == ["trade-historical-99"]
    assert evidence["read_errors"] == []




def test_unknown_entry_recent_trade_lookup_paginates_by_trade_sequence():
    class PagedRecentTrades(FakeLabelReconciliation):
        def __init__(self):
            self.end_sequences = []

        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(
            self, instrument, count=1000, strict=True, end_seq=None
        ):
            self.end_sequences.append(end_seq)
            assert count == 1000 and strict is True
            if end_seq is None:
                return [
                    {
                        "order_id": f"ETH-recent-{i}",
                        "trade_id": f"trade-{i}",
                        "trade_seq": 10000 - i,
                        "instrument_name": "ETH_USDC-PERPETUAL",
                        "label": f"unrelated-{i}",
                    }
                    for i in range(1000)
                ]
            assert end_seq == 9000
            return [{
                "order_id": "ETH-recent-page-2",
                "trade_id": "trade-page-2",
                "trade_seq": 9000,
                "instrument_name": "ETH_USDC-PERPETUAL",
                "label": "cb_recent_page_2",
            }]

    deribit = PagedRecentTrades()
    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        deribit, "ETHUSDT", "cb_recent_page_2"
    )

    assert deribit.end_sequences == [None, 9000]
    assert evidence["matched_order_ids"] == ["ETH-recent-page-2"]
    assert evidence["recent_trade_snapshot"]["pages_returned"] == 2
    assert evidence["recent_trade_snapshot"]["history_complete"] is True
    assert evidence["recent_trade_snapshot"]["matching_trade_count"] == 1


def test_unknown_entry_historical_order_lookup_paginates_to_find_older_label():
    class PagedHistorical(FakeLabelReconciliation):
        def __init__(self):
            self.offsets = []

        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(self, instrument, count, strict=True, end_seq=None):
            return []

        def get_order_history_by_instrument_name(
            self, instrument, count=1000, historical=False,
            include_old=True, include_unfilled=True, offset=0, strict=True,
        ):
            self.offsets.append(offset)
            assert count == 1000 and historical is True and strict is True
            if offset == 0:
                return [
                    {
                        "order_id": f"ETH-no-match-{i}",
                        "label": f"unrelated-{i}",
                        "instrument_name": "ETH_USDC-PERPETUAL",
                        "order_state": "cancelled",
                    }
                    for i in range(1000)
                ]
            if offset == 1000:
                return [{
                    "order_id": "ETH-historical-page-2",
                    "label": "cb_old_page_2",
                    "instrument_name": "ETH_USDC-PERPETUAL",
                    "order_state": "filled",
                    "filled_amount": 0.15,
                    "average_price": 2010.0,
                    "direction": "buy",
                }]
            return []

        def get_user_trades_by_order(self, order_id, historical=False, strict=True):
            return [{
                "order_id": order_id,
                "trade_id": "trade-page-2",
                "label": "cb_old_page_2",
                "instrument_name": "ETH_USDC-PERPETUAL",
                "amount": 0.15,
            }]

    deribit = PagedHistorical()
    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        deribit, "ETHUSDT", "cb_old_page_2"
    )

    assert deribit.offsets == [0, 1000]
    assert evidence["matched_order_ids"] == ["ETH-historical-page-2"]
    assert evidence["historical_order_snapshot"]["pages_returned"] == 2
    assert evidence["historical_order_snapshot"]["history_complete"] is True
    assert evidence["historical_order_snapshot"]["pagination_capped"] is False
    assert evidence["historical_trade_matches_count"] == 1


def test_missing_stop_repair_with_persisted_unknown_state_never_resubmits(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.return_value = 0.25
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_order_state_by_label.return_value = []
    deribit.get_open_orders_by_label.return_value = []
    trade = {
        "symbol": "ETHUSDT",
        "signal": "BUY",
        "entry": 2000.0,
        "stop": 1900.0,
        "tp1": 2100.0,
        "tp2": 0.0,
        "qty": 0.25,
        "qty_tp1": 0.25,
        "qty_tp2": 0.0,
        "order_ids": {"stop_loss_label": "cb_stop_1"},
        "stop_loss_submission_state": "UNKNOWN",
        "manual_reconciliation_required": True,
        "emergency_recovery_required": True,
    }
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)

    changed = trade_executor._replace_missing_orders(deribit, "ETHUSDT", trade)

    assert changed is True
    assert trade["manual_reconciliation_required"] is True
    deribit.place_stop_loss.assert_not_called()
    deribit.place_limit_order.assert_not_called()


def test_tp_unknown_outcome_sets_persistent_manual_reconciliation_and_no_retry(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.return_value = 0.5
    deribit.get_min_trade_amount.return_value = 0.001
    deribit.get_open_orders.return_value = []
    deribit.get_tick_size.return_value = 0.05
    deribit.round_amount.side_effect = lambda symbol, qty: qty
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_order_state_by_label.return_value = []
    deribit.get_open_orders_by_label.return_value = []
    deribit.place_limit_order.side_effect = OrderOutcomeUnknown("response timeout")
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)

    trade = {
        "pred_id": "pred-1", "qty": 0.5,
        "order_ids": {},
    }
    first = trade_executor._place_tp_with_fallback(
        deribit, "ETHUSDT", "SELL", 0.25, 2100.0, "TP1", trade, "tp1", 2
    )
    assert first == ""
    deribit.place_limit_order.assert_called_once()
    assert trade["order_submission_states"]["tp1"] == "UNKNOWN"
    assert trade["manual_reconciliation_required"] is True

    second = trade_executor._place_tp_with_fallback(
        deribit, "ETHUSDT", "SELL", 0.25, 2100.0, "TP1", trade, "tp1", 2
    )
    assert second == ""
    deribit.place_limit_order.assert_called_once()


def test_tp_label_reconciliation_requires_exact_limit_contract(monkeypatch):
    deribit = MagicMock()
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_tick_size.return_value = 0.05
    deribit.round_amount.side_effect = lambda symbol, amount: amount
    deribit.round_price.side_effect = lambda symbol, price: price
    deribit.get_order_state_by_label.return_value = [{
        "order_id": "tp-1", "label": "stable-tp-label",
        "instrument_name": "ETH_USDC-PERPETUAL", "order_type": "limit",
        "order_state": "open", "direction": "sell", "reduce_only": True,
        "amount": 0.5, "price": 2100.0,
    }]
    deribit.get_open_orders_by_label.return_value = []
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)
    trade = {
        "order_submission_states": {"tp1": "UNKNOWN"},
        "order_client_labels": {"tp1": "stable-tp-label"},
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
    }

    result = trade_executor._place_tp_with_fallback(
        deribit, "ETHUSDT", "SELL", 0.25, 2100.0, "TP1", trade, "tp1", 2
    )

    assert result == ""
    assert trade["order_submission_states"]["tp1"] == "UNKNOWN"
    assert trade["manual_reconciliation_required"] is True
    deribit.place_limit_order.assert_not_called()


def test_tp_label_reconciliation_adopts_only_exact_verified_reduce_only_limit(monkeypatch):
    deribit = MagicMock()
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_tick_size.return_value = 0.05
    deribit.round_amount.side_effect = lambda symbol, amount: amount
    deribit.round_price.side_effect = lambda symbol, price: price
    deribit.get_order_state_by_label.return_value = [{
        "order_id": "tp-1", "label": "stable-tp-label",
        "instrument_name": "ETH_USDC-PERPETUAL", "order_type": "limit",
        "order_state": "open", "direction": "sell", "reduce_only": True,
        "amount": 0.25, "price": 2100.0,
    }]
    deribit.get_open_orders_by_label.return_value = []
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)
    trade = {
        "order_submission_states": {"tp1": "UNKNOWN"},
        "order_client_labels": {"tp1": "stable-tp-label"},
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
    }

    result = trade_executor._place_tp_with_fallback(
        deribit, "ETHUSDT", "SELL", 0.25, 2100.0, "TP1", trade, "tp1", 2
    )

    assert result == "tp-1"
    assert trade["order_submission_states"]["tp1"] == "RECONCILED"
    assert trade["order_ids"]["tp1"] == "tp-1"
    assert trade.get("manual_reconciliation_required") is None
    deribit.place_limit_order.assert_not_called()


def test_manual_tp_unknown_reconciliation_is_read_only_and_never_submits_another_tp(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.return_value = 0.5
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_order_state_by_label.return_value = []
    deribit.get_open_orders_by_label.return_value = []
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)
    trade = {
        "symbol": "ETHUSDT", "signal": "BUY", "entry": 2000.0,
        "stop": 1900.0, "tp1": 2100.0, "tp2": 2200.0,
        "qty": 0.5, "qty_tp1": 0.5, "qty_tp2": 0.0,
        "order_ids": {"stop_loss": "sl-1"},
        "order_submission_states": {"tp1": "UNKNOWN"},
        "order_client_labels": {"tp1": "stable-tp-label"},
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
    }

    trade_executor._replace_missing_orders(deribit, "ETHUSDT", trade)

    assert trade["order_submission_states"]["tp1"] == "UNKNOWN"
    assert trade["manual_reconciliation_required"] is True
    deribit.place_limit_order.assert_not_called()
    deribit.place_stop_loss.assert_not_called()


def test_stop_submission_intent_persists_label_before_marking_submitting(monkeypatch):
    seen = {}

    def persist(symbol, trade):
        seen["symbol"] = symbol
        seen["label"] = trade["order_ids"]["stop_loss_label"]
        seen["state"] = trade["stop_loss_submission_state"]
        return True

    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", persist)
    trade = {"order_ids": {}}

    assert trade_executor._persist_stop_submission_intent(
        "ETHUSDT", trade, "deterministic-stop-label"
    ) is True
    assert seen == {
        "symbol": "ETHUSDT",
        "label": "deterministic-stop-label",
        "state": "SUBMITTING",
    }


def test_reduce_only_tp_ack_requires_exchange_readback_contract():
    deribit = MagicMock()
    deribit.get_order.return_value = {
        "order_id": "tp-1", "instrument_name": "ETH_USDC-PERPETUAL",
        "order_type": "limit", "direction": "sell", "reduce_only": True,
        "amount": 0.25, "price": 2100.0, "order_state": "open",
        "label": "stable-tp-label",
    }
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.round_amount.side_effect = lambda symbol, amount: amount
    deribit.round_price.side_effect = lambda symbol, price: price
    deribit.get_tick_size.return_value = 0.05

    verified = trade_executor._verify_reduce_only_limit_ack(
        deribit, "ETHUSDT", "SELL", 0.25, 2100.0,
        {"order": {"order_id": "tp-1"}}, "stable-tp-label",
    )

    assert verified["order_id"] == "tp-1"
    deribit.get_order.assert_called_once_with("tp-1")


def test_reduce_only_tp_ack_wrong_order_contract_fails_closed():
    from deribit_client import ProtectiveOrderVerificationError

    deribit = MagicMock()
    deribit.get_order.return_value = {
        "order_id": "tp-1", "instrument_name": "ETH_USDC-PERPETUAL",
        "order_type": "limit", "direction": "sell", "reduce_only": False,
        "amount": 0.25, "price": 2100.0, "order_state": "open",
        "label": "stable-tp-label",
    }
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.round_amount.side_effect = lambda symbol, amount: amount
    deribit.round_price.side_effect = lambda symbol, price: price
    deribit.get_tick_size.return_value = 0.05

    with pytest.raises(ProtectiveOrderVerificationError, match="failed exchange contract"):
        trade_executor._verify_reduce_only_limit_ack(
            deribit, "ETHUSDT", "SELL", 0.25, 2100.0,
            {"order": {"order_id": "tp-1"}}, "stable-tp-label",
        )


def test_reduce_only_tp_ack_read_timeout_is_unknown_not_rejected():
    deribit = MagicMock()
    deribit.get_order.side_effect = TimeoutError("state read timeout")
    with pytest.raises(OrderOutcomeUnknown, match="state could not be independently read"):
        trade_executor._verify_reduce_only_limit_ack(
            deribit, "ETHUSDT", "SELL", 0.25, 2100.0,
            {"order": {"order_id": "tp-1"}}, "stable-tp-label",
        )


def test_unknown_tp_with_unresolved_flags_runs_only_label_reconciliation(monkeypatch):
    deribit = MagicMock()
    deribit.get_position_size.return_value = 0.5
    deribit.get_instrument_name.return_value = "ETH_USDC-PERPETUAL"
    deribit.get_instrument_info.return_value = {"base_currency": "ETH"}
    deribit.get_order_state_by_label.return_value = []
    deribit.get_open_orders_by_label.return_value = []
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *_: True)
    trade = {
        "symbol": "ETHUSDT", "signal": "BUY", "entry": 2000.0,
        "stop": 1900.0, "tp1": 2100.0, "tp2": 2200.0,
        "qty": 0.5, "qty_tp1": 0.5, "qty_tp2": 0.0,
        "order_ids": {"stop_loss": "sl-1"},
        "order_submission_states": {"tp1": "UNKNOWN"},
        "order_client_labels": {"tp1": "stable-tp-label"},
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
    }

    trade_executor._replace_missing_orders(deribit, "ETHUSDT", trade)

    deribit.place_limit_order.assert_not_called()
    deribit.place_stop_loss.assert_not_called()
    assert trade["manual_reconciliation_required"] is True


def test_historical_recovery_marks_evidence_incomplete_when_order_history_read_fails():
    class HistoricalReadFailure(FakeLabelReconciliation):
        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(self, instrument, count, strict=True, end_seq=None):
            return []

        def get_order_history_by_instrument_name(self, *args, **kwargs):
            raise RuntimeError("historical endpoint unavailable")

    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        HistoricalReadFailure(), "ETHUSDT", "cb_old_unknown_500"
    )

    assert evidence["historical_order_snapshot"]["query_succeeded"] is False
    assert evidence["historical_order_snapshot"]["history_complete"] is False
    assert evidence["historical_trade_snapshot"]["verified"] is False
    assert evidence["historical_trade_snapshot"]["query_succeeded"] is False
    assert evidence["matched_order_ids"] == []
    assert any(row["source"] == "historical_order_history" for row in evidence["read_errors"])


def test_historical_recovery_marks_trade_coverage_incomplete_when_over_ten_orders_match():
    class TooManyHistoricalMatches(FakeLabelReconciliation):
        def get_order_state_by_label(self, currency, label):
            return []

        def get_open_orders_by_label(self, currency, label):
            return []

        def get_trade_history_by_instrument_name(self, instrument, count, strict=True, end_seq=None):
            return []

        def get_order_history_by_instrument_name(
            self, instrument, count=1000, historical=False,
            include_old=True, include_unfilled=True, offset=0, strict=True,
        ):
            return [
                {
                    "order_id": f"hist-{i}",
                    "label": "cb_many_matches",
                    "instrument_name": "ETH_USDC-PERPETUAL",
                    "order_state": "filled",
                }
                for i in range(11)
            ]

        def get_user_trades_by_order(self, order_id, historical=False, strict=True):
            return []

    evidence = trade_executor._reconcile_unknown_entry_by_label_read_only(
        TooManyHistoricalMatches(), "ETHUSDT", "cb_many_matches"
    )
    assert evidence["historical_trade_snapshot"]["verified"] is False
    assert evidence["historical_trade_snapshot"]["order_trade_lookups_complete"] is False
    assert len(evidence["historical_trade_snapshot"]["order_ids_queried"]) == 10
    assert evidence["historical_order_matches_count"] == 11


def test_deribit_decimal_sizer_caps_equity_to_active_capital_tier(monkeypatch):
    monkeypatch.setattr(trade_executor.config, "CAPITAL_TIER_CAP_USDC", "500", raising=False)
    result, _, _, effective_fraction, sizing_equity = trade_executor._calculate_deribit_decimal_size(
        FakeDeribitSizing(), "ETHUSDT", _risk_snapshot(),
        entry_price="2000", stop_price="1980", risk_multiplier=1, side="BUY",
    )
    assert effective_fraction == Decimal(str(trade_executor.RISK_PER_TRADE))
    assert sizing_equity == Decimal("500")
    assert result.notional <= Decimal("25")
