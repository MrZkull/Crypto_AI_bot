import json
from unittest.mock import MagicMock

import requests
import pytest

from deribit_client import DeribitClient, DeribitTransportError


def make_client(scope="trade:read account:read"):
    client = DeribitClient.__new__(DeribitClient)
    client.client_id = "unit-client-id"
    client.client_secret = "unit-test-secret"
    client.base = "https://test.deribit.com/api/v2"
    client.auth_scope = scope
    client.allow_mutations = "trade:read_write" in scope.split()
    client.session = requests.Session()
    client._auth_lock = __import__("threading").RLock()
    client._access_token = None
    client._refresh_token = None
    client._token_expires_at = 0.0
    client._rpc_id = 0
    return client


def response(payload, status=200):
    item = MagicMock()
    item.status_code = status
    item.json.return_value = payload
    return item


def parse_body(prepared):
    body = prepared.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def test_private_read_uses_scoped_oauth_and_json_rpc_http_post():
    client = make_client()
    client.session.send = MagicMock(side_effect=[
        response({"jsonrpc": "2.0", "id": 1, "result": {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
            "scope": "trade:read account:read connection mainaccount",
        }}),
        response({"jsonrpc": "2.0", "id": 2, "result": {"equity": 12.5}}),
    ])

    result = client._get("/private/get_account_summary", {"currency": "USDC"})

    auth_req = client.session.send.call_args_list[0].args[0]
    private_req = client.session.send.call_args_list[1].args[0]
    from urllib.parse import parse_qs, urlparse
    auth_params = parse_qs(urlparse(auth_req.url).query)
    private_body = parse_body(private_req)

    assert result == {"equity": 12.5}
    assert auth_req.method == "GET"
    assert auth_req.url.startswith("https://test.deribit.com/api/v2/public/auth?")
    assert auth_params["grant_type"] == ["client_credentials"]
    assert auth_params["scope"] == ["trade:read account:read"]
    assert auth_params["client_secret"] == ["unit-test-secret"]
    assert private_req.method == "POST"
    assert private_req.url == "https://test.deribit.com/api/v2/private/get_account_summary"
    assert private_req.headers["Authorization"] == "Bearer test-access-token"
    assert private_body["jsonrpc"] == "2.0"
    assert private_body["method"] == "private/get_account_summary"
    assert private_body["params"] == {"currency": "USDC"}
    assert "unit-test-secret" not in private_req.url
    assert "unit-test-secret" not in private_req.body.decode("utf-8")
    assert client._refresh_token == "test-refresh-token"


def test_scoped_token_is_reused_for_multiple_private_reads():
    client = make_client()
    client._access_token = "cached-token"
    client._refresh_token = "refresh-token"
    client._token_expires_at = __import__("time").time() + 3600
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {"testnet": True}
    }))

    assert client._get("/private/get_time") == {"testnet": True}
    assert client.session.send.call_count == 1
    req = client.session.send.call_args.args[0]
    assert req.method == "POST"
    assert parse_body(req)["method"] == "private/get_time"
    assert req.headers["Authorization"] == "Bearer cached-token"


def test_expired_token_refreshes_with_rotating_refresh_token():
    client = make_client()
    client._access_token = "expired-token"
    client._refresh_token = "old-refresh-token"
    client._token_expires_at = 0
    client.session.send = MagicMock(side_effect=[
        response({"jsonrpc": "2.0", "id": 1, "result": {
            "access_token": "renewed-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "scope": "trade:read account:read connection",
        }}),
        response({"jsonrpc": "2.0", "id": 2, "result": []}),
    ])

    assert client._get("/private/get_positions", {"currency": "USDC"}) == []
    refresh_req = client.session.send.call_args_list[0].args[0]
    private_req = client.session.send.call_args_list[1].args[0]
    from urllib.parse import parse_qs, urlparse
    refresh_params = parse_qs(urlparse(refresh_req.url).query)
    assert refresh_req.method == "GET"
    assert refresh_params == {
        "grant_type": ["refresh_token"],
        "refresh_token": ["old-refresh-token"],
    }
    assert private_req.method == "POST"
    assert parse_body(private_req)["params"] == {"currency": "USDC"}
    assert private_req.headers["Authorization"] == "Bearer renewed-token"
    assert client._refresh_token == "new-refresh-token"


def test_public_read_uses_json_rpc_http_post_without_authentication():
    client = make_client()
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {"testnet": True}
    }))

    assert client._get("/public/get_time") == {"testnet": True}
    req = client.session.send.call_args.args[0]
    assert req.method == "POST"
    assert req.url.endswith("/api/v2/public/get_time")
    assert parse_body(req)["method"] == "public/get_time"
    assert "Authorization" not in req.headers


def test_mutating_wrapper_uses_json_rpc_http_post_and_is_locally_gated():
    client = make_client("trade:read_write account:read")
    client._access_token = "execution-token"
    client._token_expires_at = __import__("time").time() + 3600
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {"order": {"order_id": "123"}}
    }))

    result = client._post("/private/buy", {"amount": 1, "instrument_name": "ETH_USDC-PERPETUAL"})

    req = client.session.send.call_args.args[0]
    assert result["order"]["order_id"] == "123"
    assert req.method == "POST"
    assert req.url == "https://test.deribit.com/api/v2/private/buy"
    assert parse_body(req)["method"] == "private/buy"
    assert parse_body(req)["params"] == {"amount": 1, "instrument_name": "ETH_USDC-PERPETUAL"}
    assert req.headers["Authorization"] == "Bearer execution-token"



def test_historical_reconciliation_endpoints_are_allowlisted_private_reads():
    client = make_client()
    client._access_token = "cached-read-token"
    client._token_expires_at = __import__("time").time() + 3600
    client.session.send = MagicMock(side_effect=[
        response({"jsonrpc": "2.0", "id": 1, "result": [{
            "order_id": "ETH-hist-1", "label": "cb_entry_1",
            "instrument_name": "ETH_USDC-PERPETUAL", "order_state": "filled",
        }]}),
        response({"jsonrpc": "2.0", "id": 2, "result": [{
            "order_id": "ETH-hist-1", "trade_id": "ETH-trade-1",
            "instrument_name": "ETH_USDC-PERPETUAL",
        }]}),
    ])

    orders = client.get_order_history_by_instrument_name(
        "ETH_USDC-PERPETUAL", count=1000, historical=True,
        include_old=True, include_unfilled=True, offset=0, strict=True,
    )
    fills = client.get_user_trades_by_order("ETH-hist-1", historical=True, strict=True)

    assert orders[0]["label"] == "cb_entry_1"
    assert fills[0]["trade_id"] == "ETH-trade-1"
    requests = [call.args[0] for call in client.session.send.call_args_list]
    assert [req.method for req in requests] == ["POST", "POST"]
    assert requests[0].url.endswith("/private/get_order_history_by_instrument")
    assert requests[1].url.endswith("/private/get_user_trades_by_order")
    assert all(req.headers["Authorization"] == "Bearer cached-read-token" for req in requests)
    assert "/private/get_order_history_by_instrument" not in client._mutating_methods()
    assert "/private/get_user_trades_by_order" not in client._mutating_methods()


def test_recent_trade_history_end_sequence_is_sent_for_read_only_pagination():
    client = make_client()
    client._access_token = "cached-read-token"
    client._token_expires_at = __import__("time").time() + 3600
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {
            "trades": [{
                "order_id": "ETH-2", "trade_id": "trade-2",
                "trade_seq": 9000, "instrument_name": "ETH_USDC-PERPETUAL",
            }],
            "has_more": True,
        }
    }))

    rows = client.get_trade_history_by_instrument_name(
        "ETH_USDC-PERPETUAL", count=1000, end_seq=9000, strict=True
    )

    assert rows[0]["trade_seq"] == 9000
    request = client.session.send.call_args.args[0]
    assert request.method == "POST"
    assert parse_body(request)["params"] == {
        "instrument_name": "ETH_USDC-PERPETUAL",
        "count": 1000,
        "sorting": "desc",
        "end_seq": 9000,
    }

def test_non_json_error_is_sanitized():
    client = make_client()
    client.session.send = MagicMock(return_value=MagicMock(
        status_code=502,
        json=MagicMock(side_effect=ValueError("private URL must not leak")),
    ))

    try:
        client._get("/public/get_time")
    except DeribitTransportError as exc:
        assert "status=502" in str(exc)
        assert "private URL" not in str(exc)
    else:
        raise AssertionError("Expected sanitized transport error")


def test_rpc_errors_are_reported_without_request_secrets():
    client = make_client()
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": 13009, "message": "unauthorized"},
    }))

    try:
        client._get("/public/get_time")
    except RuntimeError as exc:
        assert "13009" in str(exc)
        assert "unit-test-secret" not in str(exc)
    else:
        raise AssertionError("Expected RPC error")


def test_invalid_auth_scopes_are_rejected(monkeypatch):
    monkeypatch.setattr(DeribitClient, "_verify_instruments", lambda self: None)
    with pytest.raises(ValueError, match="least-privilege"):
        DeribitClient("client", "secret", auth_scope="trade:read_write account:read")
    with pytest.raises(ValueError, match="least-privilege"):
        DeribitClient("client", "secret", auth_scope="trade:read account:read;wallet:read_write")
    with pytest.raises(ValueError, match="production mutations"):
        DeribitClient(
            "client", "secret", testnet=False,
            auth_scope="trade:read_write account:read", allow_mutations=True,
        )


def test_read_only_client_rejects_order_mutation_before_network_access():
    client = make_client("trade:read account:read")
    client.session.send = MagicMock()
    try:
        client._post("/private/buy", {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 1})
    except PermissionError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Read-only client accepted order placement")
    client.session.send.assert_not_called()


def test_set_leverage_uses_mutating_method_and_is_gated():
    client = make_client("trade:read_write account:read")
    client.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    client._post = MagicMock(return_value={"leverage": 2})

    assert client.set_leverage("ETHUSDT", 2) is True
    client._post.assert_called_once_with(
        "/private/set_leverage",
        {"instrument_name": "ETH_USDC-PERPETUAL", "leverage": 2},
    )


def test_constructor_defaults_to_read_only_scope(monkeypatch):
    monkeypatch.setattr(DeribitClient, "_verify_instruments", lambda self: None)
    client = DeribitClient("client", "secret", testnet=True)
    assert client.auth_scope == "trade:read account:read"
    assert client.allow_mutations is False


def test_constructor_allows_explicit_testnet_write_scope_only(monkeypatch):
    monkeypatch.setattr(DeribitClient, "_verify_instruments", lambda self: None)
    monkeypatch.setenv("DERIBIT_ENVIRONMENT", "TESTNET")
    monkeypatch.setenv("DERIBIT_BASE_URL", "https://test.deribit.com/api/v2")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_ID", "client")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_SECRET", "secret")
    monkeypatch.setenv("DERIBIT_TESTNET_MUTATIONS_ENABLED", "true")
    monkeypatch.setenv(
        "DERIBIT_TESTNET_MUTATION_ACK",
        "I_ACKNOWLEDGE_DERIBIT_TESTNET_ONLY",
    )
    monkeypatch.delenv("DERIBIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("DERIBIT_CLIENT_SECRET", raising=False)
    client = DeribitClient(
        "client", "secret", testnet=True,
        auth_scope="trade:read_write account:read", allow_mutations=True,
    )
    assert client.allow_mutations is True


def test_get_order_fails_closed_when_order_state_read_is_unavailable():
    client = make_client()
    client._get = MagicMock(side_effect=TimeoutError("temporary exchange outage"))
    with pytest.raises(RuntimeError, match="order-state snapshot unavailable"):
        client.get_order("order-1")


def test_get_order_reports_explicit_not_found_state():
    client = make_client()
    client._get = MagicMock(side_effect=RuntimeError("11000 order_not_found"))
    result = client.get_order("order-1")
    assert result == {"order_state": "not_found", "order_id": "order-1"}


def test_cancel_unknown_outcome_raises_for_reconciliation():
    client = make_client("trade:read_write account:read")
    client._post = MagicMock(side_effect=TimeoutError("gateway timeout"))
    with pytest.raises(RuntimeError, match="cancel outcome unknown"):
        client.cancel_order("order-1")


def test_cancel_already_closed_is_idempotent_success():
    client = make_client("trade:read_write account:read")
    client._post = MagicMock(side_effect=RuntimeError("11044 not_open_order"))
    assert client.cancel_order("order-1") == {
        "result": "already_closed",
        "order_id": "order-1",
    }


def test_trade_history_defaults_to_strict_failure():
    client = make_client()
    client._get = MagicMock(side_effect=TimeoutError("trade history unavailable"))
    with pytest.raises(TimeoutError):
        client.get_trade_history_by_instrument_name("ETH_USDC-PERPETUAL")


def test_authentication_rejects_server_scope_downgrade():
    client = make_client()
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {
            "access_token": "token-without-trade-read",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "account:read connection",
        }
    }))
    with pytest.raises(RuntimeError, match="fewer permissions than requested"):
        client._authenticate()


def test_read_only_authentication_rejects_unexpected_write_scope():
    client = make_client()
    client.session.send = MagicMock(return_value=response({
        "jsonrpc": "2.0", "id": 1, "result": {
            "access_token": "overprivileged-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "trade:read account:read trade:read_write connection",
        }
    }))
    with pytest.raises(RuntimeError, match="unexpectedly granted write"):
        client._authenticate()


def test_write_scoped_client_cannot_call_mutation_through_read_wrapper():
    client = make_client("trade:read_write account:read")
    client.session.send = MagicMock()
    with pytest.raises(PermissionError, match="through _get"):
        client._get("/private/buy", {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 1})
    client.session.send.assert_not_called()


def test_write_scoped_client_cannot_call_unlisted_mutation_wrapper():
    client = make_client("trade:read_write account:read")
    client.session.send = MagicMock()
    with pytest.raises(PermissionError, match="mutation allowlist"):
        client._post("/private/unknown_method", {"symbol": "ETHUSDT"})
    client.session.send.assert_not_called()


def test_transport_exception_does_not_leak_client_secret():
    client = make_client()
    client.session.send = MagicMock(
        side_effect=requests.RequestException(
            "GET https://test.deribit.com/api/v2/public/auth?client_secret=unit-test-secret failed"
        )
    )
    with pytest.raises(RuntimeError, match="HTTP transport failed") as caught:
        client._get("/public/get_time")
    assert "unit-test-secret" not in str(caught.value)


@pytest.mark.parametrize("error", [
    "11000 order_not_found",
    "10004 order_not_found",
    "order_not_found",
])
def test_get_order_recognizes_documented_order_not_found_codes(error):
    client = make_client()
    client._get = MagicMock(side_effect=RuntimeError(error))
    assert client.get_order("order-42") == {
        "order_state": "not_found", "order_id": "order-42"
    }


def test_get_order_does_not_misclassify_generic_instrument_not_found():
    client = make_client()
    client._get = MagicMock(side_effect=RuntimeError("instrument not_found"))
    with pytest.raises(RuntimeError, match="order-state snapshot unavailable"):
        client.get_order("order-42")


def test_get_order_rejects_missing_or_mismatched_order_identity():
    client = make_client()
    client._get = MagicMock(return_value={"order_state": "open", "order_id": "other-id"})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        client.get_order("order-expected")


def test_get_order_rejects_empty_state():
    client = make_client()
    client._get = MagicMock(return_value={"order_id": "order-expected"})
    with pytest.raises(RuntimeError, match="response malformed"):
        client.get_order("order-expected")


def test_unreviewed_private_method_fails_closed():
    client = make_client()
    client.session.send = MagicMock()
    with pytest.raises(PermissionError, match="not in the reviewed"):
        client._get("/private/withdraw", {"currency": "USDC", "amount": 1})
    client.session.send.assert_not_called()


def test_get_server_time_ms_uses_public_read_endpoint_without_private_auth():
    client = make_client()
    client.session.send = MagicMock(return_value=response({"jsonrpc": "2.0", "id": 1, "result": 1730000000123}))

    assert client.get_server_time_ms() == 1730000000123
    request = client.session.send.call_args.args[0]
    assert request.method == "POST"
    assert request.url.endswith("/api/v2/public/get_time")
    assert "Authorization" not in request.headers


@pytest.mark.parametrize("value", [None, True, 0, -1, "not-a-time", 1730000000123.5])
def test_get_server_time_ms_rejects_invalid_timestamp(value):
    client = make_client()
    client._get = MagicMock(return_value=value)
    with pytest.raises(Exception, match="server-time response"):
        client.get_server_time_ms()

@pytest.mark.parametrize(
    "extra_scope",
    ["wallet:read", "block_trade:read", "block_rfq:read"],
)
def test_read_only_authentication_rejects_unrequested_read_permissions(extra_scope):
    client = make_client()
    result = {
        "access_token": "test-token",
        "expires_in": 60,
        "scope": f"trade:read account:read connection {extra_scope}",
    }

    with pytest.raises(RuntimeError, match="unexpected functional permissions"):
        client._store_auth_result(result)


@pytest.mark.parametrize(
    "extra_scope",
    [
        "account:read_write",
        "wallet:read_write",
        "block_trade:read_write",
        "block_rfq:read_write",
    ],
)
def test_read_only_authentication_rejects_every_unrequested_write_family(extra_scope):
    client = make_client()
    result = {
        "access_token": "test-token",
        "expires_in": 60,
        "scope": f"trade:read account:read connection {extra_scope}",
    }

    with pytest.raises(RuntimeError, match="unexpectedly granted write permissions"):
        client._store_auth_result(result)
