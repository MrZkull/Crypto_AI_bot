import hashlib
import hmac
from unittest.mock import MagicMock, patch
import requests
from deribit_client import DeribitClient


def make_client():
    c = DeribitClient.__new__(DeribitClient)
    c.client_id = "unit-client-id"
    c.client_secret = "unit-test-secret"
    c.base = "https://test.deribit.com/api/v2"
    c.session = requests.Session()
    return c


def test_hmac_signature_matches_expected():
    c = make_client()
    uri = "/api/v2/private/get_account_summary?currency=BTC"
    with patch("deribit_client.time.time", return_value=1700000000.123):
        with patch("deribit_client.secrets.token_hex", return_value="0123456789abcdef"):
            header = c._build_hmac_authorization("GET", uri, "")

    message = f"1700000000123\n0123456789abcdef\nGET\n{uri}\n\n"
    expected = hmac.new(b"unit-test-secret", message.encode(), hashlib.sha256).hexdigest()
    assert f"sig={expected}" in header
    assert "unit-test-secret" not in header


def test_private_get_signs_prepared_uri():
    c = make_client()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"result": {"equity": 12.5}}
    c.session.send = MagicMock(return_value=response)

    with patch("deribit_client.time.time", return_value=1700000000.123):
        with patch("deribit_client.secrets.token_hex", return_value="0123456789abcdef"):
            result = c._get("/private/get_account_summary", {"currency": "BTC"})

    req = c.session.send.call_args.args[0]
    assert result == {"equity": 12.5}
    assert req.method == "GET"
    assert req.path_url.endswith("currency=BTC")
    assert req.headers["Authorization"].startswith("deri-hmac-sha256 ")
    assert "unit-test-secret" not in req.url


def test_public_get_has_no_authorization_header():
    c = make_client()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"result": {"testnet": True}}
    c.session.send = MagicMock(return_value=response)

    assert c._get("/public/get_time") == {"testnet": True}
    req = c.session.send.call_args.args[0]
    assert "Authorization" not in req.headers


def test_legacy_post_wrapper_preserves_get_transport():
    c = make_client()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"result": {"order": {"order_id": "123"}}}
    c.session.send = MagicMock(return_value=response)

    with patch("deribit_client.time.time", return_value=1700000000.123):
        with patch("deribit_client.secrets.token_hex", return_value="0123456789abcdef"):
            result = c._post("/private/buy", {"amount": 1})

    req = c.session.send.call_args.args[0]
    assert result["order"]["order_id"] == "123"
    assert req.method == "GET"
    assert "amount=1" in req.path_url
    assert req.headers["Authorization"].startswith("deri-hmac-sha256 ")


def test_non_json_error_is_sanitized():
    c = make_client()
    response = MagicMock()
    response.status_code = 502
    response.json.side_effect = ValueError("private URL must not leak")
    c.session.send = MagicMock(return_value=response)

    try:
        c._get("/private/get_account_summary", {"currency": "BTC"})
    except RuntimeError as exc:
        assert "non-JSON" in str(exc)
        assert "private URL" not in str(exc)
    else:
        raise AssertionError("Expected sanitized error")

def test_set_leverage_uses_hmac_request_without_legacy_auth_refresh():
    c = make_client()
    c.get_instrument_name = MagicMock(return_value="ETH_USDC-PERPETUAL")
    c._get = MagicMock(return_value={"leverage": 2})

    assert c.set_leverage("ETHUSDT", 2) is True

    c._get.assert_called_once_with(
        "/private/set_leverage",
        {
            "instrument_name": "ETH_USDC-PERPETUAL",
            "leverage": 2,
        },
    )
