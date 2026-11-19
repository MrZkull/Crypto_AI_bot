import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import pytest

from exchange_adapters_readonly import (
    BinanceUSDMReadOnlyAdapter,
    DeltaReadOnlyAdapter,
    DeribitReadOnlyAdapter,
)
from exchange_reliability import SnapshotUnavailable


class FakeDeribit:
    def __init__(self, base="https://test.deribit.com/api/v2"):
        self.base = base
        self.order_instrument = None
        self.fill_instrument = None

    def get_account_summaries(self):
        return {"summaries": [{"currency": "USDC", "equity": 1.0}]}

    def get_positions(self):
        return [{"instrument_name": "ETH_USDC-PERPETUAL", "size": 1}]

    def get_open_orders_by_instrument(self, instrument):
        self.order_instrument = instrument
        return [{"instrument_name": instrument}]

    def get_trade_history_by_instrument_name(self, instrument, count=10, strict=False):
        self.fill_instrument = instrument
        return [{"instrument_name": instrument}]


def test_deribit_adapter_requires_environment_host_match():
    with pytest.raises(ValueError):
        DeribitReadOnlyAdapter(FakeDeribit("https://www.deribit.com/api/v2"), environment="testnet")
    with pytest.raises(ValueError):
        DeribitReadOnlyAdapter(FakeDeribit(), environment="unknown")


def test_deribit_adapter_preserves_exact_instrument_name():
    client = FakeDeribit()
    adapter = DeribitReadOnlyAdapter(client, environment="testnet")
    instrument = "ETH_USDC-PERPETUAL"
    adapter.read_open_orders(instrument)
    adapter.read_recent_fills(instrument)
    assert client.order_instrument == instrument
    assert client.fill_instrument == instrument
    assert not hasattr(adapter, "place_order")


class FakeDeltaClient:
    base = "https://cdn-ind.testnet.deltaex.org"

    def __init__(self):
        self.calls = []
        self.fail = False

    def _get(self, path, params=None):
        self.calls.append((path, params or {}))
        if self.fail:
            raise TimeoutError("network")
        return {"success": True, "result": [{"size": 1, "product_id": 22}]}


def test_delta_adapter_rejects_wrong_environment_and_host():
    client = FakeDeltaClient()
    with pytest.raises(ValueError):
        DeltaReadOnlyAdapter(client, environment="production")
    client.base = "https://api.india.delta.exchange"
    with pytest.raises(ValueError):
        DeltaReadOnlyAdapter(client, environment="testnet")


def test_delta_adapter_raises_instead_of_masking_failed_reads():
    client = FakeDeltaClient()
    adapter = DeltaReadOnlyAdapter(client)
    client.fail = True
    with pytest.raises(TimeoutError):
        adapter.read_positions()


def test_delta_adapter_requires_exchange_product_id_for_order_filter():
    adapter = DeltaReadOnlyAdapter(FakeDeltaClient())
    with pytest.raises(ValueError, match="product ID"):
        adapter.read_open_orders("ETHUSDT")
    rows = adapter.read_open_orders("22")
    assert rows and rows[0]["product_id"] == 22


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return FakeResponse(self.payload)


def test_binance_adapter_signs_exact_query_string_and_uses_get_only():
    session = FakeSession({"assets": []})
    adapter = BinanceUSDMReadOnlyAdapter(
        "key",
        "secret",
        base_url="https://testnet.binancefuture.com",
        environment="testnet",
        session=session,
        clock=lambda: 1700000000,
    )

    adapter.read_account()
    call = session.calls[0]
    params = call["params"]
    assert isinstance(params, str)
    parsed = parse_qs(params, keep_blank_values=True)
    signature = parsed.pop("signature")[0]
    # The adapter signs a sorted canonical query and sends that exact query string.
    canonical = "&".join(
        f"{key}={values[0]}" for key, values in sorted(parsed.items())
    )
    expected = hmac.new(b"secret", canonical.encode(), hashlib.sha256).hexdigest()
    assert signature == expected
    assert call["headers"]["X-MBX-APIKEY"] == "key"
    assert not hasattr(adapter, "place_order")
    assert not hasattr(adapter, "cancel_order")


def test_binance_adapter_refuses_production_without_explicit_override():
    with pytest.raises(ValueError, match="allow_production"):
        BinanceUSDMReadOnlyAdapter(
            "key", "secret", base_url="https://fapi.binance.com", environment="production"
        )


def test_binance_adapter_rejects_non_allowlisted_test_host():
    with pytest.raises(ValueError, match="Unexpected Binance"):
        BinanceUSDMReadOnlyAdapter(
            "key", "secret", base_url="https://example.com", environment="testnet"
        )


def test_binance_adapter_raises_on_exchange_error():
    session = FakeSession({"code": -2015, "msg": "Invalid API-key"})
    adapter = BinanceUSDMReadOnlyAdapter(
        "key", "secret", base_url="https://testnet.binancefuture.com",
        environment="testnet", session=session, clock=lambda: 1700000000,
    )
    with pytest.raises(SnapshotUnavailable):
        adapter.read_positions()


class PaginatedDeltaClient(FakeDeltaClient):
    def __init__(self, pages):
        super().__init__()
        self.pages = list(pages)

    def _get(self, path, params=None):
        params = dict(params or {})
        self.calls.append((path, params))
        if not self.pages:
            raise AssertionError("Unexpected extra Delta page request")
        return self.pages.pop(0)


def test_delta_fills_follow_after_cursor_until_complete():
    client = PaginatedDeltaClient([
        {"success": True, "result": [{"id": 1, "product_id": 22}], "meta": {"after": "c1"}},
        {"success": True, "result": [{"id": 2, "product_id": 22}], "meta": {"after": None}},
    ])
    adapter = DeltaReadOnlyAdapter(client)
    rows = adapter.read_recent_fills("22", limit=50)
    assert [row["id"] for row in rows] == [1, 2]
    assert client.calls[0][1]["page_size"] == 50
    assert client.calls[1][1]["after"] == "c1"


def test_delta_wallet_transactions_support_pagination_and_cashflow_types():
    client = PaginatedDeltaClient([
        {"success": True, "result": [{"transaction_type": "deposit"}], "meta": {"after": "next"}},
        {"success": True, "result": [{"transaction_type": "withdrawal"}], "meta": {"after": None}},
    ])
    adapter = DeltaReadOnlyAdapter(client)
    rows = adapter.read_wallet_transactions(start_time=100, end_time=200)
    assert [row["transaction_type"] for row in rows] == ["deposit", "withdrawal"]
    assert client.calls[1][1]["after"] == "next"
    assert client.calls[0][1]["start_time"] == 100


def test_delta_repeated_pagination_cursor_fails_closed():
    client = PaginatedDeltaClient([
        {"success": True, "result": [{"id": 1}], "meta": {"after": "same"}},
        {"success": True, "result": [{"id": 2}], "meta": {"after": "same"}},
    ])
    adapter = DeltaReadOnlyAdapter(client)
    with pytest.raises(SnapshotUnavailable, match="cursor repeated"):
        adapter.read_recent_fills("22", limit=100)
