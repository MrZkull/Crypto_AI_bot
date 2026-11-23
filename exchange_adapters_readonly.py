"""Strict read-only adapters for Deribit, Delta and Binance USD-M Futures.

No adapter in this module exposes order submission, cancellation, leverage,
margin-mode changes, transfers or any other write operation.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import requests

from exchange_reliability import SnapshotUnavailable


class AdapterError(RuntimeError):
    pass


def _strict_result(payload: Any, *, name: str, expected: type = list):
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise SnapshotUnavailable(f"{name} returned an exchange error")
    result = payload.get("result", payload)
    if not isinstance(result, expected):
        raise SnapshotUnavailable(f"{name} response is malformed")
    return result


class DeribitReadOnlyAdapter:
    exchange = "DERIBIT"

    def __init__(self, client, *, environment: str = "testnet", allow_production: bool = False):
        self.client = client
        self.environment = str(environment).lower()
        if self.environment not in {"testnet", "production"}:
            raise ValueError("Deribit environment must be testnet or production")
        if self.environment == "production" and not allow_production:
            raise ValueError("Production account reads require allow_production=True")
        expected = (
            "https://test.deribit.com/api/v2"
            if self.environment == "testnet"
            else "https://www.deribit.com/api/v2"
        )
        if getattr(client, "base", None) != expected:
            raise ValueError("Deribit client host does not match adapter environment")

    def read_account(self) -> dict:
        response = self.client.get_account_summaries()
        if not isinstance(response, dict) or not isinstance(response.get("summaries"), list):
            raise SnapshotUnavailable("Deribit account summaries malformed")
        rows = response["summaries"]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise SnapshotUnavailable("Deribit account summaries incomplete")
        currencies = [str(row.get("currency", "")).upper() for row in rows]
        if any(not c for c in currencies) or len(currencies) != len(set(currencies)):
            raise SnapshotUnavailable("Deribit account currency identity invalid")
        return {"exchange": self.exchange, "environment": self.environment, "summaries": rows}

    def read_positions(self) -> list[dict]:
        positions = self.client.get_positions()
        if not isinstance(positions, list) or any(not isinstance(p, dict) for p in positions):
            raise SnapshotUnavailable("Deribit positions malformed")
        for position in positions:
            instrument = position.get("instrument_name")
            if not isinstance(instrument, str) or not instrument.strip():
                raise SnapshotUnavailable("Deribit position has no instrument_name")
        return positions

    def read_open_orders(self, instrument_id: Optional[str] = None) -> list[dict]:
        if not instrument_id:
            raise ValueError("Deribit open-order reads require exact instrument_name")
        orders = self.client.get_open_orders_by_instrument(instrument_id)
        if not isinstance(orders, list) or any(not isinstance(o, dict) for o in orders):
            raise SnapshotUnavailable("Deribit open orders malformed")
        if any(o.get("instrument_name") not in (None, instrument_id) for o in orders):
            raise SnapshotUnavailable("Deribit returned an order for a different instrument")
        return orders

    def read_recent_fills(self, instrument_id: str, *, limit: int = 50) -> list[dict]:
        if not instrument_id:
            raise ValueError("Deribit fills require exact instrument_name")
        fills = self.client.get_trade_history_by_instrument_name(instrument_id, count=limit, strict=True)
        if not isinstance(fills, list) or any(not isinstance(f, dict) for f in fills):
            raise SnapshotUnavailable("Deribit fills malformed")
        if any(f.get("instrument_name") not in (None, instrument_id) for f in fills):
            raise SnapshotUnavailable("Deribit returned a fill for a different instrument")
        return fills


class DeltaReadOnlyAdapter:
    exchange = "DELTA"

    def __init__(self, client, *, environment: str = "testnet"):
        self.client = client
        self.environment = str(environment).lower()
        if self.environment != "testnet":
            raise ValueError("This initial Delta adapter is restricted to the existing Demo/Testnet client")
        if getattr(client, "base", None) != "https://cdn-ind.testnet.deltaex.org":
            raise ValueError("Delta client base URL is not the verified India Demo/Testnet host")

    def _read(self, path: str, params: Optional[dict] = None) -> list:
        payload = self.client._get(path, params or {})
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise SnapshotUnavailable(f"Delta {path} read failed or returned an unsuccessful envelope")
        result = payload.get("result")
        if not isinstance(result, list) or any(not isinstance(x, dict) for x in result):
            raise SnapshotUnavailable(f"Delta {path} response malformed")
        return result

    def _read_paged(
        self,
        path: str,
        params: Optional[dict] = None,
        *,
        page_size: int = 100,
        max_pages: int = 20,
        max_items: Optional[int] = None,
    ) -> list[dict]:
        """Read every cursor page or raise; never return a silent partial snapshot.

        Delta documents `meta.after` as the cursor for the next page. Fills and
        historical order endpoints currently cap page_size at 50; the callers
        below observe that cap. This method also rejects repeated cursors and
        an exhausted page budget when more data remains.
        """
        if not 1 <= page_size <= 100 or max_pages < 1:
            raise ValueError("Invalid Delta pagination bounds")
        query = dict(params or {})
        query["page_size"] = page_size
        cursor = None
        seen_cursors = set()
        rows: list[dict] = []
        for page_number in range(max_pages):
            if cursor is None:
                query.pop("after", None)
            else:
                query["after"] = cursor
            payload = self.client._get(path, dict(query))
            if not isinstance(payload, dict) or payload.get("success") is not True:
                raise SnapshotUnavailable(f"Delta {path} page {page_number + 1} failed")
            page = payload.get("result")
            meta = payload.get("meta", {})
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise SnapshotUnavailable(f"Delta {path} page {page_number + 1} malformed")
            if not isinstance(meta, dict):
                raise SnapshotUnavailable(f"Delta {path} pagination metadata malformed")
            rows.extend(page)
            if max_items is not None and len(rows) >= max_items:
                return rows[:max_items]
            next_cursor = meta.get("after")
            if next_cursor in (None, ""):
                return rows
            next_cursor = str(next_cursor)
            if next_cursor in seen_cursors or next_cursor == cursor:
                raise SnapshotUnavailable(f"Delta {path} pagination cursor repeated")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise SnapshotUnavailable(
            f"Delta {path} exceeds the configured {max_pages}-page completeness limit"
        )

    def read_account(self) -> dict:
        rows = self._read("/v2/wallet/balances")
        return {"exchange": self.exchange, "environment": self.environment, "balances": rows}

    def read_positions(self) -> list[dict]:
        rows = self._read("/v2/positions/margined")
        # Preserve zero rows in snapshot completeness; callers decide whether to omit them.
        for row in rows:
            if "size" not in row:
                raise SnapshotUnavailable("Delta position row missing size")
        return rows

    def read_products(self, *, page_size: int = 100, max_pages: int = 20) -> list[dict]:
        """Return a cursor-complete product catalog for instrument verification."""
        return self._read_paged(
            "/v2/products", page_size=page_size, max_pages=max_pages
        )

    def read_open_orders(self, instrument_id: Optional[str] = None) -> list[dict]:
        # Delta `GET /v2/orders` is cursor-paginated. Never mistake one page for
        # the account's complete open-order snapshot.
        rows = self._read_paged(
            "/v2/orders", {"states": "open"}, page_size=100, max_pages=20
        )
        if instrument_id is None:
            return rows
        # Delta uses product IDs; never infer a product ID from a symbol string.
        try:
            product_id = int(instrument_id)
        except (TypeError, ValueError):
            raise ValueError("Delta instrument_id must be an exchange product ID") from None
        return [row for row in rows if int(row.get("product_id", -1)) == product_id]

    def read_recent_fills(self, instrument_id: str, *, limit: int = 50) -> list[dict]:
        if not instrument_id:
            raise ValueError("Delta fills require an exchange product ID")
        product_id = int(instrument_id)
        if not 1 <= limit <= 1000:
            raise ValueError("Delta fill limit must be 1..1000")
        return self._read_paged(
            "/v2/fills", {"product_ids": str(product_id)},
            page_size=min(limit, 50), max_pages=20, max_items=limit,
        )

    def read_wallet_transactions(
        self,
        *,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page_size: int = 100,
        max_pages: int = 20,
    ) -> list[dict]:
        """Read wallet cashflows/deposits/withdrawals/fees/funding for reconciliation."""
        params = {}
        if start_time is not None:
            params["start_time"] = int(start_time)
        if end_time is not None:
            params["end_time"] = int(end_time)
        return self._read_paged(
            "/v2/wallet/transactions", params,
            page_size=page_size, max_pages=max_pages,
        )


class BinanceUSDMReadOnlyAdapter:
    """HMAC-signed GET-only USD-M adapter. The HTTP surface has no write method.

    The caller must provide the chosen test/demo or production base URL. Mainnet
    private reads require explicit allow_production=True and a read-only API key.
    """
    exchange = "BINANCE_USDM"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str,
        environment: str = "testnet",
        allow_production: bool = False,
        session: Optional[requests.Session] = None,
        clock: callable = time.time,
    ):
        parsed = urlparse(str(base_url).rstrip("/"))
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Binance base URL must be a clean HTTPS URL")
        host = parsed.hostname.lower()
        env = str(environment).lower()
        if env == "production" and not allow_production:
            raise ValueError("Production private reads require allow_production=True")
        if env not in {"testnet", "demo", "production"}:
            raise ValueError("Unsupported Binance environment")
        if env == "production" and host not in {"fapi.binance.com", "fapi1.binance.com", "fapi2.binance.com", "fapi3.binance.com"}:
            raise ValueError("Unexpected Binance production USD-M host")
        if env in {"testnet", "demo"} and host not in {"testnet.binancefuture.com", "demo-fapi.binance.com"}:
            raise ValueError("Unexpected Binance test/demo host; verify endpoint before use")
        if not api_key or not api_secret:
            raise ValueError("Binance read-only API credentials are required")
        self.api_key = str(api_key)
        self.api_secret = str(api_secret)
        self.base_url = str(base_url).rstrip("/")
        self.environment = env
        self.session = session or requests.Session()
        self.clock = clock

    def _get(self, path: str, params: Optional[dict] = None, *, signed: bool = True):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid Binance endpoint path")
        query = dict(params or {})
        headers = {"X-MBX-APIKEY": self.api_key}
        request_params: Any = query
        if signed:
            query["timestamp"] = int(self.clock() * 1000)
            query.setdefault("recvWindow", 5000)
            # Sign the exact query string sent on the wire. Passing a dict to
            # requests after signing a differently sorted string can produce
            # a signature mismatch when insertion order differs.
            canonical = urlencode(sorted(query.items()))
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                canonical.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            request_params = canonical + "&signature=" + signature
        response = self.session.get(
            self.base_url + path,
            params=request_params,
            headers=headers,
            timeout=10,
        )
        try:
            payload = response.json()
        except ValueError:
            raise SnapshotUnavailable(f"Binance {path} returned non-JSON response") from None
        if not response.ok or (isinstance(payload, dict) and "code" in payload and payload.get("code", 0) < 0):
            code = payload.get("code") if isinstance(payload, dict) else None
            raise SnapshotUnavailable(f"Binance {path} read failed (HTTP {response.status_code}, code={code})")
        return payload

    def read_account(self) -> dict:
        payload = self._get("/fapi/v3/account")
        if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
            raise SnapshotUnavailable("Binance account response malformed")
        return {"exchange": self.exchange, "environment": self.environment, "account": payload}

    def read_positions(self) -> list[dict]:
        payload = self._get("/fapi/v3/positionRisk")
        if not isinstance(payload, list) or any(not isinstance(p, dict) for p in payload):
            raise SnapshotUnavailable("Binance position risk response malformed")
        return payload

    def read_open_orders(self, instrument_id: Optional[str] = None) -> list[dict]:
        params = {"symbol": instrument_id} if instrument_id else {}
        payload = self._get("/fapi/v1/openOrders", params)
        if not isinstance(payload, list) or any(not isinstance(o, dict) for o in payload):
            raise SnapshotUnavailable("Binance open orders response malformed")
        return payload

    def read_recent_fills(self, instrument_id: str, *, limit: int = 50) -> list[dict]:
        if not instrument_id:
            raise ValueError("Binance userTrades requires an exact exchange symbol")
        if not 1 <= limit <= 1000:
            raise ValueError("Binance userTrades limit must be 1..1000")
        payload = self._get("/fapi/v1/userTrades", {"symbol": instrument_id, "limit": limit})
        if not isinstance(payload, list) or any(not isinstance(f, dict) for f in payload):
            raise SnapshotUnavailable("Binance user trades response malformed")
        return payload

    def read_exchange_info(self) -> dict:
        payload = self._get("/fapi/v1/exchangeInfo", signed=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise SnapshotUnavailable("Binance exchangeInfo response malformed")
        return payload


class FaultInjectingReadOnlyAdapter:
    """Deterministic fake adapter for contract/failure tests; never submits orders."""

    def __init__(self, *, account=None, positions=None, orders=None, fills=None):
        self.exchange = "SIMULATOR"
        self.environment = "simulation"
        self.account = account if account is not None else {"balances": []}
        self.positions = positions if positions is not None else []
        self.orders = orders if orders is not None else []
        self.fills = fills if fills is not None else []
        self.failures: dict[str, Exception] = {}
        self.calls: list[str] = []

    def inject_failure(self, operation: str, error: Exception) -> None:
        self.failures[operation] = error

    def _take(self, operation: str, value):
        self.calls.append(operation)
        if operation in self.failures:
            raise self.failures[operation]
        return value

    def read_account(self) -> dict:
        return self._take("account", self.account)

    def read_positions(self) -> list[dict]:
        return self._take("positions", list(self.positions))

    def read_open_orders(self, instrument_id: Optional[str] = None) -> list[dict]:
        rows = list(self.orders)
        if instrument_id is not None:
            rows = [row for row in rows if row.get("instrument_id") == instrument_id]
        return self._take("open_orders", rows)

    def read_recent_fills(self, instrument_id: str, *, limit: int = 50) -> list[dict]:
        rows = [row for row in self.fills if row.get("instrument_id") == instrument_id][:limit]
        return self._take("fills", rows)
