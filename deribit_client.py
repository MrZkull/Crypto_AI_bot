# deribit_client.py — V13.3: Unified Linear USDC Execution Engine & Authoritative Safeguards

import logging
import math
import requests
import threading
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from exchange_reliability import (
    BreakerState, CircuitBreaker, SnapshotUnavailable, retry_read,
)
from exchange_environment_guard import validate_deribit_testnet_mutation_environment
from deribit_execution_gateway import (
    ExecutionGatewayBlocked, ExecutionGatewayOutcomeUnknown, create_deribit_execution_gateway,
)

log = logging.getLogger(__name__)


class DeribitTransportError(RuntimeError):
    """Transport/server failure; a mutation may have an unknown outcome."""

    def __init__(self, message, *, retry_after_seconds=None, unknown_outcome=False):
        self.retry_after_seconds = retry_after_seconds
        self.unknown_outcome = bool(unknown_outcome)
        super().__init__(message)


class DeribitRPCError(RuntimeError):
    """Explicit JSON-RPC error response (known exchange-side rejection)."""

    def __init__(self, method, code, message):
        self.method = str(method)
        self.code = str(code)
        super().__init__(f"Deribit RPC {self.method} failed (code={self.code}): {message}")


class OrderOutcomeUnknown(RuntimeError):
    """A submit/cancel response was lost or malformed; never blind-retry."""


class ProtectiveOrderVerificationError(RuntimeError):
    """A known protective order acknowledgement failed independent verification."""


TESTNET_BASE = "https://test.deribit.com/api/v2"
PROD_BASE    = "https://www.deribit.com/api/v2"

DEFAULT_LEVERAGE         = 2
MAX_SLIPPAGE_PCT         = 0.002
WIDE_SPREAD_WARN_PCT     = 0.003
MAX_TRADEABLE_SPREAD_PCT = 0.05

SYMBOL_MAP = {
    "ETHUSDT":   {"instrument": "ETH_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.0001, "max_amount": 5000,   "tick_size": 0.05},
    "BNBUSDT":   {"instrument": "BNB_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.001,  "max_amount": 1000,   "tick_size": 0.05},
    "SOLUSDT":   {"instrument": "SOL_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.001,  "max_amount": 50000,  "tick_size": 0.01},
    "XRPUSDT":   {"instrument": "XRP_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.0001},
    "NEARUSDT":  {"instrument": "NEAR_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.1,    "max_amount": 100000, "tick_size": 0.0001},
    "LTCUSDT":   {"instrument": "LTC_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.01,   "max_amount": 1000,   "tick_size": 0.01},
    "UNIUSDT":   {"instrument": "UNI_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.01,   "max_amount": 10000,  "tick_size": 0.001},
    "BCHUSDT":   {"instrument": "BCH_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.001,  "max_amount": 500,    "tick_size": 0.01},
    "DOTUSDT":   {"instrument": "DOT_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.1,    "max_amount": 10000,  "tick_size": 0.001},
    "ALGOUSDT":  {"instrument": "ALGO_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.0001},
    "ENAUSDT":   {"instrument": "ENA_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.0001},
    "DOGEUSDT":  {"instrument": "DOGE_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1.0,    "max_amount": 1000000,"tick_size": 0.00001},
    "TRUMPUSDT": {"instrument": "TRUMP_USDC-PERPETUAL", "currency": "USDC", "min_amount": 0.01,   "max_amount": 10000,  "tick_size": 0.001},
    "PUMPUSDT":  {"instrument": "PUMP_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1.0,    "max_amount": 5000000,"tick_size": 0.00001},
    "AAVEUSDT":  {"instrument": "AAVE_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.01,   "max_amount": 1000,   "tick_size": 0.01},
    "LINKUSDT":  {"instrument": "LINK_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.01,   "max_amount": 10000,  "tick_size": 0.001},
    "SUIUSDT":   {"instrument": "SUI_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.1,    "max_amount": 100000, "tick_size": 0.0001},
    "AVAXUSDT":  {"instrument": "AVAX_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.001,  "max_amount": 10000,  "tick_size": 0.001},
    "ADAUSDT":   {"instrument": "ADA_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.0001},
    "TRXUSDT":   {"instrument": "TRX_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.00001},
    "XLMUSDT":   {"instrument": "XLM_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.00001},
    "ZECUSDT":   {"instrument": "ZEC_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.001,  "max_amount": 500,    "tick_size": 0.01},
    "HBARUSDT":  {"instrument": "HBAR_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1.0,    "max_amount": 500000, "tick_size": 0.00001},
    "CRVUSDT":   {"instrument": "CRV_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1.0,    "max_amount": 100000, "tick_size": 0.0001},
    "FILUSDT":   {"instrument": "FIL_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 0.1,    "max_amount": 10000,  "tick_size": 0.001},
}

TRADEABLE_SYMBOLS: list = []


class PositionQueryError(RuntimeError):
    """Position data could not be verified as a complete exchange snapshot."""

    def __init__(self, message: str, *, failed_currencies: tuple[str, ...] = ()):
        self.failed_currencies = tuple(failed_currencies)
        super().__init__(message)


class OpenOrdersQueryError(RuntimeError):
    """Open orders could not be verified as a complete exchange snapshot."""


def _calc_decimals(val: float) -> int:
    if val <= 0:
        return 4
    s = f"{val:.10f}".rstrip("0")
    if "." in s:
        parts = s.split(".")
        return len(parts[1]) if len(parts) > 1 else 0
    return 0


class DeribitClient:

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        testnet: bool = True,
        *,
        auth_scope: str = "trade:read account:read",
        allow_mutations: bool = False,
        execution_gateway=None,
        require_execution_gateway: bool = False,
        session=None,
    ):
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "")
        if not self.client_id or not self.client_secret:
            raise ValueError("Deribit credentials are required.")

        self.base = TESTNET_BASE if testnet else PROD_BASE
        self.allow_mutations = bool(allow_mutations)
        if self.allow_mutations and not testnet:
            raise ValueError("Deribit production mutations are disabled in this release")
        if self.allow_mutations:
            validate_deribit_testnet_mutation_environment(
                expected_client_id=self.client_id,
                expected_client_secret=self.client_secret,
            )
        self.session = session or requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.auth_scope = str(auth_scope or "").strip()
        scope_items = self.auth_scope.split()
        expected_scopes = (
            {"trade:read_write", "account:read"}
            if self.allow_mutations
            else {"trade:read", "account:read"}
        )
        if set(scope_items) != expected_scopes or len(scope_items) != len(expected_scopes):
            raise ValueError(
                "Deribit auth_scope must be exactly the least-privilege "
                "trade/account scope required by this client"
            )

        self._auth_lock = threading.RLock()
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = 0.0
        self._rpc_id = 0
        self._read_breaker = CircuitBreaker(failure_threshold=3, reset_after_seconds=30.0)
        self._order_breaker = CircuitBreaker(failure_threshold=3, reset_after_seconds=30.0)
        self.require_execution_gateway = bool(require_execution_gateway)
        self.execution_gateway = execution_gateway
        self._gateway_dispatch_depth = 0
        if self.require_execution_gateway and not self.allow_mutations:
            raise ValueError("Execution gateway requirement is valid only for a Testnet write client")
        self._instrument_cache = {}
        self._supported_symbols = set()
        # Complete all read-only initialization before acquiring a mutation-owner lease.
        self._verify_instruments()
        if self.require_execution_gateway and self.execution_gateway is None:
            self.execution_gateway = create_deribit_execution_gateway(self.client_id, self._order_breaker)

    def _next_rpc_id(self) -> int:
        with self._auth_lock:
            self._rpc_id += 1
            return self._rpc_id

    def _rpc_transport(self, path: str, params: dict = None, *, access_token: str = None):
        """Send Deribit HTTP JSON-RPC calls using the documented transport.

        The client-credentials/refresh authentication examples use GET with
        query parameters. Method calls use JSON-RPC 2.0 over HTTP POST with a
        JSON body. Private method calls carry the OAuth Bearer token in the
        Authorization header. This distinction is important: using GET for
        private trading or account RPC calls is not the documented HTTP shape.
        """
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid Deribit API endpoint path.")
        if (
            path in self._mutating_methods()
            and getattr(self, "require_execution_gateway", False)
            and not getattr(self, "_gateway_dispatch_depth", 0)
        ):
            raise PermissionError("Direct Deribit transport mutation bypass rejected")
        method = path.lstrip("/")
        if not method or "?" in method or "#" in method:
            raise ValueError("Invalid Deribit HTTP method path")

        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        try:
            if path == "/public/auth":
                prepared = self.session.prepare_request(
                    requests.Request(
                        "GET",
                        f"{self.base}{path}",
                        params=dict(params or {}),
                        headers=headers,
                    )
                )
            else:
                body = {
                    "jsonrpc": "2.0",
                    "id": self._next_rpc_id(),
                    "method": method,
                    "params": dict(params or {}),
                }
                prepared = self.session.prepare_request(
                    requests.Request(
                        "POST",
                        f"{self.base}{path}",
                        json=body,
                        headers=headers,
                    )
                )
            response = self.session.send(prepared, timeout=15)
        except requests.RequestException as exc:
            # Never include prepared URLs or request bodies in transport errors.
            raise DeribitTransportError(
                f"Deribit HTTP transport failed ({type(exc).__name__})",
                unknown_outcome=path in self._mutating_methods(),
            ) from None

        status = int(getattr(response, "status_code", 0) or 0)
        response_headers = getattr(response, "headers", {}) or {}
        retry_after_seconds = None
        retry_after = response_headers.get("Retry-After") if hasattr(response_headers, "get") else None
        if retry_after is not None:
            try:
                retry_after_seconds = max(0.0, float(retry_after))
            except (TypeError, ValueError, OverflowError):
                try:
                    retry_at = parsedate_to_datetime(str(retry_after))
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    retry_after_seconds = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    retry_after_seconds = None

        # HTTP 429 and 5xx responses are transient transport/server outcomes.
        # For mutations, they are deliberately classified as UNKNOWN because
        # the exchange may have processed the request before the response failed.
        if status == 429 or status >= 500:
            raise DeribitTransportError(
                f"Deribit HTTP request failed (status={status}).",
                retry_after_seconds=retry_after_seconds,
                unknown_outcome=path in self._mutating_methods(),
            )

        try:
            data = response.json()
        except ValueError:
            raise DeribitTransportError(
                f"Deribit returned a non-JSON response (HTTP {status or 'unknown'}).",
                unknown_outcome=path in self._mutating_methods(),
            ) from None
        if not isinstance(data, dict):
            raise DeribitTransportError(
                "Deribit returned a malformed HTTP response",
                unknown_outcome=path in self._mutating_methods(),
            )
        error = data.get("error")
        if error is not None:
            code = error.get("code", "") if isinstance(error, dict) else ""
            message = error.get("message", "request failed") if isinstance(error, dict) else "request failed"
            raise DeribitRPCError(method, code, message)
        if status >= 400:
            raise DeribitRPCError(method, status, f"HTTP status {status}")
        if "result" not in data:
            raise DeribitTransportError(
                f"Deribit HTTP method {method} returned an incomplete response",
                unknown_outcome=path in self._mutating_methods(),
            )
        return data["result"]

    def _authenticate(self) -> str:
        """Obtain or refresh a scoped OAuth token for Deribit private methods."""
        with self._auth_lock:
            now = time.time()
            if self._access_token and now < self._token_expires_at - 30:
                return self._access_token

            auth_params = None
            if self._refresh_token:
                try:
                    auth_params = {
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                    }
                    result = self._rpc_transport("/public/auth", auth_params)
                except Exception:
                    # A revoked/rotated refresh token is recoverable through a
                    # fresh client_credentials grant. This is an auth read only.
                    self._access_token = None
                    self._refresh_token = None
                    auth_params = None
                else:
                    return self._store_auth_result(result)

            if auth_params is None:
                result = self._rpc_transport(
                    "/public/auth",
                    {
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "scope": self.auth_scope,
                    },
                )
                return self._store_auth_result(result)

            raise RuntimeError("Deribit authentication failed")

    def _store_auth_result(self, result: dict) -> str:
        if not isinstance(result, dict):
            raise RuntimeError("Deribit authentication response is malformed")
        token = result.get("access_token")
        granted_scope = result.get("scope")
        if not isinstance(granted_scope, str):
            raise RuntimeError("Deribit authentication returned no granted scope")
        granted = set(granted_scope.split())
        requested = set(self.auth_scope.split())
        if not requested.issubset(granted):
            raise RuntimeError("Deribit authentication granted fewer permissions than requested")

        allow_mutations = bool(getattr(self, "allow_mutations", False))

        # Functional permission families documented by Deribit.
        # Read-only: exactly account:read + trade:read.
        # Separately gated Testnet mutation: exactly account:read + trade:read_write.
        functional_prefixes = (
            "account:", "trade:", "wallet:", "block_trade:", "block_rfq:",
        )
        functional_scopes = {
            scope for scope in granted
            if scope.startswith(functional_prefixes)
        }
        expected_functional_scopes = (
            {"trade:read_write", "account:read"}
            if allow_mutations
            else {"trade:read", "account:read"}
        )

        write_scopes = {
            scope for scope in functional_scopes
            if scope.endswith(":read_write")
        }
        if not allow_mutations and write_scopes:
            raise RuntimeError(
                "Deribit read-only authentication unexpectedly granted write permissions"
            )

        if functional_scopes != expected_functional_scopes:
            raise RuntimeError(
                "Deribit authentication granted unexpected functional permissions"
            )

        # Permit known token/session metadata, not unknown permission tokens.
        token_metadata = granted - functional_scopes

        def known_token_metadata(scope):
            if scope in {"connection", "mainaccount"}:
                return True
            if scope.startswith("session:"):
                return bool(scope.partition(":")[2])
            if scope.startswith("expires:"):
                value = scope.partition(":")[2]
                return value.isdigit() and int(value) > 0
            if scope.startswith("ip:"):
                return bool(scope.partition(":")[2])
            return False

        if any(not known_token_metadata(scope) for scope in token_metadata):
            raise RuntimeError(
                "Deribit authentication returned unknown token scope metadata"
            )
        expires_in = result.get("expires_in")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Deribit authentication returned no access token")
        if isinstance(expires_in, bool):
            raise RuntimeError("Deribit authentication returned invalid token lifetime")
        try:
            ttl = float(expires_in)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("Deribit authentication returned invalid token lifetime") from None
        if not math.isfinite(ttl) or ttl <= 0:
            raise RuntimeError("Deribit authentication returned invalid token lifetime")
        self._access_token = token
        refresh_token = result.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            self._refresh_token = refresh_token
        self._token_expires_at = time.time() + ttl
        return token

    @staticmethod
    def _mutating_methods() -> set[str]:
        return {
            "/private/buy", "/private/sell", "/private/cancel",
            "/private/cancel_all", "/private/edit", "/private/set_leverage",
        }

    @staticmethod
    def _read_only_private_methods() -> set[str]:
        # Fail closed for any private method not explicitly reviewed here.
        # New Deribit endpoints must be classified before the client can call them.
        return {
            "/private/get_account_summary",
            "/private/get_account_summaries",
            "/private/get_transaction_log",
            "/private/get_positions",
            "/private/get_order_state",
            "/private/get_order_state_by_label",
            "/private/get_open_orders_by_label",
            "/private/get_open_orders_by_instrument",
            "/private/get_user_trades_by_instrument",
            "/private/get_order_history_by_instrument",
            "/private/get_user_trades_by_order",
            "/private/get_time",
        }

    def _get_read_breaker(self):
        breaker = getattr(self, "_read_breaker", None)
        if breaker is None:
            breaker = CircuitBreaker(failure_threshold=3, reset_after_seconds=30.0)
            self._read_breaker = breaker
        return breaker

    def _get_order_breaker(self):
        breaker = getattr(self, "_order_breaker", None)
        if breaker is None:
            breaker = CircuitBreaker(failure_threshold=3, reset_after_seconds=30.0)
            self._order_breaker = breaker
        return breaker

    @staticmethod
    def _is_read_retryable(exc: BaseException) -> bool:
        # Only explicit transport/server errors are retryable. Authentication,
        # malformed payloads, exchange rejections and local validation failures
        # propagate immediately instead of being disguised as retries.
        return isinstance(exc, DeribitTransportError)

    def acknowledge_order_path_failure(self, *, acknowledged_by: str) -> None:
        """Operator-only reset after an unknown mutation outcome is reconciled."""
        self._get_order_breaker().manual_reset(acknowledged_by=acknowledged_by)

    def _latch_unknown_order_outcome(self, reason: str) -> None:
        breaker = self._get_order_breaker()
        breaker.record_failure(order_path=True)
        log.critical("Deribit order path halted; read-only reconciliation required (%s)", reason)

    def _request(self, path: str, params: dict = None) -> dict:
        """Call a Deribit method endpoint; retry only explicitly safe reads."""
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid Deribit API endpoint path.")
        if path.startswith("/private/"):
            if path in self._mutating_methods():
                if getattr(self, "require_execution_gateway", False) and not getattr(self, "_gateway_dispatch_depth", 0):
                    raise PermissionError("Direct Deribit mutation RPC bypass rejected; use the execution gateway")
                if not getattr(self, "allow_mutations", False):
                    raise PermissionError("Deribit mutation RPC rejected by read-only client")
                if getattr(self, "base", None) != TESTNET_BASE:
                    raise PermissionError("Deribit mutations are restricted to Testnet")
            elif path not in self._read_only_private_methods():
                raise PermissionError(
                    "Deribit private method is not in the reviewed read/mutation allowlists"
                )
        access_token = self._authenticate() if path.startswith("/private/") else None
        if path in self._mutating_methods():
            # Never retry a mutation. A timeout/server error may occur after
            # the exchange has already accepted the order.
            return self._rpc_transport(path, params=params, access_token=access_token)

        breaker = self._get_read_breaker()
        if not breaker.allow_read():
            raise SnapshotUnavailable("Deribit read circuit breaker is open")
        try:
            result = retry_read(
                lambda: self._rpc_transport(path, params=params, access_token=access_token),
                attempts=3,
                deadline_seconds=8.0,
                retry_if=self._is_read_retryable,
            )
        except Exception:
            breaker.record_failure()
            raise
        breaker.record_success()
        return result

    def _get(self, path: str, params: dict = None) -> dict:
        # Prevent a mutating method from being smuggled through a read wrapper,
        # including when the client has Testnet write permission.
        if path in self._mutating_methods():
            raise PermissionError("Mutating Deribit method cannot be called through _get")
        return self._request(path, params=params)

    def _post(self, path: str, body: dict) -> dict:
        """Submit one mutation attempt; ambiguous outcomes permanently latch writes."""
        if path not in self._mutating_methods():
            raise PermissionError("Deribit method is not in the mutation allowlist")
        if not getattr(self, "allow_mutations", False):
            raise PermissionError(
                "Deribit mutations are disabled on this read-only client"
            )
        if getattr(self, "base", None) != TESTNET_BASE:
            raise PermissionError("Deribit mutations are restricted to Testnet")
        if not isinstance(body, dict):
            raise ValueError("Deribit mutation body must be an object")

        breaker = self._get_order_breaker()
        if breaker.manual_ack_required or breaker.state is BreakerState.HALTED:
            raise PermissionError("Deribit order path is HALTED pending operator acknowledgement")

        reduce_only = str(body.get("reduce_only", "false")).strip().lower() in {"true", "1"}
        if path in {"/private/cancel", "/private/cancel_all"} or reduce_only:
            allowed = breaker.allow_protective_actions()
        else:
            allowed = breaker.allow_new_entries()
        if not allowed:
            raise PermissionError("Deribit circuit breaker blocks this mutation")

        try:
            if getattr(self, "require_execution_gateway", False):
                gateway = getattr(self, "execution_gateway", None)
                if gateway is None:
                    raise PermissionError("Required Deribit execution gateway is unavailable")
                def _send_through_gateway():
                    self._gateway_dispatch_depth += 1
                    try:
                        return self._request(path, params=body)
                    finally:
                        self._gateway_dispatch_depth -= 1
                if path in {"/private/buy", "/private/sell"}:
                    result = gateway.submit_order(path, body, _send_through_gateway)
                else:
                    result = gateway.execute_control_mutation(path, body, _send_through_gateway)
            else:
                result = self._request(path, params=body)
        except ExecutionGatewayBlocked as exc:
            raise PermissionError(str(exc)) from None
        except ExecutionGatewayOutcomeUnknown as exc:
            raise OrderOutcomeUnknown(str(exc)) from None
        except DeribitTransportError:
            breaker.record_failure(order_path=True)
            raise OrderOutcomeUnknown(
                "Deribit mutation transport outcome is unknown; reconcile before any further mutation"
            ) from None
        except DeribitRPCError:
            # A structured JSON-RPC error is a confirmed exchange response,
            # unlike a timeout/5xx response. It is journaled by the gateway when required.
            breaker.record_failure(order_path=False)
            raise
        except OrderOutcomeUnknown:
            raise
        except PermissionError:
            raise
        except Exception as exc:
            breaker.record_failure(order_path=True)
            raise OrderOutcomeUnknown(
                f"Deribit mutation outcome is unknown ({type(exc).__name__}); reconcile before retry"
            ) from None
        breaker.record_success()
        return result

    def verify_instrument_contract(self, symbol: str) -> dict:
        """
        Validates linear USDC perpetual contract invariants against live exchange metadata:
        - Instrument exists and matches requested target symbol
        - Active status confirmed
        - Instrument type is explicitly 'linear' (or future/perpetual)
        - Explicit USDC settlement currency
        - Contract size is positive and finite
        - Order sizing units represent base currency coin amounts
        """
        name = self.get_instrument_name(symbol)
        info = self._get("/public/get_instrument", {"instrument_name": name})

        if not info:
            raise ValueError(f"CRITICAL: Instrument {name} not found on Deribit.")

        api_name = str(info.get("instrument_name", ""))
        if api_name != name:
            raise ValueError(f"CRITICAL: Target instrument mismatch. Config='{name}', Exchange='{api_name}'.")

        if not info.get("is_active", False):
            raise ValueError(f"CRITICAL: Instrument {name} is inactive on exchange.")
        instrument_state = str(info.get("state", "")).strip().lower()
        if instrument_state != "open":
            raise ValueError(
                f"CRITICAL: Instrument {name} state='{instrument_state}', expected 'open'."
            )

        # Strict instrument classification invariants. Do not fall back across
        # independent exchange metadata fields: each invariant must pass.
        inst_type = str(info.get("instrument_type", "")).lower()
        if inst_type != "linear":
            raise ValueError(
                f"CRITICAL: {name} instrument_type='{inst_type}', expected 'linear'. "
                f"Non-linear/reversed instruments are strictly prohibited."
            )

        kind = str(info.get("kind", "")).lower()
        if kind != "future":
            raise ValueError(f"CRITICAL: {name} kind='{kind}', expected 'future'.")

        settlement_period = str(info.get("settlement_period", "")).lower()
        if settlement_period != "perpetual":
            raise ValueError(
                f"CRITICAL: {name} settlement_period='{settlement_period}', expected 'perpetual'. "
                f"Expiring/dated futures are prohibited."
            )

        # Strict Settlement Currency Check
        settlement_ccy = str(info.get("settlement_currency", "")).upper()
        if settlement_ccy != "USDC":
            raise ValueError(f"CRITICAL: {name} settlement currency is '{settlement_ccy}', expected 'USDC'.")

        # Contract Multiplier Physical Positivity Check
        contract_size = float(info.get("contract_size", 0.0))
        if not math.isfinite(contract_size) or contract_size <= 0:
            raise ValueError(f"CRITICAL: {name} invalid contract_size={contract_size} <= 0.")

        min_amt = float(info.get("min_trade_amount", 0.0))
        tick = float(info.get("tick_size", 0.0))
        steps = info.get("tick_size_steps", [])
        if (
            not math.isfinite(min_amt)
            or not math.isfinite(tick)
            or min_amt <= 0
            or tick <= 0
        ):
            raise ValueError(
                f"CRITICAL: Invalid lot/tick specs for {name}: "
                f"min={min_amt}, tick={tick}"
            )

        base_ccy = str(info.get("base_currency", "")).upper()
        step_summary = [f"(>{s.get('above_price')}={s.get('tick_size')})" for s in steps]
        step_str = f" | steps: {', '.join(step_summary)}" if steps else ""
        log.info(
            f"  ✓ Verified {name}: settlement={settlement_ccy} | mult={contract_size} {base_ccy}/contract "
            f"| order_units={base_ccy} | min_lot={min_amt} | base_tick={tick}{step_str}"
        )
        return info

    def _verify_instruments(self):
        global TRADEABLE_SYMBOLS
        confirmed = []
        for sym in SYMBOL_MAP.keys():
            try:
                info = self.verify_instrument_contract(sym)
                target = info["instrument_name"]
                self._instrument_cache[target] = info
                self._supported_symbols.add(sym)
                confirmed.append(sym)
            except Exception as e:
                log.error(f"  ❌ Instrument verification failed for {sym}: {e}")

        TRADEABLE_SYMBOLS = confirmed
        if not confirmed:
            raise RuntimeError("HALT: Zero tradeable instruments passed linear USDC contract verification.")
        log.info(f"✓ Linear USDC instruments verified: {len(confirmed)}/{len(SYMBOL_MAP)}")

    def is_supported(self, symbol: str) -> bool:
        return symbol in self._supported_symbols

    def get_tradeable(self) -> list:
        return list(self._supported_symbols)

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not in SYMBOL_MAP")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            try:
                self._instrument_cache[name] = self._get("/public/get_instrument", {"instrument_name": name})
            except Exception:
                pass
        return self._instrument_cache.get(name, {})

    def get_tick_size(self, symbol: str, price: float = None) -> float:
        info = self.get_instrument_info(symbol)
        base_tick = float(info.get("tick_size") or SYMBOL_MAP.get(symbol, {}).get("tick_size", 0.0001))
        steps = info.get("tick_size_steps", [])

        if price is not None and steps:
            applicable = [s for s in steps if price >= float(s.get("above_price", 0.0))]
            if applicable:
                best_step = max(applicable, key=lambda s: float(s.get("above_price", 0.0)))
                return float(best_step.get("tick_size", base_tick))
        return base_tick

    def get_min_trade_amount(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        return float(api_min) if api_min else float(SYMBOL_MAP[symbol].get("min_amount", 1.0))

    def get_max_trade_amount(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        api_max = info.get("max_trade_amount") or info.get("max_amount")
        return float(api_max) if api_max else float(SYMBOL_MAP.get(symbol, {}).get("max_amount", float("inf")))

    def round_price(self, symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0
        tick = self.get_tick_size(symbol, price=price)
        if tick <= 0 or tick > price / 2:
            tick = float(SYMBOL_MAP.get(symbol, {}).get("tick_size", 0.0001))
        if tick <= 0:
            tick = 0.0001
        decimals = _calc_decimals(tick)
        steps    = round(price / tick)
        rounded  = round(steps * tick, decimals)
        if rounded <= 0 and price > 0:
            rounded = round(max(price, tick), decimals)
            if rounded <= 0:
                rounded = tick
        return rounded

    def round_amount(self, symbol: str, raw: float) -> float:
        if raw <= 0:
            return 0.0
        step     = self.get_min_trade_amount(symbol)
        steps    = math.floor(raw / step)
        result   = steps * step
        decimals = _calc_decimals(step)
        if result < step:
            return 0.0
        return round(result, decimals) if decimals else int(round(result))

    def split_amount(self, symbol: str, total: float) -> tuple:
        if total <= 0:
            return 0.0, 0.0
        step = self.get_min_trade_amount(symbol)
        half_qty = math.floor((total * 0.50) / step) * step

        if half_qty >= step and (total - half_qty) >= step:
            tp1 = half_qty
            tp2 = total - tp1
        else:
            tp1 = total
            tp2 = 0.0

        decimals = _calc_decimals(step)
        return (round(tp1, decimals), round(tp2, decimals)) if decimals else (int(round(tp1)), int(round(tp2)))

    def get_server_time_ms(self) -> int:
        """Return Deribit's public server clock in Unix milliseconds.

        The endpoint is public and does not require a trading credential. A
        malformed response is a failed read, never a zero timestamp.
        """
        result = self._get("/public/get_time")
        if isinstance(result, bool):
            raise SnapshotUnavailable("Deribit server-time response is not an integer")
        try:
            timestamp = int(result)
        except (TypeError, ValueError, OverflowError):
            raise SnapshotUnavailable("Deribit server-time response is not an integer") from None
        if timestamp <= 0 or str(result).strip() not in {str(timestamp), f"{timestamp}.0"}:
            raise SnapshotUnavailable("Deribit server-time response is invalid")
        return timestamp

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0.0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}")
            return 0.0

    def get_mark_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("index_price") or 0.0)
        except Exception as e:
            log.warning(f"  mark_price {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str):
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            rate = t.get("current_funding") if t.get("current_funding") is not None else t.get("funding_8h")
            if rate is None:
                return None
            return float(rate)
        except Exception as e:
            log.warning(f"  funding_rate {symbol}: {e}")
            return None

    def get_order_book_spread(self, symbol: str) -> dict:
        try:
            book = self._get("/public/get_order_book", {"instrument_name": self.get_instrument_name(symbol), "depth": 5})
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids or not asks:
                return {"best_bid": 0.0, "best_ask": 0.0, "spread_pct": 999.0, "is_wide": True}

            best_bid   = float(bids[0][0])
            best_ask   = float(asks[0][0])
            mid        = (best_bid + best_ask) / 2.0
            spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0.0

            is_wide = spread_pct > WIDE_SPREAD_WARN_PCT
            if is_wide:
                log.warning(f"  ⚠️ Wide spread on {symbol}: {spread_pct*100:.3f}% (bid={best_bid}, ask={best_ask})")
            return {"best_bid": best_bid, "best_ask": best_ask, "spread_pct": spread_pct, "is_wide": is_wide}
        except Exception as e:
            log.warning(f"  order book {symbol}: {e}")
            return {"best_bid": 0.0, "best_ask": 0.0, "spread_pct": 999.0, "is_wide": True}

    def calc_contracts(
        self,
        symbol: str,
        risk_equity: float,
        entry: float,
        stop: float,
        risk_per_trade: float,
        risk_mult: float = 1.0,
        available_funds: float = None,
    ):
        """Size a linear USDC contract using USDC-denominated risk capital.

        Until venue-specific margin/leverage is independently verified, the
        available-funds cap is conservative 1x notional, not assumed leverage.
        """
        try:
            equity = float(risk_equity)
            available = None if available_funds is None else float(available_funds)
            entry_value = float(entry)
            stop_value = float(stop)
            risk_fraction = float(risk_per_trade)
            multiplier = float(risk_mult)
        except (TypeError, ValueError, OverflowError):
            return 0.0

        if not all(math.isfinite(v) for v in (equity, entry_value, stop_value, risk_fraction, multiplier)):
            return 0.0
        if equity <= 0 or entry_value <= 0 or risk_fraction <= 0 or multiplier <= 0:
            return 0.0
        if available is not None and (not math.isfinite(available) or available <= 0):
            return 0.0

        risk_budget_usdc = equity * risk_fraction * multiplier
        stop_distance = abs(entry_value - stop_value)
        if stop_distance <= 0 or risk_budget_usdc <= 0:
            return 0.0

        # Amount is in base-asset units for the verified linear USDC perps.
        raw = risk_budget_usdc / stop_distance
        max_notional_qty = (equity * 0.05) / entry_value
        max_risk_notional_qty = (risk_budget_usdc * 10.0) / entry_value
        raw = min(raw, max_notional_qty, max_risk_notional_qty)

        if available is not None:
            # Do not assume account leverage if the exchange has not verified
            # it for this instrument/account. One USDC available funds backs
            # at most one USDC of notional in this conservative release.
            raw = min(raw, available / entry_value)

        max_amount = self.get_max_trade_amount(symbol)
        if raw > max_amount:
            log.info("  Position capped at exchange ceiling: %s %s", max_amount, symbol)
            raw = max_amount

        result = self.round_amount(symbol, raw)
        notional_usdc = result * entry_value
        log.info(
            "  Amount: %s %s | notional≈%.2f USDC | risk budget=%.2f USDC",
            result, symbol, notional_usdc, risk_budget_usdc,
        )
        return result

    @staticmethod
    def _is_position_size_limit_error(e) -> bool:
        return "10057" in str(e) or "non_pme_max_future_position_size" in str(e)

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        *,
        client_label: str = None,
    ) -> dict:
        """Place one reduce-only stop-market and independently verify it.

        There is intentionally no fallback stop submission. If the exchange
        accepted the order but its state cannot be read, the caller receives a
        verification error and must reconcile/cancel/flatten from live exchange
        state rather than sending a second stop with a new identity.
        """
        instrument = self.get_instrument_name(symbol)
        side = str(side or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Stop-loss side must be BUY or SELL")
        try:
            amount = float(amount)
            stop_price = float(stop_price)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Stop-loss amount and trigger must be finite") from None
        if not math.isfinite(amount) or amount <= 0 or not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError("Stop-loss amount and trigger must be finite and positive")
        safe_trigger = self.round_price(symbol, stop_price)
        label = str(client_label or f"bot_sl_{uuid.uuid4().hex[:24]}").strip()
        if not label or len(label) > 64 or any(ord(ch) < 33 or ord(ch) > 126 for ch in label):
            raise ValueError("Order label must be 1-64 printable non-space ASCII characters")

        method = "/private/buy" if side == "BUY" else "/private/sell"
        body = {
            "instrument_name": instrument,
            "amount": amount,
            "type": "stop_market",
            "trigger_price": safe_trigger,
            "trigger": "mark_price",
            "reduce_only": True,
            "label": label,
        }
        result = self._post(method, body)
        order_res = result.get("order", result) if isinstance(result, dict) else None
        order_id = str(order_res.get("order_id", "")).strip() if isinstance(order_res, dict) else ""
        if not order_id:
            self._latch_unknown_order_outcome("stop-market acknowledgement has no order id")
            raise OrderOutcomeUnknown(
                f"Stop-loss submission for {symbol} returned no exchange order ID; do not resubmit"
            )

        try:
            state = self.get_order(order_id)
        except Exception:
            raise ProtectiveOrderVerificationError(
                f"Stop-loss {order_id} was acknowledged but its exchange state could not be verified"
            ) from None
        if str(state.get("order_id", "")) != order_id:
            raise ProtectiveOrderVerificationError("Stop-loss state identity mismatch")
        if str(state.get("instrument_name", "")) != instrument:
            raise ProtectiveOrderVerificationError("Stop-loss instrument identity mismatch")
        if str(state.get("order_type", "")).lower() != "stop_market":
            raise ProtectiveOrderVerificationError("Exchange did not confirm stop_market type")
        if str(state.get("direction", "")).lower() != side.lower():
            raise ProtectiveOrderVerificationError("Stop-loss direction mismatch")
        if str(state.get("trigger", "")).lower() != "mark_price":
            raise ProtectiveOrderVerificationError("Exchange did not confirm mark_price trigger")
        if state.get("reduce_only") is not True:
            raise ProtectiveOrderVerificationError("Exchange did not confirm reduce_only=True")
        try:
            confirmed_trigger = float(state.get("trigger_price"))
            confirmed_amount = float(state.get("amount"))
        except (TypeError, ValueError, OverflowError):
            raise ProtectiveOrderVerificationError("Stop-loss trigger or amount is missing") from None
        if not math.isfinite(confirmed_trigger) or not math.isclose(
            confirmed_trigger, safe_trigger, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ProtectiveOrderVerificationError("Exchange stop trigger does not match requested trigger")
        if not math.isfinite(confirmed_amount) or not math.isclose(
            confirmed_amount, amount, rel_tol=1e-8, abs_tol=max(1e-12, amount * 1e-8)
        ):
            raise ProtectiveOrderVerificationError("Exchange stop amount does not match confirmed position quantity")
        state_name = str(state.get("order_state", "")).lower()
        if state_name not in {"open", "untriggered"}:
            raise ProtectiveOrderVerificationError(
                f"Stop-loss is not resting and verified (state={state_name or 'missing'})"
            )

        log.info(
            "  ✓ Exchange verified stop-market %s [%s]: %s @ %s (reduce_only=True)",
            order_id, symbol, amount, safe_trigger,
        )
        return result

    @staticmethod
    def _validated_mutation_order(result: dict, requested_amount: float, operation: str) -> dict:
        """Require a complete, terminal exchange acknowledgement."""
        if not isinstance(result, dict):
            raise OrderOutcomeUnknown(f"{operation} returned a malformed response")
        order = result.get("order", result)
        if not isinstance(order, dict):
            raise OrderOutcomeUnknown(f"{operation} returned no order object")
        order_id = str(order.get("order_id", "")).strip()
        state = str(order.get("order_state", "")).strip().lower()
        if not order_id or state not in {"filled", "cancelled", "canceled", "rejected", "closed", "expired"}:
            raise OrderOutcomeUnknown(
                f"{operation} acknowledgement is missing an ID or terminal state"
            )
        raw_filled = order.get("filled_amount")
        if raw_filled is None or isinstance(raw_filled, bool):
            raise OrderOutcomeUnknown(f"{operation} acknowledgement has no filled_amount")
        try:
            filled = float(raw_filled)
            requested = float(requested_amount)
        except (TypeError, ValueError, OverflowError):
            raise OrderOutcomeUnknown(f"{operation} acknowledgement has invalid quantity") from None
        if not math.isfinite(filled) or not math.isfinite(requested) or requested <= 0:
            raise OrderOutcomeUnknown(f"{operation} acknowledgement has non-finite quantity")
        if filled < 0 or filled > requested + max(1e-12, requested * 1e-8):
            raise OrderOutcomeUnknown(f"{operation} fill quantity is outside requested bounds")
        return order

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
        *,
        client_label: str = None,
    ) -> dict:
        """Submit a single slippage-bounded IOC limit order.

        This method deliberately does not retry, halve, or convert a failed IOC
        into a raw market order. Any transport ambiguity is surfaced to the
        caller as UNKNOWN. A separately verified emergency market reduction is
        available only through place_reduce_only_market_order().
        """
        instrument = self.get_instrument_name(symbol)
        side = str(side or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order side must be BUY or SELL")
        try:
            amount = float(amount)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Order amount must be finite and positive") from None
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Order amount must be finite and positive")

        method = "/private/buy" if side == "BUY" else "/private/sell"
        label = str(client_label or f"bot_{uuid.uuid4().hex[:24]}").strip()
        if not label or len(label) > 64 or any(ord(ch) < 33 or ord(ch) > 126 for ch in label):
            raise ValueError("Order label must be 1-64 printable non-space ASCII characters")

        spread = self.get_order_book_spread(symbol)
        if not isinstance(spread, dict):
            return {}
        try:
            best_bid = float(spread.get("best_bid", 0.0))
            best_ask = float(spread.get("best_ask", 0.0))
            spread_pct = float(spread.get("spread_pct", 999.0))
        except (TypeError, ValueError, OverflowError):
            return {}
        effective_ceiling = 0.50 if reduce_only else MAX_TRADEABLE_SPREAD_PCT
        if (
            not all(math.isfinite(v) for v in (best_bid, best_ask, spread_pct))
            or best_bid <= 0 or best_ask <= 0 or spread_pct < 0
            or spread_pct > effective_ceiling
        ):
            log.warning("  Entry/exit IOC blocked for %s: invalid or excessive spread", symbol)
            return {}

        if side == "BUY":
            worst_price = self.round_price(symbol, best_ask * (1 + MAX_SLIPPAGE_PCT))
        else:
            worst_price = self.round_price(symbol, best_bid * (1 - MAX_SLIPPAGE_PCT))
        if not math.isfinite(float(worst_price)) or worst_price <= 0:
            return {}

        body = {
            "instrument_name": instrument,
            "amount": amount,
            "type": "limit",
            "price": worst_price,
            "time_in_force": "immediate_or_cancel",
            "label": label,
            "reduce_only": bool(reduce_only),
        }
        try:
            result = self._post(method, body)
        except DeribitRPCError as exc:
            # The exchange explicitly rejected this request. It is not retried.
            log.warning("  IOC order rejected for %s (code=%s)", symbol, exc.code)
            return {}
        except OrderOutcomeUnknown:
            raise
        except Exception as exc:
            # Any unclassified post-send error is conservatively UNKNOWN.
            self._latch_unknown_order_outcome(type(exc).__name__)
            raise OrderOutcomeUnknown(
                "Deribit IOC outcome unknown; reconcile by label/order/position before any retry"
            ) from None

        try:
            order = self._validated_mutation_order(result, amount, "Deribit IOC")
        except OrderOutcomeUnknown as exc:
            self._latch_unknown_order_outcome(str(exc))
            raise
        log.info(
            "  IOC %s %s %s id=%s state=%s filled=%s",
            side, amount, instrument, order["order_id"], order["order_state"], order["filled_amount"],
        )
        return result

    def place_reduce_only_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        client_label: str = None,
    ) -> dict:
        """Submit one reduce-only market order after caller verifies live exposure.

        No retries are permitted. A timeout or malformed acknowledgement latches
        the order path and requires read-only reconciliation plus manual reset.
        """
        instrument = self.get_instrument_name(symbol)
        side = str(side or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order side must be BUY or SELL")
        try:
            amount = float(amount)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Reduce-only market amount must be finite and positive") from None
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Reduce-only market amount must be finite and positive")
        label = str(client_label or f"bot_flat_{uuid.uuid4().hex[:20]}").strip()
        if not label or len(label) > 64 or any(ord(ch) < 33 or ord(ch) > 126 for ch in label):
            raise ValueError("Order label must be 1-64 printable non-space ASCII characters")
        method = "/private/buy" if side == "BUY" else "/private/sell"
        result = self._post(method, {
            "instrument_name": instrument,
            "amount": amount,
            "type": "market",
            "label": label,
            "reduce_only": True,
        })
        try:
            self._validated_mutation_order(result, amount, "Deribit reduce-only market order")
        except OrderOutcomeUnknown as exc:
            self._latch_unknown_order_outcome(str(exc))
            raise
        return result

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float,
        stop_price: float = None, use_reduce_only: bool = False,
        *, client_label: str = None,
    ) -> dict:
        instrument = self.get_instrument_name(symbol)
        side = str(side or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Limit order side must be BUY or SELL")
        try:
            amount = float(amount)
            price = float(price)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Limit order amount and price must be finite") from None
        if not math.isfinite(amount) or amount <= 0 or not math.isfinite(price) or price <= 0:
            raise ValueError("Limit order amount and price must be finite and positive")
        method     = "/private/buy" if side == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)
        label = str(client_label or f"bot_{uuid.uuid4().hex[:24]}").strip()
        if not label or len(label) > 64 or any(ord(ch) < 33 or ord(ch) > 126 for ch in label):
            raise ValueError("Order label must be 1-64 printable non-space ASCII characters")

        if stop_price is not None:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "stop_limit",
                "price":           safe_price,
                "trigger_price":   self.round_price(symbol, stop_price),
                "trigger":         "mark_price",
                "label":           label,
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "label":           label,
            }

        if use_reduce_only:
            body["reduce_only"] = True

        result = self._post(method, body)
        order = result.get("order", result) if isinstance(result, dict) else None
        order_id = str(order.get("order_id", "")).strip() if isinstance(order, dict) else ""
        if not order_id:
            self._latch_unknown_order_outcome("limit/TP acknowledgement has no order id")
            raise OrderOutcomeUnknown(
                "Deribit limit/stop-limit acknowledgement omitted order ID; do not resubmit"
            )
        kind = "SL" if stop_price is not None else "TP"
        log.info(
            "  ✅ %s %s %s %s @ trigger:%s id=%s state=%s",
            kind, side, amount, instrument, stop_price,
            order_id, str(order.get("order_state", "")),
        )
        return result

    def get_fill_price(self, market_result: dict, fallback: float = 0.0) -> float:
        try:
            trades = market_result.get("trades", [])
            if trades:
                total_cost = sum(float(t["price"]) * float(t["amount"]) for t in trades)
                total_qty  = sum(float(t["amount"]) for t in trades)
                return round(total_cost / total_qty, 8) if total_qty else fallback
            order = market_result.get("order", {})
            avg   = order.get("average_price") or order.get("price")
            return float(avg) if avg else fallback
        except Exception:
            return fallback

    def get_order_fill_price(self, order: dict, fallback: float = 0.0) -> float:
        """Return authoritative average/last fill from an exchange order-state response."""
        try:
            avg = order.get("average_price")
            if avg is not None and float(avg) > 0:
                return float(avg)
            lp = order.get("last_price") or order.get("price")
            if lp is not None and float(lp) > 0:
                return float(lp)
        except (TypeError, ValueError):
            pass
        return fallback

    def position_size_from_record(self, position: dict) -> float:
        """Return signed base quantity for bot-managed linear USDC perpetuals."""
        if not isinstance(position, dict):
            raise PositionQueryError("Position row is not an object")

        instrument = position.get("instrument_name")
        if not isinstance(instrument, str) or not instrument.strip():
            raise PositionQueryError("Position row has no valid instrument name")
        if not instrument.endswith("_USDC-PERPETUAL"):
            raise PositionQueryError(
                f"Unsupported instrument for linear USDC position normalization: {instrument}"
            )

        raw_size = position.get("size")
        if raw_size is None or isinstance(raw_size, bool):
            raise PositionQueryError(
                f"Position size is missing or invalid for {instrument}"
            )
        try:
            size = float(raw_size)
        except (TypeError, ValueError) as exc:
            raise PositionQueryError(
                f"Position size is invalid for {instrument}"
            ) from exc
        if not math.isfinite(size):
            raise PositionQueryError(
                f"Position size is non-finite for {instrument}"
            )
        if size < 0:
            raise PositionQueryError(
                f"Position size must be a non-negative magnitude for {instrument}"
            )

        # Deribit reports size_currency as a numeric base-currency quantity,
        # not a currency-code string. Use that native quantity for linear orders.
        raw_base_size = position.get("size_currency")
        if raw_base_size is None or isinstance(raw_base_size, bool):
            raise PositionQueryError(
                f"Base position quantity is missing or invalid for {instrument}"
            )
        try:
            base_size = float(raw_base_size)
        except (TypeError, ValueError) as exc:
            raise PositionQueryError(
                f"Base position quantity is invalid for {instrument}"
            ) from exc
        if not math.isfinite(base_size):
            raise PositionQueryError(
                f"Base position quantity is non-finite for {instrument}"
            )
        if base_size < 0:
            raise PositionQueryError(
                f"Base position quantity must be non-negative for {instrument}"
            )

        direction = position.get("direction")
        size = abs(size)
        base_size = abs(base_size)

        if size == 0 and base_size == 0 and direction == "zero":
            return 0.0

        if size == 0 or base_size == 0:
            raise PositionQueryError(
                f"Position size and base quantity are inconsistent for {instrument}"
            )

        if direction not in ("buy", "sell"):
            raise PositionQueryError(
                f"Position direction is missing or invalid for {instrument}"
            )

        return base_size if direction == "buy" else -base_size

    def get_position_size(self, symbol: str) -> float:
        """Return a verified position size; raise when the exchange read is uncertain.

        Zero is returned only after get_positions() has verified every configured
        settlement-currency query successfully and no matching position exists.
        """
        instrument = self.get_instrument_name(symbol)
        if not instrument or not isinstance(instrument, str):
            raise PositionQueryError(
                f"Cannot resolve a Deribit instrument for symbol {symbol!r}"
            )

        matches = [
            position
            for position in self.get_positions()
            if position.get("instrument_name") == instrument
        ]
        if not matches:
            return 0.0
        if len(matches) != 1:
            raise PositionQueryError(
                f"Position snapshot contains duplicate rows for {instrument}"
            )
        return self.position_size_from_record(matches[0])

    def get_account_summary(self, currency: str) -> dict:
        '''Strict private account-summary read used to verify authenticated access.'''
        currency = str(currency or "").strip().upper()
        if currency not in {"USDC", "USDT", "BTC", "ETH"}:
            raise ValueError("Unsupported account-summary currency")
        result = self._get(
            "/private/get_account_summary",
            {"currency": currency, "extended": "true"},
        )
        if not isinstance(result, dict):
            raise ValueError("Malformed private account-summary response")
        return result

    def get_account_summaries(self) -> dict:
        '''Return Deribit's complete per-currency account summary snapshot.'''
        result = self._get("/private/get_account_summaries", {"extended": "true"})
        if not isinstance(result, dict) or not isinstance(result.get("summaries"), list):
            raise ValueError("Malformed private account-summaries response")
        summaries = result["summaries"]
        if any(not isinstance(item, dict) for item in summaries):
            raise ValueError("Malformed account summary row")
        currencies = [str(item.get("currency", "")).upper() for item in summaries]
        if not summaries or any(not currency for currency in currencies) or len(currencies) != len(set(currencies)):
            raise ValueError("Account summaries contain missing or duplicate currencies")
        return result

    def get_transaction_log(
        self,
        currency: str,
        *,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
        continuation: int | None = None,
    ) -> dict:
        """Read one Deribit account transaction-log page (GET/read-only)."""
        currency = str(currency or "").strip().upper()
        if currency not in {"USDC", "USDT", "BTC", "ETH"}:
            raise ValueError("Unsupported transaction-log currency")
        start_timestamp = int(start_timestamp)
        end_timestamp = int(end_timestamp)
        count = int(count)
        if start_timestamp < 0 or end_timestamp < start_timestamp:
            raise ValueError("Invalid transaction-log time range")
        if not 1 <= count <= 1000:
            raise ValueError("Transaction-log count must be 1..1000")
        params = {
            "currency": currency,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "count": count,
        }
        if continuation is not None:
            params["continuation"] = int(continuation)
        result = self._get("/private/get_transaction_log", params)
        if not isinstance(result, dict) or not isinstance(result.get("logs"), list):
            raise ValueError("Malformed Deribit transaction-log response")
        if any(not isinstance(row, dict) for row in result["logs"]):
            raise ValueError("Malformed Deribit transaction-log row")
        return result

    def get_risk_capital_snapshot(self) -> dict:
        """Return one validated, currency-explicit account snapshot.

        This bot's configured Deribit instruments settle in USDC. Segregated
        accounts may use only USDC equity and available funds for new-entry
        sizing. Cross-collateral accounts are read successfully but new entries
        remain disabled until a verified USDC allocation/FX model exists.
        """
        response = self.get_account_summaries()
        rows = response["summaries"]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Account summaries are empty or malformed")

        seen = set()
        margin_models = set()
        assets = []
        usd_values = []
        usdc_row = None

        def finite_number(value, field, currency, *, required=True):
            if value is None and not required:
                return None
            if value is None or isinstance(value, bool):
                raise ValueError(f"Missing or invalid {field} for {currency}")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"Invalid {field} for {currency}") from None
            if not math.isfinite(number):
                raise ValueError(f"Non-finite {field} for {currency}")
            return number

        segregated_models = {"segregated_sm", "segregated_pm"}
        cross_models = {"cross_sm", "cross_pm"}

        for row in rows:
            currency = str(row.get("currency", "")).strip().upper()
            if not currency or currency in seen:
                raise ValueError("Account summaries have missing or duplicate currencies")
            seen.add(currency)

            # The active margin model is authoritative. Deribit examples can
            # report cross_collateral_enabled=true while margin_model is
            # segregated_sm; the boolean alone must not classify the account.
            margin_model = str(row.get("margin_model", "")).strip().lower()
            if margin_model not in segregated_models | cross_models:
                raise ValueError(f"Unknown or missing margin_model for {currency}")
            margin_models.add(margin_model)

            equity = finite_number(row.get("equity"), "equity", currency)
            available = finite_number(
                row.get("available_funds"), "available_funds", currency,
                required=False,
            )
            assets.append({
                "asset": currency,
                "equity": equity,
                "available": available,
                "unit": currency,
            })
            if currency == "USDC":
                usdc_row = assets[-1]

            total_usd = finite_number(
                row.get("total_equity_usd"), "total_equity_usd", currency,
                required=False,
            )
            if margin_model in cross_models and total_usd is not None:
                usd_values.append(total_usd)

        if len(margin_models) != 1:
            raise ValueError("Account summaries disagree on active margin_model")

        margin_model = next(iter(margin_models))
        cross_collateral = margin_model in cross_models
        account_equity_usd = None
        if cross_collateral:
            if not usd_values:
                raise ValueError("Cross-collateral account lacks total_equity_usd")
            account_equity_usd = usd_values[0]
            if any(
                not math.isclose(value, account_equity_usd, rel_tol=1e-9, abs_tol=1e-7)
                for value in usd_values[1:]
            ):
                raise ValueError("Inconsistent account-wide USD equity values")
            return {
                "entry_enabled": False,
                "risk_equity": None,
                "risk_currency": "USDC",
                "available_funds": None,
                "available_funds_currency": None,
                "margin_mode": "cross_collateral",
                "margin_model": margin_model,
                "account_equity_usd": account_equity_usd,
                "assets": assets,
                "source": "private/get_account_summaries",
                "risk_block_reason": "CROSS_COLLATERAL_USDC_ALLOCATION_NOT_VERIFIED",
            }

        if usdc_row is None:
            raise ValueError("Segregated account has no USDC summary")
        risk_equity = usdc_row["equity"]
        available_funds = usdc_row["available"]
        if risk_equity <= 0 or available_funds is None or available_funds <= 0:
            raise ValueError("USDC equity or available funds are not positive")

        return {
            "entry_enabled": True,
            "risk_equity": risk_equity,
            "risk_currency": "USDC",
            "available_funds": available_funds,
            "available_funds_currency": "USDC",
            "margin_mode": "segregated",
            "margin_model": margin_model,
            "account_equity_usd": None,
            "assets": assets,
            "source": "private/get_account_summaries",
            "risk_block_reason": None,
        }

    def get_all_balances(self) -> dict:
        """Compatibility read returning native per-currency units, never USD aliases."""
        snapshot = self.get_risk_capital_snapshot()
        return {
            row["asset"]: {
                "equity": round(row["equity"], 8),
                "available": (
                    round(row["available"], 8)
                    if row["available"] is not None else None
                ),
                "unit": row["unit"],
            }
            for row in snapshot["assets"]
        }

    def get_total_equity_usd(self) -> float:
        """Return USD only when the exchange supplies account-wide USD equity."""
        snapshot = self.get_risk_capital_snapshot()
        value = snapshot.get("account_equity_usd")
        if value is None or snapshot["margin_mode"] != "cross_collateral":
            raise RuntimeError(
                "Account-wide USD equity is unavailable; native currency equity must not be relabelled USD"
            )
        return round(float(value), 2)


    def get_positions(self) -> list:
        """Return a complete, verified position snapshot or raise PositionQueryError.

        A partial response must never be interpreted as a flat account. Any
        failed currency query or malformed row invalidates the whole snapshot.
        """
        positions = []
        seen_instruments: set[str] = set()
        failures: dict[str, Exception] = {}
        for cur in ("USDC", "BTC", "ETH"):
            try:
                response = self._get(
                    "/private/get_positions",
                    {"currency": cur, "kind": "future"},
                )
                if not isinstance(response, list):
                    raise ValueError("unexpected response type")

                currency_positions = []
                for position in response:
                    if not isinstance(position, dict):
                        raise ValueError("position row is not an object")

                    instrument = position.get("instrument_name")
                    if not isinstance(instrument, str) or not instrument.strip():
                        raise ValueError("position instrument name is invalid")
                    if instrument in seen_instruments:
                        raise ValueError("duplicate position instrument")
                    seen_instruments.add(instrument)

                    raw_size = position.get("size")
                    if raw_size is None or isinstance(raw_size, bool):
                        raise ValueError("position size is missing or invalid")
                    try:
                        size = float(raw_size)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("position size is invalid") from exc
                    if not math.isfinite(size):
                        raise ValueError("position size is non-finite")
                    if size < 0:
                        raise ValueError("position size must be non-negative")

                    direction = position.get("direction")
                    if direction not in ("buy", "sell", "zero"):
                        raise ValueError("position direction is missing or invalid")

                    raw_base_size = position.get("size_currency")
                    if size == 0 and direction == "zero" and raw_base_size is None:
                        continue
                    if raw_base_size is None or isinstance(raw_base_size, bool):
                        raise ValueError("position base quantity is missing or invalid")
                    try:
                        base_size = float(raw_base_size)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("position base quantity is invalid") from exc
                    if not math.isfinite(base_size):
                        raise ValueError("position base quantity is non-finite")
                    if base_size < 0:
                        raise ValueError("position base quantity must be non-negative")

                    if size == 0 and base_size == 0 and direction == "zero":
                        continue
                    if size == 0 or base_size == 0 or direction == "zero":
                        raise ValueError(
                            "position size, base quantity, and direction are inconsistent"
                        )

                    currency_positions.append(position)
                positions.extend(currency_positions)
            except Exception as exc:
                # Do not include raw exception text; HTTP/API errors may contain
                # request parameters. Preserve only currency and exception type.
                failures[cur] = exc
                log.warning(
                    "Deribit position query failed for currency %s (%s); "
                    "complete snapshot unavailable",
                    cur,
                    type(exc).__name__,
                )

        if failures:
            failed = tuple(failures)
            message = (
                "Incomplete Deribit position snapshot; failed currencies: "
                + ", ".join(failed)
            )
            raise PositionQueryError(message, failed_currencies=failed) from None

        return positions

    def get_order(self, order_id: str) -> dict:
        try:
            result = self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as exc:
            message = str(exc).lower()
            # Deribit documents order-not-found codes across API generations.
            # Only explicit order-not-found responses are terminal; network,
            # auth and generic lookup failures remain unknown.
            is_order_not_found = (
                "order_not_found" in message
                or "11000" in message
                or "10004" in message
            )
            if is_order_not_found:
                return {"order_state": "not_found", "order_id": str(order_id)}
            log.warning("Deribit order-state read unavailable for %s (%s)", order_id, type(exc).__name__)
            raise RuntimeError("Deribit order-state snapshot unavailable") from None
        if (
            not isinstance(result, dict)
            or not str(result.get("order_state", "")).strip()
            or str(result.get("order_id", "")) != str(order_id)
        ):
            raise RuntimeError("Deribit order-state response malformed or identity mismatch")
        gateway = getattr(self, "execution_gateway", None)
        if getattr(self, "require_execution_gateway", False) and gateway is not None:
            try:
                gateway.observe_order_state(result)
            except Exception as exc:
                raise RuntimeError(
                    f"Durable execution ledger could not absorb order-state evidence ({type(exc).__name__})"
                ) from None
        return result

    def confirm_execution_protection(
        self, *, entry_client_order_id: str, stop_client_order_id: str,
        protective_order_id: str, protected_quantity, evidence=None,
    ) -> dict:
        if not getattr(self, "require_execution_gateway", False) or self.execution_gateway is None:
            raise PermissionError("Execution-gateway protection accounting is not enabled")
        return self.execution_gateway.confirm_protection(
            entry_client_order_id=entry_client_order_id,
            stop_client_order_id=stop_client_order_id,
            protective_order_id=protective_order_id,
            protected_quantity=protected_quantity,
            evidence=evidence,
        )

    def reconcile_execution_order_label(self, client_order_id: str, rows: list, *, queries_complete: bool) -> dict | None:
        if not getattr(self, "require_execution_gateway", False) or self.execution_gateway is None:
            return None
        return self.execution_gateway.reconcile_order_label(
            client_order_id, rows, queries_complete=queries_complete,
        )

    def close_execution_gateway(self) -> None:
        gateway = getattr(self, "execution_gateway", None)
        if gateway is not None:
            gateway.close()

    def is_order_filled(self, order: dict) -> bool:
        state       = order.get("order_state", "").lower()
        filled_amt  = float(order.get("filled_amount", 0) or 0)
        avg_price   = float(order.get("average_price", 0) or 0)
        if state == "filled":
            return True
        if state in ("cancelled", "closed") and (filled_amt > 0 or avg_price > 0):
            return True
        return False

    def is_sl_triggered(self, order: dict) -> bool:
        state      = order.get("order_state", "").lower()
        filled_amt = float(order.get("filled_amount", 0) or 0)
        avg_price  = float(order.get("average_price", 0) or 0)
        if state in ("filled", "triggered"):
            return True
        if state == "cancelled" and filled_amt > 0 and avg_price > 0:
            return True
        return False

    def cancel_order(self, order_id: str) -> dict:
        try:
            result = self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as exc:
            err_msg = str(exc).lower()
            if "11044" in err_msg or "not_open_order" in err_msg:
                return {"result": "already_closed", "order_id": str(order_id)}
            log.warning("Deribit cancel outcome unknown for %s (%s)", order_id, type(exc).__name__)
            raise RuntimeError("Deribit cancel outcome unknown; reconcile before retry") from None
        if not isinstance(result, dict):
            raise RuntimeError("Deribit cancel response malformed; reconcile before retry")
        return result

    @staticmethod
    def _validate_label_query(currency: str, label: str) -> tuple[str, str]:
        currency = str(currency or "").strip().upper()
        label = str(label or "").strip()
        if currency not in {"BTC", "ETH", "USDC", "USDT", "EURR"}:
            raise ValueError("Unsupported Deribit label-query currency")
        if not label or len(label) > 64 or any(ord(ch) < 33 or ord(ch) > 126 for ch in label):
            raise ValueError("Order label must be 1-64 printable non-space ASCII characters")
        return currency, label

    def get_order_state_by_label(self, currency: str, label: str) -> list:
        """Return recent order states matching an exact Deribit label.

        Deribit only exposes recently closed orders in this lookup for a
        30-minute window. An empty array is evidence of no recent match, not
        proof that an earlier submission was rejected or never accepted.
        """
        currency, label = self._validate_label_query(currency, label)
        result = self._get(
            "/private/get_order_state_by_label",
            {"currency": currency, "label": label},
        )
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise ValueError("Malformed Deribit order-state-by-label response")
        for row in result:
            if str(row.get("label", "")) != label:
                raise ValueError("Deribit order-state-by-label returned a label mismatch")
        return result

    def get_open_orders_by_label(self, currency: str, label: str) -> list:
        """Return open orders matching a label; empty means no open match only."""
        currency, label = self._validate_label_query(currency, label)
        result = self._get(
            "/private/get_open_orders_by_label",
            {"currency": currency, "label": label},
        )
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise ValueError("Malformed Deribit open-orders-by-label response")
        for row in result:
            if str(row.get("label", "")) != label:
                raise ValueError("Deribit open-orders-by-label returned a label mismatch")
        return result

    def get_open_orders_by_instrument(self, instrument_name: str) -> list:
        '''Read open orders using the exact exchange instrument identifier.'''
        instrument = str(instrument_name or "").strip()
        if not instrument:
            raise OpenOrdersQueryError("Exact instrument_name is required")
        try:
            result = self._get(
                "/private/get_open_orders_by_instrument",
                {"instrument_name": instrument},
            )
        except Exception as exc:
            log.warning("  open_orders %s unavailable (%s)", instrument, type(exc).__name__)
            raise OpenOrdersQueryError(
                f"Open-order snapshot unavailable for {instrument}"
            ) from None
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise OpenOrdersQueryError(
                f"Open-order snapshot malformed for {instrument}"
            )
        return result

    def get_open_orders(self, symbol: str) -> list:
        '''Backward-compatible logical-symbol wrapper for trading code.'''
        try:
            instrument = self.get_instrument_name(symbol)
        except Exception as exc:
            raise OpenOrdersQueryError(
                f"Cannot resolve an exchange instrument for {symbol!r}"
            ) from exc
        return self.get_open_orders_by_instrument(instrument)

    def get_trade_history_by_instrument_name(
        self,
        instrument_name: str,
        count: int = 10,
        *,
        end_seq: int = None,
        historical: bool = False,
        strict: bool = True,
    ) -> list:
        '''Read exchange fills by exact instrument_name; strict mode propagates failures.'''
        instrument = str(instrument_name or "").strip()
        if not instrument:
            raise ValueError("Exact instrument_name is required")
        if isinstance(count, bool) or not 1 <= int(count) <= 1000:
            raise ValueError("count must be between 1 and 1000")
        params = {"instrument_name": instrument, "count": int(count), "sorting": "desc"}
        if end_seq is not None:
            if isinstance(end_seq, bool) or int(end_seq) < 0:
                raise ValueError("end_seq must be a non-negative integer")
            params["end_seq"] = int(end_seq)
        if historical:
            params["historical"] = True
        try:
            result = self._get("/private/get_user_trades_by_instrument", params)
            if isinstance(result, list):
                trades = result
            elif isinstance(result, dict) and isinstance(result.get("trades"), list):
                trades = result["trades"]
            else:
                raise ValueError(f"Malformed trade-history response for {instrument}")
            if any(not isinstance(item, dict) for item in trades):
                raise ValueError(f"Malformed trade row for {instrument}")
            return trades
        except Exception as exc:
            log.warning("  Trade history %s unavailable (%s)", instrument, type(exc).__name__)
            if strict:
                raise
            return []

    def get_order_history_by_instrument_name(
        self,
        instrument_name: str,
        count: int = 1000,
        *,
        historical: bool = False,
        include_old: bool = True,
        include_unfilled: bool = True,
        offset: int = 0,
        strict: bool = True,
    ) -> list:
        """Read exact-instrument order history, including Deribit historical records.

        `historical=True` queries the indexed historical store and does not
        include the recent-order window. Callers must combine it with recent
        order/position/trade reads and must not interpret an empty result as
        proof that an ambiguous submission was rejected.
        """
        instrument = str(instrument_name or "").strip()
        if not instrument:
            raise ValueError("Exact instrument_name is required")
        if isinstance(count, bool) or not 1 <= int(count) <= 1000:
            raise ValueError("count must be between 1 and 1000")
        if isinstance(offset, bool) or int(offset) < 0:
            raise ValueError("offset must be non-negative")
        params = {
            "instrument_name": instrument,
            "count": int(count),
            "offset": int(offset),
            "historical": bool(historical),
            "include_old": bool(include_old),
            "include_unfilled": bool(include_unfilled),
        }
        try:
            result = self._get("/private/get_order_history_by_instrument", params)
            if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
                raise ValueError("Malformed Deribit order-history response")
            return result
        except Exception:
            if strict:
                raise
            return []

    def get_user_trades_by_order(
        self, order_id: str, *, historical: bool = False, strict: bool = True
    ) -> list:
        """Read all fills associated with one exact order identifier."""
        order_id = str(order_id or "").strip()
        if not order_id:
            raise ValueError("order_id is required")
        try:
            result = self._get(
                "/private/get_user_trades_by_order",
                {"order_id": order_id, "historical": bool(historical), "sorting": "desc"},
            )
            if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
                raise ValueError("Malformed Deribit trades-by-order response")
            return result
        except Exception:
            if strict:
                raise
            return []

    def get_trade_history_for_instrument(
        self, symbol: str, count: int = 10, *, strict: bool = True
    ) -> list:
        '''Backward-compatible logical-symbol wrapper for trading code.'''
        try:
            instrument = self.get_instrument_name(symbol)
        except Exception as exc:
            log.warning("  Trade history %s cannot be mapped (%s)", symbol, type(exc).__name__)
            if strict:
                raise
            return []
        return self.get_trade_history_by_instrument_name(instrument, count=count, strict=strict)

    def set_leverage(self, symbol: str, leverage: int = DEFAULT_LEVERAGE) -> bool:
        try:
            instrument = self.get_instrument_name(symbol)
            result = self._post("/private/set_leverage", {"instrument_name": instrument, "leverage": leverage})
            actual = result.get("leverage", leverage) if isinstance(result, dict) else leverage
            log.info(f"  ⚡ Leverage set to {actual}x — {instrument}")
            return True
        except Exception as e:
            log.warning(f"  set_leverage {symbol}: {e} — proceeding at default margin")
            return False

    def test_connection(self) -> bool:
        try:
            snapshot = self.get_risk_capital_snapshot()
            if snapshot["entry_enabled"]:
                log.info(
                    "Deribit OK — risk equity %.2f %s | available %.2f %s | %s symbols verified",
                    snapshot["risk_equity"], snapshot["risk_currency"],
                    snapshot["available_funds"], snapshot["available_funds_currency"],
                    len(TRADEABLE_SYMBOLS),
                )
            else:
                log.warning(
                    "Deribit authenticated; new entries disabled: %s | account USD equity=%s",
                    snapshot["risk_block_reason"], snapshot["account_equity_usd"],
                )
            return True
        except Exception as exc:
            log.error("Deribit connection/account verification failed: %s", type(exc).__name__)
            raise
