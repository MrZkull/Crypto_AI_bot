# deribit_client.py — ALL BUGS FIXED
# FIX 1: _post() now sends real HTTP POST (was calling _get() silently)
# FIX 2: stop_price field correct (was "trigger_price" — wrong)
# FIX 3: extended="true" string (was Python bool True)
# FIX 4: is_order_filled() checks "order_state" (not "status")
# FIX 5: get_fill_price() reads trades[] array for actual fill price

import time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT": {"instrument": "BTC-PERPETUAL", "currency": "BTC"},
    "ETHUSDT": {"instrument": "ETH-PERPETUAL", "currency": "ETH"},
}
TRADEABLE = list(SYMBOL_MAP.keys())


class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.base          = TESTNET_BASE
        self.session       = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "CryptoBotAI/1.0"})
        self._access_token = None
        self._token_expiry = 0
        self._authenticate()

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = data.get("result")
        if not result:
            raise Exception(f"Auth failed: {data}")
        self._access_token = result["access_token"]
        self._token_expiry = time.time() + int(result.get("expires_in", 900)) - 60
        self.session.headers["Authorization"] = f"Bearer {self._access_token}"
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error GET {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        # FIX 1: Real HTTP POST with JSON body (was silently calling _get)
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error POST {path}: {data['error']}")
        return data.get("result", data)

    def get_instrument(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not on Deribit testnet. Supported: {TRADEABLE}")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_currency(self, symbol: str) -> str:
        return SYMBOL_MAP[symbol]["currency"]

    def is_supported(self, symbol: str) -> bool:
        return symbol in SYMBOL_MAP

    def get_balance(self, currency: str = "BTC") -> dict:
        # FIX 3: "true" string not Python bool True
        return self._get("/private/get_account_summary", {"currency": currency, "extended": "true"})

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH"]:
            try:
                s = self.get_balance(currency)
                eq = float(s.get("equity_usd") or s.get("equity", 0) or 0)
                balances[currency] = {
                    "equity_usd": round(eq, 2),
                    "currency":   currency,
                    "available":  float(s.get("available_funds", 0) or 0),
                }
            except Exception as e:
                log.warning(f"  Balance {currency}: {e}")
        return balances

    def get_usdt_equivalent(self) -> float:
        return round(sum(v.get("equity_usd", 0) for v in self.get_all_balances().values()), 2)

    def place_market_order(self, symbol: str, side: str, amount_usd: float) -> dict:
        instrument = self.get_instrument(symbol)
        contracts  = max(10, round(amount_usd / 10) * 10)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result = self._post(method, {
            "instrument_name": instrument,
            "amount":          contracts,
            "type":            "market",
            "label":           f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} ${contracts} {instrument} id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result  # return full result so get_fill_price can read trades[]

    def get_fill_price(self, order_result: dict, fallback: float) -> float:
        # FIX 5: fill price is in trades[], not in order dict
        trades = order_result.get("trades", [])
        if trades:
            prices = [float(t.get("price", 0)) for t in trades if t.get("price")]
            if prices:
                return round(sum(prices) / len(prices), 2)
        order = order_result.get("order", order_result)
        avg = order.get("average_price") or order.get("price")
        if avg and float(avg) > 0:
            return float(avg)
        return fallback

    def place_limit_order(self, symbol: str, side: str, amount_usd: float,
                          price: float, stop_price: float = None) -> dict:
        instrument = self.get_instrument(symbol)
        contracts  = max(10, round(amount_usd / 10) * 10)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"

        if stop_price is not None:
            # FIX 2: correct field is "stop_price" + "trigger", NOT "trigger_price"
            body = {
                "instrument_name": instrument,
                "amount":          contracts,
                "type":            "stop_limit",
                "price":           round(price, 2),
                "stop_price":      round(stop_price, 2),
                "trigger":         "last_price",
                "reduce_only":     True,
                "label":           f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          contracts,
                "type":            "limit",
                "price":           round(price, 2),
                "reduce_only":     True,
                "label":           f"bot_tp_{int(time.time())}",
            }

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  ✅ {kind} {side.upper()} ${contracts} {instrument} @ {price} id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
        # FIX 4: Deribit uses "order_state" not "status"
        return order.get("order_state") == "filled"

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  cancel_order {order_id}: {e}")
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            if symbol and symbol in SYMBOL_MAP:
                result = self._get("/private/get_open_orders_by_instrument",
                                   {"instrument_name": self.get_instrument(symbol)})
            else:
                result = []
                for cur in ["BTC", "ETH"]:
                    r = self._get("/private/get_open_orders_by_currency", {"currency": cur})
                    result.extend(r if isinstance(r, list) else [])
            return result if isinstance(result, list) else []
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}")
            return 0.0

    def calc_usd_amount(self, balance_usd: float, entry: float, stop: float, risk_mult: float = 1.0) -> float:
        risk_usd  = balance_usd * 0.01 * risk_mult
        stop_dist = abs(entry - stop) / entry
        if stop_dist <= 0: return 10.0
        amount = min(risk_usd / stop_dist, balance_usd * 0.20)
        return max(10.0, round(amount / 10) * 10)

    def test_connection(self) -> bool:
        try:
            total = self.get_usdt_equivalent()
            log.info(f"✅ Deribit Testnet — Portfolio: ~${total:.2f} USD")
            return True
        except Exception as e:
            log.error(f"✗ Deribit FAILED: {e}")
            return False
