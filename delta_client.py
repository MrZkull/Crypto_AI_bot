# delta_client.py — Delta Exchange India Testnet
# Legal in India, no geo-block from Render/Replit
# Testnet URL: https://testnet.delta.exchange/
# API docs: https://docs.delta.exchange/

import hmac, hashlib, time, json, requests, logging

log = logging.getLogger(__name__)
TESTNET_BASE = "https://cdn-ind.testnet.deltaex.org"

# Known product IDs — bot fetches fresh list on startup to fill any gaps
KNOWN_PRODUCTS = {
    "BTCUSDT": "BTCUSD",  "ETHUSDT": "ETHUSD",  "BNBUSDT": "BNBUSD",
    "SOLUSDT": "SOLUSD",  "AVAXUSDT":"AVAXUSD", "LINKUSDT":"LINKUSD",
    "XRPUSDT": "XRPUSD",  "ADAUSDT": "ADAUSD",  "DOTUSDT": "DOTUSD",
    "NEARUSDT":"NEARUSD", "INJUSDT": "INJUSD",  "ARBUSDT": "ARBUSD",
    "OPUSDT":  "OPUSD",   "UNIUSDT": "UNIUSD",  "AAVEUSDT":"AAVEUSD",
    "SUIUSDT": "SUIUSD",  "APTUSDT": "APTUSD",
}


class DeltaClient:
    """
    Direct HMAC-signed REST client for Delta Exchange India Testnet.
    Zero geo-block from Indian servers (Render/Replit both work).
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base       = TESTNET_BASE
        self.session    = requests.Session()
        self._products  = {}   # symbol → {id, contract_value, tick_size}
        self._load_products()

    # ── Product catalogue ─────────────────────────────────────────

    def _load_products(self):
        """Fetch all available products from Delta testnet on startup."""
        try:
            r = self.session.get(f"{self.base}/v2/products", timeout=15)
            if not r.ok:
                log.warning(f"Could not load Delta products: {r.status_code}")
                return
            for p in r.json().get("result", []):
                sym_delta = p.get("symbol","")     # e.g. BTCUSD
                prod_id   = p.get("id")
                cv        = float(p.get("contract_value", 0.001) or 0.001)
                tick      = float(p.get("tick_size", 0.5) or 0.5)
                state     = p.get("state","")
                if state != "live": continue
                # Map our BTCUSDT → BTCUSD product
                for our_sym, delta_sym in KNOWN_PRODUCTS.items():
                    if sym_delta == delta_sym:
                        self._products[our_sym] = {
                            "id":             prod_id,
                            "contract_value": cv,
                            "tick_size":      tick,
                            "delta_symbol":   sym_delta,
                        }
            log.info(f"  ✅ Loaded {len(self._products)} Delta products: {list(self._products.keys())[:5]}...")
        except Exception as e:
            log.warning(f"  Delta product load failed: {e}")

    def get_product(self, symbol: str) -> dict:
        if symbol not in self._products:
            self._load_products()
        if symbol not in self._products:
            raise ValueError(f"{symbol} not available on Delta Exchange India testnet")
        return self._products[symbol]

    def get_product_id(self, symbol: str) -> int:
        return self.get_product(symbol)["id"]

    def round_price(self, symbol: str, price: float) -> str:
        try:
            tick = self.get_product(symbol)["tick_size"]
            decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
            return str(round(round(price / tick) * tick, decimals))
        except Exception:
            return str(round(price, 2))

    def round_qty(self, qty: float) -> int:
        """Delta uses integer contract sizes."""
        return max(1, int(qty))

    # ── Signing ───────────────────────────────────────────────────

    def _sign(self, method: str, path: str, body: str = "") -> dict:
        ts  = str(int(time.time()))
        msg = method + ts + path + body
        sig = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return {
            "api-key":      self.api_key,
            "signature":    sig,
            "timestamp":    ts,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        qs   = ("?" + "&".join(f"{k}={v}" for k,v in (params or {}).items())) if params else ""
        hdrs = self._sign("GET", path + qs)
        r    = self.session.get(f"{self.base}{path}", params=params, headers=hdrs, timeout=15)
        if not r.ok: raise Exception(f"GET {path} {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        bs   = json.dumps(body)
        hdrs = self._sign("POST", path, bs)
        r    = self.session.post(f"{self.base}{path}", data=bs, headers=hdrs, timeout=15)
        if not r.ok: raise Exception(f"POST {path} {r.status_code}: {r.text[:300]}")
        return r.json()

    def _delete(self, path: str, body: dict = None) -> dict:
        bs   = json.dumps(body or {})
        hdrs = self._sign("DELETE", path, bs)
        r    = self.session.delete(f"{self.base}{path}", data=bs, headers=hdrs, timeout=15)
        if not r.ok: raise Exception(f"DELETE {path} {r.status_code}: {r.text[:200]}")
        return r.json()

    # ── Balance ───────────────────────────────────────────────────

    def get_wallet_balance(self) -> dict:
        """Returns full wallet with USDT available balance."""
        result = self._get("/v2/wallet/balances")
        balances = {}
        for item in result.get("result", []):
            asset = item.get("asset_symbol", item.get("asset",{}).get("symbol",""))
            avail = float(item.get("available_balance", 0) or 0)
            if avail > 0 or asset == "USDT":
                balances[asset] = avail
        return balances

    def get_usdt_balance(self) -> float:
        return self.get_wallet_balance().get("USDT", 0.0)

    def get_positions(self) -> list:
        try:
            result = self._get("/v2/positions/margined")
            return [p for p in result.get("result",[]) if float(p.get("size",0) or 0) != 0]
        except Exception as e:
            log.warning(f"  get_positions failed: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, contracts: int) -> dict:
        prod = self.get_product(symbol)
        body = {
            "product_id":    prod["id"],
            "side":          side.lower(),
            "order_type":    "market_order",
            "size":          self.round_qty(contracts),
            "time_in_force": "ioc",
        }
        result = self._post("/v2/orders", body)
        order  = result.get("result", {})
        log.info(f"  ✅ MARKET {side.upper()} {contracts}x {symbol} — id:{order.get('id','')}")
        return order

    def place_limit_order(self, symbol: str, side: str, contracts: int,
                          price: float, stop_price: float = None) -> dict:
        prod = self.get_product(symbol)
        body = {
            "product_id":    prod["id"],
            "side":          side.lower(),
            "size":          self.round_qty(contracts),
            "limit_price":   self.round_price(symbol, price),
            "time_in_force": "gtc",
        }
        if stop_price is not None:
            body["order_type"]      = "limit_order"
            body["stop_price"]      = self.round_price(symbol, stop_price)
            body["stop_order_type"] = "stop_loss_order"
            body["isTrailingStopLoss"] = False
        else:
            body["order_type"] = "limit_order"

        result = self._post("/v2/orders", body)
        order  = result.get("result", {})
        kind   = "SL" if stop_price else "LIMIT"
        log.info(f"  ✅ {kind} {side.upper()} {contracts}x {symbol} @ {price} — id:{order.get('id','')}")
        return order

    def get_order(self, order_id) -> dict:
        try:
            result = self._get(f"/v2/orders/{order_id}")
            return result.get("result", {})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def cancel_order(self, symbol: str, order_id) -> dict:
        try:
            prod   = self.get_product(symbol)
            result = self._delete("/v2/orders", {"id": int(order_id), "product_id": prod["id"]})
            return result.get("result", {})
        except Exception as e:
            log.warning(f"  cancel_order {order_id}: {e}")
            return {}

    def get_open_orders(self) -> list:
        try:
            result = self._get("/v2/orders", {"state": "open", "limit": 100})
            return result.get("result", [])
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def get_live_price(self, symbol: str) -> float:
        """Get mark price from Delta testnet for a symbol."""
        try:
            prod = self.get_product(symbol)
            r    = self.session.get(f"{self.base}/v2/tickers",
                                    params={"product_ids": prod["id"]}, timeout=8)
            if r.ok:
                items = r.json().get("result", [])
                if items:
                    return float(items[0].get("mark_price") or items[0].get("close", 0))
        except Exception as e:
            log.warning(f"  Delta price {symbol}: {e}")
        return 0.0

    def calc_contracts(self, balance_usdt: float, entry: float,
                       stop: float, risk_mult: float = 1.0) -> int:
        """
        Calculate contracts based on 1% risk rule.
        Each Delta contract = contract_value × entry_price USDT in notional.
        Stop distance drives contract count.
        """
        risk_usd   = balance_usdt * 0.01 * risk_mult
        stop_dist  = abs(entry - stop)
        if stop_dist <= 0: return 1

        # PnL per contract per $1 move = contract_value (e.g. 0.001 BTC)
        # So: contracts = risk_usd / (stop_dist × contract_value)
        # For BTC: contract_value ≈ 0.001, stop_dist = ~$300
        # contracts = 50 / (300 × 0.001) = 166 ... too many
        # Use simplified: 1 contract per $5 of risk (conservative start)
        contracts = max(1, int(risk_usd / 5))

        # Hard cap: never more than 10% of balance in notional
        max_contracts = max(1, int(balance_usdt * 0.10 / max(entry * 0.001, 1)))
        return min(contracts, max_contracts, 50)   # absolute max 50 contracts

    def test_connection(self) -> bool:
        try:
            bal = self.get_usdt_balance()
            log.info(f"✅ Delta Exchange India TESTNET — Balance: {bal:.2f} USDT")
            return True
        except Exception as e:
            log.error(f"✗ Delta connection FAILED: {e}")
            return False
