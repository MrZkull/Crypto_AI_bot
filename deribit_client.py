# deribit_client.py — All 10 bugs fixed
# FIX 1: _post() sends real HTTP POST (was calling _get() silently)
# FIX 2: stop_price field correct — was "trigger_price", Deribit needs "stop_price"
# FIX 3: get_fill_price() reads trades[] array for actual fill price
# FIX 4: is_order_filled() — removed "cancelled_with_fill" (not a real Deribit state)
# FIX 5: place_limit_order() returns full result so caller gets correct order dict
# FIX 6: _verify_instruments() checks live exchange, only confirmed symbols tradeable
# FIX 7: calc_contracts uses RISK_PER_TRADE from config (was hardcoded 0.02)
# FIX 8: order_id extracted from result["order"] after real POST
# FIX 9: reduce_only=True on all SL/TP to prevent accidental new positions
# FIX 10: BNBUSDT removed — BNB_USDC-PERPETUAL does not exist on Deribit

import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",       "currency": "BTC",  "kind": "inverse", "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",       "currency": "ETH",  "kind": "inverse", "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
}

TRADEABLE_SYMBOLS: list = []


class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "CryptoBotAI/2.0"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()
        self._authenticate()
        self._verify_instruments()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=15)
        r.raise_for_status()
        res = r.json().get("result", {})
        if not res or "access_token" not in res:
            raise Exception(f"Auth failed: {r.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
        self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    # ── HTTP ──────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error GET {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        # FIX 1: Real HTTP POST with JSON body. Never call _get() here.
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error POST {path}: {data['error']}")
        return data.get("result", data)

    # ── Instrument verification (FIX 6) ──────────────────────────────

    def _verify_instruments(self):
        """Check each symbol against the live exchange on startup."""
        global TRADEABLE_SYMBOLS
        confirmed = []
        for sym, info in SYMBOL_MAP.items():
            try:
                result = self._get("/public/get_instrument", {"instrument_name": info["instrument"]})
                self._instrument_cache[info["instrument"]] = result
                self._supported_symbols.add(sym)
                confirmed.append(sym)
            except Exception:
                log.debug(f"  {sym} ({info['instrument']}) not available — skipped")
        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} symbols — {confirmed}")

    # ── Symbol helpers ────────────────────────────────────────────────

    def is_supported(self, symbol: str) -> bool:
        return symbol in self._supported_symbols

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not in SYMBOL_MAP")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self._get("/public/get_instrument", {"instrument_name": name})
        return self._instrument_cache[name]

    def get_tick_size(self, symbol: str) -> float:
        return float(self.get_instrument_info(symbol).get("tick_size", 0.5))

    def get_min_trade_amount(self, symbol: str) -> float:
        return float(self.get_instrument_info(symbol).get("min_trade_amount", SYMBOL_MAP[symbol]["min_amount"]))

    def round_price(self, symbol: str, price: float) -> float:
        if price <= 0: return 0.0
        tick = self.get_tick_size(symbol)
        rounded = round(round(price / tick) * tick, 10)
        if float(tick).is_integer(): return float(int(rounded))
        dec = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, dec)

    def round_amount(self, symbol: str, amount: float) -> float:
        if amount <= 0: return 0.0
        step    = self.get_min_trade_amount(symbol)
        rounded = max(step, math.floor(amount / step) * step)
        return int(rounded) if float(step).is_integer() else round(rounded, 6)

    # ── Position sizing (FIX 7) ───────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float,
                       stop: float, risk_mult: float = 1.0) -> float:
        # FIX 7: import RISK_PER_TRADE from config, not hardcoded 0.02
        try:
            from config import RISK_PER_TRADE as rpt
        except ImportError:
            rpt = 0.01
        risk_usd  = balance_usd * rpt * risk_mult
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)
        if stop_dist <= 0: return self.round_amount(symbol, min_amt)

        kind = SYMBOL_MAP[symbol]["kind"]
        if kind == "inverse":
            amount = risk_usd / (stop_dist / entry)
        else:
            amount = min(risk_usd / stop_dist, balance_usd * 0.20 / entry)

        final = self.round_amount(symbol, amount)
        log.info(f"  Contracts: {final} {symbol} (risk=${risk_usd:.2f}, {kind})")
        return final

    # ── Balance ───────────────────────────────────────────────────────

    def get_account_summary(self, currency: str) -> dict:
        return self._get("/private/get_account_summary",
                         {"currency": currency, "extended": "true"})

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self.get_account_summary(currency)
                eq = float(s.get("equity_usd") or s.get("equity") or 0)
                av = float(s.get("available_funds", 0) or 0)
                if eq > 0:
                    balances[currency] = {"equity_usd": round(eq, 2), "available": round(av, 6)}
            except Exception as e:
                log.debug(f"  Balance {currency}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        return round(sum(v["equity_usd"] for v in self.get_all_balances().values()), 2)

    def get_positions(self) -> list:
        try:
            positions = []
            for currency in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions", {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(p for p in r if float(p.get("size", 0) or 0) != 0)
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Returns full result dict: {"order": {...}, "trades": [...]}"""
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result = self._post(method, {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "market",
            "label":           f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} {amount} {instrument} "
                 f"id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
        """FIX 3: Actual fill price lives in result['trades'][0]['price']."""
        trades = market_result.get("trades", [])
        if trades:
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if prices:
                return round(sum(prices) / len(prices), 8)
        order = market_result.get("order", market_result)
        avg   = order.get("average_price") or order.get("price")
        if avg and float(avg) > 0:
            return float(avg)
        return fallback

    def place_limit_order(self, symbol: str, side: str, amount: float,
                          price: float, stop_price: float = None) -> dict:
        """
        FIX 2: Uses "stop_price" not "trigger_price".
        FIX 5: Returns full result dict.
        FIX 9: reduce_only=True on all orders.
        """
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)

        if stop_price is not None:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "stop_limit",
                "price":           safe_price,
                "stop_price":      self.round_price(symbol, stop_price),  # FIX 2
                "trigger":         "last_price",
                "reduce_only":     True,                                   # FIX 9
                "label":           f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "reduce_only":     True,                                   # FIX 9
                "label":           f"bot_tp_{int(time.time())}",
            }

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ {safe_price} "
                 f"id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result  # FIX 5: full result, not just order dict

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
        # FIX 4: Only "filled" is valid. "cancelled_with_fill" does NOT exist on Deribit.
        return order.get("order_state", "").lower() == "filled"

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  cancel {order_id}: {e}")
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            if symbol and symbol in SYMBOL_MAP:
                r = self._get("/private/get_open_orders_by_instrument",
                              {"instrument_name": self.get_instrument_name(symbol)})
                return r if isinstance(r, list) else []
            orders = []
            for cur in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_open_orders_by_currency", {"currency": cur})
                orders.extend(r if isinstance(r, list) else [])
            return orders
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}")
            return 0.0

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet — ${total:.2f} USD | {len(TRADEABLE_SYMBOLS)} symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit FAILED: {e}")
            raise
