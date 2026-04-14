# deribit_client.py — Margin-aware trading + import binding fix
#
# ROOT CAUSE OF bad_request 11050 (ALL trades failing):
#   Doc 61 switched to _USDC-PERPETUAL instruments for all symbols.
#   These require USDC margin. The testnet account has BTC/ETH margin
#   but ZERO USDC — so every single trade fails immediately.
#
# FIX: Smart margin routing — trade each instrument using the margin
#   currency it actually requires. No USDC needed for BTC/ETH trades.
#   BTC-PERPETUAL  → BTC margin  (you already have this ✅)
#   ETH-PERPETUAL  → ETH margin  (you already have this ✅)
#   *_USDC-PERPETUAL → USDC margin (only if USDC balance > 0)
#
# IMPORT BINDING FIX:
#   'from deribit_client import TRADEABLE_SYMBOLS' captures [] at import time.
#   Fix: trade_executor uses deribit_client.TRADEABLE_SYMBOLS (module reference)
#   OR reads from deribit instance. Solved by exposing get_tradeable().
#
# ALL PREVIOUS FIXES RETAINED:
#   _post() sends real HTTP POST
#   Integer amounts for inverse contracts
#   Tick-rounded prices
#   reduce_only=True on SL/TP
#   _verify_instruments() on startup

import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# Two-tier instrument map:
# Tier 1 (inverse): BTC/ETH — use existing BTC/ETH margin, always work
# Tier 2 (linear USDC): everything else — only used if USDC balance available
SYMBOL_MAP = {
    # ── Tier 1: Inverse perpetuals — BTC/ETH margin (always available) ──
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",       "currency": "BTC",  "kind": "inverse", "contract_usd": 10, "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",       "currency": "ETH",  "kind": "inverse", "contract_usd": 1,  "min_amount": 1},
    # ── Tier 2: Linear USDC perpetuals — requires USDC margin ──
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
}

# Module-level list — use deribit_client.TRADEABLE_SYMBOLS (not 'from X import')
TRADEABLE_SYMBOLS: list = []


class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "CryptoBotAI/4.0"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()   # symbols with instrument + margin
        self._usdc_balance      = 0.0     # cached USDC balance
        self._authenticate()
        self._check_margins()
        self._verify_instruments()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type": "client_credentials",
            "client_id": self.client_id, "client_secret": self.client_secret,
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

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error GET {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        # Real HTTP POST with plain JSON body (NOT JSON-RPC envelope)
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error POST {path}: {data['error']}")
        return data.get("result", data)

    # ── Margin check ──────────────────────────────────────────────────

    def _check_margins(self):
        """
        Check which margin currencies have balance.
        USDC perps only enabled if USDC account has funds.
        This prevents bad_request 11050 from insufficient margin.
        """
        try:
            usdc = self._get("/private/get_account_summary",
                             {"currency": "USDC", "extended": "true"})
            self._usdc_balance = float(usdc.get("available_funds", 0) or 0)
            if self._usdc_balance > 10:
                log.info(f"✓ USDC margin: ${self._usdc_balance:.2f} — linear perps enabled")
            else:
                log.warning(
                    f"⚠️ USDC balance ${self._usdc_balance:.2f} — linear (USDC) perps disabled.\n"
                    f"   To enable all 19 coins: go to test.deribit.com → Deposit → "
                    f"select USDC → click 'Deposit' to get testnet USDC funds."
                )
        except Exception as e:
            log.warning(f"  USDC margin check failed: {e} — linear perps disabled")
            self._usdc_balance = 0.0

    def has_usdc_margin(self) -> bool:
        return self._usdc_balance > 10.0

    # ── Instrument verification ───────────────────────────────────────

    def _verify_instruments(self):
        """Check each symbol against live exchange AND available margin."""
        global TRADEABLE_SYMBOLS
        # Bulk load all instruments
        active = {}
        for currency in ["USDC", "BTC", "ETH"]:
            try:
                res = self._get("/public/get_instruments",
                                {"currency": currency, "kind": "future", "expired": "false"})
                if isinstance(res, list):
                    for inst in res:
                        name = inst.get("instrument_name")
                        if name:
                            active[name] = inst
            except Exception as e:
                log.warning(f"  Instrument list {currency}: {e}")

        confirmed = []
        skipped_margin = []
        for sym, info in SYMBOL_MAP.items():
            target = info["instrument"]
            if target not in active:
                log.debug(f"  {sym} ({target}) not on testnet — skip")
                continue
            # Check margin availability
            if info["currency"] == "USDC" and not self.has_usdc_margin():
                skipped_margin.append(sym)
                continue
            self._instrument_cache[target] = active[target]
            self._supported_symbols.add(sym)
            confirmed.append(sym)

        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} — {confirmed}")
        if skipped_margin:
            log.warning(
                f"  Skipped {len(skipped_margin)} USDC perps (no USDC margin): "
                f"{skipped_margin[:5]}...\n"
                f"  → Deposit testnet USDC at test.deribit.com to enable these."
            )

    def get_tradeable(self) -> list:
        """Use this instead of importing TRADEABLE_SYMBOLS — avoids import binding issue."""
        return list(self._supported_symbols)

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
            self._instrument_cache[name] = self._get("/public/get_instrument",
                                                      {"instrument_name": name})
        return self._instrument_cache[name]

    def get_tick_size(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("tick_size") or SYMBOL_MAP[symbol].get("tick_size", 0.5))

    def get_min_trade_amount(self, symbol: str) -> float:
        info    = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        return float(api_min) if api_min else float(SYMBOL_MAP[symbol]["min_amount"])

    def round_price(self, symbol: str, price: float) -> float:
        if price <= 0: return 0.0
        tick = self.get_tick_size(symbol)
        rounded = round(round(price / tick) * tick, 10)
        dec = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, dec)

    def round_amount(self, symbol: str, raw: float):
        """Round to valid step. Returns int for inverse (step≥1), float for linear (step<1)."""
        if raw <= 0: return 0
        step = self.get_min_trade_amount(symbol)
        steps = math.floor(raw / step)
        result = max(step, steps * step)
        dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return int(result) if dec == 0 else round(result, dec)

    def to_int_amount(self, symbol: str, raw: float):
        """Alias for round_amount — keeps trade_executor code readable."""
        return self.round_amount(symbol, raw)

    def split_amount(self, symbol: str, total) -> tuple:
        """Split into two valid halves for TP1/TP2."""
        if not total or total <= 0: return 0, 0
        step = self.get_min_trade_amount(symbol)
        half = total / 2.0
        tp1  = max(step, math.floor(half / step) * step)
        tp2  = total - tp1
        if tp2 < step: tp1, tp2 = total, 0
        dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        if dec == 0: return int(tp1), int(tp2)
        return round(tp1, dec), round(tp2, dec)

    # ── Position sizing ───────────────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float,
                       entry: float, stop: float, risk_mult: float = 1.0):
        """
        1% risk per trade, capped at 20% of balance.
        Inverse (BTC/ETH): amount in USD contracts (integers: BTC min=10, ETH min=1)
        Linear (USDC): amount in base currency units (float ok, e.g. 5.3 SOL)
        """
        try:
            from config import RISK_PER_TRADE as risk_pct
        except ImportError:
            risk_pct = 0.01

        risk_usd  = balance_usd * risk_pct * risk_mult
        stop_dist = abs(entry - stop)
        spec      = SYMBOL_MAP[symbol]
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0 or entry <= 0:
            return self.round_amount(symbol, min_amt)

        if spec["kind"] == "inverse":
            # BTC-PERPETUAL: $10/contract. ETH-PERPETUAL: $1/contract.
            # risk_usd = contracts × cs × stop_dist / entry
            cs  = spec["contract_usd"]
            raw = risk_usd * entry / (cs * stop_dist)
            cap = int((balance_usd * 0.20) / cs)
            raw = min(raw, cap)
        else:
            # Linear: risk_usd = amount × stop_dist
            raw = risk_usd / stop_dist
            cap = (balance_usd * 0.20) / entry
            raw = min(raw, cap)

        result = self.round_amount(symbol, raw)
        log.info(f"  Contracts: {result} {symbol} "
                 f"(risk=${risk_usd:.2f} stop_dist={stop_dist:.4f} raw={raw:.2f})")
        return result

    # ── Balance ───────────────────────────────────────────────────────

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self._get("/private/get_account_summary",
                               {"currency": currency, "extended": "true"})
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
                r = self._get("/private/get_positions",
                              {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(p for p in r if float(p.get("size", 0) or 0) != 0)
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount) -> dict:
        """Market order. Returns full result dict {"order": {...}, "trades": [...]}."""
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
        """Extract fill price from trades[] array (more reliable than order.average_price)."""
        trades = market_result.get("trades", [])
        if trades:
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if prices:
                return round(sum(prices) / len(prices), 8)
        order = market_result.get("order", market_result)
        avg   = order.get("average_price") or order.get("price")
        if avg and float(avg) > 0: return float(avg)
        return fallback

    def place_limit_order(self, symbol: str, side: str, amount,
                          price: float, stop_price: float = None) -> dict:
        """Limit or stop-limit. trigger_price field per Deribit REST docs. reduce_only=True."""
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)

        if stop_price is not None:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "stop_limit",
                "price":           safe_price,
                "trigger_price":   self.round_price(symbol, stop_price),
                "trigger":         "last_price",
                "reduce_only":     True,
                "label":           f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "reduce_only":     True,
                "label":           f"bot_tp_{int(time.time())}",
            }

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ {safe_price} "
                 f"id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            if "order_not_found" in str(e).lower():
                return {"order_state": "not_found"}
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
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
            usdc_status = f"USDC=${self._usdc_balance:.0f}" if self.has_usdc_margin() else "USDC=none"
            log.info(f"✅ Deribit Testnet — ${total:.2f} USD | {usdc_status} | "
                     f"{len(self._supported_symbols)} tradeable")
            return True
        except Exception as e:
            log.error(f"✗ Deribit FAILED: {e}")
            raise
