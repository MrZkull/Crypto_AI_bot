# deribit_client.py — V13.0: Institutional Deribit Client & Verification Engine

import math
import time
import logging
import requests

log = logging.getLogger(__name__)

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


def _calc_decimals(val: float) -> int:
    if val <= 0:
        return 4
    s = f"{val:.10f}".rstrip("0")
    if "." in s:
        parts = s.split(".")
        return len(parts[1]) if len(parts) > 1 else 0
    return 0


class DeribitClient:

    def __init__(self, client_id: str, client_secret: str, testnet: bool = True):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE if testnet else PROD_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()
        self._authenticate()
        self._verify_instruments()

    def _authenticate(self):
        for attempt in range(3):
            try:
                r = self.session.get(
                    f"{self.base}/public/auth",
                    params={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret
                    },
                    timeout=15
                )
                r.raise_for_status()
                res = r.json().get("result", {})
                if not res or "access_token" not in res:
                    raise Exception(f"Auth failed: {r.text[:200]}")
                self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
                self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60
                log.info("✓ Deribit authenticated successfully")
                return
            except Exception as e:
                if attempt < 2:
                    log.warning(f"Auth attempt {attempt+1} failed: {e} — retrying in 15s")
                    time.sleep(15)
                else:
                    log.error(f"Auth failed after 3 attempts: {e}")
                    raise

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        data = r.json()
        if "error" in data:
            err  = data["error"]
            msg  = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", "")          if isinstance(err, dict) else ""
            raise Exception(f"{msg} (Code:{code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=body, timeout=15)
        data = r.json()
        if "error" in data:
            err   = data["error"]
            msg   = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code  = err.get("code", "")          if isinstance(err, dict) else ""
            extra = err.get("data", "")          if isinstance(err, dict) else ""
            log.error(f"  Deribit error: {msg} | Code:{code} | Data:{extra}")
            raise Exception(f"{msg} (Code:{code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    def verify_instrument_contract(self, symbol: str) -> dict:
        """
        Validates linear USDC perpetual contract invariants against live exchange metadata:
        - Instrument exists and matches requested symbol identifier
        - Active status confirmed
        - Classified as future/perpetual
        - Explicit USDC settlement
        - Contract size is exactly 1.0 base coin
        """
        name = self.get_instrument_name(symbol)
        info = self._get("/public/get_instrument", {"instrument_name": name})
        if not info:
            raise ValueError(f"CRITICAL: Instrument {name} not found on Deribit.")

        api_name = str(info.get("instrument_name", ""))
        if api_name != name:
            raise ValueError(f"CRITICAL: Symbol mapping mismatch. Config='{name}', Exchange='{api_name}'.")

        if not info.get("is_active", False):
            raise ValueError(f"CRITICAL: Instrument {name} is inactive on exchange.")

        kind = str(info.get("kind", "")).lower()
        future_type = str(info.get("future_type", "")).lower()
        if kind not in ("future", "perpetual") and future_type != "perpetual":
            raise ValueError(f"CRITICAL: {name} classification failed (kind='{kind}', future_type='{future_type}').")

        settlement_ccy = str(info.get("settlement_currency", "")).upper()
        if settlement_ccy != "USDC":
            raise ValueError(f"CRITICAL: {name} settlement currency is '{settlement_ccy}', expected 'USDC'.")

        contract_size = float(info.get("contract_size", 0.0))
        if not math.isclose(contract_size, 1.0, rel_tol=1e-5):
            raise ValueError(f"CRITICAL: {name} contract_size={contract_size} != 1.0 (non-linear sizing invariant).")

        min_amt = float(info.get("min_trade_amount", 0.0))
        tick = float(info.get("tick_size", 0.0))
        steps = info.get("tick_size_steps", [])
        if min_amt <= 0 or tick <= 0:
            raise ValueError(f"CRITICAL: Invalid lot/tick specs for {name}: min={min_amt}, tick={tick}")

        step_summary = [f"(>{s.get('above_price')}={s.get('tick_size')})" for s in steps]
        step_str = f" | steps: {', '.join(step_summary)}" if steps else ""
        log.info(
            f"  ✓ Verified {name}: settlement={settlement_ccy} | contract_size={contract_size:.1f} "
            f"| min_lot={min_amt} | base_tick={tick}{step_str}"
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
        """Resolves dynamic tick size from price tier ladders (tick_size_steps)."""
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
        """Floors position size to step increments. Returns 0.0 if below minimum lot."""
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
        """Returns float funding rate, or None if unavailable (fail-closed semantics)."""
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

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float, stop: float, risk_per_trade: float, risk_mult: float = 1.0):
        risk_usd  = balance_usd * float(risk_per_trade) * risk_mult
        stop_dist = abs(entry - stop)

        if stop_dist <= 0 or entry <= 0 or risk_usd <= 0:
            return 0.0

        raw      = risk_usd / stop_dist
        max_pct  = (balance_usd * 0.05) / entry
        max_risk = (risk_usd * 10) / entry
        raw      = min(raw, max_pct, max_risk)

        max_amt  = self.get_max_trade_amount(symbol)
        if raw > max_amt:
            log.info(f"  ⚡ Position capped at exchange ceiling: {max_amt} {symbol}")
            raw = max_amt

        result   = self.round_amount(symbol, raw)
        notional = result * entry
        log.info(f"  Contracts: {result} {symbol} | notional≈${notional:.2f} | risk=${risk_usd:.2f}")
        return result

    @staticmethod
    def _is_position_size_limit_error(e) -> bool:
        return "10057" in str(e) or "non_pme_max_future_position_size" in str(e)

    def place_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> dict:
        """
        Dispatches native stop_market order and queries exchange to verify parameter registration.
        """
        instrument   = self.get_instrument_name(symbol)
        method       = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_trigger = self.round_price(symbol, stop_price)

        body = {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "stop_market",
            "trigger_price":   safe_trigger,
            "trigger":         "mark_price",
            "reduce_only":     True,
            "label":           f"bot_sl_{int(time.time())}",
        }
        res = self._post(method, body)
        order_res = res.get("order", res)
        order_id = str(order_res.get("order_id", ""))

        if not order_id:
            raise RuntimeError(f"SL placement on {symbol} returned no order_id.")

        # Hard Invariant: Authoritative exchange state verification
        state = self.get_order(order_id)
        if not state:
            raise RuntimeError(f"Exchange state query failed for SL {order_id} on {symbol}.")

        api_type = str(state.get("order_type", "")).lower()
        api_trigger = str(state.get("trigger", "")).lower()
        api_reduce = bool(state.get("reduce_only", False))
        api_trig_price = float(state.get("trigger_price", 0.0))

        if api_type != "stop_market":
            raise ValueError(f"Exchange state mismatch on {symbol}: Expected stop_market, found {api_type}")
        if api_trigger != "mark_price":
            raise ValueError(f"Exchange state mismatch on {symbol}: Trigger is {api_trigger}, expected mark_price")
        if not api_reduce:
            raise ValueError(f"Exchange state mismatch on {symbol}: reduce_only flag was not confirmed True.")
        if not math.isclose(api_trig_price, safe_trigger, rel_tol=1e-4):
            raise ValueError(f"Exchange trigger price mismatch on {symbol}: {api_trig_price} != {safe_trigger}")

        log.info(f"  ✓ Exchange verified SL {order_id} [{symbol}]: stop_market @ {safe_trigger} (reduce_only=True)")
        return res

    def place_market_order(self, symbol: str, side: str, amount: float, reduce_only: bool = False) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        label      = f"bot_entry_{int(time.time())}"

        cur_amount = amount
        for attempt in range(4):
            try:
                result = self._post(method, {
                    "instrument_name": instrument, "amount": cur_amount, "type": "market",
                    "label": label, "reduce_only": bool(reduce_only),
                })
                order = result.get("order", result)
                log.info(f"  ✅ MARKET {side.upper()} {cur_amount} {instrument} id={order.get('order_id','')} state={order.get('order_state','')}")
                return result
            except Exception as e:
                if self._is_position_size_limit_error(e) and attempt < 3:
                    cur_amount = self.round_amount(symbol, cur_amount * 0.5)
                    if cur_amount <= 0:
                        break
                    log.warning(f"  ⚠️ {symbol}: position-size limit (Code:10057) on market — retrying at {cur_amount}")
                    continue
                log.error(f"  Market order failed permanently for {symbol}: {e}")
                return {}
        return {}

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float,
                          stop_price: float = None, use_reduce_only: bool = False) -> dict:
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
                "trigger":         "mark_price",
                "label":           f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "label":           f"bot_tp_{int(time.time())}",
            }

        if use_reduce_only:
            body["reduce_only"] = True

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "SL" if stop_price else "TP"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ trigger:{stop_price} "
                 f"id={order.get('order_id','')} state={order.get('order_state','')}")
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

    def get_position_size(self, symbol: str) -> float:
        try:
            instrument = self.get_instrument_name(symbol)
            for p in self.get_positions():
                if p.get("instrument_name") == instrument:
                    size = float(p.get("size", 0) or 0)
                    base_ccy = instrument.split("_")[0].upper()
                    size_ccy = str(p.get("size_currency", "") or "").upper()
                    if size_ccy and base_ccy and size_ccy != base_ccy:
                        mark = float(p.get("mark_price", 0) or 0)
                        if mark > 0:
                            return size / mark
                    return size
            return 0.0
        except Exception as e:
            log.warning(f"  get_position_size {symbol}: {e}")
            return 0.0

    def get_all_balances(self) -> dict:
        balances = {}
        for cur in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self._get("/private/get_account_summary", {"currency": cur, "extended": "true"})
                eq = float(s.get("equity_usd") or s.get("equity") or 0)
                av = float(s.get("available_funds", 0) or 0)
                if eq > 0:
                    balances[cur] = {"equity_usd": round(eq, 2), "available": round(av, 6)}
            except Exception as e:
                log.debug(f"  Balance {cur}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        return round(sum(v["equity_usd"] for v in self.get_all_balances().values()), 2)

    def get_positions(self) -> list:
        try:
            positions = []
            for cur in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions", {"currency": cur, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(p for p in r if float(p.get("size", 0) or 0) != 0)
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}")
            return []

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            if "not_found" in str(e).lower():
                return {"order_state": "not_found"}
            log.warning(f"  get_order {order_id}: {e}")
            return {}

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
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            err_msg = str(e).lower()
            if "11044" in err_msg or "not_open_order" in err_msg:
                return {"result": "already_closed"}
            log.warning(f"  cancel {order_id}: {e}")
            return {}

    def get_open_orders(self, symbol: str) -> list:
        try:
            return self._get("/private/get_open_orders_by_instrument", {"instrument_name": self.get_instrument_name(symbol)}) or []
        except Exception as e:
            log.warning(f"  open_orders {symbol}: {e}")
            return []

    def get_trade_history_for_instrument(self, symbol: str, count: int = 10) -> list:
        try:
            instrument = self.get_instrument_name(symbol)
            result     = self._get("/private/get_user_trades_by_instrument", {"instrument_name": instrument, "count": count, "sorting": "desc"})
            return result if isinstance(result, list) else result.get("trades", [])
        except Exception as e:
            log.warning(f"  Trade history {symbol}: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int = DEFAULT_LEVERAGE) -> bool:
        try:
            self._ensure_auth()
            instrument = self.get_instrument_name(symbol)
            result = self._get("/private/set_leverage", {"instrument_name": instrument, "leverage": leverage})
            actual = result.get("leverage", leverage) if isinstance(result, dict) else leverage
            log.info(f"  ⚡ Leverage set to {actual}x — {instrument}")
            return True
        except Exception as e:
            log.warning(f"  set_leverage {symbol}: {e} — proceeding at default margin")
            return False

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit OK — ${total:.2f} | {len(TRADEABLE_SYMBOLS)} symbols verified")
            return True
        except Exception as e:
            log.error(f"✗ Deribit: {e}")
            raise
