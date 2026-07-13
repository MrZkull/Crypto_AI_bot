# deribit_client.py — V10: Sub-Dollar Tick Size Fix & Complete V7 Methods

import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# ── Execution Protection Constants ──
DEFAULT_LEVERAGE      = 2      # 2x leverage — preserves 1% risk, frees margin
MAX_SLIPPAGE_PCT      = 0.002  # 0.2% max acceptable slippage on entry
WIDE_SPREAD_WARN_PCT  = 0.003  # warn if spread > 0.3%
MAX_TRADEABLE_SPREAD_PCT = 0.05  # NEW: hard abort if spread > 5% — book is too thin to trade safely, not just "wide"

# ── SYMBOL MAP (Updated with Max Amounts & Tick Sizes) ──
SYMBOL_MAP = {
    # Big 3
    "BTCUSDT":    {"instrument": "BTC_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.0001, "max_amount": 100,       "tick_size": 0.5},
    "ETHUSDT":    {"instrument": "ETH_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.001,  "max_amount": 5000,      "tick_size": 0.05},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.01,   "max_amount": 1000,      "tick_size": 0.05},

    # High-Liquidity L1s
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 50000,     "tick_size": 0.01},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 10000,     "tick_size": 0.01},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 1,      "max_amount": 100000,    "tick_size": 0.0001},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 1,      "max_amount": 100000,    "tick_size": 0.0001},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 10000,     "tick_size": 0.001},
    "ATOMUSDT":   {"instrument": "ATOM_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 1,      "max_amount": 2100,      "tick_size": 0.001},
    "TRXUSDT":    {"instrument": "TRX_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 10,     "max_amount": 500000,    "tick_size": 0.00001},

    # Institutional Alts
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 1,      "max_amount": 10000,     "tick_size": 0.001},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 1,      "max_amount": 10000,     "tick_size": 0.001},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 1,      "max_amount": 10000,     "tick_size": 0.001},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 1000,      "tick_size": 0.01},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 10,     "max_amount": 500000,    "tick_size": 0.0001},
    "LTCUSDT":    {"instrument": "LTC_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 1000,      "tick_size": 0.01},
    "BCHUSDT":    {"instrument": "BCH_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 0.02,   "max_amount": 500,       "tick_size": 0.01},
    "ALGOUSDT":   {"instrument": "ALGO_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 10,     "max_amount": 500000,    "tick_size": 0.0001},

    # AI & Momentum
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 1,      "max_amount": 100000,    "tick_size": 0.0001},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL",  "currency": "USDC",
                   "min_amount": 0.1,    "max_amount": 10000,     "tick_size": 0.001},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",   "currency": "USDC",
                   "min_amount": 10,     "max_amount": 500000,    "tick_size": 0.0001},
    "HYPEUSDT":    {"instrument": "HYPE_USDC-PERPETUAL", "currency": "USDC",
                    "min_amount": 0.1,      "max_amount": 10000,     "tick_size": 0.001},
    "DOGEUSDT":    {"instrument": "DOGE_USDC-PERPETUAL", "currency": "USDC",
                    "min_amount": 100,      "max_amount": 1000000,   "tick_size": 0.00001},
}

TRADEABLE_SYMBOLS: list = []

class DeribitClient:

    def __init__(self, client_id: str, client_secret: str):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()
        self._authenticate()
        self._verify_instruments()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        for attempt in range(3):
            try:
                r = self.session.get(
                    f"{self.base}/public/auth",
                    params={
                        "grant_type":    "client_credentials",
                        "client_id":     self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=15
                )
                r.raise_for_status()
                res = r.json().get("result", {})
                if not res or "access_token" not in res:
                    raise Exception(f"Auth failed: {r.text[:200]}")
                self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
                self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60
                log.info("✓ Deribit testnet authenticated")
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
        r    = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
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
        r    = self.session.get(f"{self.base}{path}", params=body, timeout=15)
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

    # ── Instrument management ─────────────────────────────────────────

    def _verify_instruments(self):
        global TRADEABLE_SYMBOLS
        active = {}
        try:
            res = self._get("/public/get_instruments",
                            {"currency": "USDC", "kind": "future", "expired": "false"})
            if isinstance(res, list):
                active = {i["instrument_name"]: i for i in res}
        except Exception as e:
            log.warning(f"  Instrument list: {e}")
        confirmed = []
        for sym, info in SYMBOL_MAP.items():
            target = info["instrument"]
            if target in active:
                self._instrument_cache[target] = active[target]
                self._supported_symbols.add(sym)
                confirmed.append(sym)
        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} — {confirmed}")

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
            self._instrument_cache[name] = self._get(
                "/public/get_instrument", {"instrument_name": name})
        return self._instrument_cache.get(name, {})

    # ── Price + amount precision ──────────────────────────────────────

    def get_tick_size(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("tick_size") or SYMBOL_MAP[symbol].get("tick_size", 0.001))

    def get_min_trade_amount(self, symbol: str) -> float:
        info    = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        return float(api_min) if api_min else float(SYMBOL_MAP[symbol].get("min_amount", 1.0))

    def get_max_trade_amount(self, symbol: str) -> float:
        info    = self.get_instrument_info(symbol)
        api_max = info.get("max_trade_amount") or info.get("max_amount")
        return float(api_max) if api_max else float(
            SYMBOL_MAP.get(symbol, {}).get("max_amount", float("inf"))
        )

    def round_price(self, symbol: str, price: float) -> float:
        """
        Round price to exchange tick size.
        SAFETY: always falls back to SYMBOL_MAP tick if API returns wrong value.
        Never returns 0.0 for a positive input price.
        """
        if price <= 0:
            return 0.0
            
        # Try API tick first, validate it makes sense for this price
        tick = self.get_tick_size(symbol)
        
        # Sanity check: if tick > price/2, the API returned garbage 
        # (e.g. tick=1.0 for an ADA price of $0.16 would round to 0.0)
        if tick <= 0 or tick > price / 2:
            tick = float(SYMBOL_MAP.get(symbol, {}).get("tick_size", 0.0001))
            log.debug(f"  round_price: API tick invalid for {symbol}@{price:.6f} — using SYMBOL_MAP tick={tick}")
            
        if tick <= 0:
            tick = 0.0001  # absolute fallback
            
        rounded  = round(round(price / tick) * tick, 10)
        
        if rounded <= 0:
            # Still 0 — return raw price rounded conservatively
            log.warning(f"  round_price({symbol}, {price:.6f}) still 0 after tick={tick} — using raw")
            return round(price, 6)
            
        decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, decimals)

    def round_amount(self, symbol: str, raw: float) -> float:
        if raw <= 0: return 0.0
        step     = self.get_min_trade_amount(symbol)
        steps    = math.floor(raw / step)
        result   = max(step, steps * step)
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return round(result, decimals) if decimals else int(result)

    def split_amount(self, symbol: str, total) -> tuple:
        if total <= 0: return 0, 0
        step  = self.get_min_trade_amount(symbol)
        tp1   = max(step, math.floor(total / 2.0 / step) * step)
        tp2   = total - tp1
        if tp2 < step: tp1 = total; tp2 = 0
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return (round(tp1, decimals), round(tp2, decimals)) if decimals else (int(tp1), int(tp2))

    # ── Market data ───────────────────────────────────────────────────

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}"); return 0.0

    def get_mark_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("index_price") or 0)
        except Exception as e:
            log.warning(f"  mark_price {symbol}: {e}"); return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("current_funding") or t.get("funding_8h") or 0)
        except Exception as e:
            log.warning(f"  funding_rate {symbol}: {e}"); return 0.0

    def get_order_book_spread(self, symbol: str) -> dict:
        """Fetch best bid/ask and compute spread percentage."""
        try:
            book = self._get("/public/get_order_book", {
                "instrument_name": self.get_instrument_name(symbol),
                "depth": 5,
            })
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids or not asks:
                return {"best_bid": 0, "best_ask": 0, "spread_pct": 999, "is_wide": True}

            best_bid   = float(bids[0][0])
            best_ask   = float(asks[0][0])
            mid        = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0

            is_wide = spread_pct > WIDE_SPREAD_WARN_PCT
            if is_wide:
                log.warning(f"  ⚠️ Wide spread on {symbol}: {spread_pct*100:.3f}% (bid={best_bid}, ask={best_ask})")
            return {
                "best_bid":   best_bid,
                "best_ask":   best_ask,
                "spread_pct": spread_pct,
                "is_wide":    is_wide,
            }
        except Exception as e:
            log.warning(f"  order book {symbol}: {e}")
            return {"best_bid": 0, "best_ask": 0, "spread_pct": 999, "is_wide": True}

    # ── Position sizing ───────────────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float,
                       entry: float, stop: float, risk_mult: float = 1.0):
        try:
            from config import RISK_PER_TRADE as rpt
        except ImportError:
            rpt = 0.01

        risk_usd  = balance_usd * rpt * risk_mult
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0 or entry <= 0:
            return self.round_amount(symbol, min_amt)

        raw      = risk_usd / stop_dist
        max_pct  = (balance_usd * 0.05) / entry
        max_risk = (risk_usd * 10) / entry
        raw      = min(raw, max_pct, max_risk)

        # ── Exchange ceiling cap ─────────────────────────────────────────
        max_amt  = self.get_max_trade_amount(symbol)
        if raw > max_amt:
            log.info(f"  ⚡ Position capped at exchange ceiling: {max_amt} {symbol} (raw={raw:.0f}, max={max_amt})")
            raw = max_amt

        result   = self.round_amount(symbol, max(raw, min_amt))
        notional = result * entry
        log.info(f"  Contracts: {result} {symbol} | notional≈${notional:.0f} | risk=${risk_usd:.2f}")
        return result

    # ── Order execution ───────────────────────────────────────────────

    @staticmethod
    def _is_position_size_limit_error(e) -> bool:
        """Code:10057 — non-PME accounts have a max position size per instrument that's
        SEPARATE from the instrument's general max_trade_amount (which calc_contracts()
        already checks). This can reject an order calc_contracts() thought was fine —
        worst case, on a closing/emergency order, which previously had no fallback and
        just failed outright, leaving a breached-stop position with zero protection."""
        return "10057" in str(e) or "non_pme_max_future_position_size" in str(e)

    def place_market_order(self, symbol: str, side: str, amount) -> dict:
        """Entry order with slippage protection (IoC limit). Reduce-and-retry on a
        non-PME position-size rejection (Code:10057) rather than failing outright —
        critical for emergency/SL-missed closes, where partial closure beats none."""
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        label      = f"bot_entry_{int(time.time())}"

        try:
            spread = self.get_order_book_spread(symbol)
            best_bid = spread["best_bid"]
            best_ask = spread["best_ask"]

            # NEW: hard abort — a 5%+ spread means the book is too thin to fill safely,
            # not just "wide". Proceeding here is how a 54,370-contract order fills 3,592
            # (6.6%) at an unknown, likely terrible price. Skip the trade entirely.
            if spread["spread_pct"] > MAX_TRADEABLE_SPREAD_PCT:
                log.warning(f"  🚫 {symbol}: spread {spread['spread_pct']*100:.1f}% exceeds "
                            f"{MAX_TRADEABLE_SPREAD_PCT*100:.0f}% ceiling — book too thin, skipping entry")
                return {}

            if best_bid > 0 and best_ask > 0:
                if side.upper() == "BUY":
                    worst_price = self.round_price(symbol, best_ask * (1 + MAX_SLIPPAGE_PCT))
                else:
                    worst_price = self.round_price(symbol, best_bid * (1 - MAX_SLIPPAGE_PCT))

                log.info(f"  IoC limit {side} {amount} {instrument} @ max {worst_price} (spread {spread['spread_pct']*100:.3f}%)")

                cur_amount = amount
                for attempt in range(4):
                    try:
                        result = self._post(method, {
                            "instrument_name": instrument,
                            "amount":          cur_amount,
                            "type":            "limit",
                            "price":           worst_price,
                            "time_in_force":   "immediate_or_cancel",
                            "label":           label,
                        })
                        order = result.get("order", result)
                        state = order.get("order_state", "")

                        if state == "cancelled":
                            log.warning(f"  ⚠️ IoC CANCELLED — market moved >{MAX_SLIPPAGE_PCT*100:.1f}%. Skipping entry.")
                            return {}

                        log.info(f"  ✅ IoC {side.upper()} {cur_amount} {instrument} id={order.get('order_id','')} state={state}")
                        return result
                    except Exception as e:
                        if self._is_position_size_limit_error(e) and attempt < 3:
                            cur_amount = self.round_amount(symbol, cur_amount * 0.5)
                            if cur_amount <= 0:
                                log.error(f"  {symbol}: position-size limit — reduced to 0, giving up on IoC")
                                break
                            log.warning(f"  ⚠️ {symbol}: position-size limit (Code:10057) — "
                                        f"retrying IoC at reduced size {cur_amount}")
                            continue
                        raise

        except Exception as e:
            log.warning(f"  IoC order failed ({e}) — falling back to market order")

        # Fallback pure market order — same reduce-and-retry protection
        cur_amount = amount
        for attempt in range(4):
            try:
                result = self._post(method, {
                    "instrument_name": instrument,
                    "amount":          cur_amount,
                    "type":            "market",
                    "label":           label,
                })
                order = result.get("order", result)
                log.info(f"  ✅ MARKET {side.upper()} {cur_amount} {instrument} id={order.get('order_id','')} state={order.get('order_state','')}")
                if cur_amount != amount:
                    log.warning(f"  ⚠️ {symbol}: only filled {cur_amount}/{amount} due to position-size "
                                f"limit — position may be PARTIALLY closed, verify manually")
                return result
            except Exception as e:
                if self._is_position_size_limit_error(e) and attempt < 3:
                    cur_amount = self.round_amount(symbol, cur_amount * 0.5)
                    if cur_amount <= 0:
                        break
                    log.warning(f"  ⚠️ {symbol}: position-size limit (Code:10057) on market fallback — "
                                f"retrying at reduced size {cur_amount}")
                    continue
                log.error(f"  Market order failed permanently for {symbol}: {e}")
                return {}

        log.error(f"  🚨 {symbol}: could not fill even at minimum size after repeated position-size "
                  f"rejections — order NOT placed, manual intervention required")
        return {}

    # ── Fill price + positions ────────────────────────────────────────

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
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
                    return float(p.get("size", 0) or 0)
            return 0.0
        except Exception as e:
            log.warning(f"  get_position_size {symbol}: {e}"); return 0.0

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
            log.warning(f"  get_positions: {e}"); return []

    def place_limit_order(self, symbol: str, side: str, amount, price: float,
                          stop_price: float = None, use_reduce_only: bool = False) -> dict:
        """
        Place a limit or stop-limit order.

        use_reduce_only default changed to False to prevent Code:11030:
          - SL (stop_limit): caller passes use_reduce_only=True explicitly
          - TP (limit):      use_reduce_only=False — TP closing a position is
                             handled by side direction, not reduce_only flag.
                             reduce_only requires exact position-size match
                             which fails when IoC fill differs from recorded qty.
        """
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)

        if stop_price is not None:
            # Stop-limit (SL order)
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "stop_limit",
                "price":           safe_price,
                "trigger_price":   self.round_price(symbol, stop_price),
                "trigger":         "last_price",
                "label":           f"bot_sl_{int(time.time())}",
            }
            if use_reduce_only:
                body["reduce_only"] = "true"
        else:
            # Plain limit (TP order) — never reduce_only to avoid Code:11030
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "label":           f"bot_tp_{int(time.time())}",
            }
            # reduce_only intentionally omitted for TP orders

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "SL" if stop_price else "TP"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ {safe_price} "
                 f"id={order.get('order_id','')} state={order.get('order_state','')}")
        return result

    # ── V7 CRITICAL METHODS RESTORED BELOW THIS LINE ──

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
        if state == "filled": return True
        if state in ("cancelled", "closed") and (filled_amt > 0 or avg_price > 0):
            log.info(f"  Triggered order detected: state={state} filled={filled_amt} avg={avg_price}")
            return True
        return False

    def is_sl_triggered(self, order: dict) -> bool:
        state      = order.get("order_state", "").lower()
        filled_amt = float(order.get("filled_amount", 0) or 0)
        avg_price  = float(order.get("average_price", 0) or 0)

        if state in ("filled", "triggered"):
            return True
        if state == "cancelled" and filled_amt > 0 and avg_price > 0:
            log.info(f"  SL cancelled-but-filled: amt={filled_amt} avg={avg_price}")
            return True
        return False

    def get_order_fill_price(self, order: dict, fallback: float) -> float:
        avg = order.get("average_price")
        if avg and float(avg) > 0: return float(avg)
        lp = order.get("last_price") or order.get("price")
        if lp and float(lp) > 0: return float(lp)
        return fallback

    def get_trade_history_for_instrument(self, symbol: str, count: int = 10) -> list:
        try:
            instrument = self.get_instrument_name(symbol)
            result     = self._get("/private/get_user_trades_by_instrument", {
                "instrument_name": instrument,
                "count":           count,
                "sorting":         "desc",
            })
            return result if isinstance(result, list) else result.get("trades", [])
        except Exception as e:
            log.warning(f"  Trade history {symbol}: {e}"); return []

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  cancel {order_id}: {e}"); return {}

    def get_open_orders(self, symbol: str) -> list:
        try:
            return self._get("/private/get_open_orders_by_instrument", {"instrument_name": self.get_instrument_name(symbol)}) or []
        except Exception as e:
            log.warning(f"  open_orders {symbol}: {e}"); return []

    def set_leverage(self, symbol: str, leverage: int = DEFAULT_LEVERAGE) -> bool:
        try:
            self._ensure_auth()
            instrument = self.get_instrument_name(symbol)
            result = self._get("/private/set_leverage", {
                "instrument_name": instrument,
                "leverage":        leverage,
            })
            actual = result.get("leverage", leverage) if isinstance(result, dict) else leverage
            log.info(f"  ⚡ Leverage set to {actual}x — {instrument}")
            return True
        except Exception as e:
            log.warning(f"  set_leverage {symbol}: {e} — proceeding at default margin")
            return False

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit OK — ${total:.2f} | {len(TRADEABLE_SYMBOLS)} symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit: {e}"); raise
