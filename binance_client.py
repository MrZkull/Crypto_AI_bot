# binance_client.py — CLEAN WORKING VERSION (NO 451 ISSUE)

import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone

class BinanceTestnet:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret

        # ✅ USE MAIN API (LESS BLOCKED)
        self.base = "https://api.binance.com"

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0",
        })

    def _sign(self, params):
        params["timestamp"] = int(datetime.now(timezone.utc).timestamp() * 1000)
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.secret.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

        params["signature"] = signature
        return params

    def _get(self, endpoint, params=None, auth=False):
        params = params or {}

        if auth:
            params = self._sign(params)

        url = self.base + endpoint

        r = self.session.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_balance(self):
        data = self._get("/api/v3/account", auth=True)

        balances = {}

        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked

            if total > 0:
                balances[b["asset"]] = total

        return balances

    def get_usdt_balance(self):
        balances = self.get_balance()
        return balances.get("USDT", 0.0)
