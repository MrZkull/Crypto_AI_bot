"""Currency-aware account snapshot normalisation for CryptoBot AI.

No value in this module is assumed to be USD or USDT. Native collateral,
account-wide USD equity and new-entry risk capital are distinct fields.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


class RiskAccountingError(ValueError):
    """Exchange account data cannot be safely used for risk accounting."""


def _finite(value: Any, field: str, *, required: bool = True):
    if value is None and not required:
        return None
    if value is None or isinstance(value, bool):
        raise RiskAccountingError(f"{field} missing or invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RiskAccountingError(f"{field} invalid") from None
    if not math.isfinite(number):
        raise RiskAccountingError(f"{field} non-finite")
    return number


def build_balance_payload(
    snapshot: dict,
    positions: list[dict],
    *,
    updated_at: str | None = None,
) -> tuple[dict, float]:
    """Return a balance.json-compatible payload and USDC risk capital.

    A segregated account can size this bot only from USDC equity plus positive
    USDC available funds. Cross-collateral USD value is reporting-only until a
    verified USD/USDC allocation and margin-budget model exists.
    """
    if not isinstance(snapshot, dict):
        raise RiskAccountingError("risk-capital snapshot malformed")
    if not isinstance(positions, list) or any(not isinstance(item, dict) for item in positions):
        raise RiskAccountingError("position snapshot malformed")

    risk_currency = str(snapshot.get("risk_currency", "")).upper()
    if risk_currency != "USDC":
        raise RiskAccountingError("configured Deribit instruments require USDC risk capital")

    entry_enabled = snapshot.get("entry_enabled") is True
    margin_mode = str(snapshot.get("margin_mode", "unknown"))
    if margin_mode not in {"segregated", "cross_collateral"}:
        raise RiskAccountingError("margin mode missing or unsupported")

    if entry_enabled:
        if margin_mode != "segregated":
            raise RiskAccountingError("new entries require segregated USDC collateral")
        risk_equity = _finite(snapshot.get("risk_equity"), "USDC risk equity")
        available_funds = _finite(snapshot.get("available_funds"), "USDC available funds")
        if risk_equity <= 0 or available_funds <= 0:
            raise RiskAccountingError("USDC equity and available funds must be positive")
    else:
        risk_equity = 0.0
        available_funds = 0.0

    assets = snapshot.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RiskAccountingError("native asset balances missing")
    normalized_assets = []
    seen = set()
    for item in assets:
        if not isinstance(item, dict):
            raise RiskAccountingError("native asset row malformed")
        asset = str(item.get("asset", "")).strip().upper()
        unit = str(item.get("unit", "")).strip().upper()
        if not asset or unit != asset or asset in seen:
            raise RiskAccountingError("native asset identity missing, duplicated or mismatched")
        seen.add(asset)
        equity = _finite(item.get("equity"), f"{asset} native equity")
        available = _finite(item.get("available"), f"{asset} available funds", required=False)
        normalized_assets.append({
            "asset": asset,
            "free": str(round(available, 8)) if available is not None else "N/A",
            "total": str(round(equity, 8)),
            "unit": unit,
        })

    pnl_values = []
    for position in positions:
        raw = position.get("floating_profit_loss_usd")
        if raw is None or isinstance(raw, bool):
            pnl_values = None
            break
        value = _finite(raw, "floating_profit_loss_usd")
        pnl_values.append(value)
    unrealised_usd = round(sum(pnl_values), 4) if pnl_values is not None else None

    account_equity_usd = _finite(
        snapshot.get("account_equity_usd"), "account_equity_usd", required=False
    )
    if margin_mode == "cross_collateral" and account_equity_usd is None:
        raise RiskAccountingError("cross-collateral snapshot requires account-wide USD equity")
    if margin_mode == "segregated" and account_equity_usd is not None:
        raise RiskAccountingError("segregated snapshot must not relabel native equity as account-wide USD")

    timestamp = updated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        # Legacy keys remain USD-only; zero is an explicit unavailable sentinel.
        "usdt": round(account_equity_usd, 2) if account_equity_usd is not None else 0.0,
        "equity": round(account_equity_usd, 2) if account_equity_usd is not None else 0.0,
        "equity_currency": "USD" if account_equity_usd is not None else None,
        "account_equity_usd": round(account_equity_usd, 2) if account_equity_usd is not None else None,
        "risk_equity": round(risk_equity, 8),
        "risk_equity_currency": "USDC",
        "risk_entry_enabled": entry_enabled,
        "risk_block_reason": snapshot.get("risk_block_reason"),
        "margin_mode": margin_mode,
        "available_funds": round(available_funds, 8),
        "available_funds_currency": "USDC",
        "unrealised_usd": unrealised_usd,
        "assets": normalized_assets,
        "updated_at": timestamp,
        "mode": "deribit_testnet",
        "exchange": "Deribit Testnet",
        "open_positions": len(positions),
        "source": snapshot.get("source", "private/get_account_summaries"),
    }
    return payload, risk_equity


def unavailable_balance_payload(*, reason: str = "ACCOUNT_SNAPSHOT_UNAVAILABLE") -> dict:
    """Explicitly invalidate stale cached risk capital after a failed account read."""
    return {
        "usdt": 0.0,
        "equity": 0.0,
        "equity_currency": None,
        "account_equity_usd": None,
        "risk_equity": 0.0,
        "risk_equity_currency": "USDC",
        "risk_entry_enabled": False,
        "risk_block_reason": reason,
        "margin_mode": "unknown",
        "available_funds": 0.0,
        "available_funds_currency": "USDC",
        "unrealised_usd": None,
        "assets": [],
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": "deribit_testnet",
        "exchange": "Deribit Testnet",
        "source": "unavailable",
    }
