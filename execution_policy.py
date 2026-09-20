#!/usr/bin/env python3
"""Canonical execution identity + deterministic P&L math.

This module is intentionally free of market/exchange I/O. It provides the
identity contract used by training/candidate artifacts and the accounting
primitives used by tests and research.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Side = Literal["BUY", "SELL"]


def _file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_policy_hash() -> str:
    """SHA256 of this exact execution-policy source file."""
    return _file_hash(__file__)


def get_feature_code_hash() -> str:
    """SHA256 of the canonical feature-engineering source."""
    return _file_hash(Path(__file__).with_name("feature_engineering.py"))


def get_feature_schema_hash(features: list) -> str:
    payload = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate_config() -> dict:
    import config

    keys = (
        "THRESHOLD_BUY",
        "THRESHOLD_SELL",
        "RISK_MULT",
        "ATR_STOP_MULT",
        "ATR_TARGET1_MULT",
        "ATR_TARGET2_MULT",
        "MAX_OPEN_TRADES",
        "MAX_SAME_DIRECTION",
        "ENTRY_MAX_SPREAD_PCT",
        "RISK_PER_TRADE",
        "MAX_DAILY_TRADES",
    )
    out = {}
    for key in keys:
        if hasattr(config, key):
            value = getattr(config, key)
            out[key] = float(value) if isinstance(value, (int, float)) else value
    return out


def get_config_hash(config_dict: dict) -> str:
    payload = json.dumps(
        config_dict,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calculate_continuous_funding(
    side: str,
    qty_base: float,
    mark_price: float,
    funding_rate: float,
    elapsed_ms: int,
) -> float:
    """Calculate pro-rata funding cost for an elapsed interval.

    Positive funding means BUY/long pays and SELL/short receives.
    Funding is normalized to an 8-hour period.
    """
    side = str(side).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Invalid side: {side}")

    if elapsed_ms < 0:
        raise ValueError(f"elapsed_ms cannot be negative: {elapsed_ms}")

    for name, value in (
        ("qty_base", qty_base),
        ("mark_price", mark_price),
        ("funding_rate", funding_rate),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite {name}: {value}")

    if qty_base <= 0:
        raise ValueError(f"qty_base must be > 0: {qty_base}")
    if mark_price <= 0:
        raise ValueError(f"mark_price must be > 0: {mark_price}")

    if elapsed_ms == 0 or funding_rate == 0:
        return 0.0

    fraction_8h = elapsed_ms / (8.0 * 3600.0 * 1000.0)
    cost = qty_base * mark_price * funding_rate * fraction_8h

    if not math.isfinite(cost):
        raise ValueError(f"Non-finite funding result: {cost}")

    return cost if side == "BUY" else -cost


def evaluate_thesis_outcome(exit_reason: str) -> int:
    return int(exit_reason in {"HIT_TP1", "HIT_TP2", "TP1 hit", "TP2 hit"})


def finalize_outcome(trade: dict, exit_reason: str) -> dict:
    trade["thesis_label"] = evaluate_thesis_outcome(exit_reason)
    trade["exit_reason"] = exit_reason
    trade["status"] = "CLOSED"
    return trade


@dataclass
class Position:
    side: Side
    qty: float
    entry_price: float
    entry_fee: float = 0.0
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    funding: float = 0.0

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty)


def gross_pnl(side: Side, entry_price: float, exit_price: float, qty: float) -> float:
    if min(entry_price, exit_price, qty) < 0 or qty == 0:
        raise ValueError("invalid trade geometry")
    if side == "BUY":
        return (exit_price - entry_price) * qty
    if side == "SELL":
        return (entry_price - exit_price) * qty
    raise ValueError(f"unsupported side: {side}")


def fee(notional: float, rate: float) -> float:
    if notional < 0 or rate < 0:
        raise ValueError("invalid fee inputs")
    return notional * rate


def close_partial(
    pos: Position,
    exit_price: float,
    exit_qty: float,
    exit_fee_rate: float,
) -> dict:
    if exit_qty <= 0 or exit_qty > pos.qty:
        raise ValueError("exit_qty outside remaining position")
    g = gross_pnl(pos.side, pos.entry_price, exit_price, exit_qty)
    f = fee(exit_price * exit_qty, exit_fee_rate)
    pos.qty -= exit_qty
    pos.realized_gross += g
    pos.realized_fees += f
    return {
        "gross_pnl": g,
        "exit_fee": f,
        "remaining_qty": pos.qty,
        "realized_net_pnl": net_pnl(pos),
    }


def close_all(pos: Position, exit_price: float, exit_fee_rate: float) -> dict:
    return close_partial(pos, exit_price, pos.qty, exit_fee_rate)


def net_pnl(pos: Position) -> float:
    return pos.realized_gross - pos.realized_fees - pos.entry_fee - pos.funding


def net_r(pos: Position, initial_risk_usd: float) -> float:
    if initial_risk_usd <= 0:
        raise ValueError("initial_risk_usd must be positive")
    return net_pnl(pos) / initial_risk_usd
