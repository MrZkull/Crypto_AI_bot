"""Capital-tier feasibility evaluation using exchange-verified instrument specs."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from exchange_reliability import (
    InstrumentSpec, RiskCapital, RiskLimits, SizeResult, calculate_position_size,
)


@dataclass(frozen=True)
class CapitalTier:
    name: str
    capital_cap: Decimal
    max_positions: int
    allowed_symbols: tuple[str, ...]
    risk_fraction: Decimal = Decimal("0.005")
    max_notional_fraction: Decimal = Decimal("0.05")
    max_daily_loss_fraction: Decimal = Decimal("0.02")

    def __post_init__(self):
        if not self.name or Decimal(str(self.capital_cap)) <= 0 or self.max_positions < 1:
            raise ValueError("Invalid capital tier")
        for name in ("risk_fraction", "max_notional_fraction", "max_daily_loss_fraction"):
            value = Decimal(str(getattr(self, name)))
            if value <= 0 or value > 1:
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)


def build_feasibility_matrix(
    instruments: Sequence[InstrumentSpec],
    tiers: Sequence[CapitalTier],
    *,
    entry_prices: dict[str, Decimal | str | float],
    stop_distances: dict[str, Decimal | str | float],
    settlement_currency: str,
    available_funds_by_tier: dict[str, Decimal | str | float] | None = None,
    taker_fee_rate: Decimal = Decimal("0.0005"),
    available_funds_reserve_fraction: Decimal = Decimal("0.20"),
    sides_by_symbol: dict[str, str] | None = None,
) -> list[dict]:
    """Return a symbol-by-tier matrix from verified available collateral only.

    A tier's capital cap is a sizing ceiling, not proof that the exchange has
    that amount available. Missing available-funds evidence is an explicit
    non-tradable result; equity is never substituted for available margin.
    """
    matrix = []
    available_funds_by_tier = available_funds_by_tier or {}
    sides_by_symbol = sides_by_symbol or {}
    reserve = Decimal(str(available_funds_reserve_fraction))
    if not reserve.is_finite() or reserve < 0 or reserve >= 1:
        raise ValueError("available_funds_reserve_fraction must be in [0, 1)")

    for tier in tiers:
        tier_symbols = {s.upper() for s in tier.allowed_symbols}
        equity = Decimal(str(tier.capital_cap))
        daily_loss_amount = equity * tier.max_daily_loss_fraction
        for spec in instruments:
            side = str(sides_by_symbol.get(spec.symbol, "BUY")).upper()
            row = {
                "tier": tier.name,
                "capital_cap": str(equity),
                "max_positions": tier.max_positions,
                "max_daily_loss_fraction": str(tier.max_daily_loss_fraction),
                "max_daily_loss_amount": str(daily_loss_amount),
                "risk_fraction": str(tier.risk_fraction),
                "max_notional_fraction": str(tier.max_notional_fraction),
                "available_funds_reserve_fraction": str(reserve),
                "exchange": spec.exchange,
                "symbol": spec.symbol,
                "instrument_id": spec.instrument_id,
                "side": side,
                "settlement_currency": spec.settlement_currency,
                "allowed_by_tier": spec.symbol.upper() in tier_symbols,
                "available_funds_verified": tier.name in available_funds_by_tier,
                "available_funds": None,
                "feasible": False,
                "tradable": False,
                "reason": "NOT_ALLOWED_BY_TIER",
                "quantity": "0",
                "notional": "0",
                "stop_risk": "0",
                "estimated_round_trip_fees": "0",
                "total_risk": "0",
                "risk_percent_equity": None,
                "fees_percent_of_stop_risk": None,
                "risk_currency": spec.settlement_currency,
            }
            if not row["allowed_by_tier"]:
                matrix.append(row)
                continue
            if side not in {"BUY", "SELL"}:
                row["reason"] = "INVALID_SIDE"
                matrix.append(row)
                continue
            if tier.name not in available_funds_by_tier:
                row["reason"] = "AVAILABLE_FUNDS_UNVERIFIED"
                matrix.append(row)
                continue
            try:
                available = Decimal(str(available_funds_by_tier[tier.name]))
                if not available.is_finite() or available <= 0:
                    row["reason"] = "NO_VERIFIED_AVAILABLE_FUNDS"
                    matrix.append(row)
                    continue
            except Exception:
                row["reason"] = "INVALID_AVAILABLE_FUNDS"
                matrix.append(row)
                continue
            row["available_funds"] = str(available)
            if spec.settlement_currency.upper() != settlement_currency.upper():
                row["reason"] = "SETTLEMENT_CURRENCY_MISMATCH"
                matrix.append(row)
                continue
            if spec.symbol not in entry_prices or spec.symbol not in stop_distances:
                row["reason"] = "MISSING_PRICE_OR_STOP_DISTANCE"
                matrix.append(row)
                continue
            try:
                capital = RiskCapital(
                    equity=equity,
                    currency=settlement_currency,
                    available_funds=available,
                    margin_mode="segregated",
                    source="verified_feasibility_input",
                )
                entry = Decimal(str(entry_prices[spec.symbol]))
                distance = Decimal(str(stop_distances[spec.symbol]))
                if not entry.is_finite() or not distance.is_finite() or entry <= 0 or distance <= 0:
                    raise ValueError("Price and stop distance must be positive and finite")
                stop = entry - distance if side == "BUY" else entry + distance
                limits = RiskLimits(
                    risk_fraction=tier.risk_fraction,
                    max_notional_fraction=tier.max_notional_fraction,
                    tier_cap=tier.capital_cap,
                    taker_fee_rate=taker_fee_rate,
                    available_funds_reserve_fraction=reserve,
                )
                result: SizeResult = calculate_position_size(
                    capital, spec, entry_price=entry, stop_price=stop,
                    limits=limits, side=side,
                )
            except Exception as exc:
                row["reason"] = f"INVALID_FEASIBILITY_INPUT:{type(exc).__name__}"
                matrix.append(row)
                continue

            risk_pct = (result.total_risk / equity * Decimal("100")) if equity > 0 else None
            fee_pct = (
                result.estimated_round_trip_fees / result.stop_risk * Decimal("100")
                if result.stop_risk > 0 else None
            )
            row.update({
                "feasible": result.accepted,
                "tradable": result.accepted,
                "reason": result.reason,
                "quantity": str(result.quantity),
                "notional": str(result.notional),
                "stop_risk": str(result.stop_risk),
                "estimated_round_trip_fees": str(result.estimated_round_trip_fees),
                "total_risk": str(result.total_risk),
                "risk_percent_equity": str(risk_pct) if risk_pct is not None else None,
                "fees_percent_of_stop_risk": str(fee_pct) if fee_pct is not None else None,
            })
            matrix.append(row)
    return matrix



def resolve_active_capital_tier_policy(config_module, *, environment: str) -> dict:
    """Return a fully validated active capital profile or fail closed.

    This resolver is intentionally strict: only an explicitly enabled profile
    for the exact runtime environment is accepted. No future live tier is
    synthesized from equity or deposits.
    """
    name = str(getattr(config_module, "CAPITAL_TIER_ACTIVE", "") or "").strip().upper()
    policies = getattr(config_module, "CAPITAL_TIER_POLICIES", None)
    if not name or not isinstance(policies, dict) or not isinstance(policies.get(name), dict):
        raise ValueError("Active capital tier profile is missing")
    profile = dict(policies[name])
    if profile.get("enabled") is not True:
        raise ValueError("Active capital tier profile is disabled")
    expected_environment = str(environment or "").strip().lower()
    if str(profile.get("environment") or "").strip().lower() != expected_environment:
        raise ValueError("Capital tier environment mismatch")
    allowed = profile.get("allowed_symbols")
    if not isinstance(allowed, (tuple, list, set, frozenset)) or not allowed:
        raise ValueError("Capital tier symbol allowlist is missing")
    allowed_symbols = tuple(sorted({str(item).strip().upper() for item in allowed if str(item).strip()}))
    configured_symbols = {str(item).strip().upper() for item in getattr(config_module, "SYMBOLS", ())}
    if not allowed_symbols or not set(allowed_symbols).issubset(configured_symbols):
        raise ValueError("Capital tier allowlist is invalid or outside the configured universe")
    integer_fields = ("max_positions",)
    for field in integer_fields:
        try:
            value = int(profile[field])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(f"Capital tier {field} is invalid") from None
        if value < 1:
            raise ValueError(f"Capital tier {field} must be positive")
        configured_ceiling = int(getattr(config_module, "MAX_OPEN_TRADES", 8))
        if value > configured_ceiling:
            raise ValueError("Capital tier max_positions exceeds the strategy hard cap")
        profile[field] = value
    decimal_fields = (
        "risk_fraction", "max_notional_fraction", "max_daily_loss_fraction",
        "max_portfolio_risk_fraction", "available_funds_reserve_fraction",
        "max_dust_position_notional_usdc", "dust_reserve_per_position_usdc",
        "max_total_dust_reserve_usdc",
    )
    for field in decimal_fields:
        try:
            value = Decimal(str(profile[field]))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            raise ValueError(f"Capital tier {field} is invalid") from None
        if not value.is_finite() or value <= 0:
            raise ValueError(f"Capital tier {field} must be positive and finite")
        profile[field] = value
    if profile["risk_fraction"] > Decimal("0.05") or profile["max_notional_fraction"] > Decimal("1"):
        raise ValueError("Capital tier sizing fraction exceeds the hard ceiling")
    if profile["max_daily_loss_fraction"] > Decimal("0.10") or profile["max_portfolio_risk_fraction"] > Decimal("0.05"):
        raise ValueError("Capital tier risk fraction exceeds the hard ceiling")
    if profile["available_funds_reserve_fraction"] >= Decimal("1"):
        raise ValueError("Capital tier available-funds reserve must be in [0,1)")
    if (
        profile["max_dust_position_notional_usdc"] > Decimal("0.10")
        or profile["dust_reserve_per_position_usdc"] > Decimal("0.10")
        or profile["max_total_dust_reserve_usdc"] > Decimal("0.25")
    ):
        raise ValueError("Testnet dust policy exceeds its hard USDC ceilings")
    if profile["dust_reserve_per_position_usdc"] > profile["max_total_dust_reserve_usdc"]:
        raise ValueError("Per-position dust reserve exceeds total dust reserve")
    for field in ("max_daily_loss_usdc", "max_portfolio_risk_usdc", "capital_cap_usdc"):
        if profile.get(field) is not None:
            try:
                value = Decimal(str(profile[field]))
            except (InvalidOperation, TypeError, ValueError):
                raise ValueError(f"Capital tier {field} is invalid") from None
            if not value.is_finite() or value <= 0:
                raise ValueError(f"Capital tier {field} must be positive and finite")
            profile[field] = value
    profile["name"] = name
    profile["allowed_symbols"] = allowed_symbols
    return profile
