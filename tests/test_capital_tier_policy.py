from types import SimpleNamespace
from decimal import Decimal
import pytest

from capital_feasibility import resolve_active_capital_tier_policy


def profile(environment="testnet", enabled=True):
    return SimpleNamespace(
        CAPITAL_TIER_ACTIVE="T0_TESTNET",
        SYMBOLS=("ETHUSDT", "SOLUSDT"),
        MAX_OPEN_TRADES=8,
        CAPITAL_TIER_POLICIES={"T0_TESTNET": {
            "enabled": enabled, "environment": environment,
            "allowed_symbols": ("ETHUSDT", "SOLUSDT"),
            "capital_cap_usdc": None, "max_positions": 8,
            "risk_fraction": "0.03", "max_notional_fraction": "0.05",
            "max_daily_loss_fraction": "0.02", "max_daily_loss_usdc": None,
            "max_portfolio_risk_fraction": "0.02", "max_portfolio_risk_usdc": None,
            "available_funds_reserve_fraction": "0.20",
            "max_dust_position_notional_usdc": "0.10",
            "dust_reserve_per_position_usdc": "0.10",
            "max_total_dust_reserve_usdc": "0.25",
        }},
    )


def test_active_testnet_tier_resolves_with_explicit_allowlist_and_reserves():
    resolved = resolve_active_capital_tier_policy(profile(), environment="testnet")
    assert resolved["name"] == "T0_TESTNET"
    assert resolved["max_positions"] == 8
    assert resolved["allowed_symbols"] == ("ETHUSDT", "SOLUSDT")
    assert resolved["max_dust_position_notional_usdc"] == Decimal("0.10")


def test_disabled_or_wrong_environment_tier_fails_closed():
    with pytest.raises(ValueError, match="disabled"):
        resolve_active_capital_tier_policy(profile(enabled=False), environment="testnet")
    with pytest.raises(ValueError, match="environment mismatch"):
        resolve_active_capital_tier_policy(profile(environment="mainnet"), environment="testnet")


def test_future_live_tier_is_not_synthesized():
    cfg = profile()
    cfg.CAPITAL_TIER_ACTIVE = "T1_LIVE"
    with pytest.raises(ValueError, match="missing"):
        resolve_active_capital_tier_policy(cfg, environment="mainnet")


def test_tier_cannot_exceed_strategy_position_limit_or_dust_ceiling():
    cfg = profile()
    cfg.CAPITAL_TIER_POLICIES["T0_TESTNET"]["max_positions"] = 9
    with pytest.raises(ValueError, match="strategy hard cap"):
        resolve_active_capital_tier_policy(cfg, environment="testnet")
    cfg = profile()
    cfg.CAPITAL_TIER_POLICIES["T0_TESTNET"]["max_total_dust_reserve_usdc"] = "0.50"
    with pytest.raises(ValueError, match="dust policy"):
        resolve_active_capital_tier_policy(cfg, environment="testnet")
