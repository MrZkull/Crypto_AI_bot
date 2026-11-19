from decimal import Decimal

from capital_feasibility import CapitalTier, build_feasibility_matrix
from exchange_reliability import InstrumentSpec


def _spec():
    return InstrumentSpec(
        exchange="DERIBIT", symbol="ETHUSDT", instrument_id="ETH_USDC-PERPETUAL",
        settlement_currency="USDC", quantity_unit="BASE", contract_value_base=Decimal("1"),
        qty_step=Decimal("0.01"), min_qty=Decimal("0.01"), max_qty=None,
        tick_size=Decimal("0.05"), min_notional=Decimal("0"), linear=True,
    )

def _tier():
    return CapitalTier(
        name="T1", capital_cap=Decimal("100"), max_positions=1,
        allowed_symbols=("ETHUSDT",), risk_fraction=Decimal("0.005"),
        max_notional_fraction=Decimal("0.05"), max_daily_loss_fraction=Decimal("0.02"),
    )

def test_feasibility_requires_exchange_verified_available_funds():
    row = build_feasibility_matrix(
        [_spec()], [_tier()], entry_prices={"ETHUSDT": "100"},
        stop_distances={"ETHUSDT": "5"}, settlement_currency="USDC",
    )[0]
    assert row["tradable"] is False
    assert row["reason"] == "AVAILABLE_FUNDS_UNVERIFIED"
    assert row["available_funds"] is None

def test_feasibility_applies_reserve_and_reports_dollar_limits():
    row = build_feasibility_matrix(
        [_spec()], [_tier()], entry_prices={"ETHUSDT": "100"},
        stop_distances={"ETHUSDT": "2"}, settlement_currency="USDC",
        available_funds_by_tier={"T1": "100"},
    )[0]
    assert row["max_positions"] == 1
    assert Decimal(row["max_daily_loss_amount"]) == Decimal("2.00")
    assert Decimal(row["available_funds_reserve_fraction"]) == Decimal("0.20")
    assert Decimal(row["notional"]) <= Decimal("80")

def test_sell_feasibility_uses_stop_above_entry():
    row = build_feasibility_matrix(
        [_spec()], [_tier()], entry_prices={"ETHUSDT": "100"},
        stop_distances={"ETHUSDT": "2"}, settlement_currency="USDC",
        available_funds_by_tier={"T1": "100"}, sides_by_symbol={"ETHUSDT": "SELL"},
    )[0]
    assert row["side"] == "SELL"
    assert Decimal(row["stop_risk"]) > 0
