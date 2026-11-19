from decimal import Decimal

from capital_feasibility import CapitalTier, build_feasibility_matrix
from exchange_reliability import InstrumentSpec


def spec(symbol="ETHUSDT", minimum="0.001", step="0.001"):
    return InstrumentSpec(
        exchange="DERIBIT",
        symbol=symbol,
        instrument_id=symbol.replace("USDT", "_USDC-PERPETUAL"),
        settlement_currency="USDC",
        quantity_unit="BASE",
        contract_value_base=Decimal("1"),
        qty_step=Decimal(step),
        min_qty=Decimal(minimum),
        max_qty=Decimal("100"),
        tick_size=Decimal("0.05"),
    )


def test_feasibility_matrix_rejects_symbol_not_allowed_by_tier():
    tier = CapitalTier("T1", Decimal("100"), 1, ("BTCUSDT",))
    rows = build_feasibility_matrix(
        [spec("ETHUSDT")], [tier], entry_prices={"ETHUSDT": "2000"},
        stop_distances={"ETHUSDT": "20"}, settlement_currency="USDC",
    )
    assert rows[0]["reason"] == "NOT_ALLOWED_BY_TIER"


def test_feasibility_matrix_flags_risk_too_small_for_minimum_lot():
    tier = CapitalTier(
        "T1", Decimal("20"), 1, ("ETHUSDT",),
        risk_fraction=Decimal("0.005"), max_notional_fraction=Decimal("0.05"),
    )
    rows = build_feasibility_matrix(
        [spec("ETHUSDT", minimum="0.1", step="0.1")], [tier],
        entry_prices={"ETHUSDT": "2000"}, stop_distances={"ETHUSDT": "20"},
        settlement_currency="USDC", available_funds_by_tier={"T1": "20"},
    )
    assert not rows[0]["feasible"]
    assert rows[0]["quantity"] == "0"
    assert rows[0]["reason"] == "BELOW_MINIMUM_WITHIN_RISK"


def test_feasibility_matrix_refuses_wrong_settlement_currency():
    tier = CapitalTier("T2", Decimal("500"), 2, ("ETHUSDT",))
    rows = build_feasibility_matrix(
        [spec("ETHUSDT")], [tier], entry_prices={"ETHUSDT": "2000"},
        stop_distances={"ETHUSDT": "20"}, settlement_currency="USD",
        available_funds_by_tier={"T2": "500"},
    )
    assert rows[0]["reason"] == "SETTLEMENT_CURRENCY_MISMATCH"


def test_feasibility_matrix_accepts_feasible_symbol_and_returns_costs():
    tier = CapitalTier("T3", Decimal("2000"), 3, ("ETHUSDT",))
    rows = build_feasibility_matrix(
        [spec("ETHUSDT")], [tier], entry_prices={"ETHUSDT": "2000"},
        stop_distances={"ETHUSDT": "20"}, settlement_currency="USDC",
        available_funds_by_tier={"T3": "2000"},
    )
    assert rows[0]["feasible"]
    assert Decimal(rows[0]["quantity"]) > 0
    assert Decimal(rows[0]["estimated_round_trip_fees"]) > 0
