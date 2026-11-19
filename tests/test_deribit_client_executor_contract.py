from __future__ import annotations

import ast
import inspect
from pathlib import Path

from deribit_client import (
    DeribitClient,
    OrderOutcomeUnknown,
    ProtectiveOrderVerificationError,
)


def test_deribit_client_exposes_the_executor_contract():
    required_methods = (
        "get_risk_capital_snapshot",
        "position_size_from_record",
        "get_order_state_by_label",
        "get_open_orders_by_label",
        "get_trade_history_by_instrument_name",
        "get_order_history_by_instrument_name",
        "get_user_trades_by_order",
        "place_reduce_only_market_order",
    )
    missing = [name for name in required_methods if not callable(getattr(DeribitClient, name, None))]
    assert not missing, f"DeribitClient is missing executor methods: {missing}"

    constructor = inspect.signature(DeribitClient.__init__)
    assert "auth_scope" in constructor.parameters
    assert "allow_mutations" in constructor.parameters
    sizing = inspect.signature(DeribitClient.calc_contracts)
    assert "available_funds" in sizing.parameters
    assert issubclass(OrderOutcomeUnknown, RuntimeError)
    assert issubclass(ProtectiveOrderVerificationError, RuntimeError)


def test_historical_recovery_endpoints_are_read_allowlisted():
    allowlist = DeribitClient._read_only_private_methods()
    assert "/private/get_order_history_by_instrument" in allowlist
    assert "/private/get_user_trades_by_order" in allowlist
    assert "/private/get_user_trades_by_instrument" in allowlist


def test_order_lifecycle_integration_gate_remains_false():
    project_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((project_root / "trade_executor.py").read_text(encoding="utf-8"))
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ORDER_LIFECYCLE_COORDINATOR_INTEGRATED"
            for target in node.targets
        ):
            assignments.append(node.value)
    assert len(assignments) == 1
    assert isinstance(assignments[0], ast.Constant) and assignments[0].value is False
