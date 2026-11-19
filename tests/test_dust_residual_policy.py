from datetime import datetime, timezone
from decimal import Decimal

import pytest

from dust_residual_policy import (
    DustResidualError, classify_verified_dust_residual,
    validate_verified_dust_residual,
)


def make_dust():
    dust = classify_verified_dust_residual(
        symbol="ETHUSDT", signal="BUY",
        first_signed_position="0.00001", second_signed_position="0.0000100001",
        min_trade_amount="0.0001", mark_price="2500",
        first_open_orders=[], second_open_orders=[],
        entry_order_id="entry-order-1", entry_order_state="cancelled",
        entry_filled_quantity="0.00002", prediction_id="prediction-1",
        observed_at=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    return dust


def make_record():
    return {
        "symbol": "ETHUSDT", "closed": True, "pred_id": "prediction-1",
        "order_ids": {"entry": "entry-order-1"},
        "dust_residual": make_dust(),
        "emergency_recovery_required": False,
        "manual_reconciliation_required": False,
        "protection_reconciliation_required": False,
    }


def test_classifies_only_bot_attributed_stable_subminimum_position():
    dust = make_dust()
    assert dust["status"] == "VERIFIED"
    assert dust["quantity"] == "0.0000100001"
    assert Decimal(dust["mark_notional_usdc"]) <= Decimal("0.10")
    assert dust["position_read_count"] == 2
    assert dust["empty_open_order_read_count"] == 2
    assert dust["first_position_signed"] == "0.00001"
    assert dust["second_position_signed"] == "0.0000100001"


def test_rejects_direction_mismatch_or_unstable_position_reads():
    with pytest.raises(DustResidualError, match="match the bot entry direction"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00001",
            second_signed_position="-0.00001", min_trade_amount="0.0001",
            mark_price="2500", first_open_orders=[], second_open_orders=[],
            entry_order_id="entry", entry_order_state="filled",
            entry_filled_quantity="0.00002", prediction_id="pred",
        )
    with pytest.raises(DustResidualError, match="stable"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00001",
            second_signed_position="0.00002", min_trade_amount="0.0001",
            mark_price="2500", first_open_orders=[], second_open_orders=[],
            entry_order_id="entry", entry_order_state="filled",
            entry_filled_quantity="0.00002", prediction_id="pred",
        )


def test_rejects_open_orders_or_missing_entry_provenance():
    with pytest.raises(DustResidualError, match="empty open-order"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00001",
            second_signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", first_open_orders=[{"order_id": "still-open"}],
            second_open_orders=[], entry_order_id="entry", entry_order_state="filled",
            entry_filled_quantity="0.00002", prediction_id="pred",
        )
    with pytest.raises(DustResidualError, match="provenance"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00001",
            second_signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", first_open_orders=[], second_open_orders=[],
            entry_order_id="", entry_order_state="filled",
            entry_filled_quantity="0.00002", prediction_id="pred",
        )


def test_rejects_mark_notional_above_per_position_cap_and_underfilled_provenance():
    with pytest.raises(DustResidualError, match="notional"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00009",
            second_signed_position="0.00009", min_trade_amount="0.0001",
            mark_price="2000", first_open_orders=[], second_open_orders=[],
            entry_order_id="entry", entry_order_state="filled",
            entry_filled_quantity="0.0001", prediction_id="pred",
        )
    with pytest.raises(DustResidualError, match="smaller than the residual"):
        classify_verified_dust_residual(
            symbol="ETHUSDT", signal="BUY", first_signed_position="0.00001",
            second_signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", first_open_orders=[], second_open_orders=[],
            entry_order_id="entry", entry_order_state="filled",
            entry_filled_quantity="0.000001", prediction_id="pred",
        )


def test_valid_dust_record_revalidates_live_position_and_no_open_orders():
    result = validate_verified_dust_residual(
        make_record(), signed_position="0.00001", min_trade_amount="0.0001",
        mark_price="2500", open_orders=[],
    )
    assert result["reserve_usdc"] == Decimal("0.10")


def test_dust_revalidation_requires_recorded_order_and_prediction_identity():
    record = make_record()
    record["order_ids"]["entry"] = "other-order"
    with pytest.raises(DustResidualError, match="entry order does not match"):
        validate_verified_dust_residual(
            record, signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", open_orders=[],
        )
    record = make_record()
    record["pred_id"] = "other-prediction"
    with pytest.raises(DustResidualError, match="prediction does not match"):
        validate_verified_dust_residual(
            record, signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", open_orders=[],
        )


def test_dust_record_fails_closed_when_same_symbol_has_order_or_mark_moves_over_limit():
    record = make_record()
    with pytest.raises(DustResidualError, match="open or unverifiable order"):
        validate_verified_dust_residual(
            record, signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", open_orders=[{"order_id": "new-order"}],
        )
    with pytest.raises(DustResidualError, match="notional"):
        validate_verified_dust_residual(
            record, signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="11000", open_orders=[],
        )


def test_unresolved_trade_record_cannot_be_relabelled_as_dust():
    record = make_record()
    record["manual_reconciliation_required"] = True
    with pytest.raises(DustResidualError, match="recovery flags"):
        validate_verified_dust_residual(
            record, signed_position="0.00001", min_trade_amount="0.0001",
            mark_price="2500", open_orders=[],
        )


def test_nonzero_sub_1e8_residual_is_preserved_as_dust_not_flat():
    dust = classify_verified_dust_residual(
        symbol="ETHUSDT", signal="BUY",
        first_signed_position="0.00000001",
        second_signed_position="0.00000001",
        min_trade_amount="0.0001", mark_price="2500",
        first_open_orders=[], second_open_orders=[],
        entry_order_id="entry-dust-tiny", entry_order_state="cancelled",
        entry_filled_quantity="0.00001", prediction_id="pred-tiny",
    )
    assert dust["status"] == "VERIFIED"
    assert Decimal(dust["quantity"]) == Decimal("0.00000001")
