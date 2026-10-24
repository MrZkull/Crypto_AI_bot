from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from deribit_client import DeribitClient, PositionQueryError, OpenOrdersQueryError


CURRENCIES = ("USDC", "BTC", "ETH")
INSTRUMENT = "DOGE_USDC-PERPETUAL"


def make_client(response_by_currency=None, failure_by_currency=None):
    response_by_currency = response_by_currency or {currency: [] for currency in CURRENCIES}
    failure_by_currency = failure_by_currency or {}
    client = DeribitClient.__new__(DeribitClient)

    def fake_get(path, params):
        assert path == "/private/get_positions"
        currency = params["currency"]
        if currency in failure_by_currency:
            raise failure_by_currency[currency]
        return response_by_currency[currency]

    client._get = Mock(side_effect=fake_get)
    client.get_instrument_name = Mock(return_value=INSTRUMENT)
    return client


def test_successful_complete_snapshot_returns_nonzero_positions_only():
    open_position = {"instrument_name": INSTRUMENT, "size": 7.0, "size_currency": 0.7, "direction": "buy"}
    client = make_client({
        "USDC": [open_position, {"instrument_name": "ZERO_USDC-PERPETUAL", "size": 0, "size_currency": 0, "direction": "zero"}],
        "BTC": [],
        "ETH": [],
    })

    assert client.get_positions() == [open_position]
    assert client._get.call_count == 3


def test_any_currency_query_failure_invalidates_the_entire_snapshot():
    client = make_client(failure_by_currency={"BTC": RuntimeError("do-not-leak-this-detail")})

    with pytest.raises(PositionQueryError) as exc:
        client.get_positions()

    assert exc.value.failed_currencies == ("BTC",)
    assert "BTC" in str(exc.value)
    assert "do-not-leak-this-detail" not in str(exc.value)
    assert client._get.call_count == 3


def test_malformed_currency_response_is_not_treated_as_empty_positions():
    client = make_client({"USDC": {"positions": []}, "BTC": [], "ETH": []})

    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_positions()


def test_malformed_position_row_invalidates_snapshot():
    client = make_client({"USDC": ["not-a-position"], "BTC": [], "ETH": []})

    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_positions()


def test_zero_position_is_returned_only_after_all_queries_succeed():
    client = make_client()

    assert client.get_position_size("DOGEUSDT") == 0.0
    assert client._get.call_count == 3


def test_position_size_propagates_snapshot_failure_instead_of_returning_zero():
    client = make_client(failure_by_currency={"USDC": TimeoutError("network failure")})

    with pytest.raises(PositionQueryError):
        client.get_position_size("DOGEUSDT")


def test_position_row_missing_instrument_name_invalidates_snapshot():
    client = make_client({
        "USDC": [{"size": 3.0}],
        "BTC": [],
        "ETH": [],
    })

    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_position_size("DOGEUSDT")


def test_duplicate_instrument_across_currency_responses_invalidates_snapshot():
    row = {"instrument_name": INSTRUMENT, "size": 3.0, "size_currency": 0.3, "direction": "buy"}
    client = make_client({
        "USDC": [row],
        "BTC": [row],
        "ETH": [],
    })

    with pytest.raises(PositionQueryError, match="BTC"):
        client.get_positions()


def test_missing_or_null_position_size_invalidates_snapshot():
    for row in (
        {"instrument_name": INSTRUMENT},
        {"instrument_name": INSTRUMENT, "size": None},
    ):
        client = make_client({
            "USDC": [row],
            "BTC": [],
            "ETH": [],
        })
        with pytest.raises(PositionQueryError, match="USDC"):
            client.get_positions()


def test_nonfinite_base_quantity_is_rejected():
    client = DeribitClient.__new__(DeribitClient)
    row = {
        "instrument_name": "DOGE_USDC-PERPETUAL",
        "size": 1000,
        "size_currency": float("inf"),
        "direction": "buy",
        "mark_price": 0.2,
    }
    with pytest.raises(PositionQueryError, match="Base position quantity is non-finite"):
        client.position_size_from_record(row)


def test_position_query_error_suppresses_raw_exception_chaining():
    client = make_client(
        failure_by_currency={"USDC": RuntimeError("sensitive-api-detail")}
    )
    with pytest.raises(PositionQueryError) as exc:
        client.get_positions()

    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__ is True
    assert "sensitive-api-detail" not in str(exc.value)


def test_nonfinite_position_size_is_rejected():
    client = make_client({
        "USDC": [{"instrument_name": INSTRUMENT, "size": float("nan")}],
        "BTC": [],
        "ETH": [],
    })

    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_positions()


def test_inverse_instrument_is_not_normalized_as_linear_usdc_quantity():
    client = DeribitClient.__new__(DeribitClient)
    row = {
        "instrument_name": "BTC-PERPETUAL",
        "size": 50,
        "size_currency": 0.006687487,
        "mark_price": 7476.65,
    }
    with pytest.raises(PositionQueryError, match="Unsupported instrument"):
        client.position_size_from_record(row)


def _function_source(path: str, name: str) -> str:
    source = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"Missing function {name!r} in {path}")


def test_entry_guard_fails_closed_on_position_read_error():
    source = _function_source("trade_executor.py", "execute_trade")
    assert "real position check failed" in source
    assert "FAIL-CLOSED, skip entry" in source
    assert "real_pos = 0.0" not in source


def test_close_helper_never_falls_back_to_stale_tracked_quantity():
    source = _function_source("trade_executor.py", "_get_safe_close_info")
    assert "refusing tracked-quantity close fallback" in source
    assert "recorded_qty" not in source


def test_snapshot_failure_skips_whole_open_trade_management_and_ghost_cleanup():
    open_source = _function_source("trade_executor.py", "check_open_trades")
    ghost_source = _function_source("trade_executor.py", "clean_ghost_trades")
    assert "Open-trade management skipped" in open_source
    assert "Ghost-trade cleanup skipped" in ghost_source
    assert "return" in open_source
    assert "return" in ghost_source


def test_broken_ghost_record_is_retained_when_position_query_fails():
    source = _function_source("trade_executor.py", "clean_ghost_trades")
    assert "Broken ghost record %s retained: position read unavailable" in source
    assert "except Exception: actual = 0" not in source


def test_sub_lot_post_fill_position_uses_verified_recovery_path():
    source = _function_source("trade_executor.py", "execute_trade")
    assert "post-fill position below minimum lot" in source
    assert "_emergency_close_and_verify(" in source


def test_position_size_from_verified_record_normalizes_currency_units():
    client = DeribitClient.__new__(DeribitClient)
    row = {
        "instrument_name": "DOGE_USDC-PERPETUAL",
        "size": 1000,
        "size_currency": 5000.0,
        "direction": "buy",
        "mark_price": 0.2,
    }
    assert client.position_size_from_record(row) == 5000.0


def test_position_consumers_do_not_alias_inverse_contracts_to_linear_symbols():
    open_source = _function_source("trade_executor.py", "check_open_trades")
    ghost_source = _function_source("trade_executor.py", "clean_ghost_trades")
    executor = (Path(__file__).resolve().parents[1] / "trade_executor.py").read_text(encoding="utf-8")
    assert 'endswith("_USDC-PERPETUAL")' in open_source
    assert 'endswith("_USDC-PERPETUAL")' in ghost_source
    assert "unmanaged/non-linear position" in executor


def test_sell_direction_returns_negative_base_quantity():
    client = DeribitClient.__new__(DeribitClient)
    row = {
        "instrument_name": "DOGE_USDC-PERPETUAL",
        "size": 1000,
        "size_currency": 5.0,
        "direction": "sell",
    }
    assert client.position_size_from_record(row) == -5.0


def test_invalid_position_direction_is_rejected():
    client = DeribitClient.__new__(DeribitClient)
    row = {
        "instrument_name": "DOGE_USDC-PERPETUAL",
        "size": 1000,
        "size_currency": 5.0,
        "direction": "unknown",
    }
    with pytest.raises(PositionQueryError, match="direction"):
        client.position_size_from_record(row)


def test_zero_quote_size_cannot_hide_nonzero_base_position():
    client = make_client({
        "USDC": [{
            "instrument_name": INSTRUMENT,
            "size": 0,
            "size_currency": 5.0,
            "direction": "sell",
        }],
        "BTC": [],
        "ETH": [],
    })
    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_positions()


def test_negative_position_magnitudes_are_rejected():
    client = DeribitClient.__new__(DeribitClient)

    for row in (
        {
            "instrument_name": "DOGE_USDC-PERPETUAL",
            "size": -1000,
            "size_currency": 5.0,
            "direction": "buy",
        },
        {
            "instrument_name": "DOGE_USDC-PERPETUAL",
            "size": 1000,
            "size_currency": -5.0,
            "direction": "buy",
        },
    ):
        with pytest.raises(PositionQueryError, match="non-negative"):
            client.position_size_from_record(row)


def test_snapshot_rejects_negative_position_magnitudes():
    client = make_client({
        "USDC": [{
            "instrument_name": INSTRUMENT,
            "size": -1000,
            "size_currency": 5.0,
            "direction": "buy",
        }],
        "BTC": [],
        "ETH": [],
    })
    with pytest.raises(PositionQueryError, match="USDC"):
        client.get_positions()


def test_safe_close_error_does_not_chain_raw_exchange_exception():
    source = _function_source("trade_executor.py", "_get_safe_close_info")
    assert "refusing tracked-quantity close fallback" in source
    assert ") from None" in source
    assert "from last_error" not in source


def test_stale_time_exit_fails_closed_and_reads_before_cancelling_orders():
    source = _function_source("trade_executor.py", "check_stale_trades")
    assert "position read unavailable; stale-time exit deferred" in source
    read = source.find("_get_safe_close_info")
    cancel = source.find("_cancel_all_open_orders_for_symbol")
    assert read >= 0 and cancel > read


def test_tp2_does_not_remove_trade_when_flatness_is_unverified():
    source = _function_source("trade_executor.py", "check_open_trades")
    guard = source.find("TP2 observed but exchange is not verified flat")
    closed_flag = source.find('trade["tp2_hit"] = True')
    assert guard >= 0 and closed_flag > guard
    assert "retaining trade tracking" in source


def test_dashboard_kill_switch_uses_mapped_linear_positions_and_verifies():
    source = _function_source("dashboard.py", "api_kill_switch")
    assert 'endswith("_USDC-PERPETUAL")' in source
    assert "client.position_size_from_record(position)" in source
    assert "remaining = client.get_position_size(symbol)" in source
    assert "unverified_positions" in source
    assert "ignored_unmanaged_positions" in source


def test_dashboard_close_trade_verifies_flatness_before_deleting_tracker():
    source = _function_source("dashboard.py", "api_close_trade")
    verify = source.find("remaining = client.get_position_size(symbol)")
    delete = source.find("trades.pop(symbol, None)")
    assert verify >= 0 and delete > verify
    assert "trade record retained" in source


def test_flatness_verification_uses_base_quantity_epsilon():
    verify_source = _function_source("trade_executor.py", "_verify_actually_closed")
    close_source = _function_source("trade_executor.py", "_get_safe_close_info")
    open_source = _function_source("trade_executor.py", "check_open_trades")
    kill_source = _function_source("dashboard.py", "api_kill_switch")
    close_trade_source = _function_source("dashboard.py", "api_close_trade")

    assert "tolerance: float = 1e-8" in verify_source
    assert "abs(actual_pos) <= 1e-8" in close_source
    assert "real_pos_size <= 1e-8" in open_source
    assert "abs(signed_size) <= 1e-8" in kill_source
    assert "abs(remaining) > 1e-8" in kill_source
    assert "abs(position) > 1e-8" in close_trade_source
    assert "abs(remaining) > 1e-8" in close_trade_source


def test_orphan_reconciler_uses_the_verified_snapshot_row_without_requery():
    # Inspect the dedicated orphan-reconciliation source block.
    source_all = (Path(__file__).resolve().parents[1] / "trade_executor.py").read_text(encoding="utf-8")
    orphan_block = source_all[source_all.find("raw_positions = deribit.get_positions()")-250:source_all.find("if fetch_succeeded:", source_all.find("raw_positions = deribit.get_positions()"))]
    assert "position_size_from_record(position)" in orphan_block
    assert "get_position_size(sym)" not in orphan_block
    assert "normalized_positions = []" in orphan_block
    assert "preserving orphan tracker and taking no flatten action" in orphan_block



def test_open_orders_query_failure_is_not_misreported_as_empty():
    client = DeribitClient.__new__(DeribitClient)
    client.get_instrument_name = Mock(return_value=INSTRUMENT)
    client._get = Mock(side_effect=RuntimeError("private endpoint unavailable"))

    with pytest.raises(OpenOrdersQueryError) as exc:
        client.get_open_orders("DOGEUSDT")

    assert "private endpoint unavailable" not in str(exc.value)


def _load_cancel_open_orders_helper():
    source = _function_source("trade_executor.py", "_cancel_all_open_orders_for_symbol")
    namespace = {"log": Mock()}
    exec(compile("from __future__ import annotations\n" + source, "<cancel-helper>", "exec"), namespace)
    return namespace["_cancel_all_open_orders_for_symbol"]


def test_cancel_open_orders_helper_requires_verified_empty_book():
    cancel_helper = _load_cancel_open_orders_helper()

    class FakeExchange:
        def __init__(self):
            self.orders = [{"order_id": "order-1"}]

        def get_open_orders(self, symbol):
            return list(self.orders)

        def cancel_order(self, order_id):
            self.orders = [row for row in self.orders if row["order_id"] != order_id]

    exchange = FakeExchange()
    assert cancel_helper(exchange, "DOGEUSDT") is True
    assert exchange.orders == []


def test_cancel_open_orders_helper_fails_when_order_remains():
    cancel_helper = _load_cancel_open_orders_helper()

    class FakeExchange:
        def __init__(self):
            self.orders = [{"order_id": "order-1"}]

        def get_open_orders(self, symbol):
            return list(self.orders)

        def cancel_order(self, order_id):
            raise RuntimeError("cancel rejected")

    exchange = FakeExchange()
    assert cancel_helper(exchange, "DOGEUSDT") is False
    assert len(exchange.orders) == 1


def test_cancel_open_orders_helper_fails_when_final_read_is_unavailable():
    cancel_helper = _load_cancel_open_orders_helper()

    class FakeExchange:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self, symbol):
            self.calls += 1
            if self.calls == 1:
                return []
            raise RuntimeError("final order read failed")

        def cancel_order(self, order_id):
            raise AssertionError("there are no orders to cancel")

    exchange = FakeExchange()
    assert cancel_helper(exchange, "DOGEUSDT") is False


def test_emergency_close_helper_retains_tracking_unless_flat_and_orders_are_verified():
    source = _function_source("trade_executor.py", "_emergency_close_and_verify")
    assert "flat_verified and orders_cleared" in source
    assert "_verify_actually_closed(deribit, symbol)" in source
    assert "_cancel_all_open_orders_for_symbol(deribit, symbol)" in source
    assert "_retain_emergency_recovery_record(" in source


def test_emergency_execution_failures_use_common_verified_recovery_helper():
    source = _function_source("trade_executor.py", "execute_trade")
    assert source.count("_emergency_close_and_verify(") >= 6
    assert "Post-fill bracket invalid. Flattened." not in source
    assert "SL rejected. Position flattened." not in source


def test_corrupted_state_close_requires_flatness_and_order_cleanup_before_recording_closed():
    source = _function_source("trade_executor.py", "clean_ghost_trades")
    start = source.find('if float(trade.get("stop", 0)) == 0 or float(trade.get("tp1", 0)) == 0:')
    end = source.find('if float(trade.get("score", 1)) == 0 and float(trade.get("confidence", 1)) == 0:', start)
    assert start >= 0 and end > start
    block = source[start:end]
    assert "flat_verified = _verify_actually_closed(deribit, symbol)" in block
    assert "orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)" in block
    assert block.find("if flat_verified and orders_cleared:") < block.find('_close_record(')
    assert "corrupted-state close unverified; retaining trade tracking" in block


def test_time_and_stop_exit_keep_tracking_when_open_order_cleanup_fails():
    open_source = _function_source("trade_executor.py", "check_open_trades")
    stale_source = _function_source("trade_executor.py", "check_stale_trades")
    assert "stop-exit flatness or open-order cleanup unverified; retaining trade tracking" in open_source
    assert "stale-time exit flatness or order cleanup unverified; retaining trade tracking" in stale_source
