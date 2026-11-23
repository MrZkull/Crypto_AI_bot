from unittest.mock import Mock

import pytest

import trade_executor
from deribit_client import OrderOutcomeUnknown


def _signal(pred_id="pred-001", signal="BUY"):
    return {"pred_id": pred_id, "symbol": "ETHUSDT", "signal": signal, "entry": 2000.0}


def test_logical_entry_intent_is_stable_and_scoped_to_symbol_and_side():
    sig = _signal()
    first = trade_executor._logical_entry_intent_id(sig, "ETHUSDT", "BUY")
    repeated = trade_executor._logical_entry_intent_id(dict(sig), "ETHUSDT", "BUY")
    other_symbol = trade_executor._logical_entry_intent_id(sig, "BTCUSDT", "BUY")
    other_side = trade_executor._logical_entry_intent_id(sig, "ETHUSDT", "SELL")
    other_prediction = trade_executor._logical_entry_intent_id(_signal("pred-002"), "ETHUSDT", "BUY")
    assert first == repeated
    assert len({first, other_symbol, other_side, other_prediction}) == 4


def test_entry_attempt_labels_are_unique_per_attempt_and_deribit_safe():
    logical = trade_executor._logical_entry_intent_id(_signal(), "ETHUSDT", "BUY")
    first = trade_executor._entry_attempt_client_label(logical, "attempt-1")
    repeated = trade_executor._entry_attempt_client_label(logical, "attempt-1")
    second = trade_executor._entry_attempt_client_label(logical, "attempt-2")
    assert first == repeated
    assert first != second
    assert 1 <= len(first) <= 64
    assert first.isascii() and first.isprintable() and " " not in first


def test_first_entry_attempt_persists_unique_identity_and_preserves_old_history():
    previous = {"entry_attempt_history": [{
        "logical_intent_id": "old", "attempt_id": "old-1", "attempt_number": 1,
        "submission_state": "ZERO_FILL_VERIFIED", "zero_fill_position_unchanged": True,
    }]}
    attempt = trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)
    assert attempt["attempt_number"] == 1
    assert attempt["attempt_id"]
    assert attempt["client_label"].startswith("cb_")
    assert attempt["history"][0]["attempt_id"] == "old-1"
    assert attempt["history"][-1]["attempt_id"] == attempt["attempt_id"]
    record = trade_executor._build_entry_submission_intent(
        _signal(), "ETHUSDT", "BUY", 2000.0, 2001.0, 0.25,
        attempt["client_label"], attempt=attempt["attempt"], attempt_history=attempt["history"],
    )
    assert record["logical_intent_id"] == attempt["logical_intent_id"]
    assert record["entry_attempt_id"] == attempt["attempt_id"]
    assert record["entry_attempt_number"] == 1
    assert record["qty"] == 0.0 and record["filled_qty"] == 0.0
    assert record["entry_attempt_history"][-1]["submission_state"] == "SUBMITTING"


def test_same_logical_intent_replay_is_blocked_while_attempt_is_unresolved():
    first = trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25)
    previous = {
        "logical_intent_id": first["logical_intent_id"],
        "entry_attempt_history": first["history"],
    }
    with pytest.raises(trade_executor.EntryIntentReplayBlocked, match="unresolved"):
        trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)


def test_same_logical_intent_replay_needs_explicit_attempt_scoped_authorization(monkeypatch):
    first = trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25)
    prior = dict(first["attempt"])
    prior.update({
        "submission_state": "ZERO_FILL_VERIFIED",
        "zero_fill_position_unchanged": True,
        "terminal_state": "cancelled",
        "filled_qty": 0.0,
    })
    previous = {
        "logical_intent_id": first["logical_intent_id"],
        "entry_attempt_history": [prior],
    }
    monkeypatch.delenv("CRYPTOBOT_ENTRY_RETRY_AUTHORIZATION", raising=False)
    with pytest.raises(trade_executor.EntryIntentReplayBlocked, match="explicit"):
        trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)

    monkeypatch.setenv(
        "CRYPTOBOT_ENTRY_RETRY_AUTHORIZATION",
        f"{first['logical_intent_id']}:{first['attempt_id']}",
    )
    second = trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)
    assert second["attempt_number"] == 2
    assert second["attempt_id"] != first["attempt_id"]
    assert second["client_label"] != first["client_label"]
    assert second["attempt"]["retry_authorization"]["previous_attempt_id"] == first["attempt_id"]


def test_unknown_or_filled_prior_attempt_cannot_be_retried_even_with_authorization(monkeypatch):
    first = trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25)
    for status, zero_unchanged in (("UNKNOWN", False), ("FILLED", False), ("ZERO_FILL_VERIFIED", False)):
        prior = dict(first["attempt"])
        prior.update({"submission_state": status, "zero_fill_position_unchanged": zero_unchanged})
        previous = {"logical_intent_id": first["logical_intent_id"], "entry_attempt_history": [prior]}
        monkeypatch.setenv(
            "CRYPTOBOT_ENTRY_RETRY_AUTHORIZATION",
            f"{first['logical_intent_id']}:{first['attempt_id']}",
        )
        with pytest.raises(trade_executor.EntryIntentReplayBlocked):
            trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)


def test_legacy_same_prediction_record_without_verified_attempt_is_blocked():
    previous = {"pred_id": "pred-001", "signal": "BUY", "closed": True,
                "order_ids": {"entry_label": trade_executor._deterministic_order_label(_signal(), "ETHUSDT", "ENTRY")}}
    with pytest.raises(trade_executor.EntryIntentReplayBlocked, match="legacy"):
        trade_executor._prepare_entry_attempt(_signal(), "ETHUSDT", "BUY", 0.25, previous)


def test_legacy_replay_is_blocked_even_if_signal_direction_was_changed():
    previous = {"pred_id": "pred-001", "signal": "BUY", "closed": True}
    with pytest.raises(trade_executor.EntryIntentReplayBlocked, match="legacy"):
        trade_executor._prepare_entry_attempt(_signal(signal="SELL"), "ETHUSDT", "SELL", 0.25, previous)


def test_zero_fill_position_check_requires_unchanged_signed_position():
    deribit = Mock()
    deribit.get_position_size.return_value = 0.125
    evidence = trade_executor._verify_entry_position_unchanged(deribit, "ETHUSDT", 0.125)
    assert evidence["verified"] is True
    assert evidence["position_before"] == evidence["position_after"] == 0.125
    deribit.get_position_size.return_value = 0.25
    with pytest.raises(OrderOutcomeUnknown, match="position changed"):
        trade_executor._verify_entry_position_unchanged(deribit, "ETHUSDT", 0.125)


def test_zero_fill_position_check_fails_closed_on_read_error():
    deribit = Mock()
    deribit.get_position_size.side_effect = TimeoutError("position read unavailable")
    with pytest.raises(OrderOutcomeUnknown, match="unavailable"):
        trade_executor._verify_entry_position_unchanged(deribit, "ETHUSDT", 0.0)


def _order(order_id="order-1", **overrides):
    value = {
        "order_id": order_id,
        "instrument_name": "ETH_USDC-PERPETUAL",
        "direction": "buy",
        "amount": 0.25,
        "order_type": "limit",
        "time_in_force": "immediate_or_cancel",
        "reduce_only": False,
        "label": "cb_attempt_label",
        "price": 2004.0,
        "order_state": "filled",
        "filled_amount": 0.25,
    }
    value.update(overrides)
    return value


_EXPECTED = {
    "instrument_name": "ETH_USDC-PERPETUAL",
    "direction": "buy",
    "amount": 0.25,
    "order_type": "limit",
    "time_in_force": "immediate_or_cancel",
    "reduce_only": False,
    "label": "cb_attempt_label",
    "price": 2004.0,
}


@pytest.mark.parametrize("field,value", [
    ("instrument_name", "BTC_USDC-PERPETUAL"),
    ("direction", "sell"),
    ("amount", 0.5),
    ("order_type", "market"),
    ("time_in_force", "good_til_cancelled"),
    ("reduce_only", True),
    ("label", "cb_other_attempt"),
    ("price", 1999.0),
])
def test_entry_order_contract_mismatch_is_unknown(field, value):
    immediate = _order()
    observed = _order(**{field: value})
    with pytest.raises(OrderOutcomeUnknown):
        trade_executor._assert_entry_order_contract(immediate, observed, _EXPECTED)


def test_entry_order_contract_requires_all_fields_in_exchange_readback():
    immediate = _order()
    observed = _order()
    observed.pop("time_in_force")
    with pytest.raises(OrderOutcomeUnknown, match="time_in_force"):
        trade_executor._assert_entry_order_contract(immediate, observed, _EXPECTED)


def test_entry_order_contract_accepts_exact_independent_readback():
    immediate = _order()
    observed = _order()
    trade_executor._assert_entry_order_contract(immediate, observed, _EXPECTED)


def test_confirm_terminal_entry_order_checks_full_contract_before_accepting_fill():
    deribit = Mock()
    deribit.get_order.return_value = _order()
    result = {"order": _order()}
    terminal = trade_executor._confirm_terminal_entry_order(
        deribit, result, 0.25, expected_contract=_EXPECTED
    )
    assert terminal["order_id"] == "order-1"
    deribit.cancel_order.assert_not_called()


def test_confirm_terminal_entry_order_contract_mismatch_stays_unknown_without_cancel():
    deribit = Mock()
    deribit.get_order.return_value = _order(direction="sell")
    result = {"order": _order()}
    with pytest.raises(OrderOutcomeUnknown, match="direction"):
        trade_executor._confirm_terminal_entry_order(
            deribit, result, 0.25, expected_contract=_EXPECTED
        )
    deribit.cancel_order.assert_not_called()


def test_confirm_terminal_entry_order_rechecks_contract_after_one_cancel():
    deribit = Mock()
    deribit.get_order.side_effect = [
        _order(order_state="open", filled_amount=0.0),
        _order(order_state="cancelled", filled_amount=0.0),
    ]
    result = {"order": _order()}
    terminal = trade_executor._confirm_terminal_entry_order(
        deribit, result, 0.25, expected_contract=_EXPECTED
    )
    assert terminal["order_state"] == "cancelled"
    deribit.cancel_order.assert_called_once_with("order-1")
    assert deribit.get_order.call_count == 2


def test_confirm_terminal_entry_order_contract_mismatch_after_cancel_is_unknown():
    deribit = Mock()
    deribit.get_order.side_effect = [
        _order(order_state="open", filled_amount=0.0),
        _order(order_state="cancelled", filled_amount=0.0, amount=0.5),
    ]
    result = {"order": _order()}
    with pytest.raises(OrderOutcomeUnknown, match="amount"):
        trade_executor._confirm_terminal_entry_order(
            deribit, result, 0.25, expected_contract=_EXPECTED
        )
    deribit.cancel_order.assert_called_once_with("order-1")


def test_attempt_history_update_requires_exactly_one_attempt_record():
    record = {"entry_attempt_history": [{"attempt_id": "attempt-1", "submission_state": "SUBMITTING"}]}
    contract = {"instrument_name": "ETH_USDC-PERPETUAL", "amount": 0.25, "price": 2004.0}
    trade_executor._update_entry_attempt_history(
        record, "attempt-1", submission_state="ACKNOWLEDGED", order_contract=contract
    )
    assert record["entry_attempt_history"][0]["submission_state"] == "ACKNOWLEDGED"
    assert record["entry_attempt_history"][0]["order_contract"] == contract
    assert record["entry_attempt"]["attempt_id"] == "attempt-1"
    with pytest.raises(Exception, match="non-unique"):
        trade_executor._update_entry_attempt_history(
            {"entry_attempt_history": [{"attempt_id": "x"}, {"attempt_id": "x"}]},
            "x", submission_state="UNKNOWN",
        )


def _persisted_submitting_record():
    sig = _signal()
    identity = trade_executor._prepare_entry_attempt(sig, "ETHUSDT", "BUY", 0.25)
    label = identity["client_label"]
    identity["attempt"]["order_contract"] = {"label": label, "amount": 0.25}
    identity["history"][-1]["order_contract"] = {"label": label, "amount": 0.25}
    record = {
        "symbol": "ETHUSDT",
        "signal": "BUY",
        "pred_id": sig["pred_id"],
        "logical_intent_id": identity["logical_intent_id"],
        "entry_attempt_id": identity["attempt_id"],
        "entry_attempt_number": identity["attempt_number"],
        "entry_attempt": dict(identity["attempt"]),
        "entry_attempt_history": [dict(item) for item in identity["history"]],
        "entry_order_contract": {"label": label, "amount": 0.25},
        "order_ids": {"entry_label": label},
        "order_client_labels": {"entry": label},
        "order_submission_states": {"entry": "SUBMITTING"},
        "entry_submission_state": "SUBMITTING",
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
        "protection_reconciliation_reason": "entry submission in progress",
        "closed": False,
        "qty": 0.0,
        "filled_qty": 0.0,
    }
    return sig, identity, record


def test_restart_resume_reuses_same_attempt_and_stays_unknown_when_no_match(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    deribit = Mock()
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    saved = {}
    monkeypatch.setattr(
        trade_executor, "_persist_trade_snapshot",
        lambda symbol, updated: saved.update(symbol=symbol, record=updated) or True,
    )
    reconcile = Mock(return_value={
        "read_only": True, "matched_order_ids": [], "read_errors": [],
        "order_state_matches": [], "open_order_matches": [],
    })
    monkeypatch.setattr(trade_executor, "_reconcile_unknown_entry_by_label_read_only", reconcile)

    handled = trade_executor._resume_pending_entry_attempt_read_only(deribit, sig)

    assert handled is True
    reconcile.assert_called_once_with(deribit, "ETHUSDT", identity["client_label"])
    updated = saved["record"]
    attempt = updated["entry_attempt_history"][-1]
    assert attempt["attempt_id"] == identity["attempt_id"]
    assert attempt["client_label"] == identity["client_label"]
    assert attempt["submission_state"] == "UNKNOWN"
    assert updated["entry_submission_state"] == "UNKNOWN"
    assert updated["manual_reconciliation_required"] is True
    assert updated["protection_reconciliation_required"] is True
    assert updated["qty"] == 0.0 and updated["filled_qty"] == 0.0
    deribit.place_market_order.assert_not_called()
    deribit.cancel_order.assert_not_called()


def test_restart_resume_records_matching_order_but_never_auto_resolves(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    saved = {}
    monkeypatch.setattr(
        trade_executor, "_persist_trade_snapshot",
        lambda symbol, updated: saved.update(record=updated) or True,
    )
    monkeypatch.setattr(
        trade_executor,
        "_reconcile_unknown_entry_by_label_read_only",
        lambda deribit, symbol, label: {
            "read_only": True,
            "matched_order_ids": ["exchange-order-77"],
            "read_errors": [],
        },
    )

    assert trade_executor._resume_pending_entry_attempt_read_only(Mock(), sig) is True
    updated = saved["record"]
    assert updated["order_ids"]["entry"] == "exchange-order-77"
    assert updated["entry_reconciliation_matched_order_ids"] == ["exchange-order-77"]
    assert updated["entry_submission_state"] == "UNKNOWN"
    assert updated["manual_reconciliation_required"] is True
    assert updated["closed"] is False
    assert updated["qty"] == 0.0


def test_restart_resume_identity_mismatch_blocks_before_exchange_read(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    record["order_ids"]["entry_label"] = "cb_wrong_label"
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    reconcile = Mock()
    monkeypatch.setattr(trade_executor, "_reconcile_unknown_entry_by_label_read_only", reconcile)
    persist = Mock(return_value=True)
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", persist)

    assert trade_executor._resume_pending_entry_attempt_read_only(Mock(), sig) is True
    reconcile.assert_not_called()
    persist.assert_not_called()


def test_restart_resume_does_not_reconcile_a_resolved_attempt(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    record["entry_attempt_history"][-1]["submission_state"] = "ZERO_FILL_VERIFIED"
    record["entry_attempt"]["submission_state"] = "ZERO_FILL_VERIFIED"
    record["entry_submission_state"] = "ZERO_FILL_VERIFIED"
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    reconcile = Mock()
    monkeypatch.setattr(trade_executor, "_reconcile_unknown_entry_by_label_read_only", reconcile)

    assert trade_executor._resume_pending_entry_attempt_read_only(Mock(), sig) is False
    reconcile.assert_not_called()


def test_execute_trade_resumes_pending_attempt_before_new_entry_mode_gate(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    deribit = Mock()
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    monkeypatch.setattr(trade_executor, "_persist_trade_snapshot", lambda *args, **kwargs: True)
    reconcile = Mock(return_value={"read_only": True, "matched_order_ids": [], "read_errors": []})
    monkeypatch.setattr(trade_executor, "_reconcile_unknown_entry_by_label_read_only", reconcile)
    monkeypatch.setattr(
        trade_executor, "_deribit_testnet_mutations_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("new-entry gate must not run")),
    )
    monkeypatch.setattr(
        trade_executor, "_execution_allows_new_entries",
        lambda: (_ for _ in ()).throw(AssertionError("execution gate must not run")),
    )

    result = trade_executor.execute_trade(
        deribit, sig, risk_mult=1.0, balance=100.0, base_min_conf=50.0
    )

    assert result is False
    reconcile.assert_called_once_with(deribit, "ETHUSDT", identity["client_label"])
    deribit.place_market_order.assert_not_called()
    deribit.cancel_order.assert_not_called()


def test_unresolved_entry_attempt_blocks_account_even_when_trade_row_is_closed():
    assert trade_executor._has_unresolved_entry_attempt({
        "closed": True,
        "manual_reconciliation_required": False,
        "entry_attempt_history": [{
            "attempt_id": "attempt-9", "submission_state": "SUBMITTING",
        }],
    }) is True


def test_malformed_entry_attempt_history_fails_closed():
    assert trade_executor._has_unresolved_entry_attempt({
        "closed": True, "entry_attempt_history": "corrupt",
    }) is True
    assert trade_executor._has_unresolved_entry_attempt({
        "closed": True, "entry_attempt_history": [None],
    }) is True


def test_terminal_entry_attempt_history_does_not_block_account():
    assert trade_executor._has_unresolved_entry_attempt({
        "closed": True,
        "entry_attempt_history": [{
            "attempt_id": "attempt-9", "submission_state": "ZERO_FILL_VERIFIED",
        }],
    }) is False


def test_restart_resume_also_reconciles_crash_after_acknowledgement(monkeypatch):
    sig, identity, record = _persisted_submitting_record()
    record["entry_attempt_history"][-1]["submission_state"] = "ACKNOWLEDGED"
    record["entry_attempt"]["submission_state"] = "ACKNOWLEDGED"
    record["entry_submission_state"] = "ACKNOWLEDGED"
    record["order_submission_states"]["entry"] = "ACKNOWLEDGED"
    monkeypatch.setattr(trade_executor, "load_trades", lambda: {"ETHUSDT": record})
    saved = {}
    monkeypatch.setattr(
        trade_executor, "_persist_trade_snapshot",
        lambda symbol, updated: saved.update(record=updated) or True,
    )
    reconcile = Mock(return_value={"read_only": True, "matched_order_ids": ["order-ack"], "read_errors": []})
    monkeypatch.setattr(trade_executor, "_reconcile_unknown_entry_by_label_read_only", reconcile)
    deribit = Mock()

    assert trade_executor._resume_pending_entry_attempt_read_only(deribit, sig) is True
    reconcile.assert_called_once_with(deribit, "ETHUSDT", identity["client_label"])
    assert saved["record"]["entry_submission_state"] == "UNKNOWN"
    assert saved["record"]["order_ids"]["entry"] == "order-ack"
    assert saved["record"]["entry_attempt_history"][-1]["attempt_id"] == identity["attempt_id"]


def test_acknowledged_attempt_is_account_level_unresolved_state():
    assert trade_executor._has_unresolved_entry_attempt({
        "closed": True,
        "entry_attempt_history": [{"attempt_id": "ack-1", "submission_state": "ACKNOWLEDGED"}],
    }) is True
