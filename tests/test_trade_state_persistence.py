from __future__ import annotations

import json

import trade_executor


def test_persist_trade_snapshot_writes_full_verified_mirrors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trade = {
        "pred_id": "pred-2026-11-16-001",
        "symbol": "ETHUSDT",
        "qty": 0.4,
        "filled_qty": 0.4,
        "entry_submission_state": "PARTIALLY_FILLED",
        "order_submission_states": {"entry": "PARTIALLY_FILLED", "stop": "UNKNOWN"},
        "order_client_labels": {"entry": "cb_entry_001", "stop": "cb_stop_001"},
        "order_ids": {"entry": "exchange-1"},
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
        "reconciliation_evidence": {"recent_complete": False, "historical_complete": False},
    }

    assert trade_executor._persist_trade_snapshot("ETHUSDT", trade) is True
    root = json.loads((tmp_path / "trades.json").read_text(encoding="utf-8"))
    mirror = json.loads((tmp_path / "data" / "trades.json").read_text(encoding="utf-8"))
    assert root == mirror
    assert root["ETHUSDT"] == trade


def test_persist_trade_snapshot_refuses_divergent_mirrors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "trades.json").write_text('{"ETHUSDT":{"state":"OLD"}}', encoding="utf-8")
    (tmp_path / "data" / "trades.json").write_text('{"ETHUSDT":{"state":"OTHER"}}', encoding="utf-8")

    trade = {"pred_id": "new", "order_submission_states": {"entry": "SUBMITTING"}}
    assert trade_executor._persist_trade_snapshot("ETHUSDT", trade) is False
    assert json.loads((tmp_path / "trades.json").read_text(encoding="utf-8")) == {"ETHUSDT": {"state": "OLD"}}
    assert json.loads((tmp_path / "data" / "trades.json").read_text(encoding="utf-8")) == {"ETHUSDT": {"state": "OTHER"}}


def test_emergency_recovery_record_uses_strict_verified_mirrors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trade_executor, "_send", lambda *_args, **_kwargs: None)

    trades = {}
    trade_executor._retain_emergency_recovery_record(
        trades,
        {"entry": 2000.0, "pred_id": "pred-emergency-001", "confidence": 81.0},
        "ETHUSDT",
        "BUY",
        2000.0,
        0.4,
        {"entry_label": "cb_test_emergency"},
        "entry outcome unknown",
        exchange_quantity_verified=False,
        reconciliation_evidence={"read_only": True, "history_complete": False},
    )

    root = json.loads((tmp_path / "trades.json").read_text(encoding="utf-8"))
    mirror = json.loads((tmp_path / "data" / "trades.json").read_text(encoding="utf-8"))
    record = root["ETHUSDT"]
    assert root == mirror
    assert record["emergency_recovery_required"] is True
    assert record["manual_reconciliation_required"] is True
    assert record["exchange_quantity_verified"] is False
    assert record["quantity_source"] == "unknown_unreconciled"
    assert record["qty"] == 0.0
    assert record["qty_tp1"] == 0.0
    assert record["qty_tp2"] == 0.0
    assert record["order_ids"]["entry_label"] == "cb_test_emergency"


def test_emergency_recovery_refuses_to_overwrite_divergent_mirrors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trade_executor, "_send", lambda *_args, **_kwargs: None)
    (tmp_path / "data").mkdir()
    root_before = b'{"ETHUSDT":{"state":"ROOT"}}'
    data_before = b'{"ETHUSDT":{"state":"DATA"}}'
    (tmp_path / "trades.json").write_bytes(root_before)
    (tmp_path / "data" / "trades.json").write_bytes(data_before)

    trade_executor._retain_emergency_recovery_record(
        {},
        {"entry": 2000.0, "pred_id": "pred-emergency-002"},
        "ETHUSDT",
        "BUY",
        2000.0,
        0.4,
        {"entry_label": "cb_test_emergency_2"},
        "position snapshot unavailable",
        exchange_quantity_verified=False,
        reconciliation_evidence={"read_only": True},
    )

    assert (tmp_path / "trades.json").read_bytes() == root_before
    assert (tmp_path / "data" / "trades.json").read_bytes() == data_before
