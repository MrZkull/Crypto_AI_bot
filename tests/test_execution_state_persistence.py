from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import execution_state_persistence as persistence
from execution_state_persistence import (
    StateFileCorrupt,
    StateMirrorMismatch,
    StatePersistenceError,
    read_json_mirrors,
    write_json_mirrors,
)


def test_writes_and_verifies_all_state_mirrors(tmp_path):
    root = tmp_path / "trades.json"
    data = tmp_path / "data" / "trades.json"
    payload = {"ETHUSDT": {"pred_id": "p-1", "qty": 0.4, "state": "UNKNOWN"}}

    receipt = write_json_mirrors((root, data), payload)

    assert receipt.sha256
    assert receipt.size_bytes > 0
    assert root.read_bytes() == data.read_bytes()
    assert read_json_mirrors((root, data)) == payload


def test_divergent_existing_mirrors_fail_closed_without_overwrite(tmp_path):
    root = tmp_path / "trades.json"
    data = tmp_path / "data" / "trades.json"
    root.parent.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True)
    root.write_text('{"ETHUSDT":{"state":"UNKNOWN"}}', encoding="utf-8")
    data.write_text('{"ETHUSDT":{"state":"CLOSED"}}', encoding="utf-8")
    before = (root.read_bytes(), data.read_bytes())

    with pytest.raises(StateMirrorMismatch):
        read_json_mirrors((root, data), default={})
    with pytest.raises(StateMirrorMismatch):
        write_json_mirrors((root, data), {"ETHUSDT": {"state": "ACKNOWLEDGED"}})

    assert (root.read_bytes(), data.read_bytes()) == before


def test_corrupt_or_duplicate_json_fails_closed(tmp_path):
    path = tmp_path / "trades.json"
    path.write_text('{"ETHUSDT": {"state": "UNKNOWN", "state": "CLOSED"}}', encoding="utf-8")

    with pytest.raises(StateFileCorrupt):
        read_json_mirrors((path,), default={})

    path.write_text('{"ETHUSDT": NaN}', encoding="utf-8")
    with pytest.raises(StateFileCorrupt):
        read_json_mirrors((path,), default={})


def test_nonfinite_payload_is_rejected_before_existing_state_changes(tmp_path):
    root = tmp_path / "trades.json"
    data = tmp_path / "data" / "trades.json"
    initial = {"ETHUSDT": {"state": "ACKNOWLEDGED"}}
    write_json_mirrors((root, data), initial)
    before = (root.read_bytes(), data.read_bytes())

    with pytest.raises(StatePersistenceError):
        write_json_mirrors((root, data), {"ETHUSDT": {"qty": float("nan")}})

    assert (root.read_bytes(), data.read_bytes()) == before


def test_failed_second_mirror_replace_raises_and_next_read_detects_divergence(tmp_path, monkeypatch):
    root = tmp_path / "trades.json"
    data = tmp_path / "data" / "trades.json"
    write_json_mirrors((root, data), {"ETHUSDT": {"state": "OLD"}})
    real_replace = os.replace
    calls = {"count": 0}

    def fail_second_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated mirror failure")
        return real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_second_replace)
    with pytest.raises(StatePersistenceError):
        write_json_mirrors((root, data), {"ETHUSDT": {"state": "NEW"}})

    monkeypatch.setattr(persistence.os, "replace", real_replace)
    with pytest.raises(StateMirrorMismatch):
        read_json_mirrors((root, data), default={})


def test_single_legacy_mirror_can_be_read_then_repaired(tmp_path):
    root = tmp_path / "trades.json"
    data = tmp_path / "data" / "trades.json"
    payload = {"BTCUSDT": {"state": "PARTIALLY_FILLED", "filled_qty": 0.1}}
    root.write_text(json.dumps(payload), encoding="utf-8")

    assert read_json_mirrors((root, data), default={}) == payload
    write_json_mirrors((root, data), payload)
    assert root.read_bytes() == data.read_bytes()


def test_symlinked_state_path_fails_closed(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "trades-link.json"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")

    with pytest.raises(StatePersistenceError):
        read_json_mirrors((link,), default={})
