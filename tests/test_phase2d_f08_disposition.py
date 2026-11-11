from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import phase2d_f08_disposition as disposition
import phase2d_f08_resolution_provenance as resolution_provenance
from phase2d_f08_disposition import DispositionError, DispositionSpec


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _state(tmp_path):
    state = {
        "schema_version": 6,
        "locked_models": {
            "experiment_id": "TEST-F08",
            "validator_code_hash": "old-validator-hash",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
        },
        "experiment_definition": {"identity": {"validator_code_hash": "old-validator-hash"}},
        "experiment_started_at": "2026-01-01T00:00:00+00:00",
        "pending": {"candidate": [], "production": []},
        "resolved": {"candidate": [], "production": []},
        "seen": {"candidate": [], "production": []},
    }
    for model, count in (("candidate", 2), ("production", 3)):
        for i in range(count):
            row = {
                "id": f"{model}:BTCUSDT:{1_700_000_000_000 + i * 900_000}",
                "model": model,
                "symbol": "BTCUSDT",
                "open_time": 1_700_000_000_000 + i * 900_000,
                "created_at": "2026-10-04T00:00:00+00:00",
                "entry": 100.0 + i,
                "atr": 1.25,
                "confidence": 71.2,
                "signal": "BUY",
                "custom_original_field": {"x": [1, 2, 3]},
            }
            state["pending"][model].append(row)
            state["seen"][model].append(f"BTCUSDT:{row['open_time']}")
        future = {
            "id": f"{model}:ETHUSDT:1800000000000",
            "model": model,
            "symbol": "ETHUSDT",
            "open_time": 1_800_000_000_000,
            "created_at": "2026-10-20T00:00:00+00:00",
            "entry": 2000.0,
            "atr": 2.0,
            "confidence": 60.0,
            "signal": "SELL",
        }
        state["pending"][model].append(future)
        state["seen"][model].append(f"ETHUSDT:{future['open_time']}")
        state["resolved"][model].append({
            "id": f"{model}:SOLUSDT:1600000000000",
            "model": model,
            "symbol": "SOLUSDT",
            "open_time": 1_600_000_000_000,
            "status": "TP",
            "net_r": 1.0,
        })
        state["seen"][model].append("SOLUSDT:1600000000000")
    return state


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _approval_evidence(path, environment, workflow, workflow_file, run_id):
    owner_id = disposition.OWNER_ID
    context = {
        "repository": "MrZkull/Crypto_AI_bot",
        "workflow": workflow,
        "workflow_file": workflow_file,
        "event_name": "workflow_dispatch",
        "ref": "refs/heads/main",
        "run_id": str(run_id),
        "run_attempt": 1,
        "head_sha": "a" * 40,
        "actor_login": "automation-operator",
        "actor_id": "11223344",
    }
    env = {
        "name": environment,
        "protection_rules": [{
            "type": "required_reviewers",
            "prevent_self_review": True,
            "reviewers": [{"type": "User", "reviewer": {"login": "MrZkull", "id": int(owner_id)}}],
        }],
        "deployment_branch_policy": {"protected_branches": True, "custom_branch_policies": False},
    }
    approvals = [{
        "state": "approved",
        "user": {"login": "MrZkull", "id": int(owner_id)},
        "environments": [{"name": environment}],
    }]
    _write_json(path, {"context": context, "environment": env, "approval_history": approvals})
    return owner_id


@pytest.fixture()
def fixture_set(tmp_path):
    root = tmp_path
    state_path = root / "phase2d_state.json"
    ledger_path = root / "phase2d_observations.jsonl"
    journal_path = root / "experiment_ledger.jsonl"
    baseline_state_path = root / "baseline_state.json"
    baseline_ledger_path = root / "baseline_observations.jsonl"
    baseline_journal_path = root / "baseline_experiment_ledger.jsonl"
    validator_path = root / "phase2d_prospective_validator.py"
    gate_path = root / "phase2d_promotion_gate.py"
    manifest_path = root / "manifest.json"
    bundle_path = root / "observations.jsonl"
    amendment_approval_path = root / "amendment_approval.json"
    disposition_approval_path = root / "disposition_approval.json"

    state = _state(root)
    # The pinned source hash fields are injected after the bytes needed for the spec are known.
    old_validator = "old-validator-hash"
    state["locked_models"]["validator_code_hash"] = old_validator
    state_raw = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode()
    baseline_state_path.write_bytes(state_raw)
    state_path.write_bytes(state_raw)

    baseline_ledger_rows = [
        {"record_type": "event", "ledger_schema_version": 1, "event": "F08_LEDGER_INITIALIZED_FROM_STATE", "experiment_id": "TEST-F08"},
        {"record_type": "event", "ledger_schema_version": 1, "event": "F08_LEDGER_MIGRATION_METADATA_CORRECTED", "experiment_id": "TEST-F08"},
        {"record_type": "event", "ledger_schema_version": 1, "event": "F08_LEDGER_HISTORICAL_RECONCILIATION", "experiment_id": "TEST-F08", "remaining_missing": 0},
    ]
    _write_jsonl(baseline_ledger_path, baseline_ledger_rows)
    ledger_path.write_bytes(baseline_ledger_path.read_bytes())

    baseline_journal_rows = [
        {"experiment_id": "TEST-F08", "event": "PHASE2D_RUN", "run": 1},
        {"experiment_id": "TEST-F08", "event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 15, "decision": "CHECKPOINT_CONTINUE"},
        {"experiment_id": "TEST-F08", "event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 20, "decision": "CHECKPOINT_CONTINUE"},
        {"experiment_id": "TEST-F08", "event": "F08_GATE_AMENDMENT", "amendment": "approved-previous-gate-amendment"},
    ]
    _write_jsonl(baseline_journal_path, baseline_journal_rows)
    journal_path.write_bytes(baseline_journal_path.read_bytes())

    validator_path.write_text("# validator candidate\nSCHEMA = 7\n", encoding="utf-8")
    gate_path.write_text("# frozen promotion gate\nGATE = 'frozen'\n", encoding="utf-8")

    # Use a small injected spec while retaining the production algorithm.
    ids_by_model = {
        model: sorted(
            row["id"] for row in state["pending"][model]
            if row["created_at"] <= disposition.LEGACY_CUTOFF_UTC
        )
        for model in ("candidate", "production")
    }
    test_pending_digest = _sha(_canonical(ids_by_model))
    population = disposition.pending_population(state, DispositionSpec(
        experiment_id="TEST-F08",
        reconciliation_id="TEST-RECON-1",
        state_sha256=_sha(state_raw),
        ledger_sha256=_sha(baseline_ledger_path.read_bytes()),
        journal_sha256=_sha(baseline_journal_path.read_bytes()),
        manifest_sha256="0" * 64,
        bundle_sha256="0" * 64,
        pending_ids_sha256=test_pending_digest,
        old_validator_sha256=old_validator,
        promotion_gate_sha256=disposition.canonical_source_sha256(gate_path),
        counts={"candidate": 2, "production": 3},
        total=5,
        manifest_path="research_outputs/f08_historical_backfill/manifest.json",
    ))

    bundle_rows = []
    for model in ("candidate", "production"):
        for i, prediction in enumerate(population["payloads_by_model"][model]):
            bundle_rows.append({
                "record_type": "historical_backfill_observation",
                "evidence_class": "HISTORICAL_BACKFILL",
                "experiment_id": "TEST-F08",
                "reconciliation_id": "TEST-RECON-1",
                "model": model,
                "id": prediction["id"],
                "symbol": prediction["symbol"],
                "open_time": prediction["open_time"],
                "status": "TP" if i == 0 else "SL",
                "outcome_bar": 1,
                "gross_r": 2.0,
                "net_r": 1.8,
            })
    bundle_raw = b"".join(_canonical(row) + b"\n" for row in bundle_rows)
    bundle_path.write_bytes(bundle_raw)

    evidence_dir = root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    raw_windows = {}
    for row in bundle_rows:
        key = (row["symbol"], int(row["open_time"]))
        raw_windows.setdefault(key, {
            "symbol": row["symbol"],
            "open_time": int(row["open_time"]),
            "request": {"resolution": "15"},
            "candles": [
                {"timestamp": int(row["open_time"]) + bar * 900_000, "high": 101.0, "low": 99.0}
                for bar in range(26)
            ],
        })
    raw_windows_path = evidence_dir / "raw_windows.jsonl"
    raw_windows_raw = b"".join(
        _canonical(row) + b"\n" for _, row in sorted(raw_windows.items())
    )
    raw_windows_path.write_bytes(raw_windows_raw)
    raw_windows_index = resolution_provenance.index_raw_windows(
        [json.loads(line) for line in raw_windows_raw.splitlines()]
    )
    resolver_rows = [
        {**row, "resolved_at": "2026-11-10T12:00:00+00:00"}
        for row in bundle_rows
    ]
    provenance_rows = resolution_provenance.build_resolution_provenance_records(
        resolver_rows, raw_windows_index, "TEST-RECON-1",
        expected_counts={"candidate": 2, "production": 3},
    )
    provenance_raw = resolution_provenance.canonical_jsonl_bytes(provenance_rows)
    provenance_path = root / resolution_provenance.PROVENANCE_FILENAME
    provenance_path.write_bytes(provenance_raw)

    source_hashes = {
        "state": _sha(state_raw),
        "ledger": _sha(baseline_ledger_path.read_bytes()),
        "journal": _sha(baseline_journal_path.read_bytes()),
        "raw_windows": _sha(raw_windows_raw),
    }
    manifest = {
        "manifest_schema_version": 1,
        "operation": "STALE_PENDING_NOT_SCORED",
        "experiment_id": "TEST-F08",
        "evidence_class": "HISTORICAL_BACKFILL",
        "legacy_cutoff_utc": disposition.LEGACY_CUTOFF_UTC,
        "manifest_status": "READY_FOR_OWNER_REVIEW",
        "approval_status": "PENDING_OWNER_APPROVAL",
        "reconciliation_id": "TEST-RECON-1",
        "source_hashes": source_hashes,
        "source_artifacts": {
            "raw_windows": {
                "path": "evidence/raw_windows.jsonl",
                "sha256": source_hashes["raw_windows"],
            },
        },
        "resolution_provenance": {
            "schema_version": 1,
            "format": "jsonl",
            "path": resolution_provenance.PROVENANCE_FILENAME,
            "sha256": _sha(provenance_raw),
            "record_count": 5,
            "raw_windows_sha256": source_hashes["raw_windows"],
            "resolved_at_semantics": resolution_provenance.RESOLVED_AT_SEMANTICS,
            "outcome_bar_close_time_semantics": resolution_provenance.OUTCOME_BAR_CLOSE_TIME_SEMANTICS,
        },
        "historical_bundle": {
            "evidence_class": "HISTORICAL_BACKFILL",
            "format": "jsonl",
            "bundle_path": "observations.jsonl",
            "bundle_sha256": _sha(bundle_raw),
            "observation_count": 5,
            "candidate_count": 2,
            "production_count": 3,
        },
        "pending_population": {
            "ids_by_model": population["ids_by_model"],
            "pending_count": 5,
            "pending_ids_sha256": population["pending_ids_sha256"],
        },
    }
    _write_json(manifest_path, manifest)
    manifest_raw = manifest_path.read_bytes()

    spec = DispositionSpec(
        experiment_id="TEST-F08",
        cutoff_utc=disposition.LEGACY_CUTOFF_UTC,
        reconciliation_id="TEST-RECON-1",
        state_sha256=_sha(state_raw),
        ledger_sha256=_sha(baseline_ledger_path.read_bytes()),
        journal_sha256=_sha(baseline_journal_path.read_bytes()),
        manifest_sha256=_sha(manifest_raw),
        bundle_sha256=_sha(bundle_raw),
        pending_ids_sha256=population["pending_ids_sha256"],
        old_validator_sha256=old_validator,
        promotion_gate_sha256=disposition.canonical_source_sha256(gate_path),
        counts={"candidate": 2, "production": 3},
        total=5,
        manifest_path="research_outputs/f08_historical_backfill/manifest.json",
    )
    owner_id1 = _approval_evidence(
        amendment_approval_path,
        disposition.VALIDATOR_AMENDMENT_ENVIRONMENT,
        disposition.VALIDATOR_AMENDMENT_WORKFLOW,
        disposition.VALIDATOR_AMENDMENT_WORKFLOW_FILE,
        "910001",
    )
    owner_id2 = _approval_evidence(
        disposition_approval_path,
        disposition.DISPOSITION_ENVIRONMENT,
        disposition.DISPOSITION_WORKFLOW,
        disposition.DISPOSITION_WORKFLOW_FILE,
        "910002",
    )
    assert owner_id1 == owner_id2

    return {
        "root": root,
        "state": state,
        "state_path": state_path,
        "baseline_state_path": baseline_state_path,
        "ledger_path": ledger_path,
        "baseline_ledger_path": baseline_ledger_path,
        "journal_path": journal_path,
        "baseline_journal_path": baseline_journal_path,
        "validator_path": validator_path,
        "gate_path": gate_path,
        "manifest_path": manifest_path,
        "bundle_path": bundle_path,
        "resolution_provenance_path": provenance_path,
        "raw_windows_path": raw_windows_path,
        "amendment_approval_path": amendment_approval_path,
        "disposition_approval_path": disposition_approval_path,
        "owner_id": owner_id1,
        "spec": spec,
    }


def _append_amendment(f):
    return disposition.append_validator_amendment(
        state_path=f["state_path"],
        observation_ledger_path=f["ledger_path"],
        experiment_journal_path=f["journal_path"],
        baseline_journal_path=f["baseline_journal_path"],
        validator_path=f["validator_path"],
        promotion_gate_path=f["gate_path"],
        approval_evidence_path=f["amendment_approval_path"],
        expected_owner_id=f["owner_id"],
        spec=f["spec"],
    )


def _apply(f):
    return disposition.apply_disposition(
        state_path=f["state_path"],
        observation_ledger_path=f["ledger_path"],
        experiment_journal_path=f["journal_path"],
        baseline_state_path=f["baseline_state_path"],
        baseline_ledger_path=f["baseline_ledger_path"],
        baseline_journal_path=f["baseline_journal_path"],
        validator_path=f["validator_path"],
        promotion_gate_path=f["gate_path"],
        manifest_path=f["manifest_path"],
        bundle_path=f["bundle_path"],
        approval_evidence_path=f["disposition_approval_path"],
        expected_owner_id=f["owner_id"],
        spec=f["spec"],
    )


def test_preflight_is_read_only_and_reports_exact_population(fixture_set):
    f = fixture_set
    before = {p: p.read_bytes() for p in (f["state_path"], f["ledger_path"], f["journal_path"])}
    result = disposition.preflight(
        state_path=f["state_path"],
        observation_ledger_path=f["ledger_path"],
        baseline_journal_path=f["baseline_journal_path"],
        manifest_path=f["manifest_path"],
        bundle_path=f["bundle_path"],
        spec=f["spec"],
    )
    assert result["status"] == "PREFLIGHT_PASS_OWNER_APPROVALS_PENDING"
    assert result["pending_by_model"] == {"candidate": 2, "production": 3}
    assert result["pending_count"] == 5
    assert all(p.read_bytes() == before[p] for p in before)


def test_immutable_backfill_manifest_rejects_embedded_approval_provenance(fixture_set):
    f = fixture_set
    manifest = json.loads(f["manifest_path"].read_text(encoding="utf-8"))
    manifest["approval_provenance"] = {"reviewer_id": f["owner_id"]}
    _write_json(f["manifest_path"], manifest)
    spec = dataclasses.replace(
        f["spec"], manifest_sha256=_sha(f["manifest_path"].read_bytes())
    )
    source_state = json.loads(f["baseline_state_path"].read_text(encoding="utf-8"))
    population = disposition.pending_population(source_state, spec)

    with pytest.raises(DispositionError, match="Approval data must not be inserted into the immutable backfill manifest"):
        disposition._verify_backfill_bundle(
            f["manifest_path"], f["bundle_path"], population, spec
        )


def test_disposition_rejects_tampered_resolution_provenance_before_any_official_write(fixture_set):
    f = fixture_set
    _append_amendment(f)
    protected_before = {
        "state": f["state_path"].read_bytes(),
        "ledger": f["ledger_path"].read_bytes(),
        "journal": f["journal_path"].read_bytes(),
    }

    provenance_rows = disposition._read_jsonl_bytes(
        f["resolution_provenance_path"].read_bytes(), "resolution provenance"
    )
    provenance_rows[0]["outcome_bar_close_time_ms"] += 1
    _write_jsonl(f["resolution_provenance_path"], provenance_rows)

    with pytest.raises(DispositionError, match="resolution-provenance SHA256 mismatch"):
        _apply(f)

    assert f["state_path"].read_bytes() == protected_before["state"]
    assert f["ledger_path"].read_bytes() == protected_before["ledger"]
    assert f["journal_path"].read_bytes() == protected_before["journal"]


def test_validator_amendment_is_separate_and_does_not_change_state_or_observation_ledger(fixture_set):
    f = fixture_set
    state_before = f["state_path"].read_bytes()
    ledger_before = f["ledger_path"].read_bytes()
    result = _append_amendment(f)
    assert result["status"] == "APPENDED"
    assert result["event"] == "F08_VALIDATOR_AMENDMENT"
    assert f["state_path"].read_bytes() == state_before
    assert f["ledger_path"].read_bytes() == ledger_before
    rows = disposition._read_jsonl_bytes(f["journal_path"].read_bytes(), "journal")
    assert sum(r.get("event") == "F08_VALIDATOR_AMENDMENT" for r in rows) == 1


def test_disposition_moves_exact_payloads_to_schema7_without_scoring(fixture_set):
    f = fixture_set
    original = copy.deepcopy(f["state"])
    _append_amendment(f)
    result = _apply(f)
    assert result["status"] == "MIGRATED"
    assert result["dispositioned"] == {"candidate": 2, "production": 3}
    assert result["official_observation_rows_appended"] == 0
    assert result["official_events_appended"] == 1

    migrated = json.loads(f["state_path"].read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 7
    assert len(migrated["dispositioned"]["candidate"]) == 2
    assert len(migrated["dispositioned"]["production"]) == 3
    for model in ("candidate", "production"):
        stale = [row for row in original["pending"][model] if row["created_at"] <= disposition.LEGACY_CUTOFF_UTC]
        stored = {item["id"]: item["prediction_payload"] for item in migrated["dispositioned"][model]}
        assert stored == {row["id"]: row for row in stale}
        assert all("status" not in item["prediction_payload"] for item in migrated["dispositioned"][model])
        assert all(row["created_at"] > disposition.LEGACY_CUTOFF_UTC for row in migrated["pending"][model])
        assert migrated["resolved"][model] == original["resolved"][model]
        assert migrated["seen"][model] == original["seen"][model]

    ledger_rows = disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")
    assert len(ledger_rows) == 4
    assert ledger_rows[-1]["event"] == disposition.STALE_PENDING_EVENT
    assert ledger_rows[-1]["outcome_rows_appended"] == 0
    assert all(row.get("record_type") != "observation" for row in ledger_rows[-1:])
    assert not any(row.get("event") == "F08_CHECKPOINT_EVALUATED" and row.get("checkpoint_blocks", 0) >= 25 for row in disposition._read_jsonl_bytes(f["journal_path"].read_bytes(), "journal"))


def test_crash_after_event_append_before_state_save_is_recoverable(fixture_set, monkeypatch):
    f = fixture_set
    _append_amendment(f)
    original_save = disposition._atomic_save_state

    def crash_before_replace(*args, **kwargs):
        raise OSError("simulated power loss before atomic state replace")

    monkeypatch.setattr(disposition, "_atomic_save_state", crash_before_replace)
    with pytest.raises(OSError, match="simulated power loss"):
        _apply(f)
    assert json.loads(f["state_path"].read_text())["schema_version"] == 6
    rows_after_crash = disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")
    assert len(rows_after_crash) == 4
    monkeypatch.setattr(disposition, "_atomic_save_state", original_save)
    result = _apply(f)
    assert result["status"] == "MIGRATED"
    assert json.loads(f["state_path"].read_text())["schema_version"] == 7
    assert len(disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")) == 4


def test_schema7_replay_is_idempotent(fixture_set):
    f = fixture_set
    _append_amendment(f)
    first = _apply(f)
    second = _apply(f)
    assert first["status"] == "MIGRATED"
    assert second["status"] == "ALREADY_APPLIED"
    assert len(disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")) == 4


def test_validator_amendment_duplicate_exact_is_idempotent(fixture_set):
    f = fixture_set
    first = _append_amendment(f)
    second = _append_amendment(f)
    assert first["status"] == "APPENDED"
    assert second["status"] == "ALREADY_PRESENT"
    rows = disposition._read_jsonl_bytes(f["journal_path"].read_bytes(), "journal")
    assert sum(r.get("event") == "F08_VALIDATOR_AMENDMENT" for r in rows) == 1


def test_conflicting_validator_amendment_event_fails_closed(fixture_set):
    f = fixture_set
    baseline = f["baseline_journal_path"].read_bytes()
    bad = {
        "record_type": "event", "ledger_schema_version": 1,
        "event": "F08_VALIDATOR_AMENDMENT", "experiment_id": "TEST-F08",
        "event_id": "conflicting", "recorded_at": "2026-11-08T00:00:00+00:00",
        "to_validator_code_sha256": "0" * 64,
    }
    f["journal_path"].write_bytes(baseline + _canonical(bad) + b"\n")
    with pytest.raises(DispositionError, match="conflicts on|exactly one|exact authorized payload|exact authorized record"):
        _append_amendment(f)


def test_wrong_approval_reviewer_id_fails_closed(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text())
    evidence["approval_history"][0]["user"]["id"] = 123456
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="designated owner"):
        _append_amendment(f)


def test_self_review_fails_closed(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text())
    evidence["context"]["actor_id"] = f["owner_id"]
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="cannot self-approve"):
        _append_amendment(f)


def test_environment_without_protected_reviewer_rule_fails_closed(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text())
    evidence["environment"]["protection_rules"] = []
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="required_reviewers"):
        _append_amendment(f)


def test_preflight_rejects_mutated_manifest(fixture_set):
    f = fixture_set
    raw = f["manifest_path"].read_bytes()
    f["manifest_path"].write_bytes(raw + b" ")
    with pytest.raises(DispositionError, match="manifest hash mismatch"):
        disposition.preflight(
            state_path=f["state_path"],
            observation_ledger_path=f["ledger_path"],
            baseline_journal_path=f["baseline_journal_path"],
            manifest_path=f["manifest_path"],
            bundle_path=f["bundle_path"],
            spec=f["spec"],
        )


def test_preflight_rejects_tampered_state_head(fixture_set):
    f = fixture_set
    changed = json.loads(f["state_path"].read_text())
    changed["pending"]["candidate"][0]["entry"] = 9999
    _write_json(f["state_path"], changed)
    with pytest.raises(DispositionError, match="Pinned source heads mismatch"):
        disposition.preflight(
            state_path=f["state_path"],
            observation_ledger_path=f["ledger_path"],
            baseline_journal_path=f["baseline_journal_path"],
            manifest_path=f["manifest_path"],
            bundle_path=f["bundle_path"],
            spec=f["spec"],
        )


def test_duplicate_pending_id_fails_closed(fixture_set):
    f = fixture_set
    state = copy.deepcopy(f["state"])
    state["pending"]["candidate"].append(copy.deepcopy(state["pending"]["candidate"][0]))
    with pytest.raises(DispositionError, match="duplicate ID"):
        disposition.pending_population(state, f["spec"])


def test_existing_event_with_conflicting_payload_fails_closed(fixture_set):
    f = fixture_set
    _append_amendment(f)
    bad_event = {
        "record_type": "event",
        "ledger_schema_version": 1,
        "event": disposition.STALE_PENDING_EVENT,
        "experiment_id": "TEST-F08",
        "event_id": "wrong-event",
        "recorded_at": "2026-11-08T00:00:00+00:00",
        "pending_count": 5,
    }
    f["ledger_path"].write_bytes(f["baseline_ledger_path"].read_bytes() + _canonical(bad_event) + b"\n")
    with pytest.raises(DispositionError, match="conflicts on|exactly one|exact authorized payload|exact authorized record"):
        _apply(f)


def _change_approval_run(path: Path, run_id: str):
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["context"]["run_id"] = run_id
    evidence["context"]["head_sha"] = (run_id[-1] * 40) if run_id[-1].isdigit() else "b" * 40
    path.write_text(json.dumps(evidence, sort_keys=True, indent=2), encoding="utf-8")


def test_validator_amendment_retry_accepts_new_approved_run(fixture_set):
    f = fixture_set
    first = _append_amendment(f)
    _change_approval_run(f["amendment_approval_path"], "9000002")
    second = _append_amendment(f)
    assert first["event_id"] == second["event_id"]
    assert second["status"] == "ALREADY_PRESENT"
    assert f["journal_path"].read_bytes().count(b"F08_VALIDATOR_AMENDMENT") == 1


def test_environment_must_be_limited_to_protected_branches(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text(encoding="utf-8"))
    evidence["environment"]["deployment_branch_policy"] = {
        "protected_branches": False,
        "custom_branch_policies": True,
    }
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="protected branches"):
        _append_amendment(f)


def test_environment_requires_exactly_the_designated_owner(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text(encoding="utf-8"))
    evidence["environment"]["protection_rules"][0]["reviewers"].append({
        "type": "User", "reviewer": {"login": "someone-else", "id": 123456}
    })
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="exactly one designated reviewer"):
        _append_amendment(f)


def test_disposition_retry_after_append_accepts_new_approved_run(fixture_set, monkeypatch):
    f = fixture_set
    _append_amendment(f)
    original = disposition._atomic_save_state

    def crash_before_state_replace(path, state):
        raise OSError("simulated process interruption before state replace")

    monkeypatch.setattr(disposition, "_atomic_save_state", crash_before_state_replace)
    with pytest.raises(OSError, match="simulated process interruption"):
        _apply(f)

    # A new workflow dispatch creates new run identity and approval history.
    # The already appended official event must remain authoritative and be
    # replayed, not conflict with the new run's approval envelope.
    _change_approval_run(f["disposition_approval_path"], "9000003")
    monkeypatch.setattr(disposition, "_atomic_save_state", original)
    result = _apply(f)
    assert result["status"] == "MIGRATED"
    assert result["official_observation_rows_appended"] == 0
    rows = disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")
    assert sum(row.get("event") == disposition.STALE_PENDING_EVENT for row in rows) == 1


def test_schema7_retry_validates_saved_approval_and_is_idempotent(fixture_set):
    f = fixture_set
    _append_amendment(f)
    applied = _apply(f)
    assert applied["status"] == "MIGRATED"
    _change_approval_run(f["disposition_approval_path"], "9000004")
    retried = _apply(f)
    assert retried["status"] == "ALREADY_APPLIED"
    assert retried["official_state_written"] is False
    assert retried["official_observation_event_appended"] is False


def test_schema7_retry_fails_if_embedded_approval_snapshot_is_tampered(fixture_set):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    rows = disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")
    rows[-1]["approval_provenance"]["evidence_snapshot"]["environment"]["deployment_branch_policy"]["protected_branches"] = False
    _write_jsonl(f["ledger_path"], rows)
    with pytest.raises(DispositionError, match="protected branches"):
        _apply(f)


def test_workflow_rerun_approval_attestation_is_rejected(fixture_set):
    f = fixture_set
    evidence = json.loads(f["amendment_approval_path"].read_text())
    evidence["context"]["run_attempt"] = 2
    _write_json(f["amendment_approval_path"], evidence)
    with pytest.raises(DispositionError, match="first attempt of a new workflow run"):
        _append_amendment(f)


def test_validator_amendment_rejects_owner_id_override_before_any_write(fixture_set):
    f = fixture_set
    before = {
        path: path.read_bytes()
        for path in (f["state_path"], f["ledger_path"], f["journal_path"])
    }
    with pytest.raises(DispositionError, match="pinned in source"):
        disposition.append_validator_amendment(
            state_path=f["state_path"],
            observation_ledger_path=f["ledger_path"],
            experiment_journal_path=f["journal_path"],
            baseline_journal_path=f["baseline_journal_path"],
            validator_path=f["validator_path"],
            promotion_gate_path=f["gate_path"],
            approval_evidence_path=f["amendment_approval_path"],
            expected_owner_id="11223344",
            spec=f["spec"],
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_disposition_rejects_owner_id_override_before_any_write(fixture_set):
    f = fixture_set
    before = {
        path: path.read_bytes()
        for path in (f["state_path"], f["ledger_path"], f["journal_path"])
    }
    with pytest.raises(DispositionError, match="pinned in source"):
        disposition.apply_disposition(
            state_path=f["state_path"],
            observation_ledger_path=f["ledger_path"],
            experiment_journal_path=f["journal_path"],
            baseline_state_path=f["baseline_state_path"],
            baseline_ledger_path=f["baseline_ledger_path"],
            baseline_journal_path=f["baseline_journal_path"],
            validator_path=f["validator_path"],
            promotion_gate_path=f["gate_path"],
            manifest_path=f["manifest_path"],
            bundle_path=f["bundle_path"],
            approval_evidence_path=f["disposition_approval_path"],
            expected_owner_id="11223344",
            spec=f["spec"],
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_existing_event_with_unrecognized_extra_field_is_rejected(fixture_set):
    f = fixture_set
    _append_amendment(f)
    # Simulate a conflicting ledger extension with an unexpected field. The
    # apply operation must not accept an event that only partially matches its
    # canonical payload.
    event = {
        "record_type": "event",
        "ledger_schema_version": 1,
        "event": disposition.STALE_PENDING_EVENT,
        "experiment_id": f["spec"].experiment_id,
        "event_id": "F08D-" + "0" * 32,
        "recorded_at": "2026-11-08T00:00:00+00:00",
        "unexpected": "not part of the contract",
    }
    f["ledger_path"].write_bytes(
        f["baseline_ledger_path"].read_bytes() + _canonical(event) + b"\n"
    )
    with pytest.raises(DispositionError, match="conflicts on|exactly one|exact authorized payload|exact authorized record"):
        _apply(f)


def test_atomic_event_append_failure_leaves_original_jsonl_unchanged(fixture_set, monkeypatch):
    f = fixture_set
    target = f["ledger_path"]
    baseline = target.read_bytes()

    def fail_replace(src, dst):
        raise OSError("simulated atomic replace interruption")

    monkeypatch.setattr(disposition.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace interruption"):
        disposition._append_one_exact_event(
            target,
            baseline,
            "TEST_ATOMIC_APPEND",
            {"proof": "complete-record"},
            "TEST-F08",
        )

    assert target.read_bytes() == baseline
    assert not list(target.parent.glob(f".{target.name}.*.append.tmp"))


def test_validator_amendment_replay_rejects_unrecognized_extra_fields(fixture_set):
    f = fixture_set
    _append_amendment(f)
    rows = disposition._read_jsonl_bytes(f["journal_path"].read_bytes(), "journal")
    rows[-1]["unrecognized_extension"] = "must fail closed"
    _write_jsonl(f["journal_path"], rows)

    with pytest.raises(DispositionError, match="exact authorized payload"):
        _append_amendment(f)


def test_disposition_event_replay_rejects_unrecognized_extra_fields(fixture_set):
    f = fixture_set
    _append_amendment(f)
    original_save = disposition._atomic_save_state

    def crash_before_state_save(path, state):
        raise OSError("simulated stop after event append")

    # Produce a genuine canonical disposition event, then simulate interruption.
    try:
        disposition._atomic_save_state = crash_before_state_save
        with pytest.raises(OSError, match="simulated stop"):
            _apply(f)
    finally:
        disposition._atomic_save_state = original_save

    rows = disposition._read_jsonl_bytes(f["ledger_path"].read_bytes(), "ledger")
    rows[-1]["unrecognized_extension"] = "must fail closed"
    _write_jsonl(f["ledger_path"], rows)

    with pytest.raises(DispositionError, match="exact authorized payload"):
        _apply(f)

@pytest.mark.parametrize(
    "alias_argument",
    [
        "state_path",
        "observation_ledger_path",
        "baseline_journal_path",
        "validator_path",
        "promotion_gate_path",
        "amendment_approval_path",
    ],
)
def test_validator_amendment_rejects_working_journal_alias_before_any_write(fixture_set, alias_argument):
    f = fixture_set
    paths = {
        "state_path": f["state_path"],
        "observation_ledger_path": f["ledger_path"],
        "baseline_journal_path": f["baseline_journal_path"],
        "validator_path": f["validator_path"],
        "promotion_gate_path": f["gate_path"],
        "amendment_approval_path": f["amendment_approval_path"],
    }
    aliased_path = paths[alias_argument]
    before = {
        path: path.read_bytes()
        for path in {
            f["state_path"], f["ledger_path"], f["baseline_journal_path"],
            f["validator_path"], f["gate_path"], f["amendment_approval_path"],
        }
    }
    with pytest.raises(DispositionError, match="aliased paths"):
        disposition.append_validator_amendment(
            state_path=f["state_path"],
            observation_ledger_path=f["ledger_path"],
            experiment_journal_path=aliased_path,
            baseline_journal_path=f["baseline_journal_path"],
            validator_path=f["validator_path"],
            promotion_gate_path=f["gate_path"],
            approval_evidence_path=f["amendment_approval_path"],
            expected_owner_id=disposition.OWNER_ID,
            spec=f["spec"],
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize(
    ("target_argument", "alias_argument"),
    [
        ("state_path", "baseline_state_path"),
        ("state_path", "baseline_ledger_path"),
        ("observation_ledger_path", "baseline_state_path"),
        ("observation_ledger_path", "baseline_ledger_path"),
        ("experiment_journal_path", "baseline_journal_path"),
        ("state_path", "manifest_path"),
        ("observation_ledger_path", "bundle_path"),
        ("state_path", "observation_ledger_path"),
    ],
)
def test_disposition_rejects_working_and_protected_path_aliases_before_any_write(
    fixture_set, target_argument, alias_argument
):
    f = fixture_set
    _append_amendment(f)
    args = {
        "state_path": f["state_path"],
        "observation_ledger_path": f["ledger_path"],
        "experiment_journal_path": f["journal_path"],
        "baseline_state_path": f["baseline_state_path"],
        "baseline_ledger_path": f["baseline_ledger_path"],
        "baseline_journal_path": f["baseline_journal_path"],
        "validator_path": f["validator_path"],
        "promotion_gate_path": f["gate_path"],
        "manifest_path": f["manifest_path"],
        "bundle_path": f["bundle_path"],
        "approval_evidence_path": f["disposition_approval_path"],
    }
    args[target_argument] = args[alias_argument]
    before = {
        path: path.read_bytes()
        for path in {
            f["state_path"], f["ledger_path"], f["journal_path"],
            f["baseline_state_path"], f["baseline_ledger_path"],
            f["baseline_journal_path"], f["validator_path"], f["gate_path"],
            f["manifest_path"], f["bundle_path"], f["disposition_approval_path"],
        }
    }
    with pytest.raises(DispositionError, match="aliased paths"):
        disposition.apply_disposition(
            **args, expected_owner_id=disposition.OWNER_ID, spec=f["spec"]
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_path_alias_guard_detects_normalized_relative_alias(tmp_path):
    target = tmp_path / "source.json"
    target.write_text("{}", encoding="utf-8")
    alias = target.parent / "." / "source.json"
    with pytest.raises(DispositionError, match="aliased paths"):
        disposition._assert_distinct_paths(
            ((target, "source"), (alias, "alias")),
            operation="test operation",
        )



def test_committed_provenance_bundle_matches_pinned_disposition_spec():
    repo_root = Path(__file__).resolve().parents[1]
    bundle_dir = repo_root / "research_outputs" / "f08_historical_backfill"
    manifest_path = bundle_dir / "manifest.json"
    bundle_path = bundle_dir / "observations.jsonl"
    provenance_path = bundle_dir / "resolution_provenance.jsonl"
    state_path = bundle_dir / "evidence" / "state.json"

    spec = DispositionSpec()
    state_raw = state_path.read_bytes()
    assert _sha(state_raw) == spec.state_sha256
    state = json.loads(state_raw.decode("utf-8"))
    population = disposition.pending_population(state, spec)

    verified = disposition._verify_backfill_bundle(
        manifest_path, bundle_path, population, spec
    )
    assert verified["manifest_sha256"] == "1aae5f75292241298977378b81ac6297e24c3270e1afb5957776a1f8b1767ab3"
    assert verified["manifest_sha256"] == spec.manifest_sha256
    assert verified["bundle_sha256"] == spec.bundle_sha256
    assert verified["manifest"]["reconciliation_id"] == spec.reconciliation_id
    assert _sha(provenance_path.read_bytes()) == "d679494c5d58292f12227b95492ec3d47daf4b5b5f2a38b21f962ca465b994c8"
    assert verified["manifest"]["resolution_provenance"]["record_count"] == 618
