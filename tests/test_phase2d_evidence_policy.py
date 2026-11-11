from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phase2d_evidence_policy as policy
import phase2d_f08_resolution_provenance as resolution_provenance


def _row(model: str, prediction_id: str, open_time: int, *, evidence_class=None):
    row = {
        "record_type": "observation",
        "ledger_schema_version": 1,
        "experiment_id": policy.EXPERIMENT_ID,
        "observation_key": f"{policy.EXPERIMENT_ID}:{model}:BTCUSDT:{open_time}",
        "model": model,
        "id": prediction_id,
        "symbol": "BTCUSDT",
        "open_time": open_time,
        "status": "TP",
        "net_r": 1.0,
        "created_at": "2026-10-15T10:00:00+00:00",
        "resolved_at": "2026-10-16T10:00:00+00:00",
    }
    if evidence_class is not None:
        row["evidence_class"] = evidence_class
    return row


def _test_legacy_specs(candidate, production):
    return {
        "candidate": {
            "count": len(candidate),
            "payload_sha256": policy._payload_fingerprint(candidate),
        },
        "production": {
            "count": len(production),
            "payload_sha256": policy._payload_fingerprint(production),
        },
    }


def test_exact_legacy_fingerprints_and_prospective_rows_are_accepted():
    candidate = [_row("candidate", "c-1", 1), _row("candidate", "c-2", 2)]
    production = [_row("production", "p-1", 3)]
    prospective = {
        **_row("candidate", "c-new", 4),
        "created_at": "2026-10-17T00:00:00+00:00",
        "resolved_at": "2026-10-17T00:15:00+00:00",
        "evidence_class": "PROSPECTIVE",
    }
    candidate.append(prospective)

    c, p, summary = policy.validate_evidence_population(
        candidate,
        production,
        legacy_cohorts=_test_legacy_specs(candidate[:2], production),
    )

    assert c == candidate
    assert p == production
    assert summary["legacy"]["candidate"]["count"] == 2
    assert summary["legacy"]["production"]["count"] == 1
    assert summary["prospective"] == {"candidate": 1, "production": 0}
    assert summary["status"] == "PASS"


def test_one_extra_untagged_record_is_rejected():
    candidate = [_row("candidate", "c-1", 1), _row("candidate", "c-2", 2)]
    production = [_row("production", "p-1", 3)]
    expected = _test_legacy_specs(candidate, production)
    candidate.append(_row("candidate", "c-extra", 4))

    with pytest.raises(policy.EvidencePolicyError, match="legacy count mismatch"):
        policy.validate_evidence_population(candidate, production, legacy_cohorts=expected)


def test_untagged_payload_mutation_is_rejected():
    candidate = [_row("candidate", "c-1", 1)]
    production = [_row("production", "p-1", 2)]
    expected = _test_legacy_specs(candidate, production)
    candidate[0]["net_r"] = 9.0

    with pytest.raises(policy.EvidencePolicyError, match="payload fingerprint mismatch"):
        policy.validate_evidence_population(candidate, production, legacy_cohorts=expected)


def test_historical_backfill_is_rejected_from_primary_gate():
    candidate = [_row("candidate", "c-1", 1)]
    production = [_row("production", "p-1", 2)]
    expected = _test_legacy_specs(candidate, production)
    candidate.append({
        **_row("candidate", "c-backfill", 3),
        "created_at": "2026-10-17T00:00:00+00:00",
        "resolved_at": "2026-11-04T00:00:00+00:00",
        "evidence_class": "HISTORICAL_BACKFILL",
        "reconciliation_id": "R-1",
    })

    with pytest.raises(policy.EvidencePolicyError, match="HISTORICAL_BACKFILL"):
        policy.validate_evidence_population(candidate, production, legacy_cohorts=expected)


def test_unknown_or_null_evidence_class_is_rejected():
    candidate = [_row("candidate", "c-1", 1)]
    production = [_row("production", "p-1", 2)]
    expected = _test_legacy_specs(candidate, production)
    candidate[0]["evidence_class"] = None

    with pytest.raises(policy.EvidencePolicyError, match="unsupported evidence_class"):
        policy.validate_evidence_population(candidate, production, legacy_cohorts=expected)


def test_prospective_timestamp_before_cutoff_is_rejected():
    candidate = [_row("candidate", "c-1", 1)]
    production = [_row("production", "p-1", 2)]
    expected = _test_legacy_specs(candidate, production)
    candidate.append({
        **_row("candidate", "c-new", 3),
        "evidence_class": "PROSPECTIVE",
        "created_at": "2026-10-15T12:00:00+00:00",
        "resolved_at": "2026-10-17T00:00:00+00:00",
    })

    with pytest.raises(policy.EvidencePolicyError, match="timestamps must be after"):
        policy.validate_evidence_population(candidate, production, legacy_cohorts=expected)


def test_gate_wrapper_only_delegates_after_evidence_validation():
    candidate = [_row("candidate", "c-1", 1)]
    production = [_row("production", "p-1", 2)]
    expected = _test_legacy_specs(candidate, production)
    calls = []

    def fake_gate(c, p, **kwargs):
        calls.append((c, p, kwargs))
        return {"gate_status": "NOT_READY", "promotion_ready": False}

    c, p, summary = policy.validate_evidence_population(
        candidate,
        production,
        legacy_cohorts=expected,
    )
    assert c == candidate and p == production and summary["status"] == "PASS"

    # Inject the fixed cohort fixture for this isolated wrapper contract test.
    original = policy.LEGACY_COHORTS
    try:
        policy.LEGACY_COHORTS = expected
        result = policy.evaluate_promotion_with_evidence_policy(
            candidate,
            production,
            gate_callable=fake_gate,
            sentinel="kept",
        )
    finally:
        policy.LEGACY_COHORTS = original

    assert result["gate_status"] == "NOT_READY"
    assert result["evidence_policy"]["status"] == "PASS"
    assert calls == [(candidate, production, {"sentinel": "kept"})]


def _pending_state():
    pending = {"candidate": [], "production": []}
    for model, count in policy.EXPECTED_PENDING_COUNTS.items():
        for index in range(count):
            pending[model].append({
                "id": f"{model}:SYM{index}:{index}",
                "model": model,
                "symbol": f"SYM{index}",
                "open_time": 1_791_143_100_000 + index * 900_000,
                "created_at": "2026-10-16T14:08:20+00:00",
                "signal": "BUY" if model == "candidate" else "SELL",
                "entry": 100.0,
                "atr": 1.0,
            })
    return {"pending": pending}


def _historical_bundle_fixture(tmp_path: Path, state: dict, monkeypatch):
    evidence_dir = tmp_path / "research_outputs" / "f08_historical_backfill" / "evidence"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir.parent / "manifest.json"
    ids_by_model = {
        model: sorted(row["id"] for row in state["pending"][model])
        for model in ("candidate", "production")
    }

    # Source evidence copies are deliberately small fixtures, but use the same
    # artifact types and verification boundaries as the production bundle.
    state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    state_path = evidence_dir / "state.json"
    state_path.write_bytes(state_bytes)

    ledger_path_copy = evidence_dir / "ledger.jsonl"
    with ledger_path_copy.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(16208):
            row = {
                "record_type": "observation",
                "experiment_id": policy.EXPERIMENT_ID,
                "observation_key": f"{policy.EXPERIMENT_ID}:candidate:OLD{index}:{index}",
                "model": "candidate", "id": f"old-{index}",
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        for name in ("EVT_A", "EVT_B", "EVT_C"):
            handle.write(json.dumps({
                "record_type": "event", "experiment_id": policy.EXPERIMENT_ID,
                "event": name, "timestamp": "2026-01-01T00:00:00+00:00",
            }, sort_keys=True, separators=(",", ":")) + "\n")

    journal_path = evidence_dir / "journal.jsonl"
    journal_rows = [{"event": "PHASE2D_RUN", "run_id": str(i)} for i in range(95)]
    journal_rows.extend([
        {"event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 15},
        {"event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 20},
        {"event": "F08_GATE_AMENDMENT", "approved_by": "MrZkull",
         "approval_run_id": "41720710702", "workflow_event": "workflow_dispatch"},
    ])
    journal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in journal_rows), encoding="utf-8")

    results_path = evidence_dir / "results.json"
    snapshot_path = evidence_dir / "snapshot.json"
    dry_run_path = evidence_dir / "dry_run.json"
    independent_path = evidence_dir / "independent.json"
    results_path.write_text('{"results":"fixture"}\n', encoding="utf-8")
    snapshot_path.write_text('{"snapshot":"fixture"}\n', encoding="utf-8")
    dry_run_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    independent_path.write_text('{"status":"PASS"}\n', encoding="utf-8")

    raw_windows_path = evidence_dir / "raw_windows.jsonl"
    raw_windows_path.write_text("".join(
        json.dumps({
            "symbol": f"SYM{i}", "open_time": 1_791_143_100_000 + i * 900000,
            "request": {"resolution": "15"},
            "candles": [
                {"timestamp": 1_791_143_100_000 + i * 900000 + j * 900000, "high": 101.0, "low": 99.0}
                for j in range(26)
            ],
        }, sort_keys=True) + "\n"
        for i in range(573)
    ), encoding="utf-8")

    proposal_rows = []
    bundle_rows = []
    outcome_by_model = {
        "candidate": ["TP"] * 12 + ["SL"] * 62,
        "production": ["TP"] * 280 + ["SL"] * 245 + ["EXPIRED"] * 19,
    }
    gross_by_status = {"TP": 3.5, "SL": -2.5, "EXPIRED": 0.0}
    net_by_status = {"TP": 3.38, "SL": -2.62, "EXPIRED": -0.12}
    for model in ("candidate", "production"):
        for index, pending in enumerate(state["pending"][model]):
            status = outcome_by_model[model][index]
            proposal_rows.append({
                "model": model, "prediction_id": pending["id"],
                "symbol": pending["symbol"], "signal": pending["model"] and ("BUY" if model == "candidate" else "SELL"),
                "open_time": pending["open_time"], "entry": 100.0, "atr": 1.0,
                "observation_key": f"{policy.EXPERIMENT_ID}:{model}:{pending['symbol']}:{pending['open_time']}",
                "source_created_at": pending["created_at"],
                "proposed_status": status, "outcome_bar": 24 if status == "EXPIRED" else 1,
                "gross_r": gross_by_status[status], "net_r": net_by_status[status],
            })
            bundle_rows.append({
                **pending,
                "status": status,
                "outcome_bar": 24 if status == "EXPIRED" else 1,
                "gross_r": gross_by_status[status],
                "net_r": net_by_status[status],
                "experiment_id": policy.EXPERIMENT_ID,
                "record_type": "historical_backfill_observation",
                "evidence_class": "HISTORICAL_BACKFILL",
                "observation_key": f"{policy.EXPERIMENT_ID}:{model}:{pending['symbol']}:{pending['open_time']}",
            })
    proposal_path = evidence_dir / "proposal.json"
    # Placeholder hashes are replaced below once the source artifacts are fixed.
    proposal = {
        "proposal_status": "READY_FOR_REVIEW_NOT_APPLIED",
        "state_or_ledger_modified": False,
        "promotion_authorized": False,
        "proposed_observation_appends": 618,
        "pending_before": {"candidate": 74, "production": 544},
        "pending_after_if_approved": {"candidate": 0, "production": 0},
        "proposed_candidate": {"TP": 12, "SL": 62},
        "proposed_production": {"TP": 280, "SL": 245, "EXPIRED": 19},
        "proposed_total_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
        "proposal_records": proposal_rows,
    }
    proposal_path.write_text(json.dumps(proposal, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    actual_resolver_report = {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": 618,
        "candidate_verified": 74,
        "production_verified": 544,
        "mismatch_count": 0,
        "validator_git_blob": policy.HISTORICAL_RESOLVER_IDENTITY["validator_git_blob"],
        "actual_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
    }
    actual_resolver_path = evidence_dir / "actual_resolver_report.json"

    def hash_file(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # First bind the state and ledger bytes, then bind the report and proposal.
    source_hashes = {
        "state": hash_file(state_path),
        "ledger": hash_file(ledger_path_copy),
        "journal": hash_file(journal_path),
        "results": hash_file(results_path),
        "snapshot": hash_file(snapshot_path),
        "proposal": "0" * 64,
        "raw_windows": hash_file(raw_windows_path),
        "dry_run": hash_file(dry_run_path),
        "independent": hash_file(independent_path),
        "actual_resolver": "0" * 64,
    }
    proposal["source_state_sha256"] = source_hashes["state"]
    proposal["source_observation_ledger_sha256"] = source_hashes["ledger"]
    proposal["source_dry_run_sha256"] = source_hashes["dry_run"]
    proposal["source_independent_verification_sha256"] = source_hashes["independent"]
    proposal_path.write_text(json.dumps(proposal, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    source_hashes["proposal"] = hash_file(proposal_path)
    actual_resolver_report.update({
        "state_sha256_before": source_hashes["state"],
        "state_sha256_after": source_hashes["state"],
        "ledger_sha256_before": source_hashes["ledger"],
        "ledger_sha256_after": source_hashes["ledger"],
    })
    actual_resolver_path.write_text(json.dumps(actual_resolver_report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    source_hashes["actual_resolver"] = hash_file(actual_resolver_path)

    monkeypatch.setattr(policy, "EXPECTED_SOURCE_HASHES", {
        "state": source_hashes["state"],
        "ledger": source_hashes["ledger"],
    })

    source_artifacts = {}
    paths = {
        "state": state_path, "ledger": ledger_path_copy, "journal": journal_path,
        "results": results_path, "snapshot": snapshot_path, "proposal": proposal_path,
        "raw_windows": raw_windows_path, "dry_run": dry_run_path,
        "independent": independent_path, "actual_resolver": actual_resolver_path,
    }
    for name, path in paths.items():
        source_artifacts[name] = {
            "path": f"evidence/{path.name}",
            "sha256": source_hashes[name],
        }

    bundle_rows.sort(key=lambda row: (row["model"], row["id"]))
    normalized_rows_sha = hashlib.sha256(
        b"\n".join(policy._canonical_json(row) for row in bundle_rows) + b"\n"
    ).hexdigest()
    stable_identity = {
        **policy.HISTORICAL_RESOLVER_IDENTITY,
        "pinned_files_clean": True,
    }
    reconciliation_material = {
        "source_hashes": source_hashes,
        "resolver_identity": stable_identity,
        "pending_ids_sha256": policy.pending_ids_digest(ids_by_model),
        "normalized_result_rows_sha256": normalized_rows_sha,
    }
    reconciliation_id = "F08-HIST-BF-" + hashlib.sha256(
        policy._canonical_json(reconciliation_material)
    ).hexdigest()[:32]
    for row in bundle_rows:
        row["reconciliation_id"] = reconciliation_id
    bundle_path = manifest_path.parent / "observations.jsonl"
    bundle_path.write_bytes(b"\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        for row in bundle_rows
    ) + b"\n")
    bundle_sha = hash_file(bundle_path)

    raw_window_rows = [
        json.loads(line)
        for line in raw_windows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_windows_by_key = resolution_provenance.index_raw_windows(raw_window_rows)
    resolver_rows = [
        {**row, "resolved_at": "2026-11-05T12:00:00+00:00"}
        for row in bundle_rows
    ]
    provenance_rows = resolution_provenance.build_resolution_provenance_records(
        resolver_rows, raw_windows_by_key, reconciliation_id
    )
    provenance_path = manifest_path.parent / resolution_provenance.PROVENANCE_FILENAME
    provenance_path.write_bytes(
        resolution_provenance.canonical_jsonl_bytes(provenance_rows)
    )
    provenance_sha = hash_file(provenance_path)

    manifest = {
        "manifest_schema_version": 1,
        "operation": policy.STALE_PENDING_EVENT,
        "experiment_id": policy.EXPERIMENT_ID,
        "evidence_class": "HISTORICAL_BACKFILL",
        "legacy_cutoff_utc": policy.LEGACY_CUTOFF_UTC,
        "manifest_status": "READY_FOR_OWNER_REVIEW",
        "approval_status": "PENDING_OWNER_APPROVAL",
        "reconciliation_id": reconciliation_id,
        "source_hashes": source_hashes,
        "source_artifacts": source_artifacts,
        "raw_ohlc_windows_sha256": source_hashes["raw_windows"],
        "resolution_provenance": {
            "schema_version": 1,
            "format": "jsonl",
            "path": resolution_provenance.PROVENANCE_FILENAME,
            "sha256": provenance_sha,
            "record_count": 618,
            "raw_windows_sha256": source_hashes["raw_windows"],
            "resolved_at_semantics": resolution_provenance.RESOLVED_AT_SEMANTICS,
            "outcome_bar_close_time_semantics": resolution_provenance.OUTCOME_BAR_CLOSE_TIME_SEMANTICS,
        },
        "actual_resolver_report_sha256": source_hashes["actual_resolver"],
        "resolver_identity": stable_identity,
        "actual_resolver_verification": actual_resolver_report,
        "experiment_journal_review": {
            "rows": 98,
            "c15_rows": 1,
            "c20_rows": 1,
            "c25_or_later_rows": 0,
            "f08_amendment_rows": 1,
            "amendment_actor": "MrZkull",
            "amendment_run": "41720710702",
            "amendment_is_final_row": True,
            "sha256": source_hashes["journal"],
        },
        "resolution": {
            "predictions_resolved": 618,
            "candidate": {"TP": 12, "SL": 62},
            "production": {"TP": 280, "SL": 245, "EXPIRED": 19},
            "total_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
            "raw_windows_used": 573,
        },
        "safety_invariants": {
            "official_state_written": False,
            "official_ledger_written": False,
            "official_event_appended": False,
            "official_observations_appended": False,
            "pending_predictions_mutated": False,
            "experiment_journal_modified": False,
            "c15_modified": False,
            "c20_modified": False,
            "c25_evaluated": False,
            "model_promotion_authorized": False,
            "live_trading_enabled": False,
        },
        "historical_bundle": {
            "evidence_class": "HISTORICAL_BACKFILL",
            "format": "jsonl",
            "bundle_path": "observations.jsonl",
            "bundle_sha256": bundle_sha,
            "observation_count": 618,
            "candidate_count": 74,
            "production_count": 544,
            "resolved_at_policy": "omitted from deterministic observation rows; exact per-prediction resolver wall-clock timestamps and raw-OHLC-derived outcome-bar close times are preserved in the separately hashed resolution_provenance.jsonl sidecar",
            "normalized_result_rows_sha256": normalized_rows_sha,
        },
        "pending_population": {
            "ids_by_model": ids_by_model,
            "pending_count": 618,
            "pending_ids_sha256": policy.pending_ids_digest(ids_by_model),
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest_path, manifest


def test_schema6_f08_state_fails_closed_before_any_disposition_checks(tmp_path):
    state = _pending_state()
    state["schema_version"] = 6
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(policy.EvidencePolicyError, match="schema 7"):
        policy.validate_stale_pending_disposition(
            state, ledger, manifest_path=tmp_path / "manifest.json"
        )
    assert ledger.read_bytes() == b""


def test_historical_bundle_verifier_checks_per_prediction_provenance(tmp_path, monkeypatch):
    state = _pending_state()
    manifest_path, manifest = _historical_bundle_fixture(
        tmp_path, state, monkeypatch
    )
    evidence_paths = policy._verify_source_artifacts(manifest, manifest_path.parent)
    stale_ids = policy._stale_pending_ids(state)

    policy._verify_bundle_artifacts(
        manifest,
        manifest_path.parent,
        evidence_paths,
        stale_ids,
        manifest["reconciliation_id"],
        state,
    )


def test_historical_bundle_verifier_rejects_tampered_provenance_sidecar(tmp_path, monkeypatch):
    state = _pending_state()
    manifest_path, manifest = _historical_bundle_fixture(
        tmp_path, state, monkeypatch
    )
    provenance_path = manifest_path.parent / resolution_provenance.PROVENANCE_FILENAME
    rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["outcome_bar_close_time_ms"] += 1
    provenance_path.write_bytes(resolution_provenance.canonical_jsonl_bytes(rows))
    evidence_paths = policy._verify_source_artifacts(manifest, manifest_path.parent)
    stale_ids = policy._stale_pending_ids(state)

    with pytest.raises(policy.EvidencePolicyError, match="Per-prediction resolution-provenance SHA256 mismatch"):
        policy._verify_bundle_artifacts(
            manifest,
            manifest_path.parent,
            evidence_paths,
            stale_ids,
            manifest["reconciliation_id"],
            state,
        )


def test_historical_bundle_verifier_rejects_rehashed_but_semantically_invalid_provenance(tmp_path, monkeypatch):
    state = _pending_state()
    manifest_path, manifest = _historical_bundle_fixture(
        tmp_path, state, monkeypatch
    )
    provenance_path = manifest_path.parent / resolution_provenance.PROVENANCE_FILENAME
    rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["outcome_bar_close_time_ms"] += 1
    provenance_bytes = resolution_provenance.canonical_jsonl_bytes(rows)
    provenance_path.write_bytes(provenance_bytes)
    # Simulate an operator who recomputes the sidecar hash and updates the
    # manifest too; semantic verification must still reject the changed field.
    manifest["resolution_provenance"]["sha256"] = hashlib.sha256(provenance_bytes).hexdigest()
    evidence_paths = policy._verify_source_artifacts(manifest, manifest_path.parent)
    stale_ids = policy._stale_pending_ids(state)

    with pytest.raises(policy.EvidencePolicyError, match="Per-prediction resolution provenance is invalid"):
        policy._verify_bundle_artifacts(
            manifest,
            manifest_path.parent,
            evidence_paths,
            stale_ids,
            manifest["reconciliation_id"],
            state,
        )


def test_schema6_f08_state_fails_closed_before_any_disposition_checks(tmp_path):
    state = _pending_state()
    state["schema_version"] = 6
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(policy.EvidencePolicyError, match="schema 7"):
        policy.validate_stale_pending_disposition(
            state, ledger, manifest_path=tmp_path / "manifest.json"
        )
    assert ledger.read_bytes() == b""


def test_schema6_f08_state_fails_closed_even_if_manifest_claims_legacy_approval(tmp_path):
    state = _pending_state()
    state["schema_version"] = 6
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"approval_status": "APPROVED"}), encoding="utf-8")
    with pytest.raises(policy.EvidencePolicyError, match="schema 7"):
        policy.validate_stale_pending_disposition(state, ledger, manifest_path=manifest_path)
    assert ledger.read_bytes() == b""


def test_real_production_resolver_remains_neutral_and_writer_tags_prospective():
    import pandas as pd
    import phase2d_prospective_validator as validator

    times = [1_800_000_000_000 + i * 900_000 for i in range(25)]
    rows = []
    for i, ts in enumerate(times):
        rows.append({
            "open_time": ts,
            "high": 136.0 if i == 1 else 110.0,
            "low": 95.0,
        })
    prediction = {
        "id": "candidate:BTCUSDT:1800000000000",
        "model": "candidate",
        "symbol": "BTCUSDT",
        "open_time": times[0],
        "signal": "BUY",
        "entry": 100.0,
        "atr": 10.0,
        "created_at": "2026-10-17T00:00:00+00:00",
    }
    result = validator.resolve_prediction(prediction, pd.DataFrame(rows))

    assert result is not None and result["status"] == "TP"
    assert "evidence_class" not in result
    tagged = policy.tag_prospective_observation(result)
    assert tagged["evidence_class"] == "PROSPECTIVE"
    assert "evidence_class" not in result  # writer returns a new object


def test_exact_stale_ids_from_recovered_authoritative_state_can_be_disposition_checked(tmp_path):
    source = Path("/mnt/data/phase2d_authoritative_run_41720710702/research_outputs/phase2d_state.json")
    if not source.is_file():
        pytest.skip("Authoritative recovery artifact is not mounted in this test environment")
    state = json.loads(source.read_text(encoding="utf-8"))
    stale_ids = policy._stale_pending_ids(state)
    assert {model: len(ids) for model, ids in stale_ids.items()} == policy.EXPECTED_PENDING_COUNTS

    # No event is appended.  This asserts that the exact real 618-ID set is
    # detected and that the normal validator will fail closed before resolving it.
    ledger = tmp_path / "phase2d_observations.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(policy.EvidencePolicyError, match="schema 7"):
        policy.validate_stale_pending_disposition(
            state,
            ledger,
            manifest_path=tmp_path / "manifest.json",
        )


def test_authoritative_f08_fresh_start_override_is_prohibited():
    with pytest.raises(policy.EvidencePolicyError, match="F08 fresh-start override is prohibited"):
        policy.validate_f08_execution_context(policy.EXPERIMENT_ID, True)


def test_f08_state_cannot_be_reopened_under_a_different_experiment_id():
    existing = {
        "locked_models": {"experiment_id": policy.EXPERIMENT_ID},
        "experiment_definition": {"experiment_id": policy.EXPERIMENT_ID},
    }
    with pytest.raises(policy.EvidencePolicyError, match="different experiment ID"):
        policy.validate_f08_execution_context("PHASE2D-EXPERIMENT-OTHER", False, existing)


def test_non_f08_experiment_keeps_its_separate_fresh_start_contract():
    assert policy.validate_f08_execution_context("PHASE2D-EXPERIMENT-OTHER", True) is False
