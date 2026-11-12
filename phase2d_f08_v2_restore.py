#!/usr/bin/env python3
"""Create, verify, and safely extract the quarantined F08 schema-7 V2 artifact.

V1 snapshot/archive restore contracts remain frozen.  This V2 package is a
separate, explicit artifact that proves the transition from the pinned schema-6
baseline to schema-7 without introducing scored rows for the 618 stale pending
predictions.  The F08 candidate branch contains a separate V2 bootstrap and
continuation handoff in run_phase2d.yml; that handoff remains an unapproved
release candidate until independently reviewed, merged, and all owner-controlled
release gates are satisfied.  It does not authorize F08 activation, promotion,
or live trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import phase2d_f08_disposition as disposition

ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_TYPE = "PHASE2D_F08_SCHEMA7_DISPOSITION"
ARTIFACT_STATUS = "QUARANTINED_PENDING_PIPELINE_HANDOFF"
BASELINE_NAMES = {
    "phase2d_state.json": "phase2d_state.json",
    "phase2d_results.json": "phase2d_results.json",
    "phase2d_latest_snapshot.json": "phase2d_latest_snapshot.json",
    "phase2d_observations.jsonl": "phase2d_observations.jsonl",
    "experiment_ledger.jsonl": "experiment_ledger.jsonl",
}
WORKING_NAMES = {
    "phase2d_state.json": "phase2d_state.json",
    "phase2d_results.json": "phase2d_results.json",
    "phase2d_latest_snapshot.json": "phase2d_latest_snapshot.json",
    "phase2d_observations.jsonl": "phase2d_observations.jsonl",
    "experiment_ledger.jsonl": "experiment_ledger.jsonl",
}
SOURCE_FILES = (
    "phase2d_f08_disposition.py",
    "phase2d_f08_v2_restore.py",
    "phase2d_f08_v2_handoff.py",
    "phase2d_f08_historical_backfill.py",
    "phase2d_f08_resolution_provenance.py",
    "phase2d_observation_ledger.py",
    "feature_engineering.py",
    "requirements.txt",
    ".gitattributes",
    "phase2d_prospective_validator.py",
    "phase2d_promotion_gate.py",
    "phase2d_evidence_policy.py",
    "phase2d_restore_guard.py",
    "phase2d_f08_snapshot_restore.py",
    "phase2d_f08_archive_restore.py",
    "tools/collect_f08_approval_evidence.py",
    ".github/workflows/run_phase2d.yml",
    ".github/workflows/phase2d_f08_validator_amendment.yml",
    ".github/workflows/phase2d_f08_controlled_disposition.yml",
)


class V2RestoreError(RuntimeError):
    """Raised when a V2 F08 artifact is incomplete, modified, or unsafe."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path, label: str) -> dict:
    disposition._reject_symlink_components(path, label)
    if path.is_symlink() or not path.is_file():
        raise V2RestoreError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise V2RestoreError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V2RestoreError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict]:
    disposition._reject_symlink_components(path, label)
    if path.is_symlink() or not path.is_file():
        raise V2RestoreError(f"{label} is missing or unsafe: {path}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise V2RestoreError(f"{label} must end with a newline")
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise V2RestoreError(f"{label} contains a blank line at {index}")
        try:
            row = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise V2RestoreError(f"{label} has invalid JSON at line {index}: {exc}") from exc
        if not isinstance(row, dict):
            raise V2RestoreError(f"{label} line {index} is not an object")
        rows.append(row)
    return rows


def _safe_copy(source: Path, destination: Path, label: str) -> None:
    disposition._reject_symlink_components(source, label)
    if source.is_symlink() or not source.is_file():
        raise V2RestoreError(f"Required {label} is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _assert_disjoint_directories(source: Path, destination: Path, label: str) -> None:
    """Reject equal or ancestor/descendant roots before a recursive copy."""
    disposition._reject_symlink_components(source, f"{label} source")
    disposition._reject_symlink_components(destination, f"{label} destination")
    try:
        source_root = source.expanduser().resolve(strict=True)
        destination_root = destination.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise V2RestoreError(
            f"{label} source/destination cannot be safely resolved: {exc}"
        ) from exc

    if (
        source_root == destination_root
        or source_root in destination_root.parents
        or destination_root in source_root.parents
    ):
        raise V2RestoreError(
            f"{label} source and destination directories overlap: "
            f"{source_root} <-> {destination_root}"
        )


def _copy_tree_no_links(source: Path, destination: Path, label: str) -> None:
    disposition._reject_symlink_components(source, label)
    if source.is_symlink() or not source.is_dir():
        raise V2RestoreError(f"Required {label} directory is missing or unsafe: {source}")
    _assert_disjoint_directories(source, destination, label)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise V2RestoreError(f"{label} contains a symlink: {path}")
        rel = path.relative_to(source)
        target = destination / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _safe_copy(path, target, f"{label} member")


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    disposition._reject_symlink_components(root, "V2 artifact root")
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise V2RestoreError(f"V2 artifact contains a symlink: {path}")
        if path.is_dir() or path.name == "manifest_v2.json":
            continue
        rel = path.relative_to(root).as_posix()
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or "\\" in rel:
            raise V2RestoreError(f"Unsafe artifact member path: {rel}")
        raw = path.read_bytes()
        inventory[rel] = {"sha256": _sha(raw), "size_bytes": len(raw)}
    return inventory


def _verify_transition(
    root: Path,
    *,
    spec: disposition.DispositionSpec = disposition.DispositionSpec(),
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    baseline = root / "baseline"
    post = root / "disposition"
    backfill = root / "historical_backfill"
    approvals = root / "approvals"
    source = root / "source"

    baseline_state_path = baseline / "phase2d_state.json"
    baseline_ledger_path = baseline / "phase2d_observations.jsonl"
    baseline_journal_path = baseline / "experiment_ledger.jsonl"
    state_path = post / "phase2d_state.json"
    ledger_path = post / "phase2d_observations.jsonl"
    journal_path = post / "experiment_ledger.jsonl"
    manifest_path = backfill / "manifest.json"
    bundle_path = backfill / "observations.jsonl"

    baseline_state_raw = baseline_state_path.read_bytes()
    baseline_ledger_raw = baseline_ledger_path.read_bytes()
    baseline_journal_raw = baseline_journal_path.read_bytes()
    if _sha(baseline_state_raw) != spec.state_sha256:
        raise V2RestoreError("V2 baseline state SHA256 is not the frozen schema-6 source")
    if _sha(baseline_ledger_raw) != spec.ledger_sha256:
        raise V2RestoreError("V2 baseline observation-ledger SHA256 is not the frozen source")
    if _sha(baseline_journal_raw) != spec.journal_sha256:
        raise V2RestoreError("V2 baseline experiment-journal SHA256 is not the frozen source")

    # These files are runtime continuation inputs.  They must be present in
    # both package roots and must remain byte-identical across disposition.
    runtime_artifact_hashes: dict[str, str] = {}
    for name in ("phase2d_results.json", "phase2d_latest_snapshot.json"):
        baseline_path = baseline / name
        post_path = post / name
        baseline_bytes = baseline_path.read_bytes()
        post_bytes = post_path.read_bytes()
        _read_json(baseline_path, f"V2 baseline {name}")
        _read_json(post_path, f"V2 disposition {name}")
        if baseline_bytes != post_bytes:
            raise V2RestoreError(f"V2 disposition changed runtime artifact {name}")
        runtime_artifact_hashes[name] = _sha(baseline_bytes)

    source_state = _read_json(baseline_state_path, "V2 baseline schema-6 state")
    population = disposition.pending_population(source_state, spec)
    bundle_info = disposition._verify_backfill_bundle(manifest_path, bundle_path, population, spec)
    evidence_dir = backfill / "evidence"
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise V2RestoreError("V2 historical-backfill evidence directory is missing or unsafe")
    if spec.experiment_id == disposition.EXPERIMENT_ID:
        # For the real F08 experiment, require the full source artifact / resolver
        # evidence contract, not only the manifest and deterministic JSONL bundle.
        try:
            from phase2d_evidence_policy import _verify_manifest_artifacts
            _verify_manifest_artifacts(
                bundle_info["manifest"],
                manifest_path.parent,
                population["ids_by_model"],
                spec.reconciliation_id,
                source_state,
            )
        except Exception as exc:
            raise V2RestoreError(f"V2 historical-backfill evidence validation failed: {exc}") from exc

    ledger_rows = _read_jsonl(ledger_path, "V2 disposition observation ledger")
    baseline_ledger_rows = _read_jsonl(baseline_ledger_path, "V2 baseline observation ledger")
    if len(ledger_rows) != len(baseline_ledger_rows) + 1:
        raise V2RestoreError("V2 official observation ledger must be exactly one append-only event longer than baseline")
    if ledger_path.read_bytes()[: len(baseline_ledger_raw)] != baseline_ledger_raw:
        raise V2RestoreError("V2 official observation ledger is not an exact byte-prefix extension")
    event = ledger_rows[-1]
    if event.get("event") != disposition.STALE_PENDING_EVENT or event.get("record_type") != "event":
        raise V2RestoreError("V2 official observation ledger tail is not STALE_PENDING_NOT_SCORED")
    if event.get("outcome_rows_appended") != 0 or event.get("pending_count") != spec.total:
        raise V2RestoreError("V2 disposition event claims outcome rows or incorrect pending count")
    stale_observation_keys = {
        f"{spec.experiment_id}:{model}:{prediction['symbol']}:{int(prediction['open_time'])}"
        for model, predictions in population["payloads_by_model"].items()
        for prediction in predictions
    }
    stale_prediction_ids = {value for values in population["ids_by_model"].values() for value in values}
    for row in ledger_rows:
        if row.get("record_type") == "observation" and (
            row.get("observation_key") in stale_observation_keys
            or row.get("id") in stale_prediction_ids
        ):
            raise V2RestoreError("V2 official ledger contains a scored stale-pending observation")
    if any(row.get("event") == disposition.STALE_PENDING_EVENT for row in ledger_rows[:-1]):
        raise V2RestoreError("V2 baseline already contains a stale disposition event")

    state = _read_json(state_path, "V2 schema-7 disposition state")
    try:
        disposition.validate_schema7_state(state, event, spec)
    except Exception as exc:
        raise V2RestoreError(f"V2 schema-7 state validation failed: {exc}") from exc

    expected_payloads = {
        (model, prediction["id"]): prediction
        for model, rows in population["payloads_by_model"].items()
        for prediction in rows
    }
    actual_payloads = {
        (model, item["id"]): item["prediction_payload"]
        for model in ("candidate", "production")
        for item in state["dispositioned"][model]
    }
    if actual_payloads != expected_payloads:
        raise V2RestoreError("V2 schema-7 state does not preserve every original pending prediction payload exactly")

    if journal_path.read_bytes()[: len(baseline_journal_raw)] != baseline_journal_raw:
        raise V2RestoreError("V2 experiment journal is not an exact byte-prefix extension")
    journal_rows = _read_jsonl(journal_path, "V2 disposition experiment journal")
    baseline_journal_rows = _read_jsonl(baseline_journal_path, "V2 baseline experiment journal")
    if len(journal_rows) != len(baseline_journal_rows) + 1:
        raise V2RestoreError("V2 experiment journal must contain exactly one validator-amendment extension")
    if journal_rows[-1].get("event") != disposition.VALIDATOR_AMENDMENT_EVENT:
        raise V2RestoreError("V2 experiment journal tail is not F08_VALIDATOR_AMENDMENT")
    if any(int(row.get("checkpoint_blocks", 0) or 0) >= 25 for row in journal_rows if row.get("event") == "F08_CHECKPOINT_EVALUATED"):
        raise V2RestoreError("V2 experiment journal contains a C25-or-later evaluation")
    checkpoints = {
        int(row.get("checkpoint_blocks", -1)): row
        for row in journal_rows if row.get("event") == "F08_CHECKPOINT_EVALUATED"
    }
    if set(checkpoints) != {15, 20}:
        raise V2RestoreError("V2 experiment journal does not preserve exactly C15 and C20")
    for number in (15, 20):
        checkpoint = checkpoints[number]
        if checkpoint.get("decision") != "CHECKPOINT_CONTINUE":
            raise V2RestoreError(f"V2 C{number} checkpoint decision differs from the frozen record")
        if checkpoint.get("candidate_profitable") is True or checkpoint.get("candidate_beats_production") is True:
            raise V2RestoreError(f"V2 C{number} checkpoint outcome differs from the frozen record")
    if sum(row.get("event") == "F08_GATE_AMENDMENT" for row in journal_rows) != 1:
        raise V2RestoreError("V2 experiment journal must retain exactly the one pre-existing F08_GATE_AMENDMENT")

    validator_path = source / "phase2d_prospective_validator.py"
    gate_path = source / "phase2d_promotion_gate.py"
    try:
        amendment = disposition.verify_validator_amendment_record(
            state=state,
            experiment_journal_path=journal_path,
            validator_path=validator_path,
            promotion_gate_path=gate_path,
            spec=spec,
        )
        expected_event = disposition._make_disposition_event_payload(
            population=population,
            bundle_info=bundle_info,
            approval=event["approval_provenance"],
            validator_amendment=amendment["event"],
            validator_amendment_sha256=amendment["event_sha256"],
            journal_sha256_after=_sha(journal_path.read_bytes()),
            spec=spec,
        )
        if not disposition._event_payload_matches(event, disposition.STALE_PENDING_EVENT, expected_event, spec.experiment_id):
            raise V2RestoreError("V2 disposition event does not match the exact immutable source/amendment/approval bindings")
        expected_state = disposition._build_schema7_state(source_state, population, event, spec)
        if state != expected_state:
            raise V2RestoreError("V2 schema-7 state differs from the exact deterministic schema-6-to-schema-7 migration")
        if spec.experiment_id == disposition.EXPERIMENT_ID:
            from phase2d_evidence_policy import validate_stale_pending_disposition
            validate_stale_pending_disposition(
                state,
                ledger_path,
                manifest_path=manifest_path,
                experiment_id=spec.experiment_id,
                experiment_journal_path=journal_path,
            )
    except Exception as exc:
        if isinstance(exc, V2RestoreError):
            raise
        raise V2RestoreError(f"V2 validator amendment verification failed: {exc}") from exc

    try:
        disposition._validate_stored_approval(
            event.get("approval_provenance"),
            expected_owner_id=disposition.OWNER_ID,
            environment_name=disposition.DISPOSITION_ENVIRONMENT,
            workflow_name=disposition.DISPOSITION_WORKFLOW,
        )
    except Exception as exc:
        raise V2RestoreError(f"V2 disposition owner approval evidence failed: {exc}") from exc

    expected_approval_files = {
        "disposition_approval_evidence.json": event.get("approval_provenance", {}).get("evidence_snapshot"),
        "validator_amendment_approval_evidence.json": amendment["event"].get("owner_approval", {}).get("evidence_snapshot"),
    }
    for filename, snapshot in expected_approval_files.items():
        if not isinstance(snapshot, dict):
            raise V2RestoreError(f"V2 event lacks the required exact approval snapshot: {filename}")
        approval_path = approvals / filename
        if _read_json(approval_path, f"V2 {filename}") != snapshot:
            raise V2RestoreError(f"V2 external approval evidence does not exactly match event snapshot: {filename}")
    amendment_context = amendment["event"]["owner_approval"]["evidence_snapshot"]["context"]
    disposition_context = event["approval_provenance"]["evidence_snapshot"]["context"]
    if amendment_context.get("run_id") != (root / "validator_amendment_run_id.txt").read_text(encoding="utf-8").strip():
        raise V2RestoreError("V2 validator-amendment approval run ID differs from handoff run ID")
    if disposition_context.get("run_id") != (root / "disposition_run_id.txt").read_text(encoding="utf-8").strip():
        raise V2RestoreError("V2 disposition approval run ID differs from artifact run ID")

    # Confirm immutable source files and source-code identity are present.
    for rel in SOURCE_FILES:
        path = source / rel
        if path.is_symlink() or not path.is_file():
            raise V2RestoreError(f"V2 source identity member is missing or unsafe: {rel}")
    if disposition.canonical_source_sha256(validator_path) != state["locked_models"].get("validator_code_hash"):
        raise V2RestoreError("V2 validator file hash differs from migrated locked model identity")
    if disposition.canonical_source_sha256(gate_path) != spec.promotion_gate_sha256:
        raise V2RestoreError("V2 promotion gate source differs from frozen gate hash")
    if expected_source_commit is not None:
        commit_path = root / "source_commit.txt"
        if not commit_path.is_file() or commit_path.read_text(encoding="utf-8").strip() != expected_source_commit:
            raise V2RestoreError("V2 artifact source commit does not match the expected commit")

    # Backfill manifest remains immutable and explicitly unapproved.
    manifest = _read_json(manifest_path, "V2 historical-backfill manifest")
    if manifest.get("manifest_status") != "READY_FOR_OWNER_REVIEW" or manifest.get("approval_status") != "PENDING_OWNER_APPROVAL":
        raise V2RestoreError("V2 historical-backfill manifest was changed to an approved state")
    if "approval_provenance" in manifest:
        raise V2RestoreError("V2 historical-backfill manifest must not contain owner approval")

    return {
        "observation_count": len(disposition._read_jsonl_bytes(baseline_ledger_raw, "baseline ledger")),
        "official_event_extension_count": 1,
        "official_outcome_rows_appended": 0,
        "dispositioned_by_model": spec.model_counts(),
        "dispositioned_total": spec.total,
        "pending_ids_sha256": spec.pending_ids_sha256,
        "validator_amendment_event_id": amendment["event_id"],
        "validator_amendment_event_sha256": amendment["event_sha256"],
        "disposition_event_id": event["event_id"],
        "disposition_event_sha256": disposition._event_sha256(event),
        "c15_preserved": True,
        "c20_preserved": True,
        "c25_evaluated": False,
        "promotion_authorized": False,
        "live_trading_enabled": False,
        "bundle_manifest_sha256": bundle_info["manifest_sha256"],
        "historical_bundle_sha256": bundle_info["bundle_sha256"],
        "runtime_artifacts_unchanged": runtime_artifact_hashes,
    }


def _build_package(
    *, baseline_dir: Path, working_dir: Path, source_root: Path, output_dir: Path,
    source_commit: str, validator_amendment_run_id: str, disposition_run_id: str,
    spec: disposition.DispositionSpec,
) -> dict[str, Any]:
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise V2RestoreError("Source commit must be a 40-character lowercase Git SHA")
    for name, value in (("validator amendment run ID", validator_amendment_run_id), ("disposition run ID", disposition_run_id)):
        if not isinstance(value, str) or not value.isdigit():
            raise V2RestoreError(f"{name} is malformed")
    if output_dir.exists():
        raise V2RestoreError(f"Refusing to overwrite existing V2 artifact directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        # Frozen baseline copies.
        for name in BASELINE_NAMES:
            _safe_copy(baseline_dir / name, staging / "baseline" / name, f"baseline/{name}")
        # Post-disposition schema-7 copies.
        for name in WORKING_NAMES:
            _safe_copy(working_dir / name, staging / "disposition" / name, f"disposition/{name}")
        _safe_copy(working_dir / "f08_historical_backfill" / "manifest.json", staging / "historical_backfill" / "manifest.json", "historical-backfill manifest")
        _safe_copy(working_dir / "f08_historical_backfill" / "observations.jsonl", staging / "historical_backfill" / "observations.jsonl", "historical-backfill bundle")
        _safe_copy(
            working_dir / "f08_historical_backfill" / disposition.PROVENANCE_FILENAME,
            staging / "historical_backfill" / disposition.PROVENANCE_FILENAME,
            "historical-backfill resolution provenance",
        )
        _copy_tree_no_links(working_dir / "f08_historical_backfill" / "evidence", staging / "historical_backfill" / "evidence", "historical-backfill evidence")
        _safe_copy(working_dir / "disposition_approval_evidence.json", staging / "approvals" / "disposition_approval_evidence.json", "disposition approval evidence")
        _safe_copy(working_dir / "validator_amendment_approval_evidence.json", staging / "approvals" / "validator_amendment_approval_evidence.json", "validator-amendment approval evidence")
        for rel in SOURCE_FILES:
            _safe_copy(source_root / rel, staging / "source" / rel, f"source/{rel}")
        (staging / "source_commit.txt").write_text(source_commit + "\n", encoding="ascii")
        (staging / "validator_amendment_run_id.txt").write_text(validator_amendment_run_id + "\n", encoding="ascii")
        (staging / "disposition_run_id.txt").write_text(disposition_run_id + "\n", encoding="ascii")

        verification = _verify_transition(staging, spec=spec, expected_source_commit=source_commit)
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_status": ARTIFACT_STATUS,
            "experiment_id": spec.experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_commit": source_commit,
            "validator_amendment_run_id": validator_amendment_run_id,
            "disposition_run_id": disposition_run_id,
            "reconciliation_id": spec.reconciliation_id,
            "source_heads": {
                "state_sha256": spec.state_sha256,
                "observation_ledger_sha256": spec.ledger_sha256,
                "experiment_journal_sha256": spec.journal_sha256,
            },
            "post_disposition_heads": {
                "state_sha256": _sha((staging / "disposition/phase2d_state.json").read_bytes()),
                "observation_ledger_sha256": _sha((staging / "disposition/phase2d_observations.jsonl").read_bytes()),
                "experiment_journal_sha256": _sha((staging / "disposition/experiment_ledger.jsonl").read_bytes()),
            },
            "verification": verification,
            "restore_policy": {
                "v1_contract_modified": False,
                "normal_phase2d_pipeline_handoff": False,
                "c25_evaluated": False,
                "promotion_authorized": False,
                "live_trading_enabled": False,
            },
            "files": _inventory(staging),
        }
        (staging / "manifest_v2.json").write_bytes(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
        # Validate the just-sealed package before publishing its directory.
        verify_artifact(staging, spec=spec, expected_source_commit=source_commit, runtime_source_root=source_root)
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_artifact(
    artifact_dir: Path,
    *,
    spec: disposition.DispositionSpec = disposition.DispositionSpec(),
    expected_source_commit: str | None = None,
    runtime_source_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest_v2.json"
    manifest = _read_json(manifest_path, "V2 artifact manifest")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION or manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise V2RestoreError("V2 artifact manifest type/schema mismatch")
    if manifest.get("artifact_status") != ARTIFACT_STATUS:
        raise V2RestoreError("V2 artifact is not quarantined pending pipeline handoff")
    if manifest.get("experiment_id") != spec.experiment_id or manifest.get("reconciliation_id") != spec.reconciliation_id:
        raise V2RestoreError("V2 artifact experiment/reconciliation identity mismatch")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise V2RestoreError("V2 artifact source commit is malformed")
    if expected_source_commit and commit != expected_source_commit:
        raise V2RestoreError("V2 artifact source commit differs from requested source commit")
    if manifest.get("source_heads") != {
        "state_sha256": spec.state_sha256,
        "observation_ledger_sha256": spec.ledger_sha256,
        "experiment_journal_sha256": spec.journal_sha256,
    }:
        raise V2RestoreError("V2 artifact baseline source heads differ from frozen values")
    for run_key in ("validator_amendment_run_id", "disposition_run_id"):
        value = manifest.get(run_key)
        if not isinstance(value, str) or not value.isdigit():
            raise V2RestoreError(f"V2 artifact {run_key} is malformed")
    recorded_inventory = manifest.get("files")
    if not isinstance(recorded_inventory, dict) or not recorded_inventory:
        raise V2RestoreError("V2 artifact file inventory is missing")
    actual_inventory = _inventory(artifact_dir)
    if actual_inventory != recorded_inventory:
        raise V2RestoreError("V2 artifact file inventory/hash/size verification failed")
    verification = _verify_transition(artifact_dir, spec=spec, expected_source_commit=commit)
    if manifest.get("verification") != verification:
        raise V2RestoreError("V2 artifact verification summary differs from independently recomputed values")
    if manifest.get("restore_policy") != {
        "v1_contract_modified": False,
        "normal_phase2d_pipeline_handoff": False,
        "c25_evaluated": False,
        "promotion_authorized": False,
        "live_trading_enabled": False,
    }:
        raise V2RestoreError("V2 artifact restore-policy declaration is invalid")
    if runtime_source_root is not None:
        for rel in SOURCE_FILES:
            runtime_file = runtime_source_root / rel
            artifact_file = artifact_dir / "source" / rel
            disposition._reject_symlink_components(runtime_file, f"runtime source {rel}")
            if not runtime_file.is_file() or _sha(runtime_file.read_bytes()) != _sha(artifact_file.read_bytes()):
                raise V2RestoreError(f"V2 artifact source differs from the runtime checkout: {rel}")
    return {"status": "PASS", "artifact_status": ARTIFACT_STATUS, "verification": verification}


def restore_artifact(
    artifact_dir: Path,
    output_dir: Path,
    *,
    spec: disposition.DispositionSpec = disposition.DispositionSpec(),
    expected_source_commit: str | None = None,
    runtime_source_root: Path | None = None,
) -> dict[str, Any]:
    result = verify_artifact(
        artifact_dir,
        spec=spec,
        expected_source_commit=expected_source_commit,
        runtime_source_root=runtime_source_root,
    )
    _assert_disjoint_directories(artifact_dir, output_dir, "V2 restore")
    if output_dir.exists() and not output_dir.is_dir():
        raise V2RestoreError(f"Restore destination exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise V2RestoreError(f"Refusing to overwrite non-empty restore destination: {output_dir}")
    disposition._reject_symlink_components(output_dir, "V2 restore destination")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.restore.", dir=output_dir.parent))
    try:
        _copy_tree_no_links(artifact_dir, staging, "complete verified V2 artifact")
        # Verify the complete restored copy, including frozen baseline and inventory.
        verify_artifact(
            staging,
            spec=spec,
            expected_source_commit=expected_source_commit,
            runtime_source_root=runtime_source_root,
        )
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**result, "restored_to": str(output_dir), "normal_pipeline_handoff": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    for name in ("baseline-dir", "working-dir", "source-root", "output-dir"):
        seal.add_argument(f"--{name}", required=True, type=Path)
    seal.add_argument("--source-commit", required=True)
    seal.add_argument("--validator-amendment-run-id", required=True)
    seal.add_argument("--disposition-run-id", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--artifact-dir", required=True, type=Path)
    verify.add_argument("--expected-source-commit")

    restore = sub.add_parser("restore")
    restore.add_argument("--artifact-dir", required=True, type=Path)
    restore.add_argument("--output-dir", required=True, type=Path)
    restore.add_argument("--expected-source-commit")

    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = _build_package(
                baseline_dir=args.baseline_dir,
                working_dir=args.working_dir,
                source_root=args.source_root,
                output_dir=args.output_dir,
                source_commit=args.source_commit,
                validator_amendment_run_id=args.validator_amendment_run_id,
                disposition_run_id=args.disposition_run_id,
                spec=disposition.DispositionSpec(),
            )
            print("F08 V2 schema-7 artifact seal: PASS")
            print("Status:", ARTIFACT_STATUS)
            print("File count:", len(result["files"]))
            print("Disposition event:", result["verification"]["disposition_event_id"])
            print("Outcome rows appended:", result["verification"]["official_outcome_rows_appended"])
            print("Normal Phase 2D handoff: NO")
        elif args.command == "verify":
            result = verify_artifact(
                args.artifact_dir,
                expected_source_commit=args.expected_source_commit,
                runtime_source_root=Path(__file__).resolve().parent,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            result = restore_artifact(
                args.artifact_dir,
                args.output_dir,
                expected_source_commit=args.expected_source_commit,
                runtime_source_root=Path(__file__).resolve().parent,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
