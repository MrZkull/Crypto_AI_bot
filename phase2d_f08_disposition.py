#!/usr/bin/env python3
"""Controlled F08 schema-7 disposition and validator-amendment operations.

This module deliberately separates the immutable historical-backfill manifest
from the later owner approval.  The official observation ledger receives one
STALE_PENDING_NOT_SCORED event only; historical TP/SL/EXPIRED rows remain in
the separate HISTORICAL_BACKFILL bundle and are never inserted as observations.

The two write operations are intentionally explicit:
  * amend-validator: append one independently owner-approved
    F08_VALIDATOR_AMENDMENT to the experiment journal; no state/observation
    ledger mutation.
  * apply-disposition: require that amendment and a separate protected
    environment approval, append one disposition event, then atomically migrate
    the 618 stale pending payloads from pending to schema-7 dispositioned.

No default command writes.  No operation supports a bypass flag.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2d_f08_resolution_provenance import (
    PROVENANCE_FILENAME,
    RESOLVED_AT_SEMANTICS,
    OUTCOME_BAR_CLOSE_TIME_SEMANTICS,
    index_raw_windows as index_provenance_raw_windows,
    verify_resolution_provenance_records,
)

EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
LEGACY_CUTOFF_UTC = "2026-10-16T14:08:36.946361+00:00"
LEGACY_CUTOFF = datetime.fromisoformat(LEGACY_CUTOFF_UTC)

STALE_PENDING_EVENT = "STALE_PENDING_NOT_SCORED"
VALIDATOR_AMENDMENT_EVENT = "F08_VALIDATOR_AMENDMENT"
LEDGER_SCHEMA_VERSION = 1
STATE_SCHEMA_V6 = 6
STATE_SCHEMA_V7 = 7

OWNER_LOGIN = "MrZkull"
OWNER_ID = "166250462"
REPOSITORY = "MrZkull/Crypto_AI_bot"
VALIDATOR_AMENDMENT_ENVIRONMENT = "f08-validator-amendment"
DISPOSITION_ENVIRONMENT = "f08-disposition"
VALIDATOR_AMENDMENT_WORKFLOW = "Phase2D F08 Validator Amendment"
DISPOSITION_WORKFLOW = "Phase2D F08 Controlled Disposition"
VALIDATOR_AMENDMENT_WORKFLOW_FILE = ".github/workflows/phase2d_f08_validator_amendment.yml"
DISPOSITION_WORKFLOW_FILE = ".github/workflows/phase2d_f08_controlled_disposition.yml"

# These are immutable source identities from the 2026-11-06 F08 checkpoint.
EXPECTED_RECONCILIATION_ID = "F08-HIST-BF-c8a8330915c85db8fe923615e2df2a58"
EXPECTED_STATE_SHA256 = "4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5"
EXPECTED_LEDGER_SHA256 = "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9"
EXPECTED_JOURNAL_SHA256 = "cf12ac74b39e44d8487951bb1abde203f12140a626d7c78fc388ebd91e80e267"
EXPECTED_MANIFEST_SHA256 = "a32917de06548c3d4d0bcc6ec09b7c6974c00235dc90ef86db0df057b836c691"
EXPECTED_BUNDLE_SHA256 = "3f05c466810fc86a911d4105575c951a59a73e78a3b6271c9c01cb0042386878"
EXPECTED_PENDING_IDS_SHA256 = "a1751b84e4fedd5dad40348392bac0ccf8c1be00a419118d152ee561a9d1c644"
EXPECTED_OLD_VALIDATOR_SHA256 = "2b9f16f87f8222fbb00cfd0e92a0f149dfefdd107d0bfa3079a9a97040e299c5"
EXPECTED_PROMOTION_GATE_SHA256 = "f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab"
EXPECTED_COUNTS = {"candidate": 74, "production": 544}
EXPECTED_TOTAL = 618
PINNED_MANIFEST_PATH = "research_outputs/f08_historical_backfill/manifest.json"


class DispositionError(RuntimeError):
    """Raised when F08 disposition authorization or state integrity fails."""


@dataclass(frozen=True)
class DispositionSpec:
    experiment_id: str = EXPERIMENT_ID
    cutoff_utc: str = LEGACY_CUTOFF_UTC
    reconciliation_id: str = EXPECTED_RECONCILIATION_ID
    state_sha256: str = EXPECTED_STATE_SHA256
    ledger_sha256: str = EXPECTED_LEDGER_SHA256
    journal_sha256: str = EXPECTED_JOURNAL_SHA256
    manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    bundle_sha256: str = EXPECTED_BUNDLE_SHA256
    pending_ids_sha256: str = EXPECTED_PENDING_IDS_SHA256
    old_validator_sha256: str = EXPECTED_OLD_VALIDATOR_SHA256
    promotion_gate_sha256: str = EXPECTED_PROMOTION_GATE_SHA256
    counts: dict[str, int] | None = None
    total: int = EXPECTED_TOTAL
    manifest_path: str = PINNED_MANIFEST_PATH

    def model_counts(self) -> dict[str, int]:
        return dict(self.counts or EXPECTED_COUNTS)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_source_sha256(path: Path) -> str:
    """Hash text source consistently across Windows CRLF and Linux LF checkouts."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DispositionError(f"Source file is not UTF-8 text: {path}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return sha256_bytes(normalized)


def _reject_symlink_components(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        is_junction = getattr(current, "is_junction", None)
        if current.is_symlink() or (callable(is_junction) and is_junction()):
            raise DispositionError(f"{label} traverses a symlink/junction: {current}")
    return absolute


def _assert_distinct_paths(
    named_paths: tuple[tuple[Path, str], ...], *, operation: str,
) -> None:
    """Refuse path aliases between mutable destinations and protected inputs.

    Path.resolve catches relative/absolute and symlink spelling aliases.
    os.path.samefile additionally catches hard links when both paths exist.
    This check runs before any file is read for an operation or any write occurs.
    """
    identities: list[tuple[Path, str, str]] = []
    for path, label in named_paths:
        try:
            resolved = path.expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise DispositionError(
                f"{operation} cannot safely resolve {label}: {path}: {exc}"
            ) from exc
        key = os.path.normcase(os.path.normpath(os.fspath(resolved)))
        for prior_path, prior_label, prior_key in identities:
            aliases = key == prior_key
            if not aliases and path.exists() and prior_path.exists():
                try:
                    aliases = os.path.samefile(path, prior_path)
                except OSError as exc:
                    raise DispositionError(
                        f"{operation} cannot safely compare {label} and {prior_label}: {exc}"
                    ) from exc
            if aliases:
                raise DispositionError(
                    f"{operation} refuses aliased paths: {label} and {prior_label}"
                )
        identities.append((path, label, key))


def _read_json(path: Path, label: str) -> dict:
    _reject_symlink_components(path, label)
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise DispositionError(f"Cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispositionError(f"{label} must be a JSON object")
    return value


def _read_jsonl_bytes(raw: bytes, label: str, *, require_terminal_newline: bool = True) -> list[dict]:
    if require_terminal_newline and raw and not raw.endswith(b"\n"):
        raise DispositionError(f"{label} is not newline-terminated")
    rows: list[dict] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise DispositionError(f"{label} contains a blank line at {line_number}")
        try:
            value = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise DispositionError(f"Invalid {label} JSON at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise DispositionError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DispositionError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DispositionError(f"{label} is invalid: {value!r}") from exc
    if parsed.tzinfo is None:
        raise DispositionError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def pending_population(state: dict, spec: DispositionSpec = DispositionSpec()) -> dict:
    if state.get("schema_version") != STATE_SCHEMA_V6:
        raise DispositionError("Disposition source state must be schema 6")
    locked = state.get("locked_models")
    if not isinstance(locked, dict) or locked.get("experiment_id") != spec.experiment_id:
        raise DispositionError("Source state experiment identity mismatch")
    if locked.get("validator_code_hash") != spec.old_validator_sha256:
        raise DispositionError("Source state old validator identity mismatch")
    if "dispositioned" in state:
        raise DispositionError("Schema-6 source unexpectedly already contains dispositioned data")

    pending = state.get("pending")
    seen = state.get("seen")
    resolved = state.get("resolved")
    if not all(isinstance(obj, dict) for obj in (pending, seen, resolved)):
        raise DispositionError("Source state pending/seen/resolved objects are invalid")

    ids_by_model: dict[str, list[str]] = {"candidate": [], "production": []}
    payloads_by_model: dict[str, list[dict]] = {"candidate": [], "production": []}
    for model in ("candidate", "production"):
        rows = pending.get(model)
        model_seen = seen.get(model)
        model_resolved = resolved.get(model)
        if not isinstance(rows, list) or not isinstance(model_seen, list) or not isinstance(model_resolved, list):
            raise DispositionError(f"Source state {model} arrays are invalid")
        known_seen = set(model_seen)
        resolved_ids = {str(row.get("id")) for row in model_resolved if isinstance(row, dict)}
        if len(resolved_ids) != len(model_resolved):
            raise DispositionError(f"Source state {model} resolved population is malformed or duplicated")
        seen_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise DispositionError(f"Source state pending/{model} contains a non-object")
            prediction_id = row.get("id")
            if not isinstance(prediction_id, str) or not prediction_id or prediction_id in seen_ids:
                raise DispositionError(f"Source state pending/{model} has a missing or duplicate ID")
            seen_ids.add(prediction_id)
            symbol, open_time = row.get("symbol"), row.get("open_time")
            if not isinstance(symbol, str) or not symbol or open_time is None:
                raise DispositionError(f"Source state pending/{model}/{prediction_id} has invalid identity")
            try:
                key = f"{symbol}:{int(open_time)}"
            except Exception as exc:
                raise DispositionError(f"Source state pending/{model}/{prediction_id} has invalid open_time") from exc
            if key not in known_seen:
                raise DispositionError(f"Source state pending/{model}/{prediction_id} is absent from seen")
            if prediction_id in resolved_ids:
                raise DispositionError(f"Prediction {model}/{prediction_id} overlaps pending and resolved")
            created = _parse_utc(row.get("created_at"), f"pending/{model}/{prediction_id}/created_at")
            if created <= datetime.fromisoformat(spec.cutoff_utc):
                ids_by_model[model].append(prediction_id)
                payloads_by_model[model].append(copy.deepcopy(row))
        ids_by_model[model].sort()
        payloads_by_model[model].sort(key=lambda row: row["id"])

    counts = {model: len(ids) for model, ids in ids_by_model.items()}
    if counts != spec.model_counts() or sum(counts.values()) != spec.total:
        raise DispositionError(f"Stale pending cohort mismatch: expected={spec.model_counts()}, actual={counts}")

    digest = sha256_bytes(canonical_json_bytes(ids_by_model))
    if digest != spec.pending_ids_sha256:
        raise DispositionError(f"Pending-ID digest mismatch: expected={spec.pending_ids_sha256}, actual={digest}")

    return {
        "ids_by_model": ids_by_model,
        "payloads_by_model": payloads_by_model,
        "counts": counts,
        "total": sum(counts.values()),
        "pending_ids_sha256": digest,
    }


def _verify_source_heads(
    state_path: Path,
    ledger_path: Path,
    journal_baseline_path: Path,
    spec: DispositionSpec,
) -> tuple[bytes, bytes, bytes]:
    for path, label in (
        (state_path, "authoritative state"),
        (ledger_path, "official observation ledger"),
        (journal_baseline_path, "baseline experiment journal"),
    ):
        _reject_symlink_components(path, label)
        if not path.is_file():
            raise DispositionError(f"Missing {label}: {path}")
    state_raw = state_path.read_bytes()
    ledger_raw = ledger_path.read_bytes()
    journal_raw = journal_baseline_path.read_bytes()
    actuals = {
        "state": sha256_bytes(state_raw),
        "ledger": sha256_bytes(ledger_raw),
        "journal": sha256_bytes(journal_raw),
    }
    expected = {
        "state": spec.state_sha256,
        "ledger": spec.ledger_sha256,
        "journal": spec.journal_sha256,
    }
    if actuals != expected:
        raise DispositionError(f"Pinned source heads mismatch: expected={expected}, actual={actuals}")
    return state_raw, ledger_raw, journal_raw


def _verify_backfill_bundle(
    manifest_path: Path,
    bundle_path: Path,
    population: dict,
    spec: DispositionSpec,
) -> dict:
    _reject_symlink_components(manifest_path, "backfill manifest")
    _reject_symlink_components(bundle_path, "backfill bundle")
    if not manifest_path.is_file() or not bundle_path.is_file():
        raise DispositionError("Immutable historical-backfill manifest/bundle is missing")
    manifest_raw = manifest_path.read_bytes()
    bundle_raw = bundle_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_raw)
    bundle_sha = sha256_bytes(bundle_raw)
    if manifest_sha != spec.manifest_sha256:
        raise DispositionError(f"Immutable manifest hash mismatch: {manifest_sha}")
    if bundle_sha != spec.bundle_sha256:
        raise DispositionError(f"Immutable backfill bundle hash mismatch: {bundle_sha}")
    manifest = _read_json(manifest_path, "historical-backfill manifest")
    required = {
        "manifest_schema_version": 1,
        "operation": STALE_PENDING_EVENT,
        "experiment_id": spec.experiment_id,
        "evidence_class": "HISTORICAL_BACKFILL",
        "legacy_cutoff_utc": spec.cutoff_utc,
        "manifest_status": "READY_FOR_OWNER_REVIEW",
        "approval_status": "PENDING_OWNER_APPROVAL",
        "reconciliation_id": spec.reconciliation_id,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise DispositionError(f"Immutable manifest {key} mismatch: expected={expected!r}, got={manifest.get(key)!r}")
    if "approval_provenance" in manifest:
        raise DispositionError("Approval data must not be inserted into the immutable backfill manifest")

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise DispositionError("Manifest source_hashes is missing")
    for key, expected in {
        "state": spec.state_sha256,
        "ledger": spec.ledger_sha256,
        "journal": spec.journal_sha256,
    }.items():
        if source_hashes.get(key) != expected:
            raise DispositionError(f"Manifest source hash mismatch for {key}")

    bundle = manifest.get("historical_bundle")
    if not isinstance(bundle, dict):
        raise DispositionError("Manifest historical_bundle object is missing")
    bundle_expected = {
        "bundle_sha256": spec.bundle_sha256,
        "observation_count": spec.total,
        "candidate_count": spec.model_counts()["candidate"],
        "production_count": spec.model_counts()["production"],
        "evidence_class": "HISTORICAL_BACKFILL",
        "format": "jsonl",
    }
    for key, expected in bundle_expected.items():
        if bundle.get(key) != expected:
            raise DispositionError(f"Manifest historical_bundle {key} mismatch")
    if bundle.get("bundle_path") != bundle_path.name:
        raise DispositionError("Manifest bundle_path does not match supplied bundle file")

    pending = manifest.get("pending_population")
    if not isinstance(pending, dict):
        raise DispositionError("Manifest pending_population is missing")
    if pending.get("pending_count") != spec.total or pending.get("pending_ids_sha256") != population["pending_ids_sha256"]:
        raise DispositionError("Manifest pending population count/digest mismatch")
    manifest_ids = pending.get("ids_by_model")
    if manifest_ids != population["ids_by_model"]:
        raise DispositionError("Manifest pending IDs do not match source state")

    rows = _read_jsonl_bytes(bundle_raw, "historical backfill bundle")
    if len(rows) != spec.total:
        raise DispositionError(f"Historical backfill bundle count mismatch: {len(rows)}")

    # The resolver timestamp and OHLC outcome-bar close boundary live in a
    # separately hashed sidecar, never in deterministic HISTORICAL_BACKFILL rows.
    provenance_meta = manifest.get("resolution_provenance")
    if not isinstance(provenance_meta, dict):
        raise DispositionError("Per-prediction resolution_provenance metadata is missing")
    source_hashes = manifest.get("source_hashes")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_hashes, dict) or not isinstance(source_artifacts, dict):
        raise DispositionError("Manifest source hash/artifact descriptors are missing")
    provenance_rel = provenance_meta.get("path")
    if (
        provenance_meta.get("schema_version") != 1
        or provenance_meta.get("format") != "jsonl"
        or provenance_rel != PROVENANCE_FILENAME
        or provenance_meta.get("record_count") != spec.total
        or provenance_meta.get("raw_windows_sha256") != source_hashes.get("raw_windows")
        or provenance_meta.get("resolved_at_semantics") != RESOLVED_AT_SEMANTICS
        or provenance_meta.get("outcome_bar_close_time_semantics") != OUTCOME_BAR_CLOSE_TIME_SEMANTICS
    ):
        raise DispositionError("Per-prediction resolution-provenance metadata is inconsistent")
    provenance_path = manifest_path.parent / PROVENANCE_FILENAME
    _reject_symlink_components(provenance_path, "resolution-provenance sidecar")
    if not provenance_path.is_file():
        raise DispositionError("Per-prediction resolution-provenance sidecar is missing or unsafe")
    provenance_raw = provenance_path.read_bytes()
    provenance_sha = sha256_bytes(provenance_raw)
    if provenance_sha != provenance_meta.get("sha256"):
        raise DispositionError("Per-prediction resolution-provenance SHA256 mismatch")
    provenance_rows = _read_jsonl_bytes(provenance_raw, "resolution provenance")

    raw_descriptor = source_artifacts.get("raw_windows")
    if not isinstance(raw_descriptor, dict):
        raise DispositionError("Manifest raw_windows evidence descriptor is missing")
    raw_rel_value = raw_descriptor.get("path")
    if not isinstance(raw_rel_value, str) or "\\" in raw_rel_value:
        raise DispositionError("Manifest raw_windows evidence path is invalid")
    raw_rel = Path(raw_rel_value)
    if raw_rel.is_absolute() or ".." in raw_rel.parts or not raw_rel.parts:
        raise DispositionError("Manifest raw_windows evidence path is unsafe")
    raw_windows_path = manifest_path.parent / raw_rel
    _reject_symlink_components(raw_windows_path, "raw OHLC evidence")
    if not raw_windows_path.is_file():
        raise DispositionError("Manifest raw OHLC evidence copy is missing")
    raw_windows_bytes = raw_windows_path.read_bytes()
    raw_windows_sha = sha256_bytes(raw_windows_bytes)
    if (
        raw_windows_sha != source_hashes.get("raw_windows")
        or raw_windows_sha != raw_descriptor.get("sha256")
    ):
        raise DispositionError("Raw OHLC evidence SHA256 does not match the immutable manifest")
    raw_window_rows = _read_jsonl_bytes(raw_windows_bytes, "raw OHLC evidence")
    try:
        raw_windows_by_key = index_provenance_raw_windows(raw_window_rows)
        provenance_verification = verify_resolution_provenance_records(
            provenance_rows, rows, raw_windows_by_key, spec.reconciliation_id,
            expected_counts=spec.model_counts(),
        )
    except Exception as exc:
        raise DispositionError(f"Per-prediction resolution provenance is invalid: {exc}") from exc
    if provenance_verification.get("records_verified") != spec.total:
        raise DispositionError("Per-prediction resolution provenance did not verify the full population")

    observed = {"candidate": set(), "production": set()}
    for index, row in enumerate(rows, start=1):
        if row.get("record_type") != "historical_backfill_observation":
            raise DispositionError(f"Bundle row {index} is not a HISTORICAL_BACKFILL row")
        if row.get("evidence_class") != "HISTORICAL_BACKFILL":
            raise DispositionError(f"Bundle row {index} has an invalid evidence_class")
        if row.get("experiment_id") != spec.experiment_id or row.get("reconciliation_id") != spec.reconciliation_id:
            raise DispositionError(f"Bundle row {index} has invalid experiment/reconciliation identity")
        model, prediction_id = row.get("model"), row.get("id")
        if model not in observed or not isinstance(prediction_id, str) or not prediction_id:
            raise DispositionError(f"Bundle row {index} has invalid model/prediction ID")
        if prediction_id in observed[model]:
            raise DispositionError(f"Bundle has duplicate prediction ID {model}/{prediction_id}")
        observed[model].add(prediction_id)
        if "resolved_at" in row:
            raise DispositionError("Volatile resolved_at must not be embedded in deterministic backfill rows")
        # Deliberately do not consume or append outcome values to the official ledger.
    if observed != {model: set(ids) for model, ids in population["ids_by_model"].items()}:
        raise DispositionError("Historical backfill bundle IDs do not exactly match stale pending IDs")

    return {"manifest": manifest, "manifest_sha256": manifest_sha, "bundle_sha256": bundle_sha}


def _validate_approval_evidence_object(
    evidence: dict,
    *,
    expected_owner_id: str,
    environment_name: str,
    workflow_name: str,
) -> dict:
    """Validate a captured GitHub environment + run-approval API response."""
    if not isinstance(expected_owner_id, str) or not expected_owner_id.isdigit():
        raise DispositionError("Expected owner numeric GitHub user ID is missing or malformed")
    if not isinstance(evidence, dict) or set(evidence) != {"context", "environment", "approval_history"}:
        raise DispositionError("Approval evidence must contain exactly context, environment, approval_history")
    context = evidence["context"]
    environment = evidence["environment"]
    approvals = evidence["approval_history"]
    if not isinstance(context, dict) or not isinstance(environment, dict) or not isinstance(approvals, list):
        raise DispositionError("Approval evidence structures are invalid")

    context_required = {
        "repository", "workflow", "workflow_file", "event_name", "ref", "run_id",
        "run_attempt", "head_sha", "actor_login", "actor_id",
    }
    if set(context) != context_required:
        raise DispositionError("Workflow context fields are incomplete or unexpected")
    if context["repository"] != REPOSITORY:
        raise DispositionError("Approval workflow repository mismatch")
    if context["workflow"] != workflow_name:
        raise DispositionError("Approval workflow name mismatch")
    expected_workflow_file = (
        VALIDATOR_AMENDMENT_WORKFLOW_FILE
        if environment_name == VALIDATOR_AMENDMENT_ENVIRONMENT
        else DISPOSITION_WORKFLOW_FILE
    )
    if context["workflow_file"] != expected_workflow_file:
        raise DispositionError("Approval workflow file is not the pinned controlled workflow")
    if context["event_name"] != "workflow_dispatch" or context["ref"] != "refs/heads/main":
        raise DispositionError("Approval must be a workflow_dispatch from refs/heads/main")
    if not isinstance(context["workflow_file"], str) or not context["workflow_file"].startswith(".github/workflows/"):
        raise DispositionError("Approval workflow file path is invalid")
    if not isinstance(context["run_id"], str) or not context["run_id"].isdigit():
        raise DispositionError("Workflow run ID is malformed")
    if not isinstance(context["run_attempt"], int) or isinstance(context["run_attempt"], bool) or context["run_attempt"] != 1:
        raise DispositionError("Approval attestations must come from the first attempt of a new workflow run; reruns are not accepted")
    if not isinstance(context["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", context["head_sha"]):
        raise DispositionError("Workflow head SHA is malformed")
    if not isinstance(context["actor_login"], str) or not context["actor_login"]:
        raise DispositionError("Workflow initiator login is missing")
    if not isinstance(context["actor_id"], str) or not context["actor_id"].isdigit():
        raise DispositionError("Workflow initiator numeric ID is missing or malformed")
    if context["actor_id"] == expected_owner_id:
        raise DispositionError("Workflow initiator cannot self-approve the protected environment")

    if environment.get("name") != environment_name:
        raise DispositionError("GitHub environment name mismatch")
    branch_policy = environment.get("deployment_branch_policy")
    if branch_policy != {"protected_branches": True, "custom_branch_policies": False}:
        raise DispositionError("GitHub environment must be restricted to protected branches")
    protection_rules = environment.get("protection_rules")
    if not isinstance(protection_rules, list):
        raise DispositionError("GitHub environment protection_rules are missing")
    required_reviewer_rules = [
        rule for rule in protection_rules
        if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
    ]
    if len(required_reviewer_rules) != 1:
        raise DispositionError("Exactly one required_reviewers protection rule is required")
    rule = required_reviewer_rules[0]
    if rule.get("prevent_self_review") is not True:
        raise DispositionError("GitHub environment must prevent self-review")
    reviewers = rule.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 1:
        raise DispositionError("Environment must require exactly one designated reviewer")
    reviewer_config = reviewers[0]
    reviewer_obj = reviewer_config.get("reviewer") if isinstance(reviewer_config, dict) else None
    reviewer_id = reviewer_obj.get("id") if isinstance(reviewer_obj, dict) else None
    if not isinstance(reviewer_config, dict) or reviewer_config.get("type") != "User" or str(reviewer_id or "") != expected_owner_id:
        raise DispositionError("Environment must require only the designated owner user as reviewer")

    matching_history = []
    for item in approvals:
        if not isinstance(item, dict):
            raise DispositionError("Approval history contains a non-object")
        item_envs = item.get("environments")
        if not isinstance(item_envs, list):
            continue
        if any(isinstance(env, dict) and env.get("name") == environment_name for env in item_envs):
            matching_history.append(item)
    approved = [item for item in matching_history if item.get("state") == "approved"]
    if len(approved) != 1 or len(matching_history) != 1:
        raise DispositionError("Approval history must contain exactly one approved decision for this environment")
    reviewer = approved[0].get("user")
    if not isinstance(reviewer, dict):
        raise DispositionError("Environment approval has no reviewer user object")
    if str(reviewer.get("id", "")) != expected_owner_id or reviewer.get("login") != OWNER_LOGIN:
        raise DispositionError("Actual protected-environment reviewer is not the designated owner")

    normalized = {
        "environment": environment,
        "approval_history": approvals,
        "context": context,
    }
    return {
        "environment": environment_name,
        "workflow": workflow_name,
        "repository": REPOSITORY,
        "run_id": context["run_id"],
        "run_attempt": context["run_attempt"],
        "head_sha": context["head_sha"],
        "ref": context["ref"],
        "event_name": context["event_name"],
        "initiator_login": context["actor_login"],
        "initiator_id": context["actor_id"],
        "reviewer_login": reviewer["login"],
        "reviewer_id": expected_owner_id,
        "environment_config_sha256": sha256_bytes(canonical_json_bytes(environment)),
        "approval_history_sha256": sha256_bytes(canonical_json_bytes(approvals)),
        "evidence_sha256": sha256_bytes(canonical_json_bytes(normalized)),
    }


def _require_pinned_owner_id(expected_owner_id: str) -> None:
    """Prevent CLI callers from substituting a different approval principal."""
    if str(expected_owner_id) != OWNER_ID:
        raise DispositionError(
            "Owner identity is pinned in source; --owner-id cannot override it"
        )


def _validate_approval_evidence(
    evidence_path: Path,
    *,
    expected_owner_id: str,
    environment_name: str,
    workflow_name: str,
) -> dict:
    evidence = _read_json(evidence_path, "GitHub environment approval evidence")
    summary = _validate_approval_evidence_object(
        evidence,
        expected_owner_id=expected_owner_id,
        environment_name=environment_name,
        workflow_name=workflow_name,
    )
    # Persist the exact API responses with the event. Hashes alone cannot be
    # re-verified during a later crash-recovery or read-only validator run.
    return {**summary, "evidence_snapshot": evidence}


def _validate_stored_approval(
    approval: dict,
    *,
    expected_owner_id: str,
    environment_name: str,
    workflow_name: str,
) -> dict:
    if not isinstance(approval, dict):
        raise DispositionError("Stored owner approval provenance is missing")
    snapshot = approval.get("evidence_snapshot")
    if not isinstance(snapshot, dict):
        raise DispositionError("Stored approval provenance lacks its exact GitHub API evidence snapshot")
    verified = _validate_approval_evidence_object(
        snapshot,
        expected_owner_id=expected_owner_id,
        environment_name=environment_name,
        workflow_name=workflow_name,
    )
    for key, value in verified.items():
        if approval.get(key) != value:
            raise DispositionError(f"Stored approval provenance {key} does not match its evidence snapshot")
    return verified

def _event_record(event_name: str, event_payload: dict, recorded_at: str | None = None, experiment_id: str = EXPERIMENT_ID) -> dict:
    record = {
        "record_type": "event",
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "event": event_name,
        "experiment_id": experiment_id,
        **event_payload,
    }
    if recorded_at is not None:
        record["recorded_at"] = recorded_at
    return record


def _event_payload_matches(record: dict, expected_event: str, expected_payload: dict, experiment_id: str = EXPERIMENT_ID) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("record_type") != "event" or record.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
        return False
    if record.get("event") != expected_event or record.get("experiment_id") != experiment_id:
        return False
    allowed_keys = set(expected_payload) | {"record_type", "ledger_schema_version", "event", "experiment_id", "recorded_at"}
    if set(record) != allowed_keys:
        return False
    if not isinstance(record.get("recorded_at"), str):
        return False
    try:
        _parse_utc(record["recorded_at"], f"{expected_event} recorded_at")
    except DispositionError:
        return False
    return all(record.get(key) == value for key, value in expected_payload.items())


def _atomic_append_bytes(path: Path, expected_current: bytes, append_bytes: bytes) -> None:
    """Atomically publish an append-only byte extension.

    Build the complete replacement beside the target, fsync it, re-check the
    expected head, then replace atomically. A crash leaves either the exact
    prior file or the complete appended record; it cannot leave a torn JSONL
    event line. Workflow-level concurrency remains the single-writer guard.
    """
    _reject_symlink_components(path, "atomic append target")
    if path.read_bytes() != expected_current:
        raise DispositionError("Append target changed during the atomic append operation")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".append.tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(expected_current)
            handle.write(append_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temp_path, path.stat().st_mode)
        except OSError:
            pass
        if path.read_bytes() != expected_current:
            raise DispositionError("Append target changed before atomic replacement")
        os.replace(temp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def _append_one_exact_event(path: Path, baseline_bytes: bytes, event_name: str, event_payload: dict, experiment_id: str = EXPERIMENT_ID) -> dict:
    _reject_symlink_components(path, f"{event_name} target")
    if not baseline_bytes.endswith(b"\n"):
        raise DispositionError("Pinned baseline JSONL must end with a newline")
    if not path.exists():
        raise DispositionError(f"Event target does not exist: {path}")
    current = path.read_bytes()
    if current == baseline_bytes:
        record = _event_record(event_name, event_payload, datetime.now(timezone.utc).isoformat(), experiment_id)
        encoded = canonical_json_bytes(record) + b"\n"
        _atomic_append_bytes(path, baseline_bytes, encoded)
        return record

    if current.startswith(baseline_bytes):
        tail = current[len(baseline_bytes):]
        rows = _read_jsonl_bytes(tail, f"{event_name} append")
        if len(rows) != 1 or not _event_payload_matches(rows[0], event_name, event_payload, experiment_id):
            raise DispositionError(f"{event_name} append exists but does not match the exact authorized payload")
        return rows[0]

    raise DispositionError(f"{event_name} target is not an exact append-only extension of its pinned baseline")


def _event_sha256(record: dict) -> str:
    return sha256_bytes(canonical_json_bytes(record))


def _atomic_save_state(path: Path, state: dict) -> None:
    _reject_symlink_components(path, "authoritative state destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is not available on every supported platform.
            pass
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def _validator_amendment_payload(
    *, state: dict, validator_path: Path, promotion_gate_path: Path,
    approval: dict, spec: DispositionSpec,
) -> dict:
    locked = state.get("locked_models", {})
    if locked.get("validator_code_hash") != spec.old_validator_sha256:
        raise DispositionError("State locked validator identity is not the expected pre-amendment identity")
    new_validator_sha = canonical_source_sha256(validator_path)
    gate_sha = canonical_source_sha256(promotion_gate_path)
    if gate_sha != spec.promotion_gate_sha256:
        raise DispositionError(f"Frozen promotion gate hash mismatch: {gate_sha}")
    payload = {
        "event_id": "F08VA-" + sha256_bytes(canonical_json_bytes({
            "experiment_id": spec.experiment_id,
            "from_validator_sha256": spec.old_validator_sha256,
            "to_validator_sha256": new_validator_sha,
            "from_schema": STATE_SCHEMA_V6,
            "to_schema": STATE_SCHEMA_V7,
            "reconciliation_id": spec.reconciliation_id,
            "source_state_sha256": spec.state_sha256,
            "source_observation_ledger_sha256": spec.ledger_sha256,
            "source_experiment_journal_sha256": spec.journal_sha256,
            "promotion_gate_sha256": spec.promotion_gate_sha256,
        }))[:32],
        "reconciliation_id": spec.reconciliation_id,
        "state_schema_from": STATE_SCHEMA_V6,
        "state_schema_to": STATE_SCHEMA_V7,
        "from_validator_code_sha256": spec.old_validator_sha256,
        "to_validator_code_sha256": new_validator_sha,
        "promotion_gate_code_sha256": gate_sha,
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "owner_approval": approval,
    }
    return payload


def append_validator_amendment(
    *, state_path: Path, observation_ledger_path: Path, experiment_journal_path: Path,
    baseline_journal_path: Path, validator_path: Path, promotion_gate_path: Path,
    approval_evidence_path: Path, expected_owner_id: str,
    spec: DispositionSpec = DispositionSpec(),
) -> dict:
    """Append the independently approved validator-amendment event only."""
    _require_pinned_owner_id(expected_owner_id)
    _assert_distinct_paths(
        (
            (state_path, "authoritative state"),
            (observation_ledger_path, "official observation ledger"),
            (experiment_journal_path, "working experiment journal"),
            (baseline_journal_path, "baseline experiment journal"),
            (validator_path, "validator source"),
            (promotion_gate_path, "promotion gate source"),
            (approval_evidence_path, "validator-amendment approval evidence"),
        ),
        operation="Validator amendment",
    )
    state_raw, ledger_raw, baseline_journal_raw = _verify_source_heads(
        state_path, observation_ledger_path, baseline_journal_path, spec
    )
    state = _read_json(state_path, "schema-6 F08 state")
    pending_population(state, spec)
    _reject_symlink_components(experiment_journal_path, "live experiment journal")
    approval = _validate_approval_evidence(
        approval_evidence_path,
        expected_owner_id=expected_owner_id,
        environment_name=VALIDATOR_AMENDMENT_ENVIRONMENT,
        workflow_name=VALIDATOR_AMENDMENT_WORKFLOW,
    )
    payload = _validator_amendment_payload(
        state=state,
        validator_path=validator_path,
        promotion_gate_path=promotion_gate_path,
        approval=approval,
        spec=spec,
    )
    journal_current = experiment_journal_path.read_bytes()
    was_baseline = journal_current == baseline_journal_raw
    if was_baseline:
        record = _append_one_exact_event(
            experiment_journal_path,
            baseline_journal_raw,
            VALIDATOR_AMENDMENT_EVENT,
            payload,
            spec.experiment_id,
        )
    else:
        if not journal_current.startswith(baseline_journal_raw):
            raise DispositionError("Experiment journal is not an append-only extension of the frozen baseline")
        extension = _read_jsonl_bytes(journal_current[len(baseline_journal_raw):], "existing validator-amendment extension")
        if len(extension) != 1:
            raise DispositionError("Existing experiment-journal extension is not exactly one validator-amendment event")
        record = extension[0]
        stored_approval = record.get("owner_approval")
        expected_existing_payload = _validator_amendment_payload(
            state=state,
            validator_path=validator_path,
            promotion_gate_path=promotion_gate_path,
            approval=stored_approval,
            spec=spec,
        )
        if not _event_payload_matches(
            record, VALIDATOR_AMENDMENT_EVENT, expected_existing_payload, spec.experiment_id
        ):
            raise DispositionError("Existing validator-amendment event does not match the exact authorized payload")
        _validate_stored_approval(
            stored_approval,
            expected_owner_id=expected_owner_id,
            environment_name=VALIDATOR_AMENDMENT_ENVIRONMENT,
            workflow_name=VALIDATOR_AMENDMENT_WORKFLOW,
        )
    return {
        "status": "APPENDED" if was_baseline else "ALREADY_PRESENT",
        "event": VALIDATOR_AMENDMENT_EVENT,
        "event_id": record["event_id"],
        "event_sha256": _event_sha256(record),
        "to_validator_code_sha256": record["to_validator_code_sha256"],
        "journal_sha256_after": sha256_file(experiment_journal_path),
        "official_state_written": False,
        "official_observation_ledger_written": False,
    }


def _verify_validator_amendment(
    *, state: dict, journal_bytes: bytes, baseline_journal_bytes: bytes,
    validator_path: Path, promotion_gate_path: Path, spec: DispositionSpec,
) -> tuple[dict, str]:
    if not journal_bytes.startswith(baseline_journal_bytes):
        raise DispositionError("Experiment journal is not append-only from the frozen baseline")
    tail = journal_bytes[len(baseline_journal_bytes):]
    events = _read_jsonl_bytes(tail, "validator amendment journal extension")
    if len(events) != 1:
        raise DispositionError("Expected exactly one journal extension: F08_VALIDATOR_AMENDMENT")
    amendment = events[0]
    if amendment.get("event") != VALIDATOR_AMENDMENT_EVENT or amendment.get("experiment_id") != spec.experiment_id:
        raise DispositionError("Required F08_VALIDATOR_AMENDMENT event is missing")
    expected_amendment_payload = _validator_amendment_payload(
        state=state,
        validator_path=validator_path,
        promotion_gate_path=promotion_gate_path,
        approval=amendment.get("owner_approval"),
        spec=spec,
    )
    if not _event_payload_matches(
        amendment, VALIDATOR_AMENDMENT_EVENT, expected_amendment_payload, spec.experiment_id
    ):
        raise DispositionError("Validator-amendment event envelope or payload is not an exact authorized record")
    required = {
        "state_schema_from": STATE_SCHEMA_V6,
        "state_schema_to": STATE_SCHEMA_V7,
        "from_validator_code_sha256": spec.old_validator_sha256,
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "reconciliation_id": spec.reconciliation_id,
    }
    for key, value in required.items():
        if amendment.get(key) != value:
            raise DispositionError(f"Validator amendment event {key} mismatch")
    current_validator_sha = canonical_source_sha256(validator_path)
    if amendment.get("to_validator_code_sha256") != current_validator_sha:
        raise DispositionError("Validator amendment is not authorized for the current validator source")
    gate_sha = canonical_source_sha256(promotion_gate_path)
    if gate_sha != spec.promotion_gate_sha256 or amendment.get("promotion_gate_code_sha256") != gate_sha:
        raise DispositionError("Validator amendment promotion-gate identity mismatch")
    _validate_stored_approval(
        amendment.get("owner_approval"),
        expected_owner_id=OWNER_ID,
        environment_name=VALIDATOR_AMENDMENT_ENVIRONMENT,
        workflow_name=VALIDATOR_AMENDMENT_WORKFLOW,
    )
    event_hash = _event_sha256(amendment)
    return amendment, event_hash


def _make_disposition_event_payload(
    *, population: dict, bundle_info: dict, approval: dict,
    validator_amendment: dict, validator_amendment_sha256: str,
    journal_sha256_after: str, spec: DispositionSpec,
) -> dict:
    core = {
        "experiment_id": spec.experiment_id,
        "event": STALE_PENDING_EVENT,
        "reconciliation_id": spec.reconciliation_id,
        "manifest_sha256": bundle_info["manifest_sha256"],
        "manifest_path": spec.manifest_path,
        "historical_bundle_sha256": bundle_info["bundle_sha256"],
        "pending_count": population["total"],
        "pending_by_model": population["counts"],
        "pending_ids_sha256": population["pending_ids_sha256"],
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "experiment_journal_sha256_after_validator_amendment": journal_sha256_after,
        "validator_amendment_event_id": validator_amendment["event_id"],
        "validator_amendment_event_sha256": validator_amendment_sha256,
        "validator_code_sha256": validator_amendment["to_validator_code_sha256"],
        "promotion_gate_code_sha256": spec.promotion_gate_sha256,
        "approval_provenance": approval,
        "disposition": "PENDING_PREDICTIONS_MOVED_WITHOUT_SCORING",
        "outcome_rows_appended": 0,
    }
    event_id_basis = {key: value for key, value in core.items() if key != "approval_provenance"}
    core["event_id"] = "F08D-" + sha256_bytes(canonical_json_bytes(event_id_basis))[:32]
    return core


def _verify_disposition_event_extension(
    ledger_path: Path,
    baseline_ledger_raw: bytes,
    event_payload: dict,
    *,
    expected_owner_id: str,
) -> dict:
    if not baseline_ledger_raw.endswith(b"\n"):
        raise DispositionError("Pinned baseline observation ledger must end with newline")
    current = ledger_path.read_bytes()
    if not current.startswith(baseline_ledger_raw):
        raise DispositionError("Official observation ledger is not append-only from the pinned baseline")
    tail = current[len(baseline_ledger_raw):]
    if not tail:
        return {}
    records = _read_jsonl_bytes(tail, "official observation-ledger extension")
    if len(records) != 1:
        raise DispositionError("Official observation ledger must contain only the single F08 disposition event after baseline")
    record = records[0]
    expected_existing_payload = dict(event_payload)
    expected_existing_payload["approval_provenance"] = record.get("approval_provenance")
    if not _event_payload_matches(
        record, STALE_PENDING_EVENT, expected_existing_payload, event_payload.get("experiment_id", EXPERIMENT_ID)
    ):
        raise DispositionError("Existing disposition event does not match the exact authorized payload")
    _validate_stored_approval(
        record.get("approval_provenance"),
        expected_owner_id=expected_owner_id,
        environment_name=DISPOSITION_ENVIRONMENT,
        workflow_name=DISPOSITION_WORKFLOW,
    )
    return record

def _build_schema7_state(state: dict, population: dict, event: dict, spec: DispositionSpec) -> dict:
    if state.get("schema_version") != STATE_SCHEMA_V6:
        raise DispositionError("Only schema-6 state can be migrated by this operation")
    if "dispositioned" in state:
        raise DispositionError("Schema-6 state already contains a dispositioned collection")
    migrated = copy.deepcopy(state)
    migrated["dispositioned"] = {"candidate": [], "production": []}
    event_sha = _event_sha256(event)
    for model in ("candidate", "production"):
        stale_ids = set(population["ids_by_model"][model])
        remaining: list[dict] = []
        moved: list[dict] = []
        for prediction in migrated["pending"][model]:
            if prediction["id"] not in stale_ids:
                remaining.append(prediction)
                continue
            moved.append({
                "id": prediction["id"],
                "model": model,
                "prediction_payload": copy.deepcopy(prediction),
                "disposition": STALE_PENDING_EVENT,
                "disposition_reason": "STALE_PENDING_NOT_SCORED",
                "reconciliation_id": spec.reconciliation_id,
                "event_id": event["event_id"],
                "event_sha256": event_sha,
                "source_state_sha256": spec.state_sha256,
                "recorded_at": event["recorded_at"],
            })
        if len(moved) != spec.model_counts()[model] or {row["id"] for row in moved} != stale_ids:
            raise DispositionError(f"Disposition migration did not move the exact {model} ID set")
        migrated["pending"][model] = remaining
        migrated["dispositioned"][model] = moved

    migrated["schema_version"] = STATE_SCHEMA_V7
    new_validator_sha = event["validator_code_sha256"]
    migrated["locked_models"] = dict(migrated.get("locked_models", {}))
    migrated["locked_models"]["validator_code_hash"] = new_validator_sha
    experiment_definition = migrated.get("experiment_definition")
    if isinstance(experiment_definition, dict):
        experiment_definition = copy.deepcopy(experiment_definition)
        definition_identity = experiment_definition.get("identity")
        if isinstance(definition_identity, dict):
            definition_identity["validator_code_hash"] = new_validator_sha
        migrated["experiment_definition"] = experiment_definition
    migrated["f08_disposition_migration"] = {
        "event": STALE_PENDING_EVENT,
        "event_id": event["event_id"],
        "event_sha256": event_sha,
        "recorded_at": event["recorded_at"],
        "reconciliation_id": spec.reconciliation_id,
        "pending_count": spec.total,
        "pending_by_model": spec.model_counts(),
        "pending_ids_sha256": population["pending_ids_sha256"],
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "validator_amendment_event_id": event["validator_amendment_event_id"],
        "validator_amendment_event_sha256": event["validator_amendment_event_sha256"],
        "validator_code_sha256": event["validator_code_sha256"],
        "promotion_gate_code_sha256": spec.promotion_gate_sha256,
    }
    return migrated


def validate_schema7_state(state: dict, event: dict | None = None, spec: DispositionSpec = DispositionSpec()) -> dict:
    if state.get("schema_version") != STATE_SCHEMA_V7:
        raise DispositionError("State is not schema 7")
    dispositioned = state.get("dispositioned")
    if not isinstance(dispositioned, dict):
        raise DispositionError("Schema-7 state has no dispositioned collection")
    meta = state.get("f08_disposition_migration")
    if not isinstance(meta, dict):
        raise DispositionError("Schema-7 state has no F08 migration metadata")
    if meta.get("event") != STALE_PENDING_EVENT or meta.get("reconciliation_id") != spec.reconciliation_id:
        raise DispositionError("Schema-7 migration identity mismatch")
    counts = {}
    all_ids: set[tuple[str, str]] = set()
    if state.get("locked_models", {}).get("experiment_id") != spec.experiment_id:
        raise DispositionError("Schema-7 state experiment identity mismatch")
    if not isinstance(state.get("pending"), dict) or not isinstance(state.get("resolved"), dict) or not isinstance(state.get("seen"), dict):
        raise DispositionError("Schema-7 pending/resolved/seen structures are malformed")
    for model in ("candidate", "production"):
        rows = dispositioned.get(model)
        if not isinstance(rows, list):
            raise DispositionError(f"Schema-7 dispositioned/{model} must be a list")
        ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise DispositionError(f"Schema-7 dispositioned/{model} has a non-object")
            prediction = row.get("prediction_payload")
            prediction_id = row.get("id")
            if not isinstance(prediction, dict) or not isinstance(prediction_id, str):
                raise DispositionError(f"Schema-7 dispositioned/{model} entry is malformed")
            if prediction.get("id") != prediction_id or prediction.get("model") != model:
                raise DispositionError(f"Schema-7 dispositioned/{model} payload identity mismatch")
            if row.get("disposition") != STALE_PENDING_EVENT or row.get("disposition_reason") != "STALE_PENDING_NOT_SCORED":
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} reason mismatch")
            if row.get("reconciliation_id") != spec.reconciliation_id:
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} reconciliation mismatch")
            if row.get("source_state_sha256") != spec.state_sha256:
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} source-state reference mismatch")
            if row.get("event_id") != meta.get("event_id") or row.get("event_sha256") != meta.get("event_sha256"):
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} event reference mismatch")
            if row.get("recorded_at") != meta.get("recorded_at"):
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} recorded_at mismatch")
            created = _parse_utc(prediction.get("created_at"), f"dispositioned/{model}/{prediction_id}/created_at")
            if created > LEGACY_CUTOFF:
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} is not in the frozen stale cohort")
            if prediction_id in {item.get("id") for item in state["pending"].get(model, []) if isinstance(item, dict)}:
                raise DispositionError(f"Schema-7 dispositioned/{model}/{prediction_id} remains in pending")
            if prediction_id in ids:
                raise DispositionError(f"Duplicate schema-7 dispositioned ID: {model}/{prediction_id}")
            ids.add(prediction_id)
            all_ids.add((model, prediction_id))
        counts[model] = len(rows)
        if len(ids) != len(rows):
            raise DispositionError(f"Duplicate schema-7 dispositioned IDs in {model}")
        pending_rows = state.get("pending", {}).get(model, [])
        pending_ids = {row.get("id") for row in pending_rows if isinstance(row, dict)}
        resolved_ids = {row.get("id") for row in state.get("resolved", {}).get(model, []) if isinstance(row, dict)}
        if ids & pending_ids or ids & resolved_ids:
            raise DispositionError(f"Schema-7 {model} IDs overlap pending/resolved")
    if counts != spec.model_counts() or sum(counts.values()) != spec.total:
        raise DispositionError(f"Schema-7 dispositioned population mismatch: {counts}")
    if meta.get("pending_count") != spec.total or meta.get("pending_by_model") != spec.model_counts() or meta.get("pending_ids_sha256") != spec.pending_ids_sha256:
        raise DispositionError("Schema-7 migration metadata count/digest mismatch")
    required_meta_identity = {
        "event": STALE_PENDING_EVENT,
        "reconciliation_id": spec.reconciliation_id,
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "promotion_gate_code_sha256": spec.promotion_gate_sha256,
    }
    for key, expected in required_meta_identity.items():
        if meta.get(key) != expected:
            raise DispositionError(f"Schema-7 migration metadata {key} mismatch")
    if not isinstance(meta.get("event_id"), str) or not re.fullmatch(r"F08D-[0-9a-f]{32}", meta["event_id"]):
        raise DispositionError("Schema-7 disposition event ID is malformed")
    if not isinstance(meta.get("event_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", meta["event_sha256"]):
        raise DispositionError("Schema-7 disposition event hash is malformed")
    recorded_at = _parse_utc(meta.get("recorded_at"), "schema-7 migration recorded_at")
    digest_ids = {"candidate": [], "production": []}
    for model in ("candidate", "production"):
        digest_ids[model] = sorted(row["id"] for row in dispositioned[model])
    if sha256_bytes(canonical_json_bytes(digest_ids)) != spec.pending_ids_sha256:
        raise DispositionError("Schema-7 dispositioned IDs do not match the frozen cohort digest")
    required_meta = {
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "promotion_gate_code_sha256": spec.promotion_gate_sha256,
    }
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            raise DispositionError(f"Schema-7 migration metadata {key} mismatch")
    if not isinstance(meta.get("validator_amendment_event_id"), str) or not meta.get("validator_amendment_event_id"):
        raise DispositionError("Schema-7 migration lacks validator amendment event ID")
    if not isinstance(meta.get("validator_amendment_event_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", meta["validator_amendment_event_sha256"]):
        raise DispositionError("Schema-7 migration lacks validator amendment event hash")
    if not isinstance(meta.get("validator_code_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", meta["validator_code_sha256"]):
        raise DispositionError("Schema-7 migration validator code hash is malformed")
    locked = state.get("locked_models", {})
    if locked.get("validator_code_hash") != meta.get("validator_code_sha256"):
        raise DispositionError("Schema-7 locked validator identity differs from migration metadata")
    if event is not None:
        if event.get("event_id") != meta.get("event_id") or _event_sha256(event) != meta.get("event_sha256"):
            raise DispositionError("Schema-7 state does not reference the exact official disposition event")
    return {"status": "PASS", "counts": counts, "total": sum(counts.values()), "pending_ids_sha256": spec.pending_ids_sha256}


def verify_validator_amendment_record(
    *, state: dict, experiment_journal_path: Path, validator_path: Path,
    promotion_gate_path: Path, spec: DispositionSpec = DispositionSpec(),
) -> dict:
    """Verify the immutable, independently approved validator-amendment event.

    This is read-only and is used on every subsequent schema-7 validator run.
    It refuses a hash change, duplicate amendment event, altered event payload,
    non-owner reviewer, or promotion-gate drift.
    """
    _reject_symlink_components(experiment_journal_path, "experiment journal")
    rows = _read_jsonl_bytes(experiment_journal_path.read_bytes(), "experiment journal")
    matches = [row for row in rows if row.get("event") == VALIDATOR_AMENDMENT_EVENT and row.get("experiment_id") == spec.experiment_id]
    if len(matches) != 1:
        raise DispositionError(f"Expected exactly one {VALIDATOR_AMENDMENT_EVENT} event, found {len(matches)}")
    amendment = matches[0]
    meta = state.get("f08_disposition_migration")
    if not isinstance(meta, dict):
        raise DispositionError("Schema-7 state is missing F08 migration metadata")
    if amendment.get("event_id") != meta.get("validator_amendment_event_id"):
        raise DispositionError("Schema-7 state validator-amendment event ID mismatch")
    amendment_sha = _event_sha256(amendment)
    if amendment_sha != meta.get("validator_amendment_event_sha256"):
        raise DispositionError("Schema-7 state validator-amendment event hash mismatch")
    expected = {
        "state_schema_from": STATE_SCHEMA_V6,
        "state_schema_to": STATE_SCHEMA_V7,
        "from_validator_code_sha256": spec.old_validator_sha256,
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "reconciliation_id": spec.reconciliation_id,
    }
    for key, value in expected.items():
        if amendment.get(key) != value:
            raise DispositionError(f"Stored validator amendment {key} mismatch")
    validator_sha = canonical_source_sha256(validator_path)
    gate_sha = canonical_source_sha256(promotion_gate_path)
    if validator_sha != amendment.get("to_validator_code_sha256"):
        raise DispositionError("Running validator does not match the authorized amendment hash")
    if validator_sha != meta.get("validator_code_sha256") or validator_sha != state.get("locked_models", {}).get("validator_code_hash"):
        raise DispositionError("Schema-7 validator identity does not match the amendment")
    if gate_sha != spec.promotion_gate_sha256 or amendment.get("promotion_gate_code_sha256") != gate_sha:
        raise DispositionError("Frozen promotion gate identity mismatch")
    _validate_stored_approval(
        amendment.get("owner_approval"),
        expected_owner_id=OWNER_ID,
        environment_name=VALIDATOR_AMENDMENT_ENVIRONMENT,
        workflow_name=VALIDATOR_AMENDMENT_WORKFLOW,
    )
    return {"event": amendment, "event_id": amendment.get("event_id"), "event_sha256": amendment_sha, "validator_code_sha256": validator_sha, "promotion_gate_code_sha256": gate_sha}


def apply_disposition(
    *, state_path: Path, observation_ledger_path: Path, experiment_journal_path: Path,
    baseline_state_path: Path, baseline_ledger_path: Path, baseline_journal_path: Path,
    validator_path: Path, promotion_gate_path: Path, manifest_path: Path, bundle_path: Path,
    approval_evidence_path: Path, expected_owner_id: str,
    spec: DispositionSpec = DispositionSpec(),
) -> dict:
    """Append the single disposition event and atomically migrate state to v7.

    A retry after event append but before state replacement verifies the exact
    event payload and completes the same migration.  Conflicting or partial
    extensions fail closed.
    """
    _require_pinned_owner_id(expected_owner_id)
    _assert_distinct_paths(
        (
            (state_path, "working state"),
            (observation_ledger_path, "working observation ledger"),
            (experiment_journal_path, "working experiment journal"),
            (baseline_state_path, "baseline state"),
            (baseline_ledger_path, "baseline observation ledger"),
            (baseline_journal_path, "baseline experiment journal"),
            (validator_path, "validator source"),
            (promotion_gate_path, "promotion gate source"),
            (manifest_path, "historical-backfill manifest"),
            (bundle_path, "historical-backfill bundle"),
            (approval_evidence_path, "disposition approval evidence"),
        ),
        operation="Disposition",
    )
    for path, label in ((baseline_state_path, "baseline state"), (baseline_ledger_path, "baseline observation ledger"), (baseline_journal_path, "baseline experiment journal")):
        _reject_symlink_components(path, label)
        if not path.is_file():
            raise DispositionError(f"Missing {label}: {path}")
    baseline_state_raw = baseline_state_path.read_bytes()
    baseline_ledger_raw = baseline_ledger_path.read_bytes()
    baseline_journal_raw = baseline_journal_path.read_bytes()
    expected_heads = {
        "state": spec.state_sha256,
        "ledger": spec.ledger_sha256,
        "journal": spec.journal_sha256,
    }
    actual_heads = {
        "state": sha256_bytes(baseline_state_raw),
        "ledger": sha256_bytes(baseline_ledger_raw),
        "journal": sha256_bytes(baseline_journal_raw),
    }
    if actual_heads != expected_heads:
        raise DispositionError(f"Baseline source artifacts mismatch: expected={expected_heads}, actual={actual_heads}")
    if state_path.read_bytes() != baseline_state_raw:
        current_state = _read_json(state_path, "current F08 state")
        if current_state.get("schema_version") != STATE_SCHEMA_V7:
            raise DispositionError("Current authoritative state differs from the pinned schema-6 source")

        current_ledger = observation_ledger_path.read_bytes()
        if not current_ledger.startswith(baseline_ledger_raw):
            raise DispositionError("Schema-7 retry ledger is not append-only from the pinned baseline")
        ledger_tail = current_ledger[len(baseline_ledger_raw):]
        ledger_extension = _read_jsonl_bytes(ledger_tail, "schema-7 retry observation-ledger extension")
        if len(ledger_extension) != 1:
            raise DispositionError("Schema-7 retry must have exactly one official disposition event after baseline")
        event = ledger_extension[0]
        _validate_stored_approval(
            event.get("approval_provenance"),
            expected_owner_id=expected_owner_id,
            environment_name=DISPOSITION_ENVIRONMENT,
            workflow_name=DISPOSITION_WORKFLOW,
        )

        source_state = _read_json(baseline_state_path, "pinned schema-6 source state")
        population = pending_population(source_state, spec)
        bundle_info = _verify_backfill_bundle(manifest_path, bundle_path, population, spec)
        amendment = verify_validator_amendment_record(
            state=current_state,
            experiment_journal_path=experiment_journal_path,
            validator_path=validator_path,
            promotion_gate_path=promotion_gate_path,
            spec=spec,
        )
        journal_current = experiment_journal_path.read_bytes()
        if not journal_current.startswith(baseline_journal_raw):
            raise DispositionError("Schema-7 retry experiment journal is not append-only from baseline")
        journal_extension = _read_jsonl_bytes(journal_current[len(baseline_journal_raw):], "schema-7 retry journal extension")
        if len(journal_extension) != 1 or journal_extension[0].get("event") != VALIDATOR_AMENDMENT_EVENT:
            raise DispositionError("Schema-7 retry journal does not contain exactly one validator amendment")
        if sha256_bytes(journal_current) != event.get("experiment_journal_sha256_after_validator_amendment"):
            raise DispositionError("Schema-7 retry experiment journal head differs from disposition event")

        event_expected = _make_disposition_event_payload(
            population=population,
            bundle_info=bundle_info,
            approval=event["approval_provenance"],
            validator_amendment=amendment["event"],
            validator_amendment_sha256=amendment["event_sha256"],
            journal_sha256_after=sha256_bytes(journal_current),
            spec=spec,
        )
        if not _event_payload_matches(event, STALE_PENDING_EVENT, event_expected, spec.experiment_id):
            raise DispositionError("Schema-7 retry disposition event does not match its exact immutable source and approval bindings")
        state_validation = validate_schema7_state(current_state, event, spec)

        # Check the complete original payloads survived, not merely their IDs.
        expected_payloads = {
            (model, row["id"]): row
            for model, rows in population["payloads_by_model"].items()
            for row in rows
        }
        actual_payloads = {
            (model, item["id"]): item["prediction_payload"]
            for model in ("candidate", "production")
            for item in current_state["dispositioned"][model]
        }
        if actual_payloads != expected_payloads:
            raise DispositionError("Schema-7 retry original prediction payloads differ from the frozen source cohort")

        return {
            **state_validation,
            "status": "ALREADY_APPLIED",
            "event_id": event["event_id"],
            "event_sha256": _event_sha256(event),
            "official_state_written": False,
            "official_observation_event_appended": False,
            "official_observation_rows_appended": 0,
        }

    state = _read_json(state_path, "authoritative schema-6 state")
    population = pending_population(state, spec)
    if observation_ledger_path.read_bytes() != baseline_ledger_raw:
        # If state save was interrupted, the exact event must be present and no
        # other record may have been appended; the payload is constructed below.
        pass

    journal_current = experiment_journal_path.read_bytes()
    amendment, amendment_sha = _verify_validator_amendment(
        state=state,
        journal_bytes=journal_current,
        baseline_journal_bytes=baseline_journal_raw,
        validator_path=validator_path,
        promotion_gate_path=promotion_gate_path,
        spec=spec,
    )
    journal_after_sha = sha256_bytes(journal_current)
    bundle_info = _verify_backfill_bundle(manifest_path, bundle_path, population, spec)
    approval = _validate_approval_evidence(
        approval_evidence_path,
        expected_owner_id=expected_owner_id,
        environment_name=DISPOSITION_ENVIRONMENT,
        workflow_name=DISPOSITION_WORKFLOW,
    )
    event_payload = _make_disposition_event_payload(
        population=population,
        bundle_info=bundle_info,
        approval=approval,
        validator_amendment=amendment,
        validator_amendment_sha256=amendment_sha,
        journal_sha256_after=journal_after_sha,
        spec=spec,
    )

    # Ensure the separate original state/ledger paths and their exact bytes
    # agree with the canonical source checkpoint before the event append.
    if sha256_bytes(state_path.read_bytes()) != spec.state_sha256:
        raise DispositionError("Authoritative state changed during disposition preflight")
    if sha256_bytes(baseline_ledger_raw) != spec.ledger_sha256:
        raise DispositionError("Baseline observation ledger changed during disposition preflight")

    event = _verify_disposition_event_extension(
        observation_ledger_path,
        baseline_ledger_raw,
        event_payload,
        expected_owner_id=expected_owner_id,
    )
    if not event:
        event = _append_one_exact_event(
            observation_ledger_path,
            baseline_ledger_raw,
            STALE_PENDING_EVENT,
            event_payload,
            spec.experiment_id,
        )

    # Before state commit, verify no official observation rows were added.
    ledger_rows = _read_jsonl_bytes(observation_ledger_path.read_bytes(), "official observation ledger")
    if len(ledger_rows) != len(_read_jsonl_bytes(baseline_ledger_raw, "baseline observation ledger")) + 1:
        raise DispositionError("Official observation ledger extension contains unexpected rows")
    if ledger_rows[-1].get("event") != STALE_PENDING_EVENT or ledger_rows[-1].get("record_type") != "event":
        raise DispositionError("Official ledger tail is not the disposition event")
    if ledger_rows[-1].get("outcome_rows_appended") != 0:
        raise DispositionError("Disposition event incorrectly reports outcome rows")

    migrated = _build_schema7_state(state, population, event, spec)
    validate_schema7_state(migrated, event, spec)
    _atomic_save_state(state_path, migrated)

    # Re-open and verify the actual atomically saved file before reporting PASS.
    saved = _read_json(state_path, "migrated schema-7 state")
    validate_schema7_state(saved, event, spec)
    return {
        "status": "MIGRATED",
        "schema_version": STATE_SCHEMA_V7,
        "dispositioned": population["counts"],
        "dispositioned_total": population["total"],
        "pending_ids_sha256": population["pending_ids_sha256"],
        "event_id": event["event_id"],
        "event_sha256": _event_sha256(event),
        "state_sha256_after": sha256_file(state_path),
        "official_observation_rows_appended": 0,
        "official_events_appended": 1 if len(ledger_rows) else 0,
        "c15_modified": False,
        "c20_modified": False,
        "c25_evaluated": False,
        "promotion_authorized": False,
        "live_trading_enabled": False,
    }


def preflight(
    *, state_path: Path, observation_ledger_path: Path, baseline_journal_path: Path,
    manifest_path: Path, bundle_path: Path, spec: DispositionSpec = DispositionSpec(),
) -> dict:
    state_raw, ledger_raw, journal_raw = _verify_source_heads(
        state_path, observation_ledger_path, baseline_journal_path, spec
    )
    state = _read_json(state_path, "schema-6 F08 state")
    population = pending_population(state, spec)
    bundle = _verify_backfill_bundle(manifest_path, bundle_path, population, spec)
    journal = _read_jsonl_bytes(journal_raw, "baseline experiment journal")
    if any(row.get("event") == VALIDATOR_AMENDMENT_EVENT for row in journal):
        raise DispositionError("Pinned baseline journal unexpectedly already contains F08_VALIDATOR_AMENDMENT")
    if any(row.get("event") == STALE_PENDING_EVENT for row in _read_jsonl_bytes(ledger_raw, "baseline observation ledger")):
        raise DispositionError("Pinned baseline official ledger unexpectedly already contains disposition event")
    return {
        "status": "PREFLIGHT_PASS_OWNER_APPROVALS_PENDING",
        "state_schema": STATE_SCHEMA_V6,
        "expected_target_schema": STATE_SCHEMA_V7,
        "pending_by_model": population["counts"],
        "pending_count": population["total"],
        "pending_ids_sha256": population["pending_ids_sha256"],
        "reconciliation_id": spec.reconciliation_id,
        "manifest_sha256": bundle["manifest_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "source_state_sha256": sha256_bytes(state_raw),
        "source_ledger_sha256": sha256_bytes(ledger_raw),
        "source_journal_sha256": sha256_bytes(journal_raw),
        "validator_amendment_approval": "PENDING",
        "disposition_approval": "PENDING",
        "official_state_written": False,
        "official_ledger_written": False,
        "official_event_appended": False,
        "official_observations_appended": False,
        "c15_modified": False,
        "c20_modified": False,
        "c25_evaluated": False,
        "promotion_authorized": False,
        "live_trading_enabled": False,
    }


def _path_arg(parser: argparse.ArgumentParser, name: str, *, required: bool = True) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", required=required, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="Read-only source and manifest validation")
    for arg in ("state", "observation_ledger", "baseline_journal", "manifest", "bundle"):
        _path_arg(pre, arg)

    amend = sub.add_parser("amend-validator", help="Append separately approved F08_VALIDATOR_AMENDMENT")
    for arg in ("state", "observation_ledger", "experiment_journal", "baseline_journal", "validator", "promotion_gate", "approval_evidence"):
        _path_arg(amend, arg)
    amend.add_argument("--owner-id", required=True)

    apply = sub.add_parser("apply-disposition", help="Append approved event and atomically migrate the 618 stale IDs")
    for arg in ("state", "observation_ledger", "experiment_journal", "baseline_state", "baseline_ledger", "baseline_journal", "validator", "promotion_gate", "manifest", "bundle", "approval_evidence"):
        _path_arg(apply, arg)
    apply.add_argument("--owner-id", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(
                state_path=args.state,
                observation_ledger_path=args.observation_ledger,
                baseline_journal_path=args.baseline_journal,
                manifest_path=args.manifest,
                bundle_path=args.bundle,
            )
        elif args.command == "amend-validator":
            result = append_validator_amendment(
                state_path=args.state,
                observation_ledger_path=args.observation_ledger,
                experiment_journal_path=args.experiment_journal,
                baseline_journal_path=args.baseline_journal,
                validator_path=args.validator,
                promotion_gate_path=args.promotion_gate,
                approval_evidence_path=args.approval_evidence,
                expected_owner_id=args.owner_id,
            )
        else:
            result = apply_disposition(
                state_path=args.state,
                observation_ledger_path=args.observation_ledger,
                experiment_journal_path=args.experiment_journal,
                baseline_state_path=args.baseline_state,
                baseline_ledger_path=args.baseline_ledger,
                baseline_journal_path=args.baseline_journal,
                validator_path=args.validator,
                promotion_gate_path=args.promotion_gate,
                manifest_path=args.manifest,
                bundle_path=args.bundle,
                approval_evidence_path=args.approval_evidence,
                expected_owner_id=args.owner_id,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
