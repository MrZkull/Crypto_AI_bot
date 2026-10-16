"""Safely append the approved F08 coverage-gate amendment to the canonical ledger.

This is invoked only by the canonical Phase 2D workflow on main, immediately
before the prospective validator. It is append-only, idempotent, and fails
closed if source hashes or checkpoint history do not match the approved case.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
AMENDMENT_EVENT = "F08_GATE_AMENDMENT"
CHECKPOINT_EVENT = "F08_CHECKPOINT_EVALUATED"
REQUIRED_PREVIOUS_CHECKPOINTS = (15, 20)
FIRST_AFFECTED_CHECKPOINT = 25
LEDGER_PATH = Path("research_outputs/experiment_ledger.jsonl")
GATE_PATH = Path("phase2d_promotion_gate.py")

# SHA-256 of canonical UTF-8 source bytes in the repository checkout (LF files).
# The old hash is the pre-normalization main-branch gate; the new hash is the
# exact gate including only the previously tested coverage-only unit fix.
OLD_GATE_SHA256 = "02a674a35f989c604b9ea3128878731325c99bf1c084e913651e983af0908344"
NEW_GATE_SHA256 = "f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab"


class AmendmentError(RuntimeError):
    """Raised when an F08 amendment safety precondition is not satisfied."""


def _sha256(path: Path) -> str:
    # Normalize Windows CRLF to the canonical LF source representation.
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AmendmentError(f"Authoritative experiment ledger is missing: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AmendmentError(
                    f"Corrupt experiment ledger JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise AmendmentError(
                    f"Experiment ledger record at {path}:{line_no} is not an object"
                )
            records.append(record)
    return records


def _payload() -> dict[str, Any]:
    return {
        "amendment_id": "F08_ATR_PERCENT_UNIT_NORMALIZATION",
        "amendment_schema_version": 1,
        "owner_approval_confirmed": True,
        "gate_source_path": "phase2d_promotion_gate.py",
        "old_code_sha256": OLD_GATE_SHA256,
        "new_code_sha256": NEW_GATE_SHA256,
        "code_hash_method": "SHA-256 of canonical repository source bytes (UTF-8; LF checkout)",
        "reason": (
            "Candidate ATR values are fractions (ATR / close), while the legacy "
            "production ATR values and frozen volatility-regime cutoffs are in "
            "percentage points (ATR / close * 100). Without unit normalization, "
            "candidate regime coverage can be misclassified."
        ),
        "normalization": {
            "field": "mean_atr_pct",
            "scope": "temporary daily-mean copy inside _coverage_gate, on final paired days only",
            "candidate_operation": "multiply by 100.0 before volatility-tercile classification",
            "production_operation": "no change; production values already use percentage points",
            "source_observations_modified": False,
            "model_feature_inputs_modified": False,
            "frozen_cutoffs_modified": False,
            "signal_thresholds_modified": False,
            "model_weights_or_artifacts_modified": False,
        },
        "historical_checkpoint_policy": {
            "C15": "preserve existing event and intervals unchanged",
            "C20": "preserve existing event and intervals unchanged",
            "historical_decisions_recomputed": False,
            "normalization_applies_to_future_gate_evaluations": True,
        },
        "required_before_checkpoint_blocks": FIRST_AFFECTED_CHECKPOINT,
        "model_promotion_authorized": False,
    }


def record_amendment(
    ledger_path: Path = LEDGER_PATH,
    gate_path: Path = GATE_PATH,
    experiment_id: str = TARGET_EXPERIMENT_ID,
    git_ref: str = "",
    approval_confirmed: bool = False,
    approval_actor: str = "",
    approval_run_id: str = "",
    approval_event_name: str = "",
    authorized_approver: str = "",
) -> str:
    """Append the event once after strict validation; never rewrite prior rows."""
    if git_ref != "refs/heads/main":
        raise AmendmentError(
            f"Refusing official amendment write on non-main ref: {git_ref!r}"
        )

    if experiment_id != TARGET_EXPERIMENT_ID:
        return "SKIPPED_NOT_TARGET_EXPERIMENT"

    # Verify the exact approved source before either appending or treating
    # an existing event as idempotently complete. A later gate-code change
    # must fail closed rather than silently reusing an old amendment.
    if not gate_path.is_file():
        raise AmendmentError(f"Promotion-gate source is missing: {gate_path}")
    actual_gate_hash = _sha256(gate_path)
    if actual_gate_hash != NEW_GATE_SHA256:
        raise AmendmentError(
            "Promotion-gate source hash does not match the reviewed amendment. "
            f"expected={NEW_GATE_SHA256}; actual={actual_gate_hash}"
        )

    records = _read_records(ledger_path)
    expected = {
        "event": AMENDMENT_EVENT,
        "experiment_id": experiment_id,
        **_payload(),
    }

    # Validate checkpoint history before treating a pre-existing amendment as
    # idempotently complete. Ledger order is append order, so a C25+ event that
    # precedes the amendment means the amendment was recorded too late.
    checkpoint_counts = {15: 0, 20: 0}
    later_checkpoint_indices: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if (
            record.get("experiment_id") != experiment_id
            or record.get("event") != CHECKPOINT_EVENT
        ):
            continue
        try:
            checkpoint = int(record["checkpoint_blocks"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AmendmentError(
                "Found a malformed F08 checkpoint event; refusing amendment write"
            ) from exc
        if checkpoint in checkpoint_counts:
            checkpoint_counts[checkpoint] += 1
        if checkpoint >= FIRST_AFFECTED_CHECKPOINT:
            later_checkpoint_indices.append((checkpoint, index))

    if checkpoint_counts != {15: 1, 20: 1}:
        raise AmendmentError(
            "Expected exactly one immutable C15 event and one immutable C20 event; "
            f"found {checkpoint_counts}. Refusing amendment write"
        )

    existing = [
        (index, record) for index, record in enumerate(records)
        if record.get("event") == AMENDMENT_EVENT
        and record.get("experiment_id") == experiment_id
    ]
    if len(existing) > 1:
        raise AmendmentError(
            "Multiple F08_GATE_AMENDMENT events already exist for the target experiment; manual review required"
        )
    if existing:
        amendment_index, existing_record = existing[0]
        for checkpoint, checkpoint_index in later_checkpoint_indices:
            if checkpoint_index < amendment_index:
                raise AmendmentError(
                    f"C{checkpoint} ledger event precedes F08_GATE_AMENDMENT; amendment was recorded too late"
                )
        comparable = dict(existing_record)
        if not isinstance(comparable.get("timestamp"), str):
            raise AmendmentError("Existing F08_GATE_AMENDMENT is missing its timestamp")
        comparable.pop("timestamp", None)
        audit_fields = {
            "approval_actor": comparable.pop("approval_actor", None),
            "approval_run_id": comparable.pop("approval_run_id", None),
            "approval_event_name": comparable.pop("approval_event_name", None),
        }
        # Detect a conflicting amendment payload before checking audit fields.
        if comparable != expected:
            raise AmendmentError(
                "Existing F08_GATE_AMENDMENT conflicts with the approved amendment payload; refusing to alter it"
            )
        # A matching existing amendment must also have approval provenance.
        if not all(
            isinstance(value, str) and value.strip()
            for value in audit_fields.values()
        ):
            raise AmendmentError(
                "Existing F08_GATE_AMENDMENT is missing approval audit provenance"
            )
        if audit_fields["approval_event_name"] != "workflow_dispatch":
            raise AmendmentError(
                "Existing F08_GATE_AMENDMENT lacks manual workflow-dispatch provenance"
            )
        if not authorized_approver.strip():
            raise AmendmentError(
                "PHASE2D_F08_APPROVER is not configured; refusing to trust existing amendment"
            )
        if (
            audit_fields["approval_actor"].strip().casefold()
            != authorized_approver.strip().casefold()
        ):
            raise AmendmentError(
                "Existing F08_GATE_AMENDMENT approval actor does not match the configured F08 approver"
            )
        if not audit_fields["approval_run_id"].strip().isdigit():
            raise AmendmentError(
                "Existing F08_GATE_AMENDMENT has an invalid GitHub workflow run ID"
            )
        return "ALREADY_PRESENT"

    if later_checkpoint_indices:
        checkpoint = later_checkpoint_indices[0][0]
        raise AmendmentError(
            f"C{checkpoint} has already been evaluated; amendment must precede C25"
        )

    # Existing matching amendments return ALREADY_PRESENT above.
    # A first append requires explicit approval, manual dispatch, and an
    # actor matching the repository's designated F08 approver variable.
    if not approval_confirmed:
        raise AmendmentError(
            "Explicit owner/mentor approval is required before appending F08_GATE_AMENDMENT"
        )
    if approval_event_name != "workflow_dispatch":
        raise AmendmentError(
            "First F08_GATE_AMENDMENT append is allowed only from workflow_dispatch"
        )
    if not approval_actor.strip() or not approval_run_id.strip():
        raise AmendmentError(
            "Approval actor and GitHub workflow run ID are required for the audit trail"
        )
    if not authorized_approver.strip():
        raise AmendmentError(
            "PHASE2D_F08_APPROVER is not configured; refusing amendment append"
        )
    if approval_actor.strip().casefold() != authorized_approver.strip().casefold():
        raise AmendmentError(
            f"Workflow actor {approval_actor!r} does not match the configured F08 approver"
        )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": AMENDMENT_EVENT,
        "experiment_id": experiment_id,
        "approval_actor": approval_actor.strip(),
        "approval_run_id": approval_run_id.strip(),
        "approval_event_name": approval_event_name,
        **_payload(),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    return "APPENDED"


def main() -> int:
    experiment_id = os.environ.get("PHASE2D_EXPERIMENT_ID", "")
    git_ref = os.environ.get("GITHUB_REF", "")
    try:
        result = record_amendment(
            ledger_path=LEDGER_PATH,
            gate_path=GATE_PATH,
            experiment_id=experiment_id,
            git_ref=git_ref,
            approval_confirmed=(
                os.environ.get("PHASE2D_APPROVE_F08_GATE_AMENDMENT", "")
                .strip()
                .lower()
                == "true"
            ),
            approval_actor=os.environ.get("GITHUB_ACTOR", "").strip(),
            approval_run_id=os.environ.get("GITHUB_RUN_ID", "").strip(),
            approval_event_name=os.environ.get("GITHUB_EVENT_NAME", "").strip(),
            authorized_approver=os.environ.get("PHASE2D_F08_APPROVER", "").strip(),
        )
    except AmendmentError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"F08 amendment recorder result: {result}")
    if result == "APPENDED":
        print("F08_GATE_AMENDMENT appended before checkpoint evaluation; C15/C20 rows were not modified.")
    elif result == "ALREADY_PRESENT":
        print("Matching amendment already exists; ledger was not changed.")
    else:
        print("Target experiment mismatch; ledger was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())