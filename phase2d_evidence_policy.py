"""Fail-closed evidence policy for the Phase 2D F08 scoring boundary.

The statistical gate remains byte-for-byte unchanged.  This module validates
which observations may be passed to that gate and validates the disposition
of the 618 stale pending predictions before the normal validator can resolve
any pending prediction.

This module never writes state, ledgers, events, or manifests.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from phase2d_observation_ledger import load_events
from phase2d_promotion_gate import evaluate_promotion as _evaluate_promotion
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

# These fingerprints are computed over complete observation payloads,
# canonical JSON (sorted keys, compact separators), sorted by observation_key.
# They are deliberately cohort-specific.  No arbitrary row with a missing
# evidence_class is accepted as legacy.
LEGACY_COHORTS = {
    "candidate": {
        "count": 2959,
        "payload_sha256": "adaa9db710328bda2153a484f2bf9554e42e491ecbfe8e6becd1dfb2801ef888",
    },
    "production": {
        "count": 13249,
        "payload_sha256": "3a586eed79660ec32b716fd4c4362002b24f9d4feaacefabef7ae5b00fa7ea23",
    },
}

# Provenance of the actual resolver used for the separate historical audit.
# The old shared resolver remains neutral and must not infer evidence_class.
HISTORICAL_RESOLVER_IDENTITY = {
    "git_head": "e592728ed88f00c7d6f62792a03b9d74808ce289",
    "validator_git_blob": "fef7a2ac75a94fe20cd095daaa6e278e4f295c5c",
    "dependency_git_blobs": {
        "phase2d_observation_ledger.py": "278fab7560917425f0bf918b420b40a0e9da21ac",
        "phase2d_promotion_gate.py": "497eb1a019a06f1a51921c65077de35fabbb1b9c",
        "feature_engineering.py": "27d4efa2fc966086d50387b8201b5b00ac96208d",
    },
    "pr14_merge_ancestor": True,
    "pr14_merge_commit": "58a64fbdeeee2a3a2851b2ef23824240eeca24e2",
}

EXPECTED_SOURCE_HASHES = {
    "state": "4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5",
    "ledger": "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9",
}
REQUIRED_SOURCE_HASH_FIELDS = (
    "state", "ledger", "journal", "results", "snapshot", "proposal",
    "raw_windows", "dry_run", "independent", "actual_resolver",
)

STALE_PENDING_EVENT = "STALE_PENDING_NOT_SCORED"
STALE_PENDING_MANIFEST_PATH = Path(
    "research_outputs/f08_historical_backfill/manifest.json"
)
EXPECTED_PENDING_COUNTS = {"candidate": 74, "production": 544}
EXPECTED_PENDING_TOTAL = 618
EXPECTED_TOTAL_OUTCOMES = {"TP": 292, "SL": 307, "EXPIRED": 19}
EXPECTED_CANDIDATE_OUTCOMES = {"TP": 12, "SL": 62}
EXPECTED_PRODUCTION_OUTCOMES = {"TP": 280, "SL": 245, "EXPIRED": 19}
AUTHORIZED_APPROVAL_ACTOR = "MrZkull"


class EvidencePolicyError(RuntimeError):
    """Raised when observation evidence or pending disposition is unsafe."""


def validate_f08_execution_context(
    experiment_id: str,
    allow_fresh_start: bool,
    existing_state: dict | None = None,
) -> bool:
    """Identify the canonical F08 state and prohibit reset/bypass attempts.

    The state identity is inspected as well as the environment variable so a
    caller cannot evade the F08 guard by omitting or changing the experiment ID.
    Returns True when this invocation is the authoritative F08 experiment.
    """
    stored_ids: set[str] = set()
    if isinstance(existing_state, dict):
        locked = existing_state.get("locked_models")
        definition = existing_state.get("experiment_definition")
        if isinstance(locked, dict) and isinstance(locked.get("experiment_id"), str):
            stored_ids.add(locked["experiment_id"])
        if isinstance(definition, dict) and isinstance(definition.get("experiment_id"), str):
            stored_ids.add(definition["experiment_id"])

    state_is_f08 = EXPERIMENT_ID in stored_ids
    invocation_is_f08 = experiment_id == EXPERIMENT_ID
    if state_is_f08 and not invocation_is_f08:
        raise EvidencePolicyError(
            "Existing canonical F08 state cannot be opened under a different experiment ID."
        )

    is_f08 = state_is_f08 or invocation_is_f08
    if is_f08 and allow_fresh_start:
        raise EvidencePolicyError(
            "F08 fresh-start override is prohibited: preserve the recovered state, "
            "C15/C20 history, and the 618 stale pending IDs."
        )
    return is_f08


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_fingerprint(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda row: str(row.get("observation_key", "")))
    payload = b"\n".join(_canonical_json(row) for row in ordered)
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvidencePolicyError(f"Missing or invalid {field} timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidencePolicyError(f"Invalid {field} timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EvidencePolicyError(f"{field} timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _ensure_record_identity(model: str, row: dict, seen_keys: set[str], seen_ids: set[tuple[str, str]]) -> None:
    if row.get("model") != model:
        raise EvidencePolicyError(
            f"Observation passed in {model!r} cohort has model={row.get('model')!r}"
        )
    key = row.get("observation_key")
    prediction_id = row.get("id")
    if not isinstance(key, str) or not key:
        raise EvidencePolicyError(f"{model}: observation missing immutable observation_key")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise EvidencePolicyError(f"{model}: observation missing prediction id")
    if key in seen_keys:
        raise EvidencePolicyError(f"Duplicate observation_key rejected: {key}")
    identity = (model, prediction_id)
    if identity in seen_ids:
        raise EvidencePolicyError(f"Duplicate prediction id rejected: {identity}")
    seen_keys.add(key)
    seen_ids.add(identity)


def validate_evidence_population(
    candidate_records: list[dict],
    production_records: list[dict],
    *,
    legacy_cohorts: dict | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Validate exact legacy cohorts and class every other gate observation.

    Returns the unchanged eligible lists plus a concise audit summary.  The
    16,208 legacy rows remain eligible only when each full model cohort
    reproduces its fixed count and full-payload fingerprint.  New rows must be
    explicitly PROSPECTIVE.  HISTORICAL_BACKFILL is never gate-eligible.
    """
    cohorts = legacy_cohorts or LEGACY_COHORTS
    inputs = {"candidate": candidate_records, "production": production_records}
    eligible: dict[str, list[dict]] = {"candidate": [], "production": []}
    summary = {
        "policy": "F08_EVIDENCE_CLASS_V1",
        "legacy_cutoff_utc": LEGACY_CUTOFF_UTC,
        "legacy": {},
        "prospective": {"candidate": 0, "production": 0},
        "historical_backfill_rejected": 0,
    }
    all_keys: set[str] = set()
    all_ids: set[tuple[str, str]] = set()

    for model in ("candidate", "production"):
        rows = inputs[model]
        if not isinstance(rows, list):
            raise EvidencePolicyError(f"{model}: observation population must be a list")
        legacy_rows: list[dict] = []
        prospective_rows: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                raise EvidencePolicyError(f"{model}: observation record is not an object")
            _ensure_record_identity(model, row, all_keys, all_ids)
            if "evidence_class" not in row:
                legacy_rows.append(row)
                continue
            evidence_class = row.get("evidence_class")
            if evidence_class == "PROSPECTIVE":
                created_at = _parse_utc(row.get("created_at"), field="created_at")
                resolved_at = _parse_utc(row.get("resolved_at"), field="resolved_at")
                if created_at <= LEGACY_CUTOFF or resolved_at <= LEGACY_CUTOFF:
                    raise EvidencePolicyError(
                        f"{model}/{row['id']}: PROSPECTIVE evidence timestamps must be after the legacy cutoff"
                    )
                prospective_rows.append(row)
                continue
            if evidence_class == "HISTORICAL_BACKFILL":
                summary["historical_backfill_rejected"] += 1
                raise EvidencePolicyError(
                    f"{model}/{row['id']}: HISTORICAL_BACKFILL is not eligible for primary F08 scoring"
                )
            raise EvidencePolicyError(
                f"{model}/{row['id']}: unsupported evidence_class={evidence_class!r}"
            )

        expected = cohorts[model]
        observed_count = len(legacy_rows)
        observed_fingerprint = _payload_fingerprint(legacy_rows)
        if observed_count != expected["count"]:
            raise EvidencePolicyError(
                f"{model}: untagged legacy count mismatch; expected {expected['count']}, got {observed_count}"
            )
        if observed_fingerprint != expected["payload_sha256"]:
            raise EvidencePolicyError(
                f"{model}: untagged legacy payload fingerprint mismatch"
            )

        for row in legacy_rows:
            resolved_at = _parse_utc(row.get("resolved_at"), field="legacy resolved_at")
            if resolved_at > LEGACY_CUTOFF:
                raise EvidencePolicyError(
                    f"{model}/{row['id']}: untagged row is newer than the established legacy cutoff"
                )

        eligible[model] = legacy_rows + prospective_rows
        summary["legacy"][model] = {
            "count": observed_count,
            "payload_sha256": observed_fingerprint,
        }
        summary["prospective"][model] = len(prospective_rows)

    summary["total_legacy"] = sum(item["count"] for item in summary["legacy"].values())
    summary["total_prospective"] = sum(summary["prospective"].values())
    summary["eligible_candidate"] = len(eligible["candidate"])
    summary["eligible_production"] = len(eligible["production"])
    summary["status"] = "PASS"
    return eligible["candidate"], eligible["production"], summary


def evaluate_promotion_with_evidence_policy(
    candidate_records: list[dict],
    production_records: list[dict],
    *,
    gate_callable: Callable | None = None,
    **gate_kwargs,
) -> dict:
    """Apply the evidence firewall, then delegate to the frozen statistical gate."""
    candidate, production, evidence_summary = validate_evidence_population(
        candidate_records,
        production_records,
    )
    evaluator = gate_callable or _evaluate_promotion
    result = evaluator(candidate, production, **gate_kwargs)
    if not isinstance(result, dict):
        raise EvidencePolicyError("Statistical gate returned a non-object result")
    return {**result, "evidence_policy": evidence_summary}


def pending_ids_digest(ids_by_model: dict[str, list[str]]) -> str:
    canonical = {
        model: sorted(str(item) for item in ids_by_model[model])
        for model in ("candidate", "production")
    }
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _stale_pending_ids(state: dict) -> dict[str, list[str]]:
    pending = state.get("pending")
    if not isinstance(pending, dict):
        raise EvidencePolicyError("Phase 2D state has no pending object")
    stale: dict[str, list[str]] = {"candidate": [], "production": []}
    for model in ("candidate", "production"):
        rows = pending.get(model)
        if not isinstance(rows, list):
            raise EvidencePolicyError(f"Phase 2D state pending/{model} is not a list")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise EvidencePolicyError(f"pending/{model} contains a non-object record")
            prediction_id = row.get("id")
            if not isinstance(prediction_id, str) or not prediction_id:
                raise EvidencePolicyError(f"pending/{model} contains a record without an id")
            if prediction_id in seen:
                raise EvidencePolicyError(f"pending/{model} contains duplicate id {prediction_id}")
            seen.add(prediction_id)
            created_at = _parse_utc(row.get("created_at"), field=f"pending/{model} created_at")
            if created_at <= LEGACY_CUTOFF:
                stale[model].append(prediction_id)
    for model in stale:
        stale[model].sort()
    return stale



def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and is_junction())


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if _is_link_or_junction(current):
            raise EvidencePolicyError(f"{label} traverses a symlink or junction: {current}")
    return absolute


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_artifact_path(manifest_dir: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise EvidencePolicyError(f"{label} artifact path is missing")
    if "\\" in relative_path:
        raise EvidencePolicyError(f"{label} artifact path must use forward slashes")
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "evidence":
        raise EvidencePolicyError(f"{label} artifact path is unsafe: {relative_path!r}")
    root = _reject_symlink_components(manifest_dir, label=f"{label} artifact root").resolve()
    current = root
    for part in rel.parts:
        current = current / part
        if _is_link_or_junction(current):
            raise EvidencePolicyError(f"{label} artifact path traverses a symlink or junction")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidencePolicyError(f"{label} evidence artifact is missing or outside manifest directory") from exc
    if not resolved.is_file():
        raise EvidencePolicyError(f"{label} evidence artifact is not a file")
    return resolved


def _read_json_evidence(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise EvidencePolicyError(f"{label} evidence is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidencePolicyError(f"{label} evidence must be a JSON object")
    return value


def _read_jsonl_evidence(path: Path, *, label: str) -> list[dict]:
    rows: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise EvidencePolicyError(f"Cannot read {label} evidence JSONL: {exc}") from exc
    for line_number, line in enumerate(raw, 1):
        if not line.strip():
            raise EvidencePolicyError(f"Blank line in {label} evidence at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidencePolicyError(f"Invalid {label} evidence JSONL line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise EvidencePolicyError(f"{label} evidence line {line_number} is not an object")
        rows.append(value)
    return rows


def _verify_source_artifacts(manifest: dict, manifest_dir: Path) -> dict[str, Path]:
    source_hashes = manifest.get("source_hashes")
    artifacts = manifest.get("source_artifacts")
    if not isinstance(source_hashes, dict) or not isinstance(artifacts, dict):
        raise EvidencePolicyError("Manifest must include source_hashes and source_artifacts objects")
    required = set(REQUIRED_SOURCE_HASH_FIELDS)
    if set(source_hashes) != required:
        raise EvidencePolicyError("Manifest source_hashes must contain exactly the ten required evidence hashes")
    if set(artifacts) != required:
        raise EvidencePolicyError("Manifest source_artifacts must contain exactly the ten required evidence files")
    paths: dict[str, Path] = {}
    used_paths: set[Path] = set()
    for name in REQUIRED_SOURCE_HASH_FIELDS:
        expected = source_hashes.get(name)
        entry = artifacts.get(name)
        if not _is_sha256(expected):
            raise EvidencePolicyError(f"Source hash is missing or malformed for {name}")
        if not isinstance(entry, dict):
            raise EvidencePolicyError(f"Source artifact descriptor is invalid for {name}")
        if entry.get("sha256") != expected:
            raise EvidencePolicyError(f"Source artifact descriptor hash does not match source_hashes for {name}")
        path = _manifest_artifact_path(manifest_dir, entry.get("path"), label=name)
        if path in used_paths:
            raise EvidencePolicyError(f"Source artifact path is reused: {entry.get('path')}")
        used_paths.add(path)
        actual = _sha256_file(path)
        if actual != expected:
            raise EvidencePolicyError(f"Source artifact byte SHA256 mismatch for {name}")
        paths[name] = path
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        if source_hashes.get(name) != expected:
            raise EvidencePolicyError(f"Pinned source hash mismatch for {name}")
    return paths


def _checkpoint_numbers(value: object, prefix: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if "checkpoint" in name.lower():
                if isinstance(child, int) and not isinstance(child, bool):
                    found.add(child)
                elif isinstance(child, str):
                    import re
                    match = re.search(r"(?:C)?(\d+)", child, re.IGNORECASE)
                    if match:
                        found.add(int(match.group(1)))
            found.update(_checkpoint_numbers(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_checkpoint_numbers(child, f"{prefix}[{index}]"))
    return found


def _verify_journal_artifact(path: Path) -> None:
    rows = _read_jsonl_evidence(path, label="journal")
    if len(rows) != 98:
        raise EvidencePolicyError(f"Journal evidence row count mismatch: expected 98, got {len(rows)}")
    checkpoints = [row for row in rows if row.get("event") == "F08_CHECKPOINT_EVALUATED"]
    c15 = [row for row in checkpoints if 15 in _checkpoint_numbers(row)]
    c20 = [row for row in checkpoints if 20 in _checkpoint_numbers(row)]
    later = [row for row in checkpoints if any(value >= 25 for value in _checkpoint_numbers(row))]
    amendments = [row for row in rows if row.get("event") == "F08_GATE_AMENDMENT"]
    if len(c15) != 1 or len(c20) != 1 or later or len(amendments) != 1 or rows[-1] is not amendments[0]:
        raise EvidencePolicyError("Journal evidence C15/C20/C25 or gate-amendment invariant failed")
    serialized = json.dumps(amendments[0], sort_keys=True)
    for required in (AUTHORIZED_APPROVAL_ACTOR, "41720710702", "workflow_dispatch"):
        if required not in serialized:
            raise EvidencePolicyError(f"Journal evidence amendment provenance missing {required}")


def _verify_actual_resolver_evidence(path: Path, manifest: dict) -> None:
    report = _read_json_evidence(path, label="actual resolver report")
    expected = {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": EXPECTED_PENDING_TOTAL,
        "candidate_verified": EXPECTED_PENDING_COUNTS["candidate"],
        "production_verified": EXPECTED_PENDING_COUNTS["production"],
        "mismatch_count": 0,
        "validator_git_blob": HISTORICAL_RESOLVER_IDENTITY["validator_git_blob"],
        "state_sha256_before": EXPECTED_SOURCE_HASHES["state"],
        "state_sha256_after": EXPECTED_SOURCE_HASHES["state"],
        "ledger_sha256_before": EXPECTED_SOURCE_HASHES["ledger"],
        "ledger_sha256_after": EXPECTED_SOURCE_HASHES["ledger"],
        "actual_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise EvidencePolicyError(f"Actual resolver report byte-backed value mismatch for {key}")
    summary = manifest.get("actual_resolver_verification")
    if not isinstance(summary, dict):
        raise EvidencePolicyError("Manifest has no actual_resolver_verification object")
    for key in (
        "verification", "verification_type", "predictions_verified", "candidate_verified",
        "production_verified", "mismatch_count", "state_sha256_before", "state_sha256_after",
        "ledger_sha256_before", "ledger_sha256_after", "actual_outcomes",
    ):
        if summary.get(key) != report.get(key):
            raise EvidencePolicyError(f"Manifest actual-resolver summary is not backed by report field {key}")


def _verify_bundle_artifacts(
    manifest: dict,
    manifest_dir: Path,
    evidence_paths: dict[str, Path],
    stale_ids: dict[str, list[str]],
    reconciliation_id: str,
    current_state: dict,
) -> None:
    # Independently re-check the pinned recovered state and the exact pending ID set.
    recovered_state = _read_json_evidence(evidence_paths["state"], label="recovered state")
    recovered_stale = _stale_pending_ids(recovered_state)
    if recovered_stale != stale_ids:
        raise EvidencePolicyError("Copied recovered state does not reproduce the current stale pending IDs")
    recovered_pending = {
        (model, str(row["id"])): row
        for model in ("candidate", "production")
        for row in recovered_state["pending"][model]
    }
    current_pending = current_state.get("pending")
    if not isinstance(current_pending, dict):
        raise EvidencePolicyError("Current F08 state has no pending object")
    for model in ("candidate", "production"):
        current_rows = current_pending.get(model)
        recovered_rows = recovered_state["pending"].get(model)
        if not isinstance(current_rows, list) or not isinstance(recovered_rows, list):
            raise EvidencePolicyError(f"Current/recovered pending/{model} is not a list")
        stale_set = set(stale_ids[model])
        current_stale = {
            str(row.get("id")): row
            for row in current_rows
            if isinstance(row, dict) and str(row.get("id")) in stale_set
        }
        recovered_stale_rows = {
            str(row.get("id")): row
            for row in recovered_rows
            if isinstance(row, dict) and str(row.get("id")) in stale_set
        }
        if current_stale != recovered_stale_rows:
            raise EvidencePolicyError(
                f"Current stale pending/{model} payloads differ from the recovered authoritative state"
            )

    # The source ledger is an immutable 16,208-observation / three-event baseline.
    ledger_rows = _read_jsonl_evidence(evidence_paths["ledger"], label="source ledger")
    ledger_observations = [row for row in ledger_rows if row.get("record_type") == "observation"]
    ledger_events = [row for row in ledger_rows if row.get("record_type") == "event"]
    if len(ledger_observations) != 16208 or len(ledger_events) != 3 or len(ledger_rows) != 16211:
        raise EvidencePolicyError("Copied source ledger does not reproduce the pinned 16,208-observation / 3-event baseline")
    if len({row.get("observation_key") for row in ledger_observations}) != 16208:
        raise EvidencePolicyError("Copied source ledger contains duplicate observation keys")
    if any(row.get("event") == STALE_PENDING_EVENT for row in ledger_events):
        raise EvidencePolicyError("Source evidence ledger already contains the disposition event")

    journal_path = evidence_paths["journal"]
    _verify_journal_artifact(journal_path)
    journal_review = manifest.get("experiment_journal_review")
    expected_journal_review = {
        "rows": 98,
        "c15_rows": 1,
        "c20_rows": 1,
        "c25_or_later_rows": 0,
        "f08_amendment_rows": 1,
        "amendment_actor": AUTHORIZED_APPROVAL_ACTOR,
        "amendment_run": "41720710702",
        "amendment_is_final_row": True,
        "sha256": manifest.get("source_hashes", {}).get("journal"),
    }
    if not isinstance(journal_review, dict) or any(
        journal_review.get(key) != value for key, value in expected_journal_review.items()
    ):
        raise EvidencePolicyError("Historical-backfill journal review summary is not backed by the journal evidence")
    for name in ("results", "snapshot", "dry_run", "independent"):
        _read_json_evidence(evidence_paths[name], label=name)
    raw_window_rows = _read_jsonl_evidence(evidence_paths["raw_windows"], label="raw OHLC windows")
    if len(raw_window_rows) != 573:
        raise EvidencePolicyError(f"Raw OHLC evidence window count mismatch: expected 573, got {len(raw_window_rows)}")
    seen_windows: set[tuple[str, int]] = set()
    for index, window in enumerate(raw_window_rows, 1):
        try:
            key = (str(window["symbol"]), int(window["open_time"]))
            candles = window["candles"]
            if not isinstance(candles, list) or len(candles) < 26:
                raise ValueError("fewer than 26 candles")
            timestamps = [int(candle["timestamp"]) for candle in candles]
            if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
                raise ValueError("timestamps are not strictly ordered and unique")
            position = timestamps.index(key[1])
            if position + 24 >= len(timestamps):
                raise ValueError("fewer than 24 candles after prediction open_time")
            if str(window.get("request", {}).get("resolution", "15")) != "15":
                raise ValueError("unexpected candle resolution")
            for candle in candles:
                high = float(candle["high"])
                low = float(candle["low"])
                if not math.isfinite(high) or not math.isfinite(low):
                    raise ValueError("non-finite high/low")
                if high < low:
                    raise ValueError("high is below low")
        except Exception as exc:
            raise EvidencePolicyError(f"Invalid raw OHLC evidence window at row {index}: {exc}") from exc
        if key in seen_windows:
            raise EvidencePolicyError(f"Duplicate raw OHLC evidence window: {key}")
        seen_windows.add(key)
    try:
        raw_windows_by_key = index_provenance_raw_windows(raw_window_rows)
    except Exception as exc:
        raise EvidencePolicyError(f"Raw OHLC evidence cannot be indexed for provenance: {exc}") from exc

    proposal = _read_json_evidence(evidence_paths["proposal"], label="proposal")
    if proposal.get("proposal_status") != "READY_FOR_REVIEW_NOT_APPLIED":
        raise EvidencePolicyError("Copied proposal is not READY_FOR_REVIEW_NOT_APPLIED")
    if proposal.get("state_or_ledger_modified") is not False or proposal.get("promotion_authorized") is not False:
        raise EvidencePolicyError("Copied proposal claims source mutation or promotion authorization")
    if proposal.get("proposed_observation_appends") != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError("Copied proposal does not contain the expected 618-row proposal count")
    if proposal.get("source_state_sha256") != EXPECTED_SOURCE_HASHES["state"]:
        raise EvidencePolicyError("Copied proposal state hash binding mismatch")
    if proposal.get("source_observation_ledger_sha256") != EXPECTED_SOURCE_HASHES["ledger"]:
        raise EvidencePolicyError("Copied proposal ledger hash binding mismatch")
    source_hashes = manifest["source_hashes"]
    if proposal.get("source_dry_run_sha256") != source_hashes.get("dry_run"):
        raise EvidencePolicyError("Copied proposal dry-run hash binding mismatch")
    if proposal.get("source_independent_verification_sha256") != source_hashes.get("independent"):
        raise EvidencePolicyError("Copied proposal independent-report hash binding mismatch")
    if proposal.get("pending_before") != EXPECTED_PENDING_COUNTS:
        raise EvidencePolicyError("Copied proposal pending_before counts mismatch")
    if proposal.get("pending_after_if_approved") != {"candidate": 0, "production": 0}:
        raise EvidencePolicyError("Copied proposal pending_after_if_approved mismatch")
    if proposal.get("proposed_candidate") != EXPECTED_CANDIDATE_OUTCOMES:
        raise EvidencePolicyError("Copied proposal candidate outcome counts mismatch")
    if proposal.get("proposed_production") != EXPECTED_PRODUCTION_OUTCOMES:
        raise EvidencePolicyError("Copied proposal production outcome counts mismatch")
    if proposal.get("proposed_total_outcomes") != EXPECTED_TOTAL_OUTCOMES:
        raise EvidencePolicyError("Copied proposal total outcome counts mismatch")
    proposal_rows = proposal.get("proposal_records")
    if not isinstance(proposal_rows, list) or len(proposal_rows) != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError("Copied proposal records are not exactly 618 rows")
    proposal_ids: dict[str, set[str]] = {"candidate": set(), "production": set()}
    proposal_by_identity: dict[tuple[str, str], dict] = {}
    for row in proposal_rows:
        model = row.get("model")
        prediction_id = row.get("prediction_id")
        if model not in proposal_ids or not isinstance(prediction_id, str) or not prediction_id:
            raise EvidencePolicyError("Copied proposal contains an invalid prediction identity")
        if prediction_id in proposal_ids[model]:
            raise EvidencePolicyError("Copied proposal contains duplicate prediction IDs")
        proposal_ids[model].add(prediction_id)
        proposal_by_identity[(model, prediction_id)] = row
    if proposal_ids != {model: set(values) for model, values in stale_ids.items()}:
        raise EvidencePolicyError("Copied proposal IDs do not exactly match stale pending IDs")

    bundle = manifest.get("historical_bundle")
    if not isinstance(bundle, dict):
        raise EvidencePolicyError("Manifest has no historical_bundle object")
    if bundle.get("evidence_class") != "HISTORICAL_BACKFILL" or bundle.get("format") != "jsonl":
        raise EvidencePolicyError("Historical bundle evidence class or format mismatch")
    if bundle.get("observation_count") != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError("Historical bundle observation_count mismatch")
    if bundle.get("candidate_count") != EXPECTED_PENDING_COUNTS["candidate"]:
        raise EvidencePolicyError("Historical bundle candidate_count mismatch")
    if bundle.get("production_count") != EXPECTED_PENDING_COUNTS["production"]:
        raise EvidencePolicyError("Historical bundle production_count mismatch")
    if bundle.get("resolved_at_policy", "").find("omitted") < 0:
        raise EvidencePolicyError("Historical bundle must document deterministic resolved_at omission")
    bundle_sha = bundle.get("bundle_sha256")
    if not _is_sha256(bundle_sha):
        raise EvidencePolicyError("Historical bundle SHA256 is missing or malformed")
    relative = bundle.get("bundle_path")
    if not isinstance(relative, str) or relative != "observations.jsonl":
        raise EvidencePolicyError("Historical bundle path is not the pinned observations.jsonl path")
    bundle_path = manifest_dir / relative
    _reject_symlink_components(bundle_path, label="historical bundle file")
    if not bundle_path.is_file():
        raise EvidencePolicyError("Historical bundle file is missing or unsafe")
    if _sha256_file(bundle_path) != bundle_sha:
        raise EvidencePolicyError("Historical bundle actual byte SHA256 mismatch")
    rows = _read_jsonl_evidence(bundle_path, label="historical bundle")
    if len(rows) != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError(f"Historical bundle file has {len(rows)} records, expected 618")
    base_rows = []
    for row in rows:
        base = dict(row)
        base.pop("reconciliation_id", None)
        base_rows.append(base)
    normalized_payload = b"\n".join(_canonical_json(row) for row in base_rows) + b"\n"
    normalized_rows_sha = hashlib.sha256(normalized_payload).hexdigest()
    if bundle.get("normalized_result_rows_sha256") != normalized_rows_sha:
        raise EvidencePolicyError("Historical bundle normalized result-row fingerprint mismatch")
    pending_population = manifest.get("pending_population")
    if not isinstance(pending_population, dict):
        raise EvidencePolicyError("Historical-backfill manifest has no pending_population object")
    pending_digest = pending_population.get("pending_ids_sha256")
    if not _is_sha256(pending_digest):
        raise EvidencePolicyError("Historical-backfill pending-ID digest is missing or malformed")
    rec_material = {
        "source_hashes": manifest["source_hashes"],
        "resolver_identity": manifest["resolver_identity"],
        "pending_ids_sha256": pending_digest,
        "normalized_result_rows_sha256": normalized_rows_sha,
    }
    expected_reconciliation_id = "F08-HIST-BF-" + hashlib.sha256(_canonical_json(rec_material)).hexdigest()[:32]
    if reconciliation_id != expected_reconciliation_id:
        raise EvidencePolicyError("Historical bundle reconciliation_id is not the deterministic source/result digest")
    observed_ids: dict[str, set[str]] = {"candidate": set(), "production": set()}
    seen_order: list[tuple[str, str]] = []
    for row in rows:
        if row.get("record_type") != "historical_backfill_observation":
            raise EvidencePolicyError("Historical bundle contains an unexpected record_type")
        if row.get("evidence_class") != "HISTORICAL_BACKFILL":
            raise EvidencePolicyError("Historical bundle contains a non-HISTORICAL_BACKFILL record")
        if row.get("experiment_id") != EXPERIMENT_ID or row.get("reconciliation_id") != reconciliation_id:
            raise EvidencePolicyError("Historical bundle record experiment/reconciliation identity mismatch")
        model = row.get("model")
        prediction_id = row.get("id")
        if model not in observed_ids or not isinstance(prediction_id, str) or not prediction_id:
            raise EvidencePolicyError("Historical bundle contains an invalid prediction identity")
        if prediction_id in observed_ids[model]:
            raise EvidencePolicyError("Historical bundle contains duplicate prediction IDs")
        observed_ids[model].add(prediction_id)
        seen_order.append((model, prediction_id))
        if "resolved_at" in row:
            raise EvidencePolicyError("Historical bundle contains volatile resolved_at field")
        if row.get("status") not in {"TP", "SL", "EXPIRED"}:
            raise EvidencePolicyError("Historical bundle contains an unsupported outcome status")
        if row.get("observation_key") != f"{EXPERIMENT_ID}:{model}:{row.get('symbol')}:{int(row.get('open_time'))}":
            raise EvidencePolicyError("Historical bundle immutable observation_key mismatch")
        proposal_row = proposal_by_identity.get((model, prediction_id))
        pending_row = recovered_pending.get((model, prediction_id))
        if proposal_row is None or pending_row is None:
            raise EvidencePolicyError("Historical bundle record is absent from the source pending/proposal population")
        # The resolver returns a copy of the original pending prediction plus
        # outcome fields.  Verify every original prediction field survives
        # unchanged, except the explicit bundle envelope fields below.
        bundle_envelope_fields = {
            "experiment_id", "record_type", "evidence_class",
            "observation_key", "reconciliation_id",
        }
        for field, expected_value in pending_row.items():
            if field in bundle_envelope_fields:
                continue
            if row.get(field) != expected_value:
                raise EvidencePolicyError(
                    f"Historical bundle/source pending payload mismatch for {model}/{prediction_id}/{field}"
                )
        for field in ("symbol", "signal", "created_at"):
            if row.get(field) != pending_row.get(field):
                raise EvidencePolicyError(f"Historical bundle/source pending mismatch for {model}/{prediction_id}/{field}")
        if int(row.get("open_time")) != int(pending_row.get("open_time")):
            raise EvidencePolicyError(f"Historical bundle/source pending mismatch for {model}/{prediction_id}/open_time")
        if proposal_row.get("symbol") != pending_row.get("symbol") or str(proposal_row.get("signal")) != str(pending_row.get("signal")):
            raise EvidencePolicyError(f"Proposal/source pending identity mismatch for {model}/{prediction_id}")
        expected_proposal_key = f"{EXPERIMENT_ID}:{model}:{pending_row.get('symbol')}:{int(pending_row.get('open_time'))}"
        if proposal_row.get("observation_key") != expected_proposal_key:
            raise EvidencePolicyError(f"Proposal immutable observation_key mismatch for {model}/{prediction_id}")
        if int(proposal_row.get("open_time")) != int(pending_row.get("open_time")):
            raise EvidencePolicyError(f"Proposal/source pending open_time mismatch for {model}/{prediction_id}")
        if proposal_row.get("source_created_at") != pending_row.get("created_at"):
            raise EvidencePolicyError(f"Proposal/source pending created_at mismatch for {model}/{prediction_id}")
        for field in ("entry", "atr"):
            try:
                bundle_matches = math.isclose(float(row.get(field)), float(pending_row.get(field)), rel_tol=1e-10, abs_tol=1e-10)
                proposal_matches = math.isclose(float(proposal_row.get(field)), float(pending_row.get(field)), rel_tol=1e-10, abs_tol=1e-10)
            except (TypeError, ValueError):
                bundle_matches = proposal_matches = False
            if not bundle_matches or not proposal_matches:
                raise EvidencePolicyError(f"Historical bundle/proposal/source pending mismatch for {model}/{prediction_id}/{field}")
        for field in ("status", "outcome_bar", "gross_r", "net_r"):
            actual = row.get(field)
            expected = proposal_row.get("proposed_status" if field == "status" else field)
            if field in {"gross_r", "net_r"}:
                try:
                    matches = math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise EvidencePolicyError(f"Historical bundle/proposal outcome mismatch for {model}/{prediction_id}/{field}")
    if seen_order != sorted(seen_order):
        raise EvidencePolicyError("Historical bundle records are not in deterministic model/ID order")
    if observed_ids != {model: set(values) for model, values in stale_ids.items()}:
        raise EvidencePolicyError("Historical bundle IDs do not exactly match stale pending IDs")

    provenance_meta = manifest.get("resolution_provenance")
    if not isinstance(provenance_meta, dict):
        raise EvidencePolicyError("Manifest has no per-prediction resolution_provenance object")
    if (
        provenance_meta.get("schema_version") != 1
        or provenance_meta.get("format") != "jsonl"
        or provenance_meta.get("path") != PROVENANCE_FILENAME
        or provenance_meta.get("record_count") != EXPECTED_PENDING_TOTAL
        or provenance_meta.get("raw_windows_sha256") != source_hashes.get("raw_windows")
        or provenance_meta.get("resolved_at_semantics") != RESOLVED_AT_SEMANTICS
        or provenance_meta.get("outcome_bar_close_time_semantics") != OUTCOME_BAR_CLOSE_TIME_SEMANTICS
    ):
        raise EvidencePolicyError("Per-prediction resolution-provenance metadata is inconsistent")
    provenance_path = manifest_dir / PROVENANCE_FILENAME
    _reject_symlink_components(provenance_path, label="resolution-provenance sidecar")
    if not provenance_path.is_file():
        raise EvidencePolicyError("Per-prediction resolution-provenance sidecar is missing or unsafe")
    provenance_sha = provenance_meta.get("sha256")
    if not _is_sha256(provenance_sha) or _sha256_file(provenance_path) != provenance_sha:
        raise EvidencePolicyError("Per-prediction resolution-provenance SHA256 mismatch")
    provenance_rows = _read_jsonl_evidence(
        provenance_path, label="per-prediction resolution provenance"
    )
    try:
        provenance_verification = verify_resolution_provenance_records(
            provenance_rows, rows, raw_windows_by_key, reconciliation_id,
            expected_counts=EXPECTED_PENDING_COUNTS,
        )
    except Exception as exc:
        raise EvidencePolicyError(f"Per-prediction resolution provenance is invalid: {exc}") from exc
    if provenance_verification.get("records_verified") != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError("Per-prediction resolution provenance did not verify all 618 IDs")

    total_outcomes = Counter(row.get("status") for row in rows)
    candidate_outcomes = Counter(row.get("status") for row in rows if row.get("model") == "candidate")
    production_outcomes = Counter(row.get("status") for row in rows if row.get("model") == "production")
    if dict(total_outcomes) != EXPECTED_TOTAL_OUTCOMES:
        raise EvidencePolicyError(f"Historical bundle total outcome counts mismatch: {dict(total_outcomes)}")
    if dict(candidate_outcomes) != EXPECTED_CANDIDATE_OUTCOMES:
        raise EvidencePolicyError(f"Historical bundle candidate outcome counts mismatch: {dict(candidate_outcomes)}")
    if dict(production_outcomes) != EXPECTED_PRODUCTION_OUTCOMES:
        raise EvidencePolicyError(f"Historical bundle production outcome counts mismatch: {dict(production_outcomes)}")

    resolution = manifest.get("resolution")
    expected_resolution = {
        "predictions_resolved": EXPECTED_PENDING_TOTAL,
        "candidate": dict(EXPECTED_CANDIDATE_OUTCOMES),
        "production": dict(EXPECTED_PRODUCTION_OUTCOMES),
        "total_outcomes": dict(EXPECTED_TOTAL_OUTCOMES),
        "raw_windows_used": 573,
    }
    if not isinstance(resolution, dict) or any(resolution.get(key) != value for key, value in expected_resolution.items()):
        raise EvidencePolicyError("Historical-backfill resolution summary is inconsistent with the verified bundle")

    safety = manifest.get("safety_invariants")
    expected_safety = {
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
    }
    if safety != expected_safety:
        raise EvidencePolicyError("Historical-backfill safety_invariants are missing or unsafe")


def _verify_manifest_artifacts(
    manifest: dict,
    manifest_dir: Path,
    stale_ids: dict[str, list[str]],
    reconciliation_id: str,
    current_state: dict,
) -> None:
    evidence_paths = _verify_source_artifacts(manifest, manifest_dir)
    if manifest.get("raw_ohlc_windows_sha256") != manifest.get("source_hashes", {}).get("raw_windows"):
        raise EvidencePolicyError("Historical-backfill raw OHLC hash does not match source_hashes")
    if manifest.get("actual_resolver_report_sha256") != manifest.get("source_hashes", {}).get("actual_resolver"):
        raise EvidencePolicyError("Historical-backfill actual-resolver report hash mismatch")
    # Validate the byte-backed resolver report before the deterministic bundle
    # digest, so a re-hashed but semantically altered report is diagnosed here.
    _verify_actual_resolver_evidence(evidence_paths["actual_resolver"], manifest)
    _verify_bundle_artifacts(
        manifest, manifest_dir, evidence_paths, stale_ids, reconciliation_id, current_state
    )

def validate_stale_pending_disposition(
    state: dict,
    observation_ledger_path: Path,
    *,
    manifest_path: Path = STALE_PENDING_MANIFEST_PATH,
    experiment_id: str = EXPERIMENT_ID,
    experiment_journal_path: Path = Path("research_outputs/experiment_ledger.jsonl"),
) -> dict:
    """Read-only validation of the schema-7, not-scored disposition.

    The immutable historical manifest stays READY_FOR_OWNER_REVIEW /
    PENDING_OWNER_APPROVAL.  Actual approval is verified independently from
    GitHub environment protection and workflow-run approval history, then
    bound into the official event.  The official observation ledger receives
    no TP/SL/EXPIRED rows for the 618 stale predictions.
    """
    if experiment_id != EXPERIMENT_ID:
        raise EvidencePolicyError("F08 disposition policy cannot validate a different experiment ID")
    if state.get("schema_version") != 7:
        raise EvidencePolicyError(
            "Canonical F08 state must be schema 7 after the separately authorized "
            "validator amendment and stale-pending disposition; schema-6 state is not runnable."
        )

    _reject_symlink_components(manifest_path, label="Historical-backfill manifest file")
    _reject_symlink_components(observation_ledger_path, label="official observation ledger")
    if not manifest_path.is_file() or not observation_ledger_path.is_file():
        raise EvidencePolicyError("Required F08 disposition manifest or official ledger is missing")

    try:
        from phase2d_f08_disposition import (
            DispositionSpec,
            validate_schema7_state,
            verify_validator_amendment_record,
            _validate_stored_approval,
        )
        validate_schema7_state(state)
    except Exception as exc:
        raise EvidencePolicyError(f"Schema-7 F08 state integrity failed: {exc}") from exc

    spec = DispositionSpec()
    migration = state.get("f08_disposition_migration")
    dispositioned = state.get("dispositioned")
    if not isinstance(migration, dict) or not isinstance(dispositioned, dict):
        raise EvidencePolicyError("Schema-7 state is missing durable F08 disposition metadata")

    stale_ids: dict[str, list[str]] = {"candidate": [], "production": []}
    source_pending: dict[str, list[dict]] = {"candidate": [], "production": []}
    for model in ("candidate", "production"):
        rows = dispositioned.get(model)
        if not isinstance(rows, list):
            raise EvidencePolicyError(f"Schema-7 dispositioned/{model} is not a list")
        for entry in rows:
            if not isinstance(entry, dict) or not isinstance(entry.get("prediction_payload"), dict):
                raise EvidencePolicyError(f"Schema-7 dispositioned/{model} has a malformed original payload")
            payload = entry["prediction_payload"]
            if payload.get("id") != entry.get("id") or payload.get("model") != model:
                raise EvidencePolicyError(f"Schema-7 dispositioned/{model} payload identity mismatch")
            created = _parse_utc(payload.get("created_at"), field=f"dispositioned/{model}/created_at")
            if created > LEGACY_CUTOFF:
                raise EvidencePolicyError(f"Schema-7 dispositioned/{model}/{entry.get('id')} is not in the stale cohort")
            stale_ids[model].append(str(entry["id"]))
            source_pending[model].append(payload)
        stale_ids[model].sort()
        source_pending[model].sort(key=lambda row: str(row.get("id", "")))

    stale_counts = {model: len(stale_ids[model]) for model in stale_ids}
    if stale_counts != EXPECTED_PENDING_COUNTS or sum(stale_counts.values()) != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError(
            f"Schema-7 dispositioned population mismatch: observed={stale_counts}, expected={EXPECTED_PENDING_COUNTS}"
        )
    digest = pending_ids_digest(stale_ids)
    if digest != spec.pending_ids_sha256 or migration.get("pending_ids_sha256") != digest:
        raise EvidencePolicyError("Schema-7 dispositioned pending-ID digest mismatch")

    # The ordinary pending queues must not still contain any pre-cutoff entry.
    remaining_stale = _stale_pending_ids(state)
    if remaining_stale != {"candidate": [], "production": []}:
        raise EvidencePolicyError("Schema-7 state still contains stale predictions in pending")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != spec.manifest_sha256:
        raise EvidencePolicyError("Immutable historical-backfill manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise EvidencePolicyError(f"Historical-backfill manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvidencePolicyError("Historical-backfill manifest must be a JSON object")

    required_manifest = {
        "manifest_schema_version": 1,
        "operation": STALE_PENDING_EVENT,
        "experiment_id": experiment_id,
        "evidence_class": "HISTORICAL_BACKFILL",
        "legacy_cutoff_utc": LEGACY_CUTOFF_UTC,
        "manifest_status": "READY_FOR_OWNER_REVIEW",
        "approval_status": "PENDING_OWNER_APPROVAL",
        "reconciliation_id": spec.reconciliation_id,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise EvidencePolicyError(
                f"Immutable historical-backfill manifest {key} mismatch: expected {expected!r}, got {manifest.get(key)!r}"
            )
    if "approval_provenance" in manifest:
        raise EvidencePolicyError("Owner approval must remain separate from the immutable historical-backfill manifest")
    if manifest_sha256 != spec.manifest_sha256:
        raise EvidencePolicyError("Historical-backfill manifest identity changed")

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise EvidencePolicyError("Historical-backfill manifest has no source_hashes object")
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        if source_hashes.get(name) != expected:
            raise EvidencePolicyError(f"Historical-backfill source hash mismatch for {name}")
    for name in REQUIRED_SOURCE_HASH_FIELDS:
        value = source_hashes.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise EvidencePolicyError(f"Historical-backfill source hash is missing or malformed for {name}")
    if manifest.get("raw_ohlc_windows_sha256") != source_hashes.get("raw_windows"):
        raise EvidencePolicyError("Historical-backfill raw OHLC hash does not match source_hashes")
    if manifest.get("actual_resolver_report_sha256") != source_hashes.get("actual_resolver"):
        raise EvidencePolicyError("Historical-backfill actual-resolver report hash mismatch")

    resolver_identity = manifest.get("resolver_identity")
    expected_resolver_identity = {**HISTORICAL_RESOLVER_IDENTITY, "pinned_files_clean": True}
    if resolver_identity != expected_resolver_identity:
        raise EvidencePolicyError("Historical resolver identity does not exactly match the pinned source identity")
    verification = manifest.get("actual_resolver_verification")
    verification_expected = {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": EXPECTED_PENDING_TOTAL,
        "candidate_verified": EXPECTED_PENDING_COUNTS["candidate"],
        "production_verified": EXPECTED_PENDING_COUNTS["production"],
        "mismatch_count": 0,
        "state_sha256_before": EXPECTED_SOURCE_HASHES["state"],
        "state_sha256_after": EXPECTED_SOURCE_HASHES["state"],
        "ledger_sha256_before": EXPECTED_SOURCE_HASHES["ledger"],
        "ledger_sha256_after": EXPECTED_SOURCE_HASHES["ledger"],
    }
    if not isinstance(verification, dict) or any(verification.get(k) != v for k, v in verification_expected.items()):
        raise EvidencePolicyError("Actual historical resolver verification is incomplete or inconsistent")

    pending_population = manifest.get("pending_population")
    if not isinstance(pending_population, dict) or pending_population.get("pending_count") != EXPECTED_PENDING_TOTAL:
        raise EvidencePolicyError("Historical-backfill manifest pending population is invalid")
    manifest_ids = pending_population.get("ids_by_model")
    if not isinstance(manifest_ids, dict):
        raise EvidencePolicyError("Historical-backfill manifest has no ids_by_model object")
    for model in ("candidate", "production"):
        values = manifest_ids.get(model)
        if not isinstance(values, list) or sorted(values) != stale_ids[model] or len(values) != EXPECTED_PENDING_COUNTS[model]:
            raise EvidencePolicyError(f"Immutable manifest IDs do not match dispositioned/{model}")
    if pending_population.get("pending_ids_sha256") != digest:
        raise EvidencePolicyError("Immutable manifest pending-ID digest mismatch")

    events = load_events(observation_ledger_path, experiment_id=experiment_id)
    disposition_events = [row for row in events if row.get("event") == STALE_PENDING_EVENT]
    if len(disposition_events) != 1:
        raise EvidencePolicyError(f"Expected exactly one {STALE_PENDING_EVENT} event, found {len(disposition_events)}")
    event = disposition_events[0]
    expected_event_values = {
        "record_type": "event",
        "ledger_schema_version": 1,
        "reconciliation_id": spec.reconciliation_id,
        "manifest_sha256": manifest_sha256,
        "manifest_path": spec.manifest_path,
        "historical_bundle_sha256": spec.bundle_sha256,
        "pending_count": EXPECTED_PENDING_TOTAL,
        "pending_by_model": EXPECTED_PENDING_COUNTS,
        "pending_ids_sha256": digest,
        "source_state_sha256": spec.state_sha256,
        "source_observation_ledger_sha256": spec.ledger_sha256,
        "source_experiment_journal_sha256": spec.journal_sha256,
        "validator_amendment_event_id": migration.get("validator_amendment_event_id"),
        "validator_amendment_event_sha256": migration.get("validator_amendment_event_sha256"),
        "validator_code_sha256": migration.get("validator_code_sha256"),
        "promotion_gate_code_sha256": spec.promotion_gate_sha256,
        "disposition": "PENDING_PREDICTIONS_MOVED_WITHOUT_SCORING",
        "outcome_rows_appended": 0,
    }
    for key, expected in expected_event_values.items():
        if event.get(key) != expected:
            raise EvidencePolicyError(f"Disposition event {key} mismatch")
    if event.get("event_id") != migration.get("event_id"):
        raise EvidencePolicyError("Disposition event ID does not match schema-7 migration metadata")
    event_id_basis = {
        key: value for key, value in event.items()
        if key not in {"record_type", "ledger_schema_version", "recorded_at", "event_id", "approval_provenance"}
    }
    expected_event_id = "F08D-" + hashlib.sha256(_canonical_json(event_id_basis)).hexdigest()[:32]
    if event.get("event_id") != expected_event_id:
        raise EvidencePolicyError("Disposition event ID does not match its deterministic source binding")
    event_hash = hashlib.sha256(_canonical_json(event)).hexdigest()
    if event_hash != migration.get("event_sha256"):
        raise EvidencePolicyError("Disposition event payload hash does not match schema-7 state metadata")

    approval = event.get("approval_provenance")
    required_approval = {
        "environment": "f08-disposition",
        "workflow": "Phase2D F08 Controlled Disposition",
        "repository": "MrZkull/Crypto_AI_bot",
        "reviewer_login": "MrZkull",
        "reviewer_id": "166250462",
        "event_name": "workflow_dispatch",
        "ref": "refs/heads/main",
    }
    if not isinstance(approval, dict) or any(approval.get(k) != v for k, v in required_approval.items()):
        raise EvidencePolicyError("Disposition event lacks the required protected-environment owner approval")
    try:
        _validate_stored_approval(
            approval,
            expected_owner_id="166250462",
            environment_name="f08-disposition",
            workflow_name="Phase2D F08 Controlled Disposition",
        )
    except Exception as exc:
        raise EvidencePolicyError(f"Stored protected-environment approval evidence is invalid: {exc}") from exc
    if not isinstance(approval.get("initiator_id"), str) or not approval["initiator_id"].isdigit() or approval["initiator_id"] == "166250462":
        raise EvidencePolicyError("Disposition approval is self-approved or has invalid initiating actor identity")
    if not isinstance(approval.get("run_id"), str) or not approval["run_id"].isdigit():
        raise EvidencePolicyError("Disposition approval workflow run ID is malformed")
    if not isinstance(approval.get("run_attempt"), int) or isinstance(approval.get("run_attempt"), bool) or approval["run_attempt"] < 1:
        raise EvidencePolicyError("Disposition approval workflow run attempt is malformed")
    if not isinstance(approval.get("head_sha"), str) or len(approval["head_sha"]) != 40 or any(ch not in "0123456789abcdef" for ch in approval["head_sha"]):
        raise EvidencePolicyError("Disposition approval workflow head SHA is malformed")
    for key in ("environment_config_sha256", "approval_history_sha256", "evidence_sha256"):
        value = approval.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise EvidencePolicyError(f"Disposition approval {key} is malformed")

    try:
        amendment_result = verify_validator_amendment_record(
            state=state,
            experiment_journal_path=experiment_journal_path,
            validator_path=Path(__file__).resolve().with_name("phase2d_prospective_validator.py"),
            promotion_gate_path=Path(__file__).resolve().with_name("phase2d_promotion_gate.py"),
        )
    except Exception as exc:
        raise EvidencePolicyError(f"Validator amendment evidence is invalid: {exc}") from exc
    amendment_event = amendment_result.get("event", {})
    if amendment_event.get("event_id") != event.get("validator_amendment_event_id"):
        raise EvidencePolicyError("Disposition event references a different validator amendment")
    if amendment_result.get("event_sha256") != event.get("validator_amendment_event_sha256"):
        raise EvidencePolicyError("Disposition event validator-amendment hash mismatch")

    # Reconstruct the exact schema-6 pending view from the preserved payloads.
    # This lets the immutable review artifacts be re-verified without putting
    # historical outcomes back into the official observation ledger.
    source_view = dict(state)
    source_view["pending"] = source_pending
    _verify_manifest_artifacts(manifest, manifest_path.parent, stale_ids, spec.reconciliation_id, source_view)

    # The official ledger must not contain outcome observations for this stale
    # cohort.  Later genuine prospective observations remain valid.
    all_lines = observation_ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_rows = []
    for line_number, line in enumerate(all_lines, 1):
        try:
            row = json.loads(line)
        except Exception as exc:
            raise EvidencePolicyError(f"Official observation ledger line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise EvidencePolicyError(f"Official observation ledger line {line_number} is not an object")
        ledger_rows.append(row)
    stale_key_set = {(model, prediction_id) for model, ids in stale_ids.items() for prediction_id in ids}
    scored_stale = [
        row for row in ledger_rows
        if row.get("record_type") == "observation"
        and (row.get("model"), row.get("id")) in stale_key_set
    ]
    if scored_stale:
        raise EvidencePolicyError("Official observation ledger contains scored rows for the stale not-scored cohort")

    return {
        "status": "PASS",
        "reconciliation_id": spec.reconciliation_id,
        "manifest_sha256": manifest_sha256,
        "pending_count": EXPECTED_PENDING_TOTAL,
        "pending_by_model": dict(EXPECTED_PENDING_COUNTS),
        "dispositioned_count": EXPECTED_PENDING_TOTAL,
        "dispositioned_by_model": dict(EXPECTED_PENDING_COUNTS),
        "skip_ids": {"candidate": set(), "production": set()},
        "event_id": event.get("event_id"),
        "event_sha256": event_hash,
        "outcome_rows_appended": 0,
    }


def tag_prospective_observation(result: dict) -> dict:
    """Tag only the official prospective writer output, never the shared resolver."""
    if not isinstance(result, dict):
        raise EvidencePolicyError("Prospective resolver result must be an object")
    if result.get("model") not in {"candidate", "production"}:
        raise EvidencePolicyError("Prospective resolver result has an invalid model")
    created_at = _parse_utc(result.get("created_at"), field="prospective created_at")
    resolved_at = _parse_utc(result.get("resolved_at"), field="prospective resolved_at")
    if created_at <= LEGACY_CUTOFF or resolved_at <= LEGACY_CUTOFF:
        raise EvidencePolicyError("Cannot tag an observation PROSPECTIVE when its timestamps predate the legacy cutoff")
    if "evidence_class" in result:
        raise EvidencePolicyError("Resolver result already has evidence_class; refusing implicit reclassification")
    tagged = dict(result)
    tagged["evidence_class"] = "PROSPECTIVE"
    return tagged
