#!/usr/bin/env python3
"""Build a deterministic, standalone F08 HISTORICAL_BACKFILL research bundle.

This utility is read-only with respect to every input artifact. It never writes
state, observation ledgers, experiment journals, results, or snapshots; never
appends an event; and never moves or removes pending predictions. The only
write target is a new research_outputs/f08_historical_backfill directory
containing observations.jsonl, resolution_provenance.jsonl, manifest.json,
and read-only evidence copies.

Default execution is read-only preflight. Use --write-bundle to create the
separate research bundle. Repeated execution is idempotent: identical source
inputs yield byte-identical output; an existing different output is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from phase2d_f08_resolution_provenance import (
    PROVENANCE_FILENAME,
    RESOLVED_AT_SEMANTICS,
    OUTCOME_BAR_CLOSE_TIME_SEMANTICS,
    build_resolution_provenance_records,
    canonical_jsonl_bytes as provenance_jsonl_bytes,
    verify_resolution_provenance_records,
)

EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
OPERATION = "STALE_PENDING_NOT_SCORED"
EXPECTED_STATE_SHA256 = "4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5"
EXPECTED_LEDGER_SHA256 = "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9"
EXPECTED_VALIDATOR_GIT_BLOB = "fef7a2ac75a94fe20cd095daaa6e278e4f295c5c"
EXPECTED_DEPENDENCY_BLOBS = {
    "phase2d_observation_ledger.py": "278fab7560917425f0bf918b420b40a0e9da21ac",
    "phase2d_promotion_gate.py": "497eb1a019a06f1a51921c65077de35fabbb1b9c",
    "feature_engineering.py": "27d4efa2fc966086d50387b8201b5b00ac96208d",
}
EXPECTED_PR14_MERGE = "58a64fbdeeee2a3a2851b2ef23824240eeca24e2"
EXPECTED_REVIEW_HEAD = "e592728ed88f00c7d6f62792a03b9d74808ce289"
EXPECTED_CANDIDATE_PENDING = 74
EXPECTED_PRODUCTION_PENDING = 544
EXPECTED_OBSERVATIONS = 16208
EXPECTED_EVENTS = 3
EXPECTED_TOTAL_OUTCOMES = {"TP": 292, "SL": 307, "EXPIRED": 19}
EXPECTED_CANDIDATE_OUTCOMES = {"TP": 12, "SL": 62}
EXPECTED_PRODUCTION_OUTCOMES = {"TP": 280, "SL": 245, "EXPIRED": 19}
EXPECTED_JOURNAL_ROWS = 98
# Preserve the previously approved C15/C20 gate-amendment provenance.
# These identify the historical amendment, not the later disposition approval.
EXPECTED_APPROVAL_ACTOR = "MrZkull"
EXPECTED_APPROVAL_RUN = "41720710702"
LEGACY_CUTOFF_UTC = "2026-10-16T14:08:36.946361+00:00"
LEGACY_CUTOFF = datetime.fromisoformat(LEGACY_CUTOFF_UTC)
EXPECTED_PENDING_COUNTS = {"candidate": 74, "production": 544}
EXPECTED_SOURCE_NAMES = (
    "state", "ledger", "journal", "results", "snapshot", "proposal",
    "raw_windows", "dry_run", "independent", "actual_resolver",
)

class BackfillBundleError(RuntimeError):
    """Raised when the historical bundle preflight or write is unsafe."""


# Reuse the source utility function messages while keeping one public exception type.
ReconciliationError = BackfillBundleError



def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and is_junction())


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    """Reject symlinks/junctions in lexical path components before resolving."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if _is_link_or_junction(current):
            raise BackfillBundleError(f"Symlink path component (symlink or junction) is not allowed for {label}: {current}")
    return absolute


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconciliationError(f"Cannot parse JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconciliationError(f"Expected JSON object in {path}")
    return value

def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReconciliationError(
                        f"Malformed JSONL at {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ReconciliationError(
                        f"JSONL row is not an object at {path}:{line_no}"
                    )
                rows.append(row)
    except OSError as exc:
        raise ReconciliationError(f"Cannot read {path}: {exc}") from exc
    return rows

def _values_by_key(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.append((name, child))
            found.extend(_values_by_key(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_values_by_key(child, f"{prefix}[{index}]"))
    return found

def _checkpoint_numbers(row: dict) -> set[int]:
    numbers: set[int] = set()
    for key, value in _values_by_key(row):
        if "checkpoint" not in key.lower():
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            numbers.add(value)
        elif isinstance(value, str):
            match = re.search(r"(?:C)?(\d+)", value, re.IGNORECASE)
            if match:
                numbers.add(int(match.group(1)))
    # Also support the historical explicit checkpoint_blocks field.
    value = row.get("checkpoint_blocks")
    if isinstance(value, int):
        numbers.add(value)
    return numbers

def validate_experiment_journal(path: Path) -> dict:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_JOURNAL_ROWS:
        raise ReconciliationError(
            f"Experiment journal row count changed: expected {EXPECTED_JOURNAL_ROWS}, got {len(rows)}"
        )

    checkpoint_rows = [
        row for row in rows if row.get("event") == "F08_CHECKPOINT_EVALUATED"
    ]
    c15 = [row for row in checkpoint_rows if 15 in _checkpoint_numbers(row)]
    c20 = [row for row in checkpoint_rows if 20 in _checkpoint_numbers(row)]
    later = [
        row for row in checkpoint_rows
        if any(number >= 25 for number in _checkpoint_numbers(row))
    ]
    amendments = [row for row in rows if row.get("event") == "F08_GATE_AMENDMENT"]

    if len(c15) != 1 or len(c20) != 1:
        raise ReconciliationError(
            f"Checkpoint journal invariant failed: C15={len(c15)}, C20={len(c20)}"
        )
    if later:
        raise ReconciliationError("C25 or later checkpoint exists in experiment journal")
    if len(amendments) != 1:
        raise ReconciliationError(
            f"Expected exactly one F08_GATE_AMENDMENT, found {len(amendments)}"
        )
    if rows[-1] is not amendments[0]:
        raise ReconciliationError("F08_GATE_AMENDMENT is not the final journal row")

    amendment_json = json.dumps(amendments[0], sort_keys=True)
    for required in (EXPECTED_APPROVAL_ACTOR, EXPECTED_APPROVAL_RUN, "workflow_dispatch"):
        if required not in amendment_json:
            raise ReconciliationError(
                f"F08 amendment provenance missing required value: {required}"
            )

    return {
        "rows": len(rows),
        "c15_rows": len(c15),
        "c20_rows": len(c20),
        "c25_or_later_rows": len(later),
        "f08_amendment_rows": len(amendments),
        "amendment_actor": EXPECTED_APPROVAL_ACTOR,
        "amendment_run": EXPECTED_APPROVAL_RUN,
        "amendment_is_final_row": True,
        "sha256": sha256_file(path),
    }

def load_resolver_module(resolver_repo: Path):
    resolver_repo = _reject_symlink_components(
        resolver_repo, label="pinned resolver repository"
    ).resolve(strict=True)
    validator_path = resolver_repo / "phase2d_prospective_validator.py"
    if not validator_path.is_file():
        raise ReconciliationError(f"Validator not found: {validator_path}")

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(resolver_repo), *args],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise ReconciliationError(f"Git verification failed: {exc.output}") from exc

    head = git("rev-parse", "HEAD")
    if head != EXPECTED_REVIEW_HEAD:
        raise ReconciliationError(
            f"Review worktree HEAD mismatch: expected {EXPECTED_REVIEW_HEAD}, got {head}"
        )
    pinned_files = {
        "phase2d_prospective_validator.py": EXPECTED_VALIDATOR_GIT_BLOB,
        **EXPECTED_DEPENDENCY_BLOBS,
    }
    actual_blobs = {}
    for relative_path, expected_blob in pinned_files.items():
        actual_blob = git("rev-parse", f"HEAD:{relative_path}")
        actual_blobs[relative_path] = actual_blob
        if actual_blob != expected_blob:
            raise ReconciliationError(
                f"Pinned source blob mismatch for {relative_path}: "
                f"expected {expected_blob}, got {actual_blob}"
            )
        diff = subprocess.run(
            ["git", "-C", str(resolver_repo), "diff", "--quiet", "HEAD", "--", relative_path],
            capture_output=True,
        )
        cached_diff = subprocess.run(
            ["git", "-C", str(resolver_repo), "diff", "--cached", "--quiet", "HEAD", "--", relative_path],
            capture_output=True,
        )
        if diff.returncode != 0 or cached_diff.returncode != 0:
            raise ReconciliationError(
                f"Pinned source file has uncommitted changes: {relative_path}"
            )
    blob = actual_blobs["phase2d_prospective_validator.py"]
    ancestor = subprocess.run(
        ["git", "-C", str(resolver_repo), "merge-base", "--is-ancestor", EXPECTED_PR14_MERGE, "HEAD"],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ReconciliationError("PR #14 merge commit is not an ancestor of the review worktree")
    # Unrelated untracked files (including this standalone utility and its tests)
    # are permitted.  The validator and each imported source dependency above
    # must match their exact pinned Git blobs and have no staged/unstaged diff.

    # Put the pinned review worktree first on sys.path, and remove prior imports
    # that could otherwise resolve to a different checkout.
    sys.path.insert(0, str(resolver_repo))
    for name in (
        "phase2d_prospective_validator",
        "phase2d_observation_ledger",
        "phase2d_promotion_gate",
        "feature_engineering",
    ):
        sys.modules.pop(name, None)
    try:
        module = importlib.import_module("phase2d_prospective_validator")
    except Exception as exc:
        raise ReconciliationError(f"Could not import pinned production validator: {exc}") from exc
    if Path(module.__file__).resolve() != validator_path.resolve():
        raise ReconciliationError(
            f"Imported wrong validator module: {module.__file__}"
        )
    imported_dependencies = {}
    for relative_path in EXPECTED_DEPENDENCY_BLOBS:
        module_name = Path(relative_path).stem
        imported = sys.modules.get(module_name)
        expected_path = (resolver_repo / relative_path).resolve()
        if imported is None or Path(imported.__file__).resolve() != expected_path:
            raise ReconciliationError(
                f"Imported wrong pinned dependency for {module_name}: "
                f"{getattr(imported, '__file__', None)}"
            )
        imported_dependencies[relative_path] = EXPECTED_DEPENDENCY_BLOBS[relative_path]
    return module, {
        "git_head": head,
        "validator_git_blob": blob,
        "dependency_git_blobs": imported_dependencies,
        "validator_path": str(validator_path.resolve()),
        "pr14_merge_ancestor": True,
        "pinned_files_clean": True,
    }

def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)
    except (TypeError, ValueError):
        return False

def verify_actual_resolver_report(report: dict, state_sha: str, ledger_sha: str) -> dict:
    expected = {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": 618,
        "mismatch_count": 0,
        "validator_git_blob": EXPECTED_VALIDATOR_GIT_BLOB,
        "state_sha256_before": state_sha,
        "state_sha256_after": state_sha,
        "ledger_sha256_before": ledger_sha,
        "ledger_sha256_after": ledger_sha,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ReconciliationError(
                f"Actual production resolver report mismatch for {key}: "
                f"expected {value!r}, got {report.get(key)!r}"
            )
    if report.get("candidate_verified") != EXPECTED_CANDIDATE_PENDING:
        raise ReconciliationError("Actual resolver report candidate count mismatch")
    if report.get("production_verified") != EXPECTED_PRODUCTION_PENDING:
        raise ReconciliationError("Actual resolver report production count mismatch")
    if report.get("actual_outcomes") != EXPECTED_TOTAL_OUTCOMES:
        raise ReconciliationError("Actual resolver report outcome counts mismatch")
    return {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": 618,
        "mismatch_count": 0,
        "candidate_verified": EXPECTED_CANDIDATE_PENDING,
        "production_verified": EXPECTED_PRODUCTION_PENDING,
        "state_sha256_before": state_sha,
        "state_sha256_after": state_sha,
        "ledger_sha256_before": ledger_sha,
        "ledger_sha256_after": ledger_sha,
        "actual_outcomes": dict(EXPECTED_TOTAL_OUTCOMES),
        "report_git_head": report.get("git_head"),
    }

def validate_state_baseline(state: dict, validator_module) -> dict:
    if state.get("schema_version") != 6:
        raise ReconciliationError(
            f"Unexpected Phase 2D state schema: {state.get('schema_version')}"
        )
    experiment = (
        state.get("experiment_definition", {}).get("experiment_id")
        or state.get("locked_models", {}).get("experiment_id")
    )
    if experiment != EXPERIMENT_ID:
        raise ReconciliationError(f"Unexpected experiment identity: {experiment}")

    try:
        validator_module.validate_state_integrity(state)
    except Exception as exc:
        raise ReconciliationError(f"Baseline state integrity failed: {exc}") from exc

    pending_counts = {}
    for model in ("candidate", "production"):
        pending = state.get("pending", {}).get(model, [])
        pending_counts[model] = len(pending)
    if pending_counts != {
        "candidate": EXPECTED_CANDIDATE_PENDING,
        "production": EXPECTED_PRODUCTION_PENDING,
    }:
        raise ReconciliationError(f"Pending counts changed: {pending_counts}")
    return pending_counts

def load_raw_windows(path: Path) -> dict[tuple[str, int], dict]:
    windows: dict[tuple[str, int], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["symbol"]), int(row["open_time"]))
            except Exception as exc:
                raise ReconciliationError(
                    f"Invalid raw OHLC record at {path}:{line_no}: {exc}"
                ) from exc
            if key in windows:
                raise ReconciliationError(f"Duplicate raw OHLC window: {key}")
            candles = row.get("candles")
            if not isinstance(candles, list) or len(candles) < 26:
                raise ReconciliationError(
                    f"OHLC window {key} has fewer than 26 candles"
                )
            times = [int(candle["timestamp"]) for candle in candles]
            if times != sorted(times) or len(times) != len(set(times)):
                raise ReconciliationError(f"OHLC timestamps invalid for {key}")
            if key[1] not in times:
                raise ReconciliationError(
                    f"OHLC window does not contain exact prediction open_time: {key}"
                )
            entry_index = times.index(key[1])
            if entry_index + 24 >= len(times):
                raise ReconciliationError(
                    f"OHLC window does not contain 24 subsequent candles: {key}"
                )
            resolution = str(row.get("request", {}).get("resolution", "15"))
            if resolution != "15":
                raise ReconciliationError(
                    f"Unexpected OHLC resolution for {key}: {resolution}"
                )
            for candle in candles:
                high = float(candle["high"])
                low = float(candle["low"])
                if not math.isfinite(high) or not math.isfinite(low):
                    raise ReconciliationError(f"Non-finite OHLC high/low in {key}")
                if high < low:
                    raise ReconciliationError(f"OHLC high is below low in {key}")
            windows[key] = row
    if len(windows) != 573:
        raise ReconciliationError(f"Expected 573 OHLC windows, got {len(windows)}")
    return windows

def build_resolution_plan(
    state: dict,
    proposal: dict,
    windows: dict[tuple[str, int], dict],
    resolver: Callable,
    pandas_module,
) -> tuple[list[dict], dict]:
    if proposal.get("proposal_status") != "READY_FOR_REVIEW_NOT_APPLIED":
        raise ReconciliationError(
            f"Proposal is not in review-ready/unapplied state: {proposal.get('proposal_status')}"
        )
    if proposal.get("state_or_ledger_modified") is not False:
        raise ReconciliationError("Proposal says source state or ledger was modified")
    if proposal.get("promotion_authorized") is not False:
        raise ReconciliationError("Proposal unexpectedly authorizes model promotion")
    if proposal.get("proposed_observation_appends") != 618:
        raise ReconciliationError("Proposal append count is not 618")
    if proposal.get("pending_before") != {
        "candidate": EXPECTED_CANDIDATE_PENDING,
        "production": EXPECTED_PRODUCTION_PENDING,
    }:
        raise ReconciliationError("Proposal pending_before counts changed")
    if proposal.get("pending_after_if_approved") != {
        "candidate": 0,
        "production": 0,
    }:
        raise ReconciliationError("Proposal projected pending_after counts changed")
    if proposal.get("proposed_candidate") != EXPECTED_CANDIDATE_OUTCOMES:
        raise ReconciliationError("Proposal candidate outcome counts changed")
    if proposal.get("proposed_production") != EXPECTED_PRODUCTION_OUTCOMES:
        raise ReconciliationError("Proposal production outcome counts changed")
    if proposal.get("proposed_total_outcomes") != EXPECTED_TOTAL_OUTCOMES:
        raise ReconciliationError("Proposal total outcome counts changed")

    proposal_records = proposal.get("proposal_records", [])
    if len(proposal_records) != 618:
        raise ReconciliationError(
            f"Expected 618 proposal records, got {len(proposal_records)}"
        )

    pending_by_model: dict[str, dict[str, dict]] = {}
    for model in ("candidate", "production"):
        mapped: dict[str, dict] = {}
        for pred in state.get("pending", {}).get(model, []):
            pid = str(pred.get("id"))
            if pid in mapped:
                raise ReconciliationError(f"Duplicate pending id: {model}/{pid}")
            mapped[pid] = pred
        pending_by_model[model] = mapped

    seen_ids: set[tuple[str, str]] = set()
    result_records: list[dict] = []
    status_counts: Counter = Counter()
    per_model: dict[str, Counter] = {
        "candidate": Counter(), "production": Counter()
    }
    dataframes: dict[tuple[str, int], Any] = {}

    for row_number, proposal_row in enumerate(proposal_records, 1):
        model = str(proposal_row.get("model"))
        if model not in pending_by_model:
            raise ReconciliationError(
                f"Unexpected model in proposal row {row_number}: {model}"
            )
        pid = str(proposal_row.get("prediction_id"))
        identity = (model, pid)
        if identity in seen_ids:
            raise ReconciliationError(f"Duplicate proposal prediction: {identity}")
        seen_ids.add(identity)
        pred = pending_by_model[model].get(pid)
        if pred is None:
            raise ReconciliationError(f"Proposal prediction missing from pending: {identity}")

        for field in ("symbol", "signal"):
            if str(pred.get(field)) != str(proposal_row.get(field)):
                raise ReconciliationError(f"{field} mismatch for {identity}")
        if int(pred.get("open_time")) != int(proposal_row.get("open_time")):
            raise ReconciliationError(f"open_time mismatch for {identity}")
        for field in ("entry", "atr"):
            if not _close(pred.get(field), proposal_row.get(field)):
                raise ReconciliationError(f"{field} mismatch for {identity}")
        if pred.get("created_at") != proposal_row.get("source_created_at"):
            raise ReconciliationError(f"created_at provenance mismatch for {identity}")

        expected_observation_key = (
            f"{EXPERIMENT_ID}:{model}:{pred['symbol']}:{int(pred['open_time'])}"
        )
        if proposal_row.get("observation_key") != expected_observation_key:
            raise ReconciliationError(f"Immutable observation-key mismatch for {identity}")

        key = (str(pred["symbol"]), int(pred["open_time"]))
        window = windows.get(key)
        if window is None:
            raise ReconciliationError(f"Missing raw OHLC window: {key}")
        if key not in dataframes:
            dataframes[key] = pandas_module.DataFrame({
                "open_time": [int(c["timestamp"]) for c in window["candles"]],
                "high": [float(c["high"]) for c in window["candles"]],
                "low": [float(c["low"]) for c in window["candles"]],
            })
        resolved = resolver(pred, dataframes[key])
        if resolved is None:
            raise ReconciliationError(f"Production resolver returned None for {identity}")
        if not isinstance(resolved, dict):
            raise ReconciliationError(f"Production resolver returned a non-object for {identity}")
        for field in ("id", "model", "symbol", "open_time", "signal", "created_at"):
            if resolved.get(field) != pred.get(field):
                raise ReconciliationError(
                    f"Production resolver changed source prediction identity/payload for {identity}/{field}"
                )
        for field in ("entry", "atr"):
            if not _close(resolved.get(field), pred.get(field)):
                raise ReconciliationError(
                    f"Production resolver changed source prediction value for {identity}/{field}"
                )

        expected_fields = {
            "status": proposal_row.get("proposed_status"),
            "outcome_bar": proposal_row.get("outcome_bar"),
            "gross_r": proposal_row.get("gross_r"),
            "net_r": proposal_row.get("net_r"),
        }
        actual_fields = {
            "status": resolved.get("status"),
            "outcome_bar": resolved.get("outcome_bar"),
            "gross_r": resolved.get("gross_r"),
            "net_r": resolved.get("net_r"),
        }
        for field, expected_value in expected_fields.items():
            actual_value = actual_fields[field]
            matches = (
                actual_value == expected_value
                if field in ("status", "outcome_bar")
                else _close(actual_value, expected_value)
            )
            if not matches:
                raise ReconciliationError(
                    f"Actual production resolver mismatch for {identity}/{field}: "
                    f"actual={actual_value!r}, proposal={expected_value!r}"
                )

        if resolved.get("status") not in {"TP", "SL", "EXPIRED"}:
            raise ReconciliationError(
                f"Unexpected status for pending reconciliation: {resolved.get('status')}"
            )
        if not resolved.get("resolved_at"):
            raise ReconciliationError(f"Missing actual resolver timestamp for {identity}")
        try:
            timestamp = datetime.fromisoformat(str(resolved["resolved_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReconciliationError(f"Invalid resolved_at timestamp for {identity}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ReconciliationError(f"resolved_at must include an explicit UTC offset for {identity}")
        if timestamp.astimezone(timezone.utc) <= LEGACY_CUTOFF:
            raise ReconciliationError(f"resolved_at must be after the legacy cutoff for {identity}")

        # Preserve the entire original prediction payload and actual production
        # resolver output.  Only the resolver's real current UTC timestamp is used;
        # market/prediction time is never substituted or backdated.
        result_records.append(dict(resolved))
        status_counts[resolved["status"]] += 1
        per_model[model][resolved["status"]] += 1

    if seen_ids != {
        (model, pid)
        for model, rows in pending_by_model.items()
        for pid in rows
    }:
        raise ReconciliationError("Proposal prediction IDs do not exactly equal pending IDs")

    if dict(status_counts) != EXPECTED_TOTAL_OUTCOMES:
        raise ReconciliationError(f"Production resolver outcome counts changed: {dict(status_counts)}")
    if dict(per_model["candidate"]) != EXPECTED_CANDIDATE_OUTCOMES:
        raise ReconciliationError(f"Candidate resolver outcomes changed: {dict(per_model['candidate'])}")
    if dict(per_model["production"]) != EXPECTED_PRODUCTION_OUTCOMES:
        raise ReconciliationError(f"Production model outcomes changed: {dict(per_model['production'])}")

    return result_records, {
        "predictions_resolved": len(result_records),
        "candidate": dict(per_model["candidate"]),
        "production": dict(per_model["production"]),
        "total_outcomes": dict(status_counts),
        "raw_windows_used": len(dataframes),
        "resolution_timestamp_policy": "actual UTC timestamp returned by the production resolver at reconciliation execution; no backdating",
    }

def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pending_population(state: dict) -> dict:
    pending = state.get("pending")
    if not isinstance(pending, dict):
        raise BackfillBundleError("Source state has no pending object")
    ids_by_model: dict[str, list[str]] = {"candidate": [], "production": []}
    for model in ("candidate", "production"):
        rows = pending.get(model)
        if not isinstance(rows, list):
            raise BackfillBundleError(f"Source state pending/{model} is not a list")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise BackfillBundleError(f"Source state pending/{model} contains a non-object")
            prediction_id = row.get("id")
            if not isinstance(prediction_id, str) or not prediction_id or prediction_id in seen:
                raise BackfillBundleError(f"Source state pending/{model} has missing or duplicate IDs")
            seen.add(prediction_id)
            created_at = row.get("created_at")
            if not isinstance(created_at, str):
                raise BackfillBundleError(f"Source state pending/{model}/{prediction_id} has no created_at")
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise BackfillBundleError(f"Invalid created_at for {model}/{prediction_id}") from exc
            if created.tzinfo is None:
                raise BackfillBundleError(f"created_at lacks UTC offset for {model}/{prediction_id}")
            if created.astimezone(timezone.utc) > LEGACY_CUTOFF:
                raise BackfillBundleError(f"Pending prediction is not stale: {model}/{prediction_id}")
            ids_by_model[model].append(prediction_id)
        ids_by_model[model].sort()
    counts = {model: len(ids) for model, ids in ids_by_model.items()}
    if counts != EXPECTED_PENDING_COUNTS:
        raise BackfillBundleError(f"Stale pending population mismatch: {counts}")
    digest = hashlib.sha256(canonical_json_bytes(ids_by_model)).hexdigest()
    return {"ids_by_model": ids_by_model, "pending_count": sum(counts.values()), "pending_ids_sha256": digest}


def stable_bundle_rows(result_records: list[dict], reconciliation_id: str | None = None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for result in result_records:
        if not isinstance(result, dict):
            raise BackfillBundleError("Resolver output contains a non-object")
        row = dict(result)
        # resolved_at is the resolver process wall-clock time, not an outcome input.
        # Keep the actual resolver report as separate evidence; omit this volatile
        # field from the canonical research rows so repeated runs are byte-identical.
        resolved_at = row.pop("resolved_at", None)
        if not isinstance(resolved_at, str) or not resolved_at:
            raise BackfillBundleError("Resolver output has no resolved_at evidence")
        try:
            parsed = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BackfillBundleError("Resolver output has invalid resolved_at") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise BackfillBundleError("Resolver output resolved_at must have an explicit UTC offset")
        model = row.get("model")
        prediction_id = row.get("id")
        if model not in EXPECTED_PENDING_COUNTS or not isinstance(prediction_id, str) or not prediction_id:
            raise BackfillBundleError("Resolver output has invalid model or prediction ID")
        identity = (model, prediction_id)
        if identity in seen:
            raise BackfillBundleError(f"Duplicate resolver output: {identity}")
        seen.add(identity)
        if row.get("status") not in {"TP", "SL", "EXPIRED"}:
            raise BackfillBundleError(f"Unexpected historical resolution status: {row.get('status')!r}")
        if row.get("experiment_id") not in (None, EXPERIMENT_ID):
            raise BackfillBundleError("Resolver output has unexpected experiment_id")
        row["experiment_id"] = EXPERIMENT_ID
        row["record_type"] = "historical_backfill_observation"
        row["evidence_class"] = "HISTORICAL_BACKFILL"
        row["observation_key"] = (
            f"{EXPERIMENT_ID}:{model}:{row['symbol']}:{int(row['open_time'])}"
        )
        if reconciliation_id is not None:
            row["reconciliation_id"] = reconciliation_id
        rows.append(row)
    rows.sort(key=lambda row: (row["model"], row["id"]))
    if len(rows) != EXPECTED_CANDIDATE_PENDING + EXPECTED_PRODUCTION_PENDING:
        raise BackfillBundleError(f"Historical bundle row count mismatch: {len(rows)}")
    counts = Counter(row["model"] for row in rows)
    if dict(counts) != EXPECTED_PENDING_COUNTS:
        raise BackfillBundleError(f"Historical bundle model counts mismatch: {dict(counts)}")
    return rows


def stable_resolver_identity(identity: dict) -> dict:
    expected = {
        "git_head": EXPECTED_REVIEW_HEAD,
        "validator_git_blob": EXPECTED_VALIDATOR_GIT_BLOB,
        "dependency_git_blobs": dict(EXPECTED_DEPENDENCY_BLOBS),
        "pr14_merge_ancestor": True,
        "pr14_merge_commit": EXPECTED_PR14_MERGE,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise BackfillBundleError(f"Resolver identity mismatch for {key}")
    return {**expected, "pinned_files_clean": True}


def _tree_hashes(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if _is_link_or_junction(path):
            raise BackfillBundleError(f"Symlink/junction is not allowed in output bundle: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def _verify_written_bundle(
    output_dir: Path, manifest: dict, expected_ids: dict,
    raw_windows: dict[tuple[str, int], dict],
) -> None:
    bundle_meta = manifest["historical_bundle"]
    bundle_path = output_dir / bundle_meta["bundle_path"]
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise BackfillBundleError("Written historical bundle is missing or unsafe")
    if sha256_file(bundle_path) != bundle_meta["bundle_sha256"]:
        raise BackfillBundleError("Written historical bundle SHA256 mismatch")
    records = read_jsonl(bundle_path)
    if len(records) != 618:
        raise BackfillBundleError(f"Written historical bundle count mismatch: {len(records)}")
    observed: dict[str, set[str]] = {"candidate": set(), "production": set()}
    for row in records:
        if row.get("record_type") != "historical_backfill_observation":
            raise BackfillBundleError("Unexpected record_type in historical bundle")
        if row.get("evidence_class") != "HISTORICAL_BACKFILL":
            raise BackfillBundleError("Historical bundle record has incorrect evidence_class")
        if row.get("experiment_id") != EXPERIMENT_ID:
            raise BackfillBundleError("Historical bundle record experiment_id mismatch")
        if row.get("reconciliation_id") != manifest["reconciliation_id"]:
            raise BackfillBundleError("Historical bundle record reconciliation_id mismatch")
        model = row.get("model")
        prediction_id = row.get("id")
        if model not in observed or not isinstance(prediction_id, str):
            raise BackfillBundleError("Historical bundle record has invalid identity")
        if prediction_id in observed[model]:
            raise BackfillBundleError(f"Duplicate bundle prediction ID: {model}/{prediction_id}")
        observed[model].add(prediction_id)
        if "resolved_at" in row:
            raise BackfillBundleError("Volatile resolved_at must not be stored in deterministic backfill rows")
    if observed != {model: set(ids) for model, ids in expected_ids.items()}:
        raise BackfillBundleError("Written bundle IDs do not exactly match source pending population")

    provenance_meta = manifest.get("resolution_provenance")
    if not isinstance(provenance_meta, dict):
        raise BackfillBundleError("Resolution-provenance metadata is missing")
    if (
        provenance_meta.get("schema_version") != 1
        or provenance_meta.get("format") != "jsonl"
        or provenance_meta.get("path") != PROVENANCE_FILENAME
        or provenance_meta.get("record_count") != 618
        or provenance_meta.get("raw_windows_sha256") != manifest["source_hashes"].get("raw_windows")
        or provenance_meta.get("resolved_at_semantics") != RESOLVED_AT_SEMANTICS
        or provenance_meta.get("outcome_bar_close_time_semantics") != OUTCOME_BAR_CLOSE_TIME_SEMANTICS
    ):
        raise BackfillBundleError("Resolution-provenance metadata is inconsistent")
    provenance_path = output_dir / PROVENANCE_FILENAME
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise BackfillBundleError("Resolution-provenance sidecar is missing or unsafe")
    provenance_sha = sha256_file(provenance_path)
    if provenance_sha != provenance_meta.get("sha256"):
        raise BackfillBundleError("Resolution-provenance sidecar SHA256 mismatch")
    provenance_rows = read_jsonl(provenance_path)
    try:
        verification = verify_resolution_provenance_records(
            provenance_rows, records, raw_windows, manifest["reconciliation_id"]
        )
    except Exception as exc:
        raise BackfillBundleError(f"Resolution-provenance verification failed: {exc}") from exc
    if verification.get("records_verified") != 618:
        raise BackfillBundleError("Resolution-provenance record count verification failed")

    artifacts = manifest["source_artifacts"]
    for name in EXPECTED_SOURCE_NAMES:
        entry = artifacts[name]
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or "\\" in entry["path"]:
            raise BackfillBundleError(f"Unsafe source artifact path: {entry['path']}")
        path = output_dir / rel
        if not path.is_file() or path.is_symlink():
            raise BackfillBundleError(f"Missing evidence copy for source artifact: {name}")
        actual = sha256_file(path)
        if actual != entry["sha256"] or actual != manifest["source_hashes"][name]:
            raise BackfillBundleError(f"Evidence copy hash mismatch: {name}")


def write_historical_backfill_bundle(
    *,
    repo_root: Path,
    output_dir: Path,
    source_paths: dict[str, Path],
    source_hashes: dict[str, str],
    state: dict,
    result_records: list[dict],
    plan_info: dict,
    journal_info: dict,
    resolver_identity: dict,
    resolver_verification: dict,
) -> dict:
    repo_root = _reject_symlink_components(repo_root, label="repository root").resolve(strict=True)
    output_dir = _reject_symlink_components(output_dir, label="output directory")
    research_dir = repo_root / "research_outputs"
    if research_dir.is_symlink():
        raise BackfillBundleError("research_outputs cannot be a symlink")
    expected_output = research_dir / "f08_historical_backfill"
    if output_dir != expected_output:
        raise BackfillBundleError(
            "Output directory must be exactly <repo_root>/research_outputs/f08_historical_backfill"
        )
    if not output_dir.parent.is_dir():
        raise BackfillBundleError(f"Output parent directory does not exist: {output_dir.parent}")
    if set(source_paths) != set(EXPECTED_SOURCE_NAMES) or set(source_hashes) != set(EXPECTED_SOURCE_NAMES):
        raise BackfillBundleError("Source path/hash fields do not match the required artifact set")
    for name, source in source_paths.items():
        source = _reject_symlink_components(source, label=f"source artifact {name}")
        resolved_source = source.resolve(strict=True)
        if not resolved_source.is_file():
            raise BackfillBundleError(f"Invalid source artifact: {name}={source}")
        if output_dir == resolved_source or output_dir in resolved_source.parents:
            raise BackfillBundleError(f"Output directory would contain source artifact: {name}")
        if sha256_file(resolved_source) != source_hashes[name]:
            raise BackfillBundleError(f"Source changed after preflight: {name}")
    if source_hashes["state"] != EXPECTED_STATE_SHA256 or source_hashes["ledger"] != EXPECTED_LEDGER_SHA256:
        raise BackfillBundleError("Pinned state/ledger source hashes do not match")

    # Build the provenance from the exact byte-hashed raw_windows source copy,
    # rather than accepting a caller-supplied approximation of candle data.
    raw_windows = load_raw_windows(source_paths["raw_windows"])

    population = pending_population(state)
    stable_identity = stable_resolver_identity(resolver_identity)
    base_rows = stable_bundle_rows(result_records)
    if {model: sorted(row["id"] for row in base_rows if row["model"] == model)
        for model in ("candidate", "production")} != population["ids_by_model"]:
        raise BackfillBundleError("Resolver result IDs do not exactly match stale pending population")

    base_rows_sha = hashlib.sha256(b"\n".join(canonical_json_bytes(row) for row in base_rows) + b"\n").hexdigest()
    rec_material = {
        "source_hashes": source_hashes,
        "resolver_identity": stable_identity,
        "pending_ids_sha256": population["pending_ids_sha256"],
        "normalized_result_rows_sha256": base_rows_sha,
    }
    reconciliation_digest = hashlib.sha256(canonical_json_bytes(rec_material)).hexdigest()
    reconciliation_id = "F08-HIST-BF-" + reconciliation_digest[:32]
    bundle_rows = stable_bundle_rows(result_records, reconciliation_id)
    bundle_bytes = b"\n".join(canonical_json_bytes(row) for row in bundle_rows) + b"\n"
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    try:
        provenance_rows = build_resolution_provenance_records(
            result_records, raw_windows, reconciliation_id
        )
        provenance_bytes = provenance_jsonl_bytes(provenance_rows)
    except Exception as exc:
        raise BackfillBundleError(f"Cannot construct per-prediction resolution provenance: {exc}") from exc
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()

    temp_dir = Path(tempfile.mkdtemp(prefix=".f08_historical_backfill.tmp-", dir=output_dir.parent))
    try:
        evidence_dir = temp_dir / "evidence"
        evidence_dir.mkdir()
        source_artifacts: dict[str, dict] = {}
        for name in EXPECTED_SOURCE_NAMES:
            source = source_paths[name].resolve(strict=True)
            suffix = source.suffix if source.suffix else ".bin"
            relative = Path("evidence") / f"{name}{suffix}"
            destination = temp_dir / relative
            shutil.copyfile(source, destination)
            digest = sha256_file(destination)
            if digest != source_hashes[name]:
                raise BackfillBundleError(f"Evidence copy did not reproduce source hash: {name}")
            source_artifacts[name] = {"path": relative.as_posix(), "sha256": digest}

        bundle_relative = "observations.jsonl"
        (temp_dir / bundle_relative).write_bytes(bundle_bytes)
        (temp_dir / PROVENANCE_FILENAME).write_bytes(provenance_bytes)
        manifest = {
            "manifest_schema_version": 1,
            "operation": OPERATION,
            "experiment_id": EXPERIMENT_ID,
            "evidence_class": "HISTORICAL_BACKFILL",
            "legacy_cutoff_utc": LEGACY_CUTOFF_UTC,
            "manifest_status": "READY_FOR_OWNER_REVIEW",
            "reconciliation_id": reconciliation_id,
            "source_hashes": dict(source_hashes),
            "source_artifacts": source_artifacts,
            "resolver_identity": stable_identity,
            "actual_resolver_report_sha256": source_hashes["actual_resolver"],
            "actual_resolver_verification": dict(resolver_verification),
            "raw_ohlc_windows_sha256": source_hashes["raw_windows"],
            "resolution_provenance": {
                "schema_version": 1,
                "format": "jsonl",
                "path": PROVENANCE_FILENAME,
                "sha256": provenance_sha,
                "record_count": len(provenance_rows),
                "raw_windows_sha256": source_hashes["raw_windows"],
                "resolved_at_semantics": RESOLVED_AT_SEMANTICS,
                "outcome_bar_close_time_semantics": OUTCOME_BAR_CLOSE_TIME_SEMANTICS,
            },
            "historical_bundle": {
                "evidence_class": "HISTORICAL_BACKFILL",
                "format": "jsonl",
                "bundle_path": bundle_relative,
                "bundle_sha256": bundle_sha,
                "observation_count": len(bundle_rows),
                "candidate_count": EXPECTED_CANDIDATE_PENDING,
                "production_count": EXPECTED_PRODUCTION_PENDING,
                "resolved_at_policy": "omitted from deterministic observation rows; exact per-prediction resolver wall-clock timestamps are preserved in the separately hashed resolution_provenance.jsonl sidecar",
                "normalized_result_rows_sha256": base_rows_sha,
            },
            "pending_population": population,
            "experiment_journal_review": journal_info,
            "resolution": plan_info,
            "source_artifact_policy": "all ten source files are copied byte-for-byte under evidence/ and verified by SHA256; resolution_provenance.jsonl is a separate generated sidecar bound to exact raw OHLC windows",
            "approval_status": "PENDING_OWNER_APPROVAL",
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
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        (temp_dir / "manifest.json").write_bytes(manifest_bytes)
        _verify_written_bundle(temp_dir, manifest, population["ids_by_model"], raw_windows)

        # Re-hash every original input immediately before publication.
        for name, source in source_paths.items():
            if sha256_file(source) != source_hashes[name]:
                raise BackfillBundleError(f"Source artifact changed during bundle creation: {name}")

        staged_tree = _tree_hashes(temp_dir)
        if output_dir.exists():
            existing_tree = _tree_hashes(output_dir)
            if existing_tree == staged_tree:
                shutil.rmtree(temp_dir)
                return {
                    "status": "ALREADY_PRESENT_IDENTICAL",
                    "output_dir": str(output_dir),
                    "reconciliation_id": reconciliation_id,
                    "bundle_sha256": bundle_sha,
                    "resolution_provenance_sha256": provenance_sha,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                }
            raise BackfillBundleError("Output directory already exists with different contents; refusing overwrite")
        os.replace(temp_dir, output_dir)
        return {
            "status": "CREATED",
            "output_dir": str(output_dir),
            "reconciliation_id": reconciliation_id,
            "bundle_sha256": bundle_sha,
            "resolution_provenance_sha256": provenance_sha,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def run_preflight(args) -> dict:
    raw_source_paths = {
        "state": Path(args.state),
        "ledger": Path(args.ledger),
        "journal": Path(args.journal),
        "results": Path(args.results),
        "snapshot": Path(args.snapshot),
        "proposal": Path(args.proposal),
        "raw_windows": Path(args.raw_windows),
        "dry_run": Path(args.dry_run_report),
        "independent": Path(args.independent_report),
        "actual_resolver": Path(args.actual_resolver_report),
    }
    source_paths: dict[str, Path] = {}
    for name, raw_path in raw_source_paths.items():
        checked_path = _reject_symlink_components(raw_path, label=f"source input {name}")
        try:
            resolved_path = checked_path.resolve(strict=True)
        except OSError as exc:
            raise BackfillBundleError(f"Missing {name} input: {checked_path}") from exc
        if not resolved_path.is_file():
            raise BackfillBundleError(f"Missing or unsafe {name} input: {resolved_path}")
        source_paths[name] = resolved_path
    before_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if before_hashes["state"] != EXPECTED_STATE_SHA256:
        raise BackfillBundleError("Authoritative source state SHA256 mismatch")
    if before_hashes["ledger"] != EXPECTED_LEDGER_SHA256:
        raise BackfillBundleError("Authoritative source observation ledger SHA256 mismatch")

    state = read_json(source_paths["state"])
    proposal = read_json(source_paths["proposal"])
    dry_run = read_json(source_paths["dry_run"])
    independent = read_json(source_paths["independent"])
    actual_resolver = read_json(source_paths["actual_resolver"])
    if proposal.get("source_state_sha256") != before_hashes["state"]:
        raise BackfillBundleError("Proposal does not bind to the expected source state")
    if proposal.get("source_observation_ledger_sha256") != before_hashes["ledger"]:
        raise BackfillBundleError("Proposal does not bind to the expected source ledger")
    if proposal.get("source_dry_run_sha256") != before_hashes["dry_run"]:
        raise BackfillBundleError("Proposal dry-run hash mismatch")
    if proposal.get("source_independent_verification_sha256") != before_hashes["independent"]:
        raise BackfillBundleError("Proposal independent-verification hash mismatch")
    if not isinstance(dry_run, dict) or not isinstance(independent, dict):
        raise BackfillBundleError("Dry-run or independent verification evidence is invalid")

    validator, code_identity = load_resolver_module(Path(args.resolver_repo))
    validate_state_baseline(state, validator)
    ledger_module = importlib.import_module("phase2d_observation_ledger")
    observations = ledger_module.load_observations(source_paths["ledger"], experiment_id=EXPERIMENT_ID)
    events = ledger_module.load_events(source_paths["ledger"], experiment_id=EXPERIMENT_ID)
    if len(observations) != EXPECTED_OBSERVATIONS:
        raise BackfillBundleError(f"Existing observation count mismatch: {len(observations)}")
    if len(events) != EXPECTED_EVENTS:
        raise BackfillBundleError(f"Existing observation event count mismatch: {len(events)}")
    if any(row.get("event") == OPERATION for row in events):
        raise BackfillBundleError("Disposition event already exists in source ledger; bundle preflight must precede event append")
    existing_keys = {row.get("observation_key") for row in observations}
    if len(existing_keys) != EXPECTED_OBSERVATIONS:
        raise BackfillBundleError("Official baseline observation keys are not unique")

    journal_info = validate_experiment_journal(source_paths["journal"])
    resolver_verification = verify_actual_resolver_report(
        actual_resolver, before_hashes["state"], before_hashes["ledger"]
    )
    windows = load_raw_windows(source_paths["raw_windows"])
    import pandas as pd
    result_records, plan_info = build_resolution_plan(
        state, proposal, windows, validator.resolve_prediction, pd
    )
    proposed_keys: set[str] = set()
    for result in result_records:
        key = ledger_module.observation_key(EXPERIMENT_ID, result)
        if key in proposed_keys:
            raise BackfillBundleError(f"Duplicate proposed immutable observation key: {key}")
        proposed_keys.add(key)
        if key in existing_keys:
            raise BackfillBundleError(f"Proposed observation already exists in official ledger: {key}")

    population = pending_population(state)
    result_ids = {
        model: sorted(str(row["id"]) for row in result_records if row.get("model") == model)
        for model in ("candidate", "production")
    }
    if result_ids != population["ids_by_model"]:
        raise BackfillBundleError("Production resolver IDs do not exactly equal stale pending IDs")

    after_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if after_hashes != before_hashes:
        raise BackfillBundleError("One or more source inputs changed during read-only preflight")
    # Remove machine-specific absolute checkout path from the durable identity.
    code_identity = stable_resolver_identity({
        **code_identity,
        "pr14_merge_commit": EXPECTED_PR14_MERGE,
    })
    return {
        "source_paths": source_paths,
        "source_hashes": before_hashes,
        "state": state,
        "result_records": result_records,
        "plan_info": plan_info,
        "journal_info": journal_info,
        "resolver_identity": code_identity,
        "resolver_verification": resolver_verification,
        "pending_population": population,
        "raw_windows": windows,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Offline F08 historical-backfill bundle builder; read-only preflight is the default."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--raw-windows", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--independent-report", required=True)
    parser.add_argument("--actual-resolver-report", required=True)
    parser.add_argument("--resolver-repo", required=True)
    parser.add_argument("--write-bundle", action="store_true", help="Write only a standalone research bundle; never writes official state or ledger.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = run_preflight(args)
    print("=" * 72)
    print("F08 HISTORICAL_BACKFILL BUNDLE PREFLIGHT")
    print("=" * 72)
    print("Preflight: PASS")
    print("Resolver: exact pinned production implementation")
    print("Pending predictions: 618 (candidate 74, production 544)")
    print("Production outcomes:", plan["plan_info"]["total_outcomes"])
    print("Raw windows used:", plan["plan_info"]["raw_windows_used"])
    print("Official state written: NO")
    print("Official ledger written: NO")
    print("Official event appended: NO")
    print("Pending predictions mutated: NO")
    print("C15/C20 modified: NO")
    print("C25 evaluated: NO")
    print("Promotion/live trading: NO")
    if not args.write_bundle:
        print("Mode: READ-ONLY PREFLIGHT; no output files written")
        return 0

    repo_root = _reject_symlink_components(
        Path(args.repo_root), label="repository root"
    ).resolve(strict=True)
    output_dir = repo_root / "research_outputs" / "f08_historical_backfill"
    result = write_historical_backfill_bundle(
        repo_root=repo_root,
        output_dir=output_dir,
        source_paths=plan["source_paths"],
        source_hashes=plan["source_hashes"],
        state=plan["state"],
        result_records=plan["result_records"],
        plan_info=plan["plan_info"],
        journal_info=plan["journal_info"],
        resolver_identity=plan["resolver_identity"],
        resolver_verification=plan["resolver_verification"],
    )
    print("Bundle write:", result["status"])
    print("Output directory:", result["output_dir"])
    print("Reconciliation ID:", result["reconciliation_id"])
    print("Bundle SHA256:", result["bundle_sha256"])
    print("Resolution provenance SHA256:", result["resolution_provenance_sha256"])
    print("Manifest SHA256:", result["manifest_sha256"])
    print("Mode: STANDALONE HISTORICAL_BACKFILL; owner approval still required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackfillBundleError as exc:
        print(f"BACKFILL BUNDLE BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BACKFILL BUNDLE FAILED CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
