#!/usr/bin/env python3
"""F08 schema-7 runtime bootstrap and continuation contracts.

The first F08 runtime install is sourced only from the fixed immutable GitHub
Release asset.  Every later run is sourced only from the exact state artifact
named by the bot-state branch pointer.  This module deliberately contains no
workflow-run search or fallback-to-older-run behavior.

Runtime installation is verify-before-write and verify-after-write.  A failed
install is allowed to leave partial files in the ephemeral runner workspace;
the calling workflow must stop and must not upload a continuation artifact or
advance the bot-state pointer.  No backup/rollback state is created.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import phase2d_f08_v2_restore as v2
import phase2d_f08_disposition as disposition

REPOSITORY = "MrZkull/Crypto_AI_bot"
WORKFLOW_NAME = "CryptoBot AI — Phase 2D Prospective Validation"
WORKFLOW_PATH = ".github/workflows/run_phase2d.yml"
MAIN_BRANCH = "main"
STATE_BRANCH = "bot-state"
STATE_POINTER_PATH = "f08/phase2d-state-f08-pointer.json"
STATE_ARTIFACT_NAME = "phase2d-state-f08"
RELEASE_TAG = "f08-schema7-disposition-v1"
RELEASE_ASSET_NAME = "f08-schema7-runtime-handoff.zip"
EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
STATE_SCHEMA_VERSION = 7
DISPOSITION_WORKFLOW_NAME = "Phase2D F08 Controlled Disposition"
DISPOSITION_WORKFLOW_PATH = ".github/workflows/phase2d_f08_controlled_disposition.yml"
EXCLUDED_DISPOSITION_ACTOR_ID = "166250462"
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "PHASE2D_F08_RUNTIME_CONTINUATION"
ARTIFACT_STATUS = "VERIFIED_SCHEMA7_CONTINUATION"
BOOTSTRAP_MANIFEST_NAME = "f08_v2_handoff_manifest.json"
CONTINUATION_MANIFEST_NAME = "phase2d_state_f08_manifest.json"
RUNTIME_FILES = (
    "phase2d_state.json",
    "phase2d_results.json",
    "phase2d_latest_snapshot.json",
    "phase2d_observations.jsonl",
    "experiment_ledger.jsonl",
)
BACKFILL_DIR = "f08_historical_backfill"
LOCKED_MODEL_FIELDS = ("candidate_sha256", "production_sha256")
MAX_RELEASE_MEMBERS = 20_000
MAX_RELEASE_UNCOMPRESSED_BYTES = 1_500_000_000


class HandoffError(RuntimeError):
    """Raised when an F08 runtime source or continuation cannot be trusted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HandoffError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HandoffError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def _is_run_id(value: Any) -> bool:
    return not isinstance(value, bool) and str(value).isdigit() and int(value) > 0


def _experiment_id(state: dict[str, Any]) -> str | None:
    locked = state.get("locked_models")
    if isinstance(locked, dict) and isinstance(locked.get("experiment_id"), str):
        return locked["experiment_id"]
    definition = state.get("experiment_definition")
    if isinstance(definition, dict) and isinstance(definition.get("experiment_id"), str):
        return definition["experiment_id"]
    return None


def _locked_model_hashes(state: dict[str, Any]) -> dict[str, str]:
    locked = state.get("locked_models")
    if not isinstance(locked, dict):
        raise HandoffError("Schema-7 runtime state has no locked_models object")
    hashes = {key: locked.get(key) for key in LOCKED_MODEL_FIELDS}
    if any(not _is_sha(value) for value in hashes.values()):
        raise HandoffError("Schema-7 runtime state has missing or malformed locked model hashes")
    return hashes  # type: ignore[return-value]


def _verify_state_identity(
    state_path: Path,
    *,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    state = _load_object(state_path, "F08 runtime state")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise HandoffError(f"F08 runtime schema mismatch: expected 7, got {state.get('schema_version')!r}")
    actual_experiment = _experiment_id(state)
    if actual_experiment != expected_experiment_id:
        raise HandoffError(f"F08 runtime experiment mismatch: {actual_experiment!r}")
    hashes = _locked_model_hashes(state)
    if expected_model_hashes is not None and hashes != expected_model_hashes:
        raise HandoffError("F08 runtime locked model hashes differ from the approved identities")
    return state, hashes


def _assert_no_links(root: Path, label: str) -> None:
    if root.is_symlink():
        raise HandoffError(f"{label} is a symlink: {root}")
    if root.exists() and root.is_dir():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise HandoffError(f"{label} contains a symlink: {path}")


def _safe_copy_tree(source: Path, destination: Path, label: str) -> None:
    _assert_no_links(source, label)
    if not source.is_dir():
        raise HandoffError(f"{label} directory is missing: {source}")
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved or source_resolved in destination_resolved.parents or destination_resolved in source_resolved.parents:
        raise HandoffError(f"{label} source and destination overlap")
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise HandoffError(f"{label} contains a symlink: {item}")
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def _runtime_inventory(root: Path, *, include_bootstrap_manifest: bool) -> dict[str, str]:
    _assert_no_links(root, "F08 runtime directory")
    inventory: dict[str, str] = {}
    for name in RUNTIME_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise HandoffError(f"F08 runtime file is missing or unsafe: {name}")
        inventory[name] = _sha_file(path)
    backfill = root / BACKFILL_DIR
    if backfill.is_symlink() or not backfill.is_dir():
        raise HandoffError("F08 historical-backfill directory is missing or unsafe")
    required_backfill = ("manifest.json", "observations.jsonl")
    for name in required_backfill:
        path = backfill / name
        if path.is_symlink() or not path.is_file():
            raise HandoffError(f"F08 historical-backfill file is missing: {name}")
    evidence = backfill / "evidence"
    if evidence.is_symlink() or not evidence.is_dir():
        raise HandoffError("F08 historical-backfill evidence directory is missing")
    for path in sorted(backfill.rglob("*")):
        if path.is_symlink():
            raise HandoffError(f"F08 historical-backfill contains a symlink: {path}")
        if path.is_file():
            rel = f"{BACKFILL_DIR}/{path.relative_to(backfill).as_posix()}"
            inventory[rel] = _sha_file(path)
    if include_bootstrap_manifest:
        marker = root / BOOTSTRAP_MANIFEST_NAME
        if marker.is_symlink() or not marker.is_file():
            raise HandoffError("F08 bootstrap provenance manifest is missing")
        inventory[BOOTSTRAP_MANIFEST_NAME] = _sha_file(marker)
    return inventory


def _copy_runtime_payload(source: Path, destination: Path, *, include_continuation_manifest: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        src = source / name
        if src.is_symlink() or not src.is_file():
            raise HandoffError(f"Runtime package is missing required file: {name}")
        shutil.copyfile(src, destination / name)
    _safe_copy_tree(source / BACKFILL_DIR, destination / BACKFILL_DIR, "F08 historical-backfill")
    marker = source / BOOTSTRAP_MANIFEST_NAME
    if marker.is_symlink() or not marker.is_file():
        raise HandoffError("Runtime package is missing the immutable bootstrap provenance manifest")
    shutil.copyfile(marker, destination / BOOTSTRAP_MANIFEST_NAME)
    if include_continuation_manifest:
        manifest = source / CONTINUATION_MANIFEST_NAME
        if manifest.is_symlink() or not manifest.is_file():
            raise HandoffError("Continuation package manifest is missing")
        shutil.copyfile(manifest, destination / CONTINUATION_MANIFEST_NAME)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    if archive.is_symlink() or not archive.is_file():
        raise HandoffError("Fixed Release asset ZIP is missing or unsafe")
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            if not infos or len(infos) > MAX_RELEASE_MEMBERS:
                raise HandoffError("Release ZIP has an invalid member count")
            for info in infos:
                raw_name = info.filename
                if "\\" in raw_name:
                    raise HandoffError(f"Release ZIP member uses an unsafe path separator: {raw_name}")
                pure = PurePosixPath(raw_name)
                if pure.is_absolute() or not pure.parts or ".." in pure.parts or ":" in pure.parts[0]:
                    raise HandoffError(f"Release ZIP contains an unsafe path: {raw_name}")
                normalized = pure.as_posix().rstrip("/")
                if not normalized or normalized in seen:
                    raise HandoffError(f"Release ZIP contains a duplicate or empty path: {raw_name}")
                seen.add(normalized)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise HandoffError(f"Release ZIP contains a symlink: {raw_name}")
                total += info.file_size
                if total > MAX_RELEASE_UNCOMPRESSED_BYTES:
                    raise HandoffError("Release ZIP exceeds the maximum uncompressed size")
                target = destination.joinpath(*pure.parts)
                resolved = target.resolve(strict=False)
                if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                    raise HandoffError(f"Release ZIP member escapes extraction root: {raw_name}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except HandoffError:
        raise
    except Exception as exc:
        raise HandoffError(f"Unable to safely extract fixed Release asset: {exc}") from exc


def _release_asset(release: dict[str, Any], archive: Path, *, expected_tag: str, expected_name: str) -> dict[str, Any]:
    if release.get("tag_name") != expected_tag:
        raise HandoffError("GitHub Release tag does not match the fixed F08 bootstrap tag")
    if release.get("immutable") is not True:
        raise HandoffError("F08 bootstrap Release is not reported immutable by the GitHub API")
    if release.get("draft") is not False or release.get("prerelease") is not False:
        raise HandoffError("F08 bootstrap Release must be published and not a prerelease")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise HandoffError("F08 bootstrap Release has malformed asset metadata")
    matches = [row for row in assets if isinstance(row, dict) and row.get("name") == expected_name]
    if len(matches) != 1:
        raise HandoffError(f"Expected exactly one fixed Release asset {expected_name!r}; found {len(matches)}")
    asset = matches[0]
    digest = asset.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or not _is_sha(digest[7:]):
        raise HandoffError("GitHub Release asset does not expose a valid SHA-256 digest")
    if asset.get("state") not in (None, "uploaded"):
        raise HandoffError("Fixed Release asset is not in uploaded state")
    if isinstance(asset.get("size"), bool) or not isinstance(asset.get("size"), int) or asset["size"] <= 0:
        raise HandoffError("Fixed Release asset has invalid size metadata")
    if archive.stat().st_size != asset["size"]:
        raise HandoffError("Downloaded fixed Release asset size differs from GitHub metadata")
    if _sha_file(archive) != digest[7:]:
        raise HandoffError("Downloaded fixed Release asset SHA-256 differs from GitHub's published digest")
    return asset


def _validate_bootstrap_manifest(marker: dict[str, Any], *, expected_experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    """Validate immutable bootstrap provenance fields without reading mutable runtime files."""
    expected = {
        "schema_version": 2,
        "handoff_status": "VERIFIED_SCHEMA7_RUNTIME_HANDOFF",
        "bootstrap_type": "IMMUTABLE_GITHUB_RELEASE",
        "repository": REPOSITORY,
        "release_tag": RELEASE_TAG,
        "release_immutable": True,
        "asset_name": RELEASE_ASSET_NAME,
        "experiment_id": expected_experiment_id,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "verification_status": "PASS",
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise HandoffError(f"F08 bootstrap manifest {key} mismatch")
    if not _is_commit(marker.get("source_commit")) or not _is_run_id(marker.get("disposition_run_id")):
        raise HandoffError("F08 bootstrap manifest source commit or disposition run ID is malformed")
    if not _is_run_id(marker.get("validator_amendment_run_id")):
        raise HandoffError("F08 bootstrap manifest validator-amendment run ID is malformed")
    if not _is_sha(marker.get("asset_sha256")):
        raise HandoffError("F08 bootstrap manifest Release asset digest is malformed")
    size = marker.get("asset_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HandoffError("F08 bootstrap manifest Release asset size is malformed")
    hashes = marker.get("locked_model_hashes")
    if not isinstance(hashes, dict) or any(not _is_sha(hashes.get(k)) for k in LOCKED_MODEL_FIELDS):
        raise HandoffError("F08 bootstrap manifest locked model hashes are malformed")
    inventory = marker.get("files_sha256")
    if not isinstance(inventory, dict) or not inventory or any(not _is_sha(v) for v in inventory.values()):
        raise HandoffError("F08 bootstrap file inventory is missing or malformed")
    return marker


def verify_runtime_handoff(handoff_dir: Path, *, expected_experiment_id: str = EXPERIMENT_ID,
                           expected_model_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    marker = _validate_bootstrap_manifest(
        _load_object(handoff_dir / BOOTSTRAP_MANIFEST_NAME, "F08 bootstrap manifest"),
        expected_experiment_id=expected_experiment_id,
    )
    expected = marker["files_sha256"]
    actual = _runtime_inventory(handoff_dir, include_bootstrap_manifest=False)
    if actual != expected:
        raise HandoffError("F08 bootstrap runtime file hash inventory mismatch")
    _, hashes = _verify_state_identity(
        handoff_dir / "phase2d_state.json",
        expected_experiment_id=expected_experiment_id,
        expected_model_hashes=expected_model_hashes,
    )
    if hashes != marker.get("locked_model_hashes"):
        raise HandoffError("F08 bootstrap manifest locked model hashes mismatch")
    return marker


def prepare_bootstrap_release(
    release_json: Path,
    archive_path: Path,
    tag_commit: str,
    output_dir: Path,
    *,
    workflow_commit: str,
    expected_tag: str = RELEASE_TAG,
    expected_asset_name: str = RELEASE_ASSET_NAME,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    release = _load_object(release_json, "GitHub fixed Release metadata")
    asset = _release_asset(release, archive_path, expected_tag=expected_tag, expected_name=expected_asset_name)
    if not _is_commit(tag_commit):
        raise HandoffError("Resolved Release tag commit is malformed")
    if not _is_commit(workflow_commit):
        raise HandoffError("Current bootstrap workflow commit is malformed")
    if workflow_commit != tag_commit:
        raise HandoffError(
            "Current bootstrap workflow commit differs from the immutable Release source commit"
        )
    if output_dir.exists():
        raise HandoffError(f"Refusing to overwrite existing bootstrap staging directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    extracted = Path(tempfile.mkdtemp(prefix="f08_release_extract_", dir=output_dir.parent))
    try:
        _safe_extract_zip(archive_path, extracted / "package")
        package = extracted / "package"
        package_manifest = _load_object(package / "manifest_v2.json", "F08 V2 Release package manifest")
        if package_manifest.get("source_commit") != tag_commit:
            raise HandoffError("Fixed Release tag commit differs from the V2 package's approved source commit")
        if package_manifest.get("experiment_id") != expected_experiment_id:
            raise HandoffError("Fixed Release package experiment identity mismatch")
        if package_manifest.get("artifact_status") != v2.ARTIFACT_STATUS:
            raise HandoffError("Fixed Release package is not a verified quarantined V2 disposition package")
        v2.verify_artifact(package, expected_source_commit=tag_commit, runtime_source_root=None)
        post = package / "disposition"
        state, hashes = _verify_state_identity(
            post / "phase2d_state.json",
            expected_experiment_id=expected_experiment_id,
            expected_model_hashes=expected_model_hashes,
        )
        if package_manifest.get("disposition_run_id") is None or not _is_run_id(package_manifest.get("disposition_run_id")):
            raise HandoffError("F08 V2 Release package disposition run ID is malformed")
        if package_manifest.get("validator_amendment_run_id") is None or not _is_run_id(package_manifest.get("validator_amendment_run_id")):
            raise HandoffError("F08 V2 Release package validator amendment run ID is malformed")
        for name in RUNTIME_FILES:
            source = post / name
            if source.is_symlink() or not source.is_file():
                raise HandoffError(f"Verified F08 Release package is missing disposition/{name}")
            shutil.copyfile(source, staging / name)
        _safe_copy_tree(package / "historical_backfill", staging / BACKFILL_DIR, "F08 Release historical-backfill")
        marker = {
            "schema_version": 2,
            "handoff_status": "VERIFIED_SCHEMA7_RUNTIME_HANDOFF",
            "bootstrap_type": "IMMUTABLE_GITHUB_RELEASE",
            "repository": REPOSITORY,
            "release_tag": expected_tag,
            "release_immutable": True,
            "asset_name": expected_asset_name,
            "asset_sha256": asset["digest"][7:],
            "asset_size_bytes": asset["size"],
            "source_commit": tag_commit,
            "experiment_id": expected_experiment_id,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "validator_amendment_run_id": str(package_manifest["validator_amendment_run_id"]),
            "disposition_run_id": str(package_manifest["disposition_run_id"]),
            "locked_model_hashes": hashes,
            "files_sha256": _runtime_inventory(staging, include_bootstrap_manifest=False),
            "verification_status": "PASS",
        }
        _write_json(staging / BOOTSTRAP_MANIFEST_NAME, marker)
        verify_runtime_handoff(staging, expected_experiment_id=expected_experiment_id,
                               expected_model_hashes=expected_model_hashes)
        os.replace(staging, output_dir)
        return verify_runtime_handoff(output_dir, expected_experiment_id=expected_experiment_id,
                                      expected_model_hashes=expected_model_hashes)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def select_latest_disposition_run(runs_payload: Any) -> dict[str, Any]:
    """Select the newest F08 disposition dispatch run without stale-run fallback.

    The caller queries the workflow-specific GitHub Actions runs endpoint with
    branch=main and event=workflow_dispatch.  We still independently validate
    every returned row's workflow/repository/ref identity, select by the
    monotonically increasing workflow run_number, and require that newest run
    to be a first-attempt success.  A failed, queued, in-progress, rerun, or
    owner-initiated newest run blocks selection; older successful runs are
    never substituted.
    """
    if not isinstance(runs_payload, dict):
        raise HandoffError("F08 disposition workflow-runs response must be an object")
    rows = runs_payload.get("workflow_runs")
    if not isinstance(rows, list) or not rows:
        raise HandoffError("No F08 controlled-disposition workflow dispatch runs were returned")

    validated: list[dict[str, Any]] = []
    for index, run in enumerate(rows):
        if not isinstance(run, dict):
            raise HandoffError(f"F08 disposition workflow run row {index} is malformed")
        expected = {
            "name": DISPOSITION_WORKFLOW_NAME,
            "path": DISPOSITION_WORKFLOW_PATH,
            "head_branch": MAIN_BRANCH,
            "event": "workflow_dispatch",
        }
        for key, value in expected.items():
            if run.get(key) != value:
                raise HandoffError(f"F08 disposition workflow run {index} {key} mismatch")
        repository = run.get("repository")
        if not isinstance(repository, dict) or repository.get("full_name") != REPOSITORY:
            raise HandoffError(f"F08 disposition workflow run {index} repository mismatch")
        run_id = run.get("id")
        run_number = run.get("run_number")
        if not _is_run_id(run_id) or isinstance(run_number, bool) or not isinstance(run_number, int) or run_number < 1:
            raise HandoffError(f"F08 disposition workflow run {index} ID/run number is malformed")
        head_sha = run.get("head_sha")
        if not _is_commit(head_sha):
            raise HandoffError(f"F08 disposition workflow run {index} head SHA is malformed")
        actor = run.get("actor")
        actor_id = actor.get("id") if isinstance(actor, dict) else None
        if not _is_run_id(actor_id):
            raise HandoffError(f"F08 disposition workflow run {index} actor identity is missing or malformed")
        validated.append(run)

    newest_number = max(run["run_number"] for run in validated)
    newest_rows = [run for run in validated if run["run_number"] == newest_number]
    if len(newest_rows) != 1:
        raise HandoffError("Newest F08 disposition workflow run number is ambiguous")
    newest = newest_rows[0]

    if str(newest.get("actor", {}).get("id")) == EXCLUDED_DISPOSITION_ACTOR_ID:
        raise HandoffError("Newest F08 disposition run was initiated by the excluded owner identity")
    if newest.get("status") != "completed" or newest.get("conclusion") != "success":
        raise HandoffError(
            "Newest F08 disposition run is not completed successfully; refusing fallback to an older run"
        )
    if newest.get("run_attempt") != 1:
        raise HandoffError("Newest F08 disposition run is a rerun; a new first-attempt dispatch is required")

    return {
        "selection_status": "PASS",
        "selection_policy": "NEWEST_MATCHING_DISPATCH_ONLY_NO_OLDER_RUN_FALLBACK",
        "selected_run_id": str(newest["id"]),
        "selected_run_number": newest_number,
        "selected_head_sha": newest["head_sha"],
        "selected_run": newest,
    }


def select_disposition_run_file(runs_json: Path, output_json: Path) -> str:
    payload = _load_object(runs_json, "F08 disposition workflow-runs response")
    selected = select_latest_disposition_run(payload)
    _write_json(output_json, selected)
    return selected["selected_run_id"]


def _validate_pointer(pointer: dict[str, Any], *, repository: str = REPOSITORY) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": repository,
        "state_branch": STATE_BRANCH,
        "state_artifact_name": STATE_ARTIFACT_NAME,
        "workflow_path": WORKFLOW_PATH,
        "experiment_id": EXPERIMENT_ID,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if pointer.get(key) != value:
            raise HandoffError(f"F08 state pointer {key} mismatch")
    if not _is_run_id(pointer.get("producer_run_id")) or not _is_commit(pointer.get("producer_commit")):
        raise HandoffError("F08 state pointer producer run/commit is malformed")
    if not _is_run_id(pointer.get("artifact_id")) or not _is_sha(pointer.get("manifest_sha256")):
        raise HandoffError("F08 state pointer artifact ID/manifest SHA is malformed")
    if not _is_run_id(pointer.get("previous_run_id")) or not _is_commit(pointer.get("previous_commit")):
        raise HandoffError("F08 state pointer predecessor run/commit is malformed")
    if not _is_sha(pointer.get("previous_manifest_sha256")):
        raise HandoffError("F08 state pointer predecessor manifest SHA256 is malformed")
    if not _is_sha(pointer.get("bootstrap_asset_sha256")):
        raise HandoffError("F08 state pointer bootstrap asset SHA256 is malformed")
    hashes = pointer.get("locked_model_hashes")
    if not isinstance(hashes, dict) or any(not _is_sha(hashes.get(k)) for k in LOCKED_MODEL_FIELDS):
        raise HandoffError("F08 state pointer locked model hashes are malformed")
    return pointer


def _normalize_artifact_payload(payload: Any, *, artifact_id: int | None = None) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("artifacts"), list):
        artifacts = [row for row in payload["artifacts"] if isinstance(row, dict)]
        named = [row for row in artifacts if row.get("name") == STATE_ARTIFACT_NAME]
        if len(named) != 1:
            raise HandoffError(
                f"Expected exactly one {STATE_ARTIFACT_NAME!r} artifact in the selected run; found {len(named)}"
            )
        if artifact_id is None:
            rows = named
        else:
            rows = [row for row in named if row.get("id") == artifact_id]
        if len(rows) != 1:
            raise HandoffError(f"Expected the exact pointed F08 artifact ID {artifact_id!r}; found {len(rows)}")
        return rows[0]
    if isinstance(payload, dict):
        return payload
    raise HandoffError("F08 state artifact metadata response is malformed")


def _workflow_path_matches(value: Any) -> bool:
    """Accept GitHub API path encodings only for the pinned main workflow.

    The REST API commonly returns ".github/workflows/run_phase2d.yml@main"
    rather than the bare workflow path.  No other branch/ref is accepted.
    """
    return value in {
        WORKFLOW_PATH,
        f"{WORKFLOW_PATH}@main",
        f"{WORKFLOW_PATH}@refs/heads/main",
    }


def _validate_run_metadata(run: dict[str, Any], pointer: dict[str, Any]) -> None:
    expected = {
        "id": int(pointer["producer_run_id"]),
        "name": WORKFLOW_NAME,
        "head_branch": MAIN_BRANCH,
        "head_sha": pointer["producer_commit"],
        "status": "completed",
        "conclusion": "success",
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise HandoffError(f"F08 predecessor workflow run {key} mismatch")
    if not _workflow_path_matches(run.get("path")):
        raise HandoffError("F08 predecessor workflow run path mismatch")
    if run.get("repository", {}).get("full_name") != REPOSITORY:
        raise HandoffError("F08 predecessor run belongs to an unrelated repository")
    if run.get("event") not in {"workflow_dispatch", "repository_dispatch"}:
        raise HandoffError("F08 predecessor run event is not an approved dispatch type")
    if run.get("run_attempt") != 1:
        raise HandoffError("F08 predecessor workflow run is not first attempt")


def verify_continuation_artifact(
    artifact_dir: Path,
    *,
    pointer: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    artifact_metadata: Any | None = None,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest_path = artifact_dir / CONTINUATION_MANIFEST_NAME
    manifest = _load_object(manifest_path, "F08 continuation manifest")
    expected_fixed = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_status": ARTIFACT_STATUS,
        "repository": REPOSITORY,
        "workflow_name": WORKFLOW_NAME,
        "workflow_path": WORKFLOW_PATH,
        "experiment_id": expected_experiment_id,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    for key, value in expected_fixed.items():
        if manifest.get(key) != value:
            raise HandoffError(f"F08 continuation manifest {key} mismatch")
    if not _is_run_id(manifest.get("producer_run_id")) or not _is_commit(manifest.get("producer_commit")):
        raise HandoffError("F08 continuation producer run/commit is malformed")
    if not _is_run_id(manifest.get("previous_run_id")) or not _is_commit(manifest.get("previous_commit")):
        raise HandoffError("F08 continuation predecessor run/commit is malformed")
    if int(manifest["producer_run_id"]) <= int(manifest["previous_run_id"]):
        raise HandoffError("F08 continuation producer run ID does not advance beyond predecessor")
    if not _is_sha(manifest.get("previous_manifest_sha256")) or not _is_sha(manifest.get("bootstrap_asset_sha256")):
        raise HandoffError("F08 continuation predecessor/bootstrap manifest digest is malformed")
    bootstrap_marker = _validate_bootstrap_manifest(
        _load_object(artifact_dir / BOOTSTRAP_MANIFEST_NAME, "F08 bootstrap provenance manifest"),
        expected_experiment_id=expected_experiment_id,
    )
    if manifest.get("bootstrap_asset_sha256") != bootstrap_marker.get("asset_sha256"):
        raise HandoffError("F08 continuation bootstrap Release asset digest mismatch")
    inventory = manifest.get("state_files_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise HandoffError("F08 continuation state-file inventory is missing")
    actual = _runtime_inventory(artifact_dir, include_bootstrap_manifest=True)
    if actual != inventory:
        raise HandoffError("F08 continuation state-file hash inventory mismatch")
    state, hashes = _verify_state_identity(
        artifact_dir / "phase2d_state.json",
        expected_experiment_id=expected_experiment_id,
        expected_model_hashes=expected_model_hashes,
    )
    if hashes != manifest.get("locked_model_hashes"):
        raise HandoffError("F08 continuation locked model hashes mismatch")
    manifest_sha = _sha_file(manifest_path)
    if pointer is not None:
        _validate_pointer(pointer)
        pointer_bindings = {
            "producer_run_id": manifest["producer_run_id"],
            "producer_commit": manifest["producer_commit"],
            "previous_run_id": manifest["previous_run_id"],
            "previous_commit": manifest["previous_commit"],
            "previous_manifest_sha256": manifest["previous_manifest_sha256"],
            "bootstrap_asset_sha256": manifest["bootstrap_asset_sha256"],
            "experiment_id": expected_experiment_id,
            "manifest_sha256": manifest_sha,
        }
        for key, expected in pointer_bindings.items():
            if pointer.get(key) != expected:
                raise HandoffError(f"F08 continuation pointer {key} does not bind the downloaded artifact")
    if run_metadata is not None:
        if pointer is None:
            raise HandoffError("F08 run metadata cannot be checked without its authoritative pointer")
        _validate_run_metadata(run_metadata, pointer)
    if artifact_metadata is not None:
        if pointer is None:
            raise HandoffError("F08 artifact metadata cannot be checked without its authoritative pointer")
        meta = _normalize_artifact_payload(artifact_metadata, artifact_id=int(pointer["artifact_id"]))
        if meta.get("id") != int(pointer["artifact_id"]) or meta.get("name") != STATE_ARTIFACT_NAME:
            raise HandoffError("F08 continuation artifact ID/name does not match the pointer")
        if meta.get("expired") is not False:
            raise HandoffError("F08 continuation artifact is missing, expired or has unknown expiry")
        if isinstance(meta.get("size_in_bytes"), bool) or not isinstance(meta.get("size_in_bytes"), int) or meta["size_in_bytes"] <= 0:
            raise HandoffError("F08 continuation artifact is empty or has invalid size metadata")
        run = meta.get("workflow_run")
        if not isinstance(run, dict):
            raise HandoffError("F08 continuation artifact lacks workflow-run provenance")
        for key, expected in (("id", int(pointer["producer_run_id"])), ("head_branch", MAIN_BRANCH), ("head_sha", pointer["producer_commit"])):
            if run.get(key) != expected:
                raise HandoffError(f"F08 continuation artifact workflow_run {key} mismatch")
    return {"status": "PASS", "manifest": manifest, "manifest_sha256": manifest_sha, "state": state}


def prepare_continuation(
    pointer_json: Path,
    run_json: Path,
    artifact_metadata_json: Path,
    artifact_dir: Path,
    output_dir: Path,
    *,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    pointer = _load_object(pointer_json, "bot-state F08 pointer")
    run = _load_object(run_json, "F08 predecessor workflow run metadata")
    artifact_metadata = _load_object(artifact_metadata_json, "F08 predecessor artifact metadata")
    if output_dir.exists():
        raise HandoffError(f"Refusing to overwrite F08 continuation staging directory: {output_dir}")
    verify = verify_continuation_artifact(
        artifact_dir,
        pointer=pointer,
        run_metadata=run,
        artifact_metadata=artifact_metadata,
        expected_experiment_id=expected_experiment_id,
        expected_model_hashes=expected_model_hashes,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _copy_runtime_payload(artifact_dir, staging, include_continuation_manifest=True)
        verify_continuation_artifact(
            staging,
            expected_experiment_id=expected_experiment_id,
            expected_model_hashes=expected_model_hashes,
        )
        os.replace(staging, output_dir)
        return {"status": "PASS", "producer_run_id": pointer["producer_run_id"], "manifest_sha256": verify["manifest_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_runtime_installation(
    target_dir: Path,
    *,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    continuation_manifest = target_dir / CONTINUATION_MANIFEST_NAME
    if continuation_manifest.exists() or continuation_manifest.is_symlink():
        if continuation_manifest.is_symlink() or not continuation_manifest.is_file():
            raise HandoffError("F08 continuation manifest is present but unsafe")
        return verify_continuation_artifact(
            target_dir,
            expected_experiment_id=expected_experiment_id,
            expected_model_hashes=expected_model_hashes,
        )
    return {"status": "PASS", "bootstrap": verify_runtime_handoff(
        target_dir,
        expected_experiment_id=expected_experiment_id,
        expected_model_hashes=expected_model_hashes,
    )}


def install_runtime_state(
    prepared_dir: Path,
    target_dir: Path,
    *,
    expected_experiment_id: str = EXPERIMENT_ID,
    expected_model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Install only the F08 runtime contract, then verify installed hashes.

    No backup or rollback code is used. If any copy or post-install check fails,
    an exception is raised and the workflow's success-gated upload/pointer jobs
    must not run.
    """
    continuation_marker = prepared_dir / CONTINUATION_MANIFEST_NAME
    if continuation_marker.exists() or continuation_marker.is_symlink():
        if continuation_marker.is_symlink() or not continuation_marker.is_file():
            raise HandoffError("Prepared F08 continuation manifest is present but unsafe")
        prepared_is_continuation = True
    else:
        prepared_is_continuation = False
    if prepared_is_continuation:
        verify_continuation_artifact(prepared_dir, expected_experiment_id=expected_experiment_id,
                                    expected_model_hashes=expected_model_hashes)
    else:
        verify_runtime_handoff(prepared_dir, expected_experiment_id=expected_experiment_id,
                               expected_model_hashes=expected_model_hashes)
    if target_dir.is_symlink():
        raise HandoffError("F08 runtime installation target is a symlink")
    prepared_root = prepared_dir.resolve(strict=True)
    target_root = target_dir.resolve(strict=False)
    if prepared_root == target_root or prepared_root in target_root.parents or target_root in prepared_root.parents:
        raise HandoffError("F08 prepared runtime and installation target overlap")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Inspect every path this installer may replace before its first write.
    # Unrelated files in research_outputs are preserved and need not be scanned.
    for name in (*RUNTIME_FILES, BACKFILL_DIR, BOOTSTRAP_MANIFEST_NAME, CONTINUATION_MANIFEST_NAME):
        existing = target_dir / name
        if existing.is_symlink():
            raise HandoffError(f"Existing F08 runtime target is a symlink: {existing}")
        if existing.is_dir():
            _assert_no_links(existing, f"Existing F08 runtime target {name}")
    if not prepared_is_continuation:
        existing_f08 = [target_dir / name for name in RUNTIME_FILES]
        existing_f08 += [target_dir / BACKFILL_DIR, target_dir / CONTINUATION_MANIFEST_NAME]
        if any(path.exists() or path.is_symlink() for path in existing_f08):
            raise HandoffError("One-time F08 bootstrap target already contains runtime state")
    # Copy validated files into a separate temporary staging directory first.
    stage = Path(tempfile.mkdtemp(prefix=".f08-install-stage-", dir=target_dir.parent))
    try:
        _copy_runtime_payload(prepared_dir, stage, include_continuation_manifest=prepared_is_continuation)
        for name in RUNTIME_FILES:
            shutil.copyfile(stage / name, target_dir / name)
        destination_backfill = target_dir / BACKFILL_DIR
        if destination_backfill.is_symlink():
            raise HandoffError("Existing F08 historical-backfill destination is a symlink")
        if destination_backfill.exists():
            _assert_no_links(destination_backfill, "Existing F08 historical-backfill destination")
            shutil.rmtree(destination_backfill)
        _safe_copy_tree(stage / BACKFILL_DIR, destination_backfill, "F08 installed historical-backfill")
        shutil.copyfile(stage / BOOTSTRAP_MANIFEST_NAME, target_dir / BOOTSTRAP_MANIFEST_NAME)
        if prepared_is_continuation:
            # Provenance manifest is written last: its absence means installation
            # did not reach the final verification boundary.
            shutil.copyfile(stage / CONTINUATION_MANIFEST_NAME, target_dir / CONTINUATION_MANIFEST_NAME)
        result = verify_runtime_installation(
            target_dir,
            expected_experiment_id=expected_experiment_id,
            expected_model_hashes=expected_model_hashes,
        )
        return {"status": "PASS", "installed_to": str(target_dir), "verification": result}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _parent_lineage(runtime_dir: Path) -> dict[str, str]:
    """Return the authenticated predecessor identity for the next continuation."""
    _, current_hashes = _verify_state_identity(runtime_dir / "phase2d_state.json")
    parent_manifest = runtime_dir / CONTINUATION_MANIFEST_NAME
    if parent_manifest.exists() or parent_manifest.is_symlink():
        if parent_manifest.is_symlink() or not parent_manifest.is_file():
            raise HandoffError("F08 predecessor continuation manifest is present but unsafe")
        manifest = _load_object(parent_manifest, "F08 predecessor continuation manifest")
        required = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_status": ARTIFACT_STATUS,
            "repository": REPOSITORY,
            "workflow_name": WORKFLOW_NAME,
            "workflow_path": WORKFLOW_PATH,
            "experiment_id": EXPERIMENT_ID,
            "state_schema_version": STATE_SCHEMA_VERSION,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise HandoffError("F08 predecessor continuation manifest identity is invalid")
        if not _is_run_id(manifest.get("producer_run_id")) or not _is_commit(manifest.get("producer_commit")):
            raise HandoffError("F08 predecessor continuation producer identity is malformed")
        if not _is_run_id(manifest.get("previous_run_id")) or not _is_commit(manifest.get("previous_commit")):
            raise HandoffError("F08 predecessor continuation parent identity is malformed")
        if manifest.get("locked_model_hashes") != current_hashes:
            raise HandoffError("F08 predecessor continuation locked model hashes changed")
        inventory = manifest.get("state_files_sha256")
        if not isinstance(inventory, dict) or not inventory or any(not _is_sha(v) for v in inventory.values()):
            raise HandoffError("F08 predecessor continuation manifest lost its state-file inventory")
        bootstrap = _validate_bootstrap_manifest(
            _load_object(runtime_dir / BOOTSTRAP_MANIFEST_NAME, "F08 bootstrap provenance manifest")
        )
        if manifest.get("bootstrap_asset_sha256") != bootstrap.get("asset_sha256"):
            raise HandoffError("F08 predecessor continuation bootstrap asset identity changed")
        return {
            "previous_run_id": str(manifest["producer_run_id"]),
            "previous_commit": str(manifest["producer_commit"]),
            "previous_manifest_sha256": _sha_file(parent_manifest),
            "bootstrap_asset_sha256": str(bootstrap["asset_sha256"]),
        }

    bootstrap = _validate_bootstrap_manifest(
        _load_object(runtime_dir / BOOTSTRAP_MANIFEST_NAME, "F08 bootstrap provenance manifest")
    )
    if bootstrap.get("locked_model_hashes") != current_hashes:
        raise HandoffError("F08 bootstrap locked model hashes changed")
    return {
        "previous_run_id": str(bootstrap["disposition_run_id"]),
        "previous_commit": str(bootstrap["source_commit"]),
        "previous_manifest_sha256": _sha_file(runtime_dir / BOOTSTRAP_MANIFEST_NAME),
        "bootstrap_asset_sha256": str(bootstrap["asset_sha256"]),
    }


def seal_continuation(
    runtime_dir: Path,
    output_dir: Path,
    *,
    producer_run_id: str,
    producer_commit: str,
    expected_experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if not _is_run_id(producer_run_id) or not _is_commit(producer_commit):
        raise HandoffError("Current F08 producer run ID/commit is malformed")
    if output_dir.exists():
        raise HandoffError(f"Refusing to overwrite F08 continuation artifact directory: {output_dir}")
    state, locked_hashes = _verify_state_identity(runtime_dir / "phase2d_state.json", expected_experiment_id=expected_experiment_id)
    lineage = _parent_lineage(runtime_dir)
    previous_run_id = lineage["previous_run_id"]
    previous_commit = lineage["previous_commit"]
    if int(producer_run_id) <= int(previous_run_id):
        raise HandoffError("Current F08 run ID does not follow its predecessor")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _copy_runtime_payload(runtime_dir, staging, include_continuation_manifest=False)
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_status": ARTIFACT_STATUS,
            "repository": REPOSITORY,
            "workflow_name": WORKFLOW_NAME,
            "workflow_path": WORKFLOW_PATH,
            "producer_run_id": producer_run_id,
            "producer_commit": producer_commit,
            "previous_run_id": previous_run_id,
            "previous_commit": previous_commit,
            "previous_manifest_sha256": lineage["previous_manifest_sha256"],
            "bootstrap_asset_sha256": lineage["bootstrap_asset_sha256"],
            "experiment_id": expected_experiment_id,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "locked_model_hashes": locked_hashes,
            "state_files_sha256": _runtime_inventory(staging, include_bootstrap_manifest=True),
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _write_json(staging / CONTINUATION_MANIFEST_NAME, manifest)
        verify_continuation_artifact(staging, expected_experiment_id=expected_experiment_id)
        os.replace(staging, output_dir)
        return verify_continuation_artifact(output_dir, expected_experiment_id=expected_experiment_id)["manifest"]
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_current_producer_run(run: dict[str, Any], *, producer_run_id: str, producer_commit: str) -> None:
    expected = {
        "id": int(producer_run_id),
        "name": WORKFLOW_NAME,
        "head_branch": MAIN_BRANCH,
        "head_sha": producer_commit,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise HandoffError(f"Current F08 producer workflow run {key} mismatch")
    if not _workflow_path_matches(run.get("path")):
        raise HandoffError("Current F08 producer workflow run path mismatch")
    if run.get("repository", {}).get("full_name") != REPOSITORY:
        raise HandoffError("Current F08 producer run belongs to an unrelated repository")
    if run.get("event") not in {"workflow_dispatch", "repository_dispatch"}:
        raise HandoffError("Current F08 producer run event is not an approved dispatch type")
    if run.get("run_attempt") != 1:
        raise HandoffError("Current F08 producer run is not first attempt")
    if run.get("status") not in {"in_progress", "completed"}:
        raise HandoffError("Current F08 producer run has an invalid workflow status")
    if run.get("status") == "completed" and run.get("conclusion") != "success":
        raise HandoffError("Current F08 producer run did not complete successfully")


def make_state_pointer(artifact_metadata_payload: Any, artifact_dir: Path, *, producer_run_id: str,
                       producer_commit: str, run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = verify_continuation_artifact(artifact_dir)["manifest"]
    if manifest.get("producer_run_id") != producer_run_id or manifest.get("producer_commit") != producer_commit:
        raise HandoffError("Current F08 artifact manifest does not match this workflow run")
    if run_metadata is not None:
        _validate_current_producer_run(run_metadata, producer_run_id=producer_run_id, producer_commit=producer_commit)
    artifact = _normalize_artifact_payload(artifact_metadata_payload)
    if artifact.get("name") != STATE_ARTIFACT_NAME or artifact.get("expired") is not False:
        raise HandoffError("Current F08 continuation artifact is missing, expired or incorrectly named")
    if not _is_run_id(artifact.get("id")) or not isinstance(artifact.get("size_in_bytes"), int) or artifact["size_in_bytes"] <= 0:
        raise HandoffError("Current F08 continuation artifact metadata is malformed")
    run = artifact.get("workflow_run")
    if not isinstance(run, dict) or run.get("id") != int(producer_run_id) or run.get("head_branch") != MAIN_BRANCH or run.get("head_sha") != producer_commit:
        raise HandoffError("Current F08 continuation artifact is not bound to this main-branch run")
    return {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": REPOSITORY,
        "state_branch": STATE_BRANCH,
        "state_artifact_name": STATE_ARTIFACT_NAME,
        "workflow_path": WORKFLOW_PATH,
        "producer_run_id": producer_run_id,
        "producer_commit": producer_commit,
        "previous_run_id": manifest["previous_run_id"],
        "previous_commit": manifest["previous_commit"],
        "previous_manifest_sha256": manifest["previous_manifest_sha256"],
        "bootstrap_asset_sha256": manifest["bootstrap_asset_sha256"],
        "artifact_id": int(artifact["id"]),
        "manifest_sha256": _sha_file(artifact_dir / CONTINUATION_MANIFEST_NAME),
        "experiment_id": manifest["experiment_id"],
        "state_schema_version": manifest["state_schema_version"],
        "locked_model_hashes": manifest["locked_model_hashes"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def validate_pointer_transition(new_pointer: dict[str, Any], existing_pointer: dict[str, Any] | None, *, bootstrap: bool) -> None:
    _validate_pointer(new_pointer)
    if bootstrap:
        if existing_pointer is not None:
            raise HandoffError("F08 bootstrap cannot replace an existing bot-state pointer")
        return
    if existing_pointer is None:
        raise HandoffError("F08 continuation pointer is missing; refusing automatic bootstrap")
    _validate_pointer(existing_pointer)
    if new_pointer.get("previous_run_id") != existing_pointer.get("producer_run_id"):
        raise HandoffError("F08 continuation predecessor run does not match the current bot-state pointer")
    if new_pointer.get("previous_commit") != existing_pointer.get("producer_commit"):
        raise HandoffError("F08 continuation predecessor commit does not match the current bot-state pointer")
    if new_pointer.get("previous_manifest_sha256") != existing_pointer.get("manifest_sha256"):
        raise HandoffError("F08 continuation predecessor manifest digest does not match the current bot-state pointer")
    if new_pointer.get("bootstrap_asset_sha256") != existing_pointer.get("bootstrap_asset_sha256"):
        raise HandoffError("F08 continuation bootstrap asset digest changed across the pointer chain")
    if int(new_pointer["producer_run_id"]) <= int(existing_pointer["producer_run_id"]):
        raise HandoffError("F08 state pointer producer run ID must advance monotonically")
    if new_pointer.get("locked_model_hashes") != existing_pointer.get("locked_model_hashes"):
        raise HandoffError("F08 state pointer cannot change locked model hashes during continuation")


def decode_github_contents_response(response_json: Path, output_json: Path) -> str:
    response = _load_object(response_json, "GitHub bot-state contents response")
    if response.get("path") != STATE_POINTER_PATH or response.get("encoding") != "base64":
        raise HandoffError("GitHub bot-state pointer response path/encoding mismatch")
    content = response.get("content")
    if not isinstance(content, str):
        raise HandoffError("GitHub bot-state pointer response has no base64 content")
    try:
        compact_content = "".join(content.split())
        decoded = base64.b64decode(compact_content, validate=True)
        pointer = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise HandoffError(f"Unable to decode bot-state pointer: {exc}") from exc
    if not isinstance(pointer, dict):
        raise HandoffError("Decoded bot-state pointer is not a JSON object")
    _validate_pointer(pointer)
    file_sha = response.get("sha")
    if not isinstance(file_sha, str) or len(file_sha) != 40 or any(ch not in "0123456789abcdef" for ch in file_sha):
        raise HandoffError("GitHub bot-state contents response lacks a valid Git blob SHA")
    _write_json(output_json, pointer)
    return file_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap-release", help="Verify and stage the one-time immutable Release asset")
    bootstrap.add_argument("--release-json", required=True, type=Path)
    bootstrap.add_argument("--archive", required=True, type=Path)
    bootstrap.add_argument("--tag-commit", required=True)
    bootstrap.add_argument("--workflow-commit", required=True)
    bootstrap.add_argument("--output-dir", required=True, type=Path)
    bootstrap.add_argument("--candidate-sha", required=True)
    bootstrap.add_argument("--production-sha", required=True)

    continuation = sub.add_parser("prepare-continuation", help="Verify the exact predecessor pointer, run and artifact")
    continuation.add_argument("--pointer-json", required=True, type=Path)
    continuation.add_argument("--run-json", required=True, type=Path)
    continuation.add_argument("--artifact-metadata-json", required=True, type=Path)
    continuation.add_argument("--artifact-dir", required=True, type=Path)
    continuation.add_argument("--output-dir", required=True, type=Path)
    continuation.add_argument("--candidate-sha", required=True)
    continuation.add_argument("--production-sha", required=True)

    install = sub.add_parser("install-runtime", help="Install verified staged runtime and verify installed hashes")
    install.add_argument("--prepared-dir", required=True, type=Path)
    install.add_argument("--target-dir", required=True, type=Path)
    install.add_argument("--candidate-sha", required=True)
    install.add_argument("--production-sha", required=True)

    seal = sub.add_parser("seal-continuation", help="Seal the successful F08 runtime as a continuation artifact")
    seal.add_argument("--runtime-dir", required=True, type=Path)
    seal.add_argument("--output-dir", required=True, type=Path)
    seal.add_argument("--producer-run-id", required=True)
    seal.add_argument("--producer-commit", required=True)

    select = sub.add_parser("select-run", help="Select only the newest successful F08 disposition dispatch run")
    select.add_argument("--runs-json", required=True, type=Path)
    select.add_argument("--output-json", required=True, type=Path)

    pointer = sub.add_parser("make-pointer", help="Build the bot-state pointer from the just-uploaded artifact")
    pointer.add_argument("--artifact-metadata-json", required=True, type=Path)
    pointer.add_argument("--artifact-dir", required=True, type=Path)
    pointer.add_argument("--producer-run-id", required=True)
    pointer.add_argument("--producer-commit", required=True)
    pointer.add_argument("--run-json", required=True, type=Path)
    pointer.add_argument("--output-json", required=True, type=Path)

    decode = sub.add_parser("decode-pointer", help="Decode a GitHub Contents API pointer response")
    decode.add_argument("--response-json", required=True, type=Path)
    decode.add_argument("--output-json", required=True, type=Path)

    transition = sub.add_parser("check-pointer-transition", help="Fail closed unless the bot-state pointer advances from the exact predecessor")
    transition.add_argument("--new-pointer", required=True, type=Path)
    transition.add_argument("--existing-response", type=Path)
    transition.add_argument("--bootstrap", choices=("true", "false"), required=True)

    args = parser.parse_args(argv)
    expected_hashes = {"candidate_sha256": args.candidate_sha, "production_sha256": args.production_sha} if hasattr(args, "candidate_sha") else None
    try:
        if args.command == "bootstrap-release":
            result = prepare_bootstrap_release(
                args.release_json, args.archive, args.tag_commit, args.output_dir,
                workflow_commit=args.workflow_commit,
                expected_model_hashes=expected_hashes,
            )
            print(json.dumps({"status": "PASS", "bootstrap": result}, sort_keys=True))
        elif args.command == "prepare-continuation":
            result = prepare_continuation(args.pointer_json, args.run_json, args.artifact_metadata_json,
                                          args.artifact_dir, args.output_dir,
                                          expected_model_hashes=expected_hashes)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "install-runtime":
            result = install_runtime_state(args.prepared_dir, args.target_dir, expected_model_hashes=expected_hashes)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "seal-continuation":
            result = seal_continuation(args.runtime_dir, args.output_dir,
                                       producer_run_id=args.producer_run_id,
                                       producer_commit=args.producer_commit)
            print(json.dumps({"status": "PASS", "manifest": result}, sort_keys=True))
        elif args.command == "select-run":
            print(select_disposition_run_file(args.runs_json, args.output_json))
        elif args.command == "make-pointer":
            metadata = _load_object(args.artifact_metadata_json, "current F08 artifact metadata")
            run_metadata = _load_object(args.run_json, "current F08 workflow run metadata")
            result = make_state_pointer(metadata, args.artifact_dir,
                                        producer_run_id=args.producer_run_id,
                                        producer_commit=args.producer_commit,
                                        run_metadata=run_metadata)
            _write_json(args.output_json, result)
            print(json.dumps({"status": "PASS", "pointer": result}, sort_keys=True))
        elif args.command == "decode-pointer":
            sha = decode_github_contents_response(args.response_json, args.output_json)
            print(sha)
        elif args.command == "check-pointer-transition":
            new_pointer = _load_object(args.new_pointer, "new F08 state pointer")
            existing_pointer = None
            if args.existing_response is not None:
                existing_response = _load_object(args.existing_response, "existing GitHub pointer response")
                temp_pointer = args.new_pointer.parent / ".existing_f08_pointer_decoded.json"
                decode_github_contents_response(args.existing_response, temp_pointer)
                existing_pointer = _load_object(temp_pointer, "existing F08 state pointer")
                temp_pointer.unlink(missing_ok=True)
            validate_pointer_transition(new_pointer, existing_pointer, bootstrap=args.bootstrap == "true")
            print("PASS")
        return 0
    except Exception as exc:
        print(f"F08 runtime handoff: BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
