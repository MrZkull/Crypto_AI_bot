from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import phase2d_f08_v2_handoff as handoff
import phase2d_f08_v2_restore as v2

CANDIDATE_SHA = "1" * 64
PRODUCTION_SHA = "2" * 64
SOURCE_COMMIT = "a" * 40
RUN_COMMIT = "b" * 40


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _package(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "package"
    post = root / "disposition"
    backfill = root / "historical_backfill"
    post.mkdir(parents=True)
    (backfill / "evidence").mkdir(parents=True)

    state = {
        "schema_version": 7,
        "locked_models": {
            "experiment_id": handoff.EXPERIMENT_ID,
            "candidate_sha256": CANDIDATE_SHA,
            "production_sha256": PRODUCTION_SHA,
        },
    }
    _json(post / "phase2d_state.json", state)
    _json(post / "phase2d_results.json", {"overall_status": "CONTINUE"})
    _json(post / "phase2d_latest_snapshot.json", {"snapshot": 1})
    (post / "phase2d_observations.jsonl").write_text('{"record_type":"event"}\n', encoding="utf-8")
    (post / "experiment_ledger.jsonl").write_text('{"event":"F08_VALIDATOR_AMENDMENT"}\n', encoding="utf-8")
    _json(backfill / "manifest.json", {"manifest_status": "READY_FOR_OWNER_REVIEW"})
    (backfill / "observations.jsonl").write_text('{"id":"historical"}\n', encoding="utf-8")
    (backfill / "evidence" / "raw_windows.jsonl").write_text('{"window":1}\n', encoding="utf-8")

    manifest = {
        "artifact_schema_version": v2.ARTIFACT_SCHEMA_VERSION,
        "artifact_type": v2.ARTIFACT_TYPE,
        "artifact_status": v2.ARTIFACT_STATUS,
        "experiment_id": handoff.EXPERIMENT_ID,
        "source_commit": SOURCE_COMMIT,
        "validator_amendment_run_id": "910001",
        "disposition_run_id": "910002",
        "restore_policy": {"normal_phase2d_pipeline_handoff": False},
    }
    _json(root / "manifest_v2.json", manifest)

    archive = tmp_path / handoff.RELEASE_ASSET_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    release = {
        "tag_name": handoff.RELEASE_TAG,
        "immutable": True,
        "draft": False,
        "prerelease": False,
        "assets": [{
            "name": handoff.RELEASE_ASSET_NAME,
            "digest": f"sha256:{digest}",
            "size": archive.stat().st_size,
            "state": "uploaded",
        }],
    }
    release_json = tmp_path / "release.json"
    _json(release_json, release)
    return archive, {"root": root, "release_json": release_json, "release": release, "state": state}


def _bootstrap(tmp_path: Path, monkeypatch) -> Path:
    archive, package = _package(tmp_path)
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    output = tmp_path / "bootstrap"
    handoff.prepare_bootstrap_release(
        package["release_json"], archive, SOURCE_COMMIT, output,
        workflow_commit=SOURCE_COMMIT,
        expected_model_hashes={"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    )
    return output


def _workflow_run(run_id: int, commit: str = RUN_COMMIT) -> dict:
    return {
        "id": run_id,
        "name": handoff.WORKFLOW_NAME,
        "path": handoff.WORKFLOW_PATH,
        "head_branch": "main",
        "head_sha": commit,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "repository": {"full_name": handoff.REPOSITORY},
    }


def _disposition_run(
    run_id: int,
    run_number: int,
    *,
    status: str = "completed",
    conclusion: str | None = "success",
    run_attempt: int = 1,
    actor_id: int = 12345,
    repository: str | None = None,
    path: str | None = None,
) -> dict:
    return {
        "id": run_id,
        "name": handoff.DISPOSITION_WORKFLOW_NAME,
        "path": path or handoff.DISPOSITION_WORKFLOW_PATH,
        "head_branch": "main",
        "head_sha": RUN_COMMIT,
        "event": "workflow_dispatch",
        "status": status,
        "conclusion": conclusion,
        "run_attempt": run_attempt,
        "run_number": run_number,
        "repository": {"full_name": repository or handoff.REPOSITORY},
        "actor": {"id": actor_id},
    }


def _artifact_metadata(run_id: int, commit: str, artifact_id: int = 77) -> dict:
    return {"artifacts": [{
        "id": artifact_id,
        "name": handoff.STATE_ARTIFACT_NAME,
        "expired": False,
        "size_in_bytes": 1234,
        "workflow_run": {"id": run_id, "head_branch": "main", "head_sha": commit},
    }]}


def test_select_run_accepts_only_newest_successful_disposition_dispatch(tmp_path):
    older = _disposition_run(910001, 101)
    newest = _disposition_run(910002, 102)
    payload = {"total_count": 2, "workflow_runs": [newest, older]}
    result = handoff.select_latest_disposition_run(payload)

    assert result["selection_status"] == "PASS"
    assert result["selection_policy"] == "NEWEST_MATCHING_DISPATCH_ONLY_NO_OLDER_RUN_FALLBACK"
    assert result["selected_run_id"] == "910002"
    assert result["selected_run"]["run_number"] == 102


def test_select_run_fails_closed_when_newest_run_failed_instead_of_falling_back():
    older_success = _disposition_run(910001, 101)
    newest_failure = _disposition_run(910002, 102, status="completed", conclusion="failure")
    with pytest.raises(handoff.HandoffError, match="refusing fallback to an older run"):
        handoff.select_latest_disposition_run({"workflow_runs": [newest_failure, older_success]})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"run_attempt": 2}, "rerun"),
        ({"status": "in_progress", "conclusion": None}, "not completed successfully"),
        ({"actor_id": int(handoff.EXCLUDED_DISPOSITION_ACTOR_ID)}, "excluded owner identity"),
        ({"repository": "attacker/fork"}, "repository mismatch"),
        ({"path": ".github/workflows/other.yml"}, "path mismatch"),
    ],
)
def test_select_run_rejects_unsafe_newest_metadata(overrides, message):
    kwargs = dict(overrides)
    actor_id = kwargs.pop("actor_id", 12345)
    newest = _disposition_run(910002, 102, actor_id=actor_id, **kwargs)
    with pytest.raises(handoff.HandoffError, match=message):
        handoff.select_latest_disposition_run({"workflow_runs": [newest]})


def test_select_run_cli_writes_audit_json_and_prints_only_run_id(tmp_path, capsys):
    runs_path = tmp_path / "runs.json"
    output_path = tmp_path / "selected.json"
    _json(runs_path, {"workflow_runs": [_disposition_run(910002, 102)]})

    assert handoff.main([
        "select-run", "--runs-json", str(runs_path), "--output-json", str(output_path)
    ]) == 0
    assert capsys.readouterr().out.strip() == "910002"
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert stored["selected_run_id"] == "910002"
    assert stored["selection_status"] == "PASS"


def test_select_run_rejects_empty_or_ambiguous_run_lists():
    with pytest.raises(handoff.HandoffError, match="No F08 controlled-disposition"):
        handoff.select_latest_disposition_run({"workflow_runs": []})
    first = _disposition_run(910001, 102)
    second = _disposition_run(910002, 102)
    with pytest.raises(handoff.HandoffError, match="ambiguous"):
        handoff.select_latest_disposition_run({"workflow_runs": [first, second]})


def test_bootstrap_requires_immutable_fixed_release_and_published_digest(tmp_path, monkeypatch):
    archive, package = _package(tmp_path)
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    output = tmp_path / "prepared"
    result = handoff.prepare_bootstrap_release(
        package["release_json"], archive, SOURCE_COMMIT, output,
        workflow_commit=SOURCE_COMMIT,
        expected_model_hashes={"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    )
    assert result["bootstrap_type"] == "IMMUTABLE_GITHUB_RELEASE"
    assert result["release_tag"] == handoff.RELEASE_TAG
    assert result["disposition_run_id"] == "910002"
    assert handoff.verify_runtime_handoff(output)["handoff_status"] == "VERIFIED_SCHEMA7_RUNTIME_HANDOFF"


def test_bootstrap_rejects_mutable_release(tmp_path, monkeypatch):
    archive, package = _package(tmp_path)
    package["release"]["immutable"] = False
    _json(package["release_json"], package["release"])
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    with pytest.raises(handoff.HandoffError, match="not reported immutable"):
        handoff.prepare_bootstrap_release(
            package["release_json"], archive, SOURCE_COMMIT, tmp_path / "out",
            workflow_commit=SOURCE_COMMIT,
        )


def test_bootstrap_rejects_download_digest_mismatch(tmp_path, monkeypatch):
    archive, package = _package(tmp_path)
    package["release"]["assets"][0]["digest"] = "sha256:" + "0" * 64
    _json(package["release_json"], package["release"])
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    with pytest.raises(handoff.HandoffError, match="SHA-256"):
        handoff.prepare_bootstrap_release(
            package["release_json"], archive, SOURCE_COMMIT, tmp_path / "out",
            workflow_commit=SOURCE_COMMIT,
        )


def test_bootstrap_rejects_wrong_tag_commit_and_model_identity(tmp_path, monkeypatch):
    archive, package = _package(tmp_path)
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    with pytest.raises(handoff.HandoffError, match="approved source commit"):
        handoff.prepare_bootstrap_release(
            package["release_json"], archive, "c" * 40, tmp_path / "bad_commit",
            workflow_commit="c" * 40,
        )
    with pytest.raises(handoff.HandoffError, match="locked model hashes"):
        handoff.prepare_bootstrap_release(
            package["release_json"], archive, SOURCE_COMMIT, tmp_path / "bad_models",
            workflow_commit=SOURCE_COMMIT,
            expected_model_hashes={"candidate_sha256": "3" * 64, "production_sha256": PRODUCTION_SHA},
        )


def test_bootstrap_rejects_workflow_checkout_from_different_commit(tmp_path, monkeypatch):
    archive, package = _package(tmp_path)
    monkeypatch.setattr(v2, "verify_artifact", lambda *a, **k: {"status": "PASS"})
    with pytest.raises(handoff.HandoffError, match="workflow commit differs"):
        handoff.prepare_bootstrap_release(
            package["release_json"], archive, SOURCE_COMMIT, tmp_path / "wrong_checkout",
            workflow_commit=RUN_COMMIT,
            expected_model_hashes={"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
        )


def test_installation_verifies_installed_files_and_preserves_unrelated_files(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    target.mkdir()
    (target / "unrelated.txt").write_text("keep", encoding="utf-8")
    result = handoff.install_runtime_state(
        prepared, target,
        expected_model_hashes={"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    )
    assert result["status"] == "PASS"
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert handoff.verify_runtime_handoff(target)["asset_sha256"] == handoff.verify_runtime_handoff(prepared)["asset_sha256"]


def test_interrupted_installation_leaves_no_bootstrap_completion_marker(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    original_copyfile = handoff.shutil.copyfile

    def interrupted(src, dst, *args, **kwargs):
        if Path(dst) == target / "phase2d_results.json":
            raise OSError("simulated runner interruption")
        return original_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(handoff.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="interruption"):
        handoff.install_runtime_state(prepared, target)
    assert not (target / handoff.BOOTSTRAP_MANIFEST_NAME).exists()


def test_installation_rejects_existing_symlink_before_writing_runtime_files(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    target.mkdir()
    outside = tmp_path / "outside_state.json"
    outside.write_text("external data must remain untouched", encoding="utf-8")
    link = target / "phase2d_state.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(handoff.HandoffError, match="Existing F08 runtime target is a symlink"):
        handoff.install_runtime_state(prepared, target)

    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "external data must remain untouched"
    assert not (target / "phase2d_results.json").exists()


def test_continuation_manifest_binds_previous_run_commit_experiment_schema_models_and_files(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    # Install first, as the first successful F08 scan would do.
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(
        runtime, target,
        expected_model_hashes={"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    )
    # Simulate legitimate prospective progress after installation: the state
    # bytes change, while the locked experiment/model identities stay fixed.
    state_path = target / "phase2d_state.json"
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    updated_state["prospective_progress_marker"] = "after-first-scan"
    _json(state_path, updated_state)

    output = tmp_path / "artifact"
    manifest = handoff.seal_continuation(target, output, producer_run_id="910010", producer_commit=RUN_COMMIT)
    assert manifest["previous_run_id"] == "910002"
    assert manifest["previous_commit"] == SOURCE_COMMIT
    assert manifest["previous_manifest_sha256"] == hashlib.sha256(
        (target / handoff.BOOTSTRAP_MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert manifest["bootstrap_asset_sha256"] == handoff.verify_runtime_handoff(runtime)["asset_sha256"]
    assert manifest["experiment_id"] == handoff.EXPERIMENT_ID
    assert manifest["state_schema_version"] == 7
    assert manifest["locked_model_hashes"] == {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA}
    assert manifest["state_files_sha256"]


def test_continuation_rejects_unrelated_run_and_expired_or_missing_artifact(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(runtime, target)
    artifact_dir = tmp_path / "artifact"
    handoff.seal_continuation(target, artifact_dir, producer_run_id="910010", producer_commit=RUN_COMMIT)
    run = _workflow_run(910010)
    pointer = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": handoff.REPOSITORY,
        "state_branch": handoff.STATE_BRANCH,
        "state_artifact_name": handoff.STATE_ARTIFACT_NAME,
        "workflow_path": handoff.WORKFLOW_PATH,
        "producer_run_id": "910010",
        "producer_commit": RUN_COMMIT,
        "previous_run_id": "910002",
        "previous_commit": SOURCE_COMMIT,
        "artifact_id": 77,
        "manifest_sha256": hashlib.sha256((artifact_dir / handoff.CONTINUATION_MANIFEST_NAME).read_bytes()).hexdigest(),
        "previous_manifest_sha256": json.loads((artifact_dir / handoff.CONTINUATION_MANIFEST_NAME).read_text(encoding="utf-8"))["previous_manifest_sha256"],
        "bootstrap_asset_sha256": json.loads((artifact_dir / handoff.CONTINUATION_MANIFEST_NAME).read_text(encoding="utf-8"))["bootstrap_asset_sha256"],
        "experiment_id": handoff.EXPERIMENT_ID,
        "state_schema_version": 7,
        "locked_model_hashes": {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    }
    _json(tmp_path / "pointer.json", pointer)
    _json(tmp_path / "run.json", run)
    _json(tmp_path / "artifact.json", _artifact_metadata(910010, RUN_COMMIT))
    result = handoff.prepare_continuation(tmp_path / "pointer.json", tmp_path / "run.json", tmp_path / "artifact.json",
                                         artifact_dir, tmp_path / "prepared")
    assert result["status"] == "PASS"

    wrong_run = dict(run, id=910011)
    _json(tmp_path / "wrong_run.json", wrong_run)
    with pytest.raises(handoff.HandoffError, match="run id mismatch"):
        handoff.prepare_continuation(tmp_path / "pointer.json", tmp_path / "wrong_run.json", tmp_path / "artifact.json",
                                     artifact_dir, tmp_path / "bad_run")

    expired = _artifact_metadata(910010, RUN_COMMIT)
    expired["artifacts"][0]["expired"] = True
    _json(tmp_path / "expired.json", expired)
    with pytest.raises(handoff.HandoffError, match="expired"):
        handoff.prepare_continuation(tmp_path / "pointer.json", tmp_path / "run.json", tmp_path / "expired.json",
                                     artifact_dir, tmp_path / "bad_expired")

    missing = {"artifacts": []}
    _json(tmp_path / "missing.json", missing)
    with pytest.raises(handoff.HandoffError, match="exactly one"):
        handoff.prepare_continuation(tmp_path / "pointer.json", tmp_path / "run.json", tmp_path / "missing.json",
                                     artifact_dir, tmp_path / "bad_missing")


def test_continuation_rejects_modified_file_and_schema_mismatch(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(runtime, target)
    artifact_dir = tmp_path / "artifact"
    handoff.seal_continuation(target, artifact_dir, producer_run_id="910010", producer_commit=RUN_COMMIT)
    (artifact_dir / "phase2d_results.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(handoff.HandoffError, match="hash inventory"):
        handoff.verify_continuation_artifact(artifact_dir)

    # Rebuild and corrupt the schema-7 payload, then recalculate its inventory.
    # This proves the schema validator rejects schema 6 independently of hashes.
    artifact_dir2 = tmp_path / "artifact2"
    handoff.seal_continuation(target, artifact_dir2, producer_run_id="910011", producer_commit="c" * 40)
    state_path = artifact_dir2 / "phase2d_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = 6
    _json(state_path, state)
    manifest_path = artifact_dir2 / handoff.CONTINUATION_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_files_sha256"] = handoff._runtime_inventory(
        artifact_dir2, include_bootstrap_manifest=True
    )
    _json(manifest_path, manifest)
    with pytest.raises(handoff.HandoffError, match="schema mismatch: expected 7, got 6"):
        handoff.verify_continuation_artifact(artifact_dir2)


def test_github_workflow_run_api_path_ref_suffix_is_accepted_only_for_main():
    run = _workflow_run(910010)
    pointer = {"producer_run_id": "910010", "producer_commit": RUN_COMMIT}
    run["path"] = f"{handoff.WORKFLOW_PATH}@main"
    handoff._validate_run_metadata(run, pointer)
    run["path"] = f"{handoff.WORKFLOW_PATH}@refs/heads/main"
    handoff._validate_run_metadata(run, pointer)
    run["path"] = f"{handoff.WORKFLOW_PATH}@refs/heads/release"
    with pytest.raises(handoff.HandoffError, match="path mismatch"):
        handoff._validate_run_metadata(run, pointer)


def test_current_producer_accepts_github_api_main_path_suffix():
    run = _workflow_run(910010)
    run["path"] = f"{handoff.WORKFLOW_PATH}@main"
    handoff._validate_current_producer_run(run, producer_run_id="910010", producer_commit=RUN_COMMIT)


def test_pointer_transition_rejects_wrong_predecessor_manifest_or_bootstrap_digest():
    base = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": handoff.REPOSITORY,
        "state_branch": handoff.STATE_BRANCH,
        "state_artifact_name": handoff.STATE_ARTIFACT_NAME,
        "workflow_path": handoff.WORKFLOW_PATH,
        "experiment_id": handoff.EXPERIMENT_ID,
        "state_schema_version": 7,
        "producer_run_id": "910020",
        "producer_commit": "d" * 40,
        "previous_run_id": "910010",
        "previous_commit": "c" * 40,
        "artifact_id": 100,
        "manifest_sha256": "e" * 64,
        "previous_manifest_sha256": "f" * 64,
        "bootstrap_asset_sha256": "a" * 64,
        "locked_model_hashes": {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    }
    existing = dict(base, producer_run_id="910010", producer_commit="c" * 40,
                    previous_run_id="910002", previous_commit=SOURCE_COMMIT,
                    manifest_sha256="b" * 64)
    with pytest.raises(handoff.HandoffError, match="predecessor manifest digest"):
        handoff.validate_pointer_transition(base, existing, bootstrap=False)
    wrong_bootstrap = dict(base, previous_manifest_sha256="b" * 64, bootstrap_asset_sha256="9" * 64)
    with pytest.raises(handoff.HandoffError, match="bootstrap asset digest"):
        handoff.validate_pointer_transition(wrong_bootstrap, existing, bootstrap=False)


def test_predecessor_artifact_selection_rejects_duplicate_fixed_names():
    one = _artifact_metadata(910010, RUN_COMMIT)["artifacts"][0]
    duplicate = dict(one, id=78)
    with pytest.raises(handoff.HandoffError, match="exactly one 'phase2d-state-f08'"):
        handoff._normalize_artifact_payload({"artifacts": [one, duplicate]}, artifact_id=77)


def test_pointer_transition_requires_exact_predecessor_and_no_implicit_bootstrap():
    base = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": handoff.REPOSITORY,
        "state_branch": handoff.STATE_BRANCH,
        "state_artifact_name": handoff.STATE_ARTIFACT_NAME,
        "workflow_path": handoff.WORKFLOW_PATH,
        "experiment_id": handoff.EXPERIMENT_ID,
        "state_schema_version": 7,
        "producer_run_id": "910020",
        "producer_commit": "d" * 40,
        "previous_run_id": "910010",
        "previous_commit": "c" * 40,
        "artifact_id": 100,
        "manifest_sha256": "e" * 64,
        "previous_manifest_sha256": "e" * 64,
        "bootstrap_asset_sha256": "a" * 64,
        "locked_model_hashes": {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    }
    existing = dict(base, producer_run_id="910010", producer_commit="c" * 40,
                    previous_run_id="910002", previous_commit=SOURCE_COMMIT)
    handoff.validate_pointer_transition(base, existing, bootstrap=False)
    with pytest.raises(handoff.HandoffError, match="pointer is missing"):
        handoff.validate_pointer_transition(base, None, bootstrap=False)
    with pytest.raises(handoff.HandoffError, match="cannot replace"):
        handoff.validate_pointer_transition(base, existing, bootstrap=True)
    unrelated = dict(existing, producer_run_id="910009")
    with pytest.raises(handoff.HandoffError, match="predecessor run"):
        handoff.validate_pointer_transition(base, unrelated, bootstrap=False)


def test_contents_pointer_decoder_validates_repository_path_and_content(tmp_path):
    pointer = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": handoff.REPOSITORY,
        "state_branch": handoff.STATE_BRANCH,
        "state_artifact_name": handoff.STATE_ARTIFACT_NAME,
        "workflow_path": handoff.WORKFLOW_PATH,
        "producer_run_id": "910020",
        "producer_commit": "d" * 40,
        "previous_run_id": "910010",
        "previous_commit": "c" * 40,
        "artifact_id": 100,
        "manifest_sha256": "e" * 64,
        "previous_manifest_sha256": "e" * 64,
        "bootstrap_asset_sha256": "a" * 64,
        "experiment_id": handoff.EXPERIMENT_ID,
        "state_schema_version": 7,
        "locked_model_hashes": {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    }
    import base64
    response = {
        "path": handoff.STATE_POINTER_PATH,
        "encoding": "base64",
        "content": base64.b64encode(json.dumps(pointer).encode()).decode(),
        "sha": "f" * 40,
    }
    _json(tmp_path / "response.json", response)
    file_sha = handoff.decode_github_contents_response(tmp_path / "response.json", tmp_path / "pointer.json")
    assert file_sha == "f" * 40
    assert json.loads((tmp_path / "pointer.json").read_text()) == pointer

    response["sha"] = "not-a-git-blob-sha"
    _json(tmp_path / "bad_sha_response.json", response)
    bad_output = tmp_path / "bad_pointer.json"
    with pytest.raises(handoff.HandoffError, match="valid Git blob SHA"):
        handoff.decode_github_contents_response(tmp_path / "bad_sha_response.json", bad_output)
    assert not bad_output.exists()



def test_bootstrap_manifest_rejects_changed_release_identity(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    marker_path = prepared / handoff.BOOTSTRAP_MANIFEST_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["release_immutable"] = False
    _json(marker_path, marker)
    with pytest.raises(handoff.HandoffError, match="release_immutable mismatch"):
        handoff.verify_runtime_handoff(prepared)


def test_bootstrap_refuses_to_overwrite_any_existing_f08_runtime_state(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    target.mkdir()
    (target / "phase2d_state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(handoff.HandoffError, match="already contains runtime state"):
        handoff.install_runtime_state(prepared, target)


def test_installed_runtime_hashes_are_rechecked_after_copy(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(prepared, target)
    (target / "phase2d_results.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(handoff.HandoffError, match="hash inventory"):
        handoff.verify_runtime_installation(target)


def test_interrupted_continuation_install_fails_post_install_verification(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(runtime, target)
    state_path = target / "phase2d_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["prospective_progress_marker"] = "first-scan"
    _json(state_path, state)
    continuation = tmp_path / "continuation"
    handoff.seal_continuation(target, continuation, producer_run_id="910010", producer_commit=RUN_COMMIT)

    original_copyfile = handoff.shutil.copyfile
    def interrupted(src, dst, *args, **kwargs):
        if Path(dst) == target / "phase2d_results.json":
            raise OSError("simulated continuation installation interruption")
        return original_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(handoff.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="interruption"):
        handoff.install_runtime_state(continuation, target)
    assert not (target / handoff.CONTINUATION_MANIFEST_NAME).exists()
    with pytest.raises(handoff.HandoffError):
        handoff.verify_runtime_installation(target)


def test_pointer_transition_rejects_rollback_and_model_identity_change():
    base = {
        "schema_version": 1,
        "pointer_type": "PHASE2D_F08_STATE_POINTER",
        "repository": handoff.REPOSITORY,
        "state_branch": handoff.STATE_BRANCH,
        "state_artifact_name": handoff.STATE_ARTIFACT_NAME,
        "workflow_path": handoff.WORKFLOW_PATH,
        "experiment_id": handoff.EXPERIMENT_ID,
        "state_schema_version": 7,
        "producer_run_id": "910020",
        "producer_commit": "d" * 40,
        "previous_run_id": "910010",
        "previous_commit": "c" * 40,
        "artifact_id": 100,
        "manifest_sha256": "e" * 64,
        "previous_manifest_sha256": "e" * 64,
        "bootstrap_asset_sha256": "a" * 64,
        "locked_model_hashes": {"candidate_sha256": CANDIDATE_SHA, "production_sha256": PRODUCTION_SHA},
    }
    existing = dict(base, producer_run_id="910010", producer_commit="c" * 40,
                    previous_run_id="910002", previous_commit=SOURCE_COMMIT)
    rollback = dict(base, producer_run_id="910009")
    with pytest.raises(handoff.HandoffError, match="monotonically"):
        handoff.validate_pointer_transition(rollback, existing, bootstrap=False)
    changed_models = dict(base, locked_model_hashes={"candidate_sha256": "3" * 64, "production_sha256": PRODUCTION_SHA})
    with pytest.raises(handoff.HandoffError, match="locked model hashes"):
        handoff.validate_pointer_transition(changed_models, existing, bootstrap=False)


def test_runtime_installation_does_not_fall_back_when_continuation_marker_is_unsafe(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    target = tmp_path / "research_outputs"
    handoff.install_runtime_state(runtime, target)
    marker = target / handoff.CONTINUATION_MANIFEST_NAME
    marker.mkdir()
    with pytest.raises(handoff.HandoffError, match="present but unsafe"):
        handoff.verify_runtime_installation(target)


def test_predecessor_lineage_does_not_fall_back_when_continuation_marker_is_directory(tmp_path, monkeypatch):
    runtime = _bootstrap(tmp_path, monkeypatch)
    marker = runtime / handoff.CONTINUATION_MANIFEST_NAME
    marker.mkdir()
    with pytest.raises(handoff.HandoffError, match="present but unsafe"):
        handoff._parent_lineage(runtime)


def test_installation_rejects_unsafe_prepared_continuation_marker(tmp_path, monkeypatch):
    prepared = _bootstrap(tmp_path, monkeypatch)
    marker = prepared / handoff.CONTINUATION_MANIFEST_NAME
    marker.mkdir()
    target = tmp_path / "research_outputs"
    with pytest.raises(handoff.HandoffError, match="Prepared F08 continuation manifest is present but unsafe"):
        handoff.install_runtime_state(prepared, target)
    assert not (target / "phase2d_state.json").exists()
