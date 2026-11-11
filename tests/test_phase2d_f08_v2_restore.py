from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import phase2d_f08_disposition as disposition
import phase2d_f08_v2_restore as v2
from test_phase2d_f08_disposition import _append_amendment, _apply, fixture_set


def _package_from_fixture(f, tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline"
    working = tmp_path / "working"
    source_base = Path(v2.__file__).resolve().parent
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    baseline.mkdir()
    working.mkdir()
    source.mkdir()
    for rel in v2.SOURCE_FILES:
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_base / rel, target)
    # The unit fixture amendment binds to these two fixture-specific source files.
    shutil.copyfile(f["validator_path"], source / "phase2d_prospective_validator.py")
    shutil.copyfile(f["gate_path"], source / "phase2d_promotion_gate.py")

    for src, name in (
        (f["baseline_state_path"], "phase2d_state.json"),
        (f["baseline_ledger_path"], "phase2d_observations.jsonl"),
        (f["baseline_journal_path"], "experiment_ledger.jsonl"),
    ):
        shutil.copyfile(src, baseline / name)

    for src, name in (
        (f["state_path"], "phase2d_state.json"),
        (f["ledger_path"], "phase2d_observations.jsonl"),
        (f["journal_path"], "experiment_ledger.jsonl"),
    ):
        shutil.copyfile(src, working / name)

    backfill = working / "f08_historical_backfill"
    (backfill / "evidence").mkdir(parents=True)
    shutil.copyfile(f["manifest_path"], backfill / "manifest.json")
    shutil.copyfile(f["bundle_path"], backfill / "observations.jsonl")
    shutil.copyfile(
        f["resolution_provenance_path"],
        backfill / disposition.PROVENANCE_FILENAME,
    )
    shutil.copyfile(f["raw_windows_path"], backfill / "evidence" / "raw_windows.jsonl")
    shutil.copyfile(f["disposition_approval_path"], working / "disposition_approval_evidence.json")
    shutil.copyfile(f["amendment_approval_path"], working / "validator_amendment_approval_evidence.json")

    v2._build_package(
        baseline_dir=baseline,
        working_dir=working,
        source_root=source,
        output_dir=output,
        source_commit="a" * 40,
        validator_amendment_run_id="910001",
        disposition_run_id="910002",
        spec=f["spec"],
    )
    return output


def test_v2_seal_and_verify_complete_schema7_transition(fixture_set, tmp_path):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    artifact = _package_from_fixture(f, tmp_path)

    result = v2.verify_artifact(
        artifact,
        spec=f["spec"],
        expected_source_commit="a" * 40,
        runtime_source_root=artifact / "source",
    )
    manifest = json.loads((artifact / "manifest_v2.json").read_text())

    assert result["status"] == "PASS"
    assert manifest["artifact_schema_version"] == 2
    assert manifest["artifact_status"] == v2.ARTIFACT_STATUS
    assert manifest["verification"]["dispositioned_by_model"] == {"candidate": 2, "production": 3}
    assert manifest["verification"]["official_outcome_rows_appended"] == 0
    assert (artifact / "historical_backfill" / disposition.PROVENANCE_FILENAME).is_file()
    assert (artifact / "historical_backfill" / "evidence" / "raw_windows.jsonl").is_file()
    assert (artifact / "source" / "phase2d_f08_resolution_provenance.py").is_file()
    assert manifest["restore_policy"]["normal_phase2d_pipeline_handoff"] is False


def test_v2_detects_tampered_disposition_state(fixture_set, tmp_path):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    artifact = _package_from_fixture(f, tmp_path)
    state_path = artifact / "disposition" / "phase2d_state.json"
    state = json.loads(state_path.read_text())
    state["dispositioned"]["candidate"][0]["prediction_payload"]["confidence"] = 0.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(v2.V2RestoreError, match="inventory/hash/size"):
        v2.verify_artifact(artifact, spec=f["spec"])


def test_v2_detects_extra_official_observation_ledger_record(fixture_set, tmp_path):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    artifact = _package_from_fixture(f, tmp_path)
    ledger_path = artifact / "disposition" / "phase2d_observations.jsonl"
    with ledger_path.open("ab") as handle:
        handle.write(b'{"record_type":"observation","status":"TP"}\n')
    with pytest.raises(v2.V2RestoreError, match="inventory/hash/size"):
        v2.verify_artifact(artifact, spec=f["spec"])


def test_v2_restore_is_verify_before_write_and_preserves_quarantine(fixture_set, tmp_path):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    artifact = _package_from_fixture(f, tmp_path)
    destination = tmp_path / "restored"
    result = v2.restore_artifact(
        artifact,
        destination,
        spec=f["spec"],
        runtime_source_root=artifact / "source",
    )
    assert result["status"] == "PASS"
    assert result["normal_pipeline_handoff"] is False
    assert (destination / "baseline" / "phase2d_state.json").is_file()
    assert (destination / "disposition" / "phase2d_state.json").is_file()
    assert v2.verify_artifact(
        destination,
        spec=f["spec"],
        runtime_source_root=destination / "source",
    )["status"] == "PASS"


def test_v2_refuses_nonempty_restore_destination(fixture_set, tmp_path):
    f = fixture_set
    _append_amendment(f)
    _apply(f)
    artifact = _package_from_fixture(f, tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "do-not-overwrite.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(v2.V2RestoreError, match="Refusing to overwrite"):
        v2.restore_artifact(artifact, destination, spec=f["spec"])
    assert (destination / "do-not-overwrite.txt").read_text() == "preserve"
