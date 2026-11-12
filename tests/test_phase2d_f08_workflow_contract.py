from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / ".github/workflows/phase2d_f08_validator_amendment.yml"
DISPOSITION = ROOT / ".github/workflows/phase2d_f08_controlled_disposition.yml"
WORKFLOW = ROOT / ".github/workflows/run_phase2d.yml"
REGRESSION = ROOT / ".github/workflows/pull_request_regression.yml"


def test_validator_amendment_handoff_contains_backfill_folder_and_approval_evidence():
    text = AMENDMENT.read_text(encoding="utf-8")
    assert "cp -a working/f08_historical_backfill output/f08_historical_backfill" in text
    assert "output/validator_amendment_approval_evidence.json" in text
    assert "f08_backfill_manifest.json" not in text
    assert "f08_backfill_observations.jsonl" not in text


def test_disposition_consumes_same_backfill_folder_layout_and_amendment_approval():
    text = DISPOSITION.read_text(encoding="utf-8")
    assert "approval_handoff/f08_historical_backfill/manifest.json" in text
    assert "approval_handoff/f08_historical_backfill/observations.jsonl" in text
    assert "cp -a approval_handoff/f08_historical_backfill working/f08_historical_backfill" in text
    assert "cp approval_handoff/validator_amendment_approval_evidence.json working/validator_amendment_approval_evidence.json" in text


def test_resumed_attempt_reuses_original_disposition_approval_snapshot():
    text = DISPOSITION.read_text(encoding="utf-8")
    assert "prior_disposition_attempt/working/disposition_approval_evidence.json" in text
    assert "working/disposition_approval_evidence.json" in text
    assert "if: ${{ inputs.resume_attempt_run_id == '' }}" in text


def test_user_inputs_are_not_interpolated_into_shell_script_bodies():
    for workflow in (AMENDMENT, DISPOSITION):
        text = workflow.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "run: |" in line:
                continue
            # Input expressions are allowed in env/with/if fields, not in shell script text.
            assert not ("${{ inputs." in line and line.lstrip().startswith(("run:", "- run:")))
    assert '"${{ inputs.' not in AMENDMENT.read_text(encoding="utf-8")
    assert '"${{ inputs.' not in DISPOSITION.read_text(encoding="utf-8")


def test_protected_workflows_checkout_exact_dispatch_commit():
    for workflow in (AMENDMENT, DISPOSITION):
        text = workflow.read_text(encoding="utf-8")
        assert "ref: ${{ github.sha }}" in text
        assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in text
        assert "ref: main" not in text


def test_validator_amendment_pins_authoritative_source_run_identity():
    text = AMENDMENT.read_text(encoding="utf-8")
    assert '"head_sha": "58a64fbdeeee2a3a2851b2ef23824240eeca24e2"' in text
    assert '"path": ".github/workflows/run_phase2d.yml"' in text
    assert '"run_attempt": 1' in text
    assert 'str(run.get("actor", {}).get("id", "")) != "166250462"' in text


PINNED_ACTIONS = {
    "actions/checkout": (
        "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "v7.0.1",
    ),
    "actions/setup-python": (
        "6d4fd22b20465ccc0f7354476588e8df7c9ca0e0",
        "v7.1.0",
    ),
    "actions/download-artifact": (
        "9000827ccba6bdab643e8b6fd33ac0654aef8333",
        "v8.0.2",
    ),
    "actions/upload-artifact": (
        "cf430e030ddbb5b0abf93d22962f4752f3646cd9",
        "v7.0.2",
    ),
}


def test_external_actions_are_pinned_to_verified_full_commit_shas():
    import re

    for workflow in (AMENDMENT, DISPOSITION, WORKFLOW, REGRESSION):
        for line_number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = re.match(r"\s*uses:\s*([^\s#]+)(?:\s+#\s*(v[0-9][^\s]*))?\s*$", line)
            if not match:
                continue
            action_ref, release_comment = match.groups()
            action, separator, revision = action_ref.partition("@")
            assert separator, f"Unpinned action at {workflow}:{line_number}: {action_ref}"
            assert action in PINNED_ACTIONS, f"Unreviewed action at {workflow}:{line_number}: {action}"
            expected_sha, expected_release = PINNED_ACTIONS[action]
            assert revision == expected_sha, f"Wrong commit SHA at {workflow}:{line_number}: {revision}"
            assert release_comment == expected_release, f"Missing or incorrect release annotation at {workflow}:{line_number}"


def test_all_required_pinned_action_families_are_used():
    combined = (
        AMENDMENT.read_text(encoding="utf-8")
        + "\n" + DISPOSITION.read_text(encoding="utf-8")
        + "\n" + WORKFLOW.read_text(encoding="utf-8")
        + "\n" + REGRESSION.read_text(encoding="utf-8")
    )
    for action in PINNED_ACTIONS:
        assert f"uses: {action}@" in combined



def test_controlled_disposition_packages_baseline_and_working_results_snapshot():
    text = DISPOSITION.read_text(encoding="utf-8")
    assert "cp approval_handoff/phase2d_results.baseline.json baseline/phase2d_results.json" in text
    assert "cp approval_handoff/phase2d_latest_snapshot.baseline.json baseline/phase2d_latest_snapshot.json" in text
    assert "cp baseline/phase2d_results.json working/phase2d_results.json" in text
    assert "cp baseline/phase2d_latest_snapshot.json working/phase2d_latest_snapshot.json" in text


def test_normal_f08_restore_uses_fixed_release_bootstrap_and_bot_state_continuation():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "bootstrap_f08_runtime:" in text
    assert "default: false" in text[text.index("bootstrap_f08_runtime:"):text.index("repository_dispatch:")]
    start = text.index("- name: Restore F08 schema-7 runtime — fixed Release bootstrap or pinned continuation")
    end = text.index("- name: Restore previous Phase 2D state", start)
    f08_block = text[start:end]
    assert "f08-schema7-disposition-v1" in f08_block
    assert "f08-schema7-runtime-handoff.zip" in f08_block
    assert "releases/tags/${RELEASE_TAG}" in f08_block
    assert "branches/${STATE_BRANCH}" in f08_block
    assert "phase2d-state-f08-pointer.json" in f08_block
    assert "prepare-continuation" in f08_block
    assert "install-runtime" in f08_block
    assert "gh run list" not in f08_block
    assert 'gh run download "$PREVIOUS_RUN_ID" --name "phase2d-state-f08"' in f08_block
    assert "select-run" not in f08_block
    assert "rollback" not in f08_block.lower()
    legacy_start = text.index("- name: Restore previous Phase 2D state")
    legacy_end = text.index("# PINNED F08 AUTHORITATIVE ARCHIVE RESTORE", legacy_start)
    legacy_block = text[legacy_start:legacy_end]
    assert "!= 'PHASE2D-HARDENED-20260924-F08'" in legacy_block
    assert 'state.get("schema_version") not in {5, 6}' in legacy_block


def test_bootstrap_requires_current_workflow_commit_to_equal_release_tag_commit():
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("RELEASE_TAG_COMMIT=\"$(git rev-parse")
    end = text.index("else\n            echo \"F08 MODE: NORMAL SCHEMA-7 CONTINUATION", start)
    block = text[start:end]
    assert '[[ "$RELEASE_TAG_COMMIT" != "$GITHUB_SHA" ]]' in block
    assert '--workflow-commit "$GITHUB_SHA"' in block


def test_actionlint_ci_includes_modified_normal_phase2d_workflow():
    text = REGRESSION.read_text(encoding="utf-8")
    assert text.count(".github/workflows/run_phase2d.yml") == 1


def test_fixed_release_asset_is_verified_and_packaged_for_owner_publication():
    text = DISPOSITION.read_text(encoding="utf-8")
    assert "phase2d_f08_v2_restore.py verify" in text
    assert "f08-schema7-runtime-handoff.zip" in text
    assert "f08-schema7-runtime-handoff.zip.sha256" in text
    assert "f08-schema7-runtime-handoff-release-candidate" in text
    assert "gh release create" not in text
    assert "gh release upload" not in text  # publication is an owner/admin operation after immutability is enabled


def test_f08_state_artifact_is_sealed_uploaded_then_pointer_is_advanced():
    text = WORKFLOW.read_text(encoding="utf-8")
    seal_at = text.index("- name: Seal F08 schema-7 continuation state")
    upload_at = text.index("- name: Upload dedicated F08 schema-7 continuation artifact")
    pointer_job_at = text.index("publish_f08_state_pointer:")
    assert seal_at < upload_at < pointer_job_at
    assert "name: phase2d-state-f08" in text[upload_at:pointer_job_at]
    pointer_job = text[pointer_job_at:text.index("# ONE-TIME F08 HISTORICAL LEDGER RECONCILIATION", pointer_job_at)]
    assert "needs: prospective" in pointer_job
    assert "contents: write" in pointer_job
    assert "check-pointer-transition" in pointer_job
    assert '--field sha="$EXISTING_SHA"' in pointer_job
    assert "bootstrap cannot replace" in pointer_job.lower()


def test_f08_passphrase_is_not_exposed_as_a_workflow_global():
    text = WORKFLOW.read_text(encoding="utf-8")
    global_env = text[text.index("env:"):text.index("jobs:")]
    assert "F08_SNAPSHOT_PASSPHRASE" not in global_env
    archive_step = text[text.index("- name: Restore verified encrypted F08 authoritative archive"):]
    assert "F08_SNAPSHOT_PASSPHRASE: ${{ secrets.F08_SNAPSHOT_PASSPHRASE }}" in archive_step


def test_f08_continuation_pointer_branch_is_not_main():
    text = WORKFLOW.read_text(encoding="utf-8")
    pointer_job = text[text.index("publish_f08_state_pointer:"):text.index("# ONE-TIME F08 HISTORICAL LEDGER RECONCILIATION")]
    assert "branch=bot-state" in pointer_job
    assert "contents: write" in pointer_job



def test_normal_f08_is_disabled_by_default_and_fresh_start_is_rejected():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "PHASE2D_F08_RUNTIME_ENABLED: ${{ vars.PHASE2D_F08_RUNTIME_ENABLED }}" in text
    assert "F08 cannot use allow_fresh_start" in text
    assert 'inputs.allow_fresh_start != true' in text
    assert "vars.PHASE2D_F08_RUNTIME_ENABLED == 'true'" in text


def test_continuation_artifact_and_pointer_are_success_gated_after_install_verification():
    text = WORKFLOW.read_text(encoding="utf-8")
    install_at = text.index("python phase2d_f08_v2_handoff.py install-runtime")
    seal_at = text.index("- name: Seal F08 schema-7 continuation state")
    upload_at = text.index("- name: Upload dedicated F08 schema-7 continuation artifact")
    pointer_at = text.index("publish_f08_state_pointer:")
    assert install_at < seal_at < upload_at < pointer_at
    assert "if: ${{ success()" in text[seal_at:upload_at]
    assert "if: ${{ success()" in text[upload_at:pointer_at]
    pointer_job = text[pointer_at:text.index("# ONE-TIME F08 HISTORICAL LEDGER RECONCILIATION", pointer_at)]
    assert "needs: prospective" in pointer_job
    assert "needs.prospective.result == 'success'" in pointer_job


def test_no_backup_or_rollback_installation_logic_remains():
    import ast

    helper = (ROOT / "phase2d_f08_v2_handoff.py").read_text(encoding="utf-8")
    tree = ast.parse(helper)
    install = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_runtime_state"
    )
    identifiers = {
        node.id.lower() for node in ast.walk(install) if isinstance(node, ast.Name)
    }
    calls = {
        node.func.id.lower() for node in ast.walk(install) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any("rollback" in value or "backup" in value for value in identifiers | calls)
    assert "verify_runtime_installation" in calls


def test_f08_payload_modes_cannot_bypass_the_normal_continuation_gates():
    text = WORKFLOW.read_text(encoding="utf-8")
    blocks = [
        text[text.index("- name: Enforce F08 disabled-by-default runtime and no-fresh-start policy"):text.index("# PYTHON", text.index("- name: Enforce F08 disabled-by-default runtime and no-fresh-start policy"))],
        text[text.index("- name: Restore F08 schema-7 runtime — fixed Release bootstrap or pinned continuation"):text.index("- name: Restore previous Phase 2D state")],
        text[text.index("- name: Seal F08 schema-7 continuation state"):text.index("# UPLOAD PHASE 2D STATE + AUDIT ARTIFACTS")],
        text[text.index("publish_f08_state_pointer:"):text.index("# ONE-TIME F08 HISTORICAL LEDGER RECONCILIATION")],
    ]
    for block in blocks:
        assert "github.event.client_payload.approve_f08_gate_amendment != 'true'" in block
        assert "github.event.client_payload.restore_f08_authoritative_archive != 'true'" in block
