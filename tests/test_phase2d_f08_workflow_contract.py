from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / ".github/workflows/phase2d_f08_validator_amendment.yml"
DISPOSITION = ROOT / ".github/workflows/phase2d_f08_controlled_disposition.yml"
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

    for workflow in (AMENDMENT, DISPOSITION, REGRESSION):
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
    combined = AMENDMENT.read_text(encoding="utf-8") + "\n" + DISPOSITION.read_text(encoding="utf-8")
    for action in PINNED_ACTIONS:
        assert f"uses: {action}@" in combined
