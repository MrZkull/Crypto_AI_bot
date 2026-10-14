from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "hourly_research_pipeline.yml"
CANONICAL = ROOT / ".github" / "workflows" / "run_phase2d.yml"


def read_workflow():
    assert WORKFLOW.is_file()
    return WORKFLOW.read_text(encoding="utf-8")


def test_hourly_pipeline_no_longer_owns_phase2d():
    content = read_workflow().lower()
    for token in (
        "phase2d", "phase 2d", "f08", "research_outputs", "phase2d-state",
        "phase2d_prospective_validator.py", "candidate_model.pkl",
        "pro_crypto_ai_model.pkl", "deribit_client",
    ):
        assert token not in content, f"Competing F08 reference remains: {token}"


def test_non_f08_audit_scripts_are_preserved():
    content = read_workflow()
    assert "run: python prediction_auditor.py" in content
    assert "run: python failure_taxonomy.py" in content


def test_non_f08_state_sync_is_preserved():
    content = read_workflow()
    for token in (
        "Snapshot research state", "Sync and Commit Research State",
        "/tmp/cryptobot_research/predictions.json",
        "/tmp/cryptobot_research/coin_dossier.json",
        "/tmp/cryptobot_research/trade_history.json",
        "/tmp/cryptobot_research/adaptation_proposals.json",
        "git add predictions.json", "git add coin_dossier.json",
        "git add trade_history.json", "git add adaptation_proposals.json",
        "git add data/",
    ):
        assert token in content, f"Required non-F08 state sync missing: {token}"


def test_commit_message_scopes_the_sync():
    content = read_workflow()
    assert "chore(research): hourly auxiliary state sync [skip ci]" in content
    assert "hourly audit and Phase 2D sync" not in content


def test_hourly_workflow_does_not_stage_or_upload_f08_state():
    content = read_workflow().lower()
    assert "git add research_outputs/" not in content
    assert "uses: actions/upload-artifact@v4" not in content
    assert "phase2d-state" not in content


def test_canonical_f08_workflow_still_exists():
    assert CANONICAL.is_file()

def test_hourly_sync_guard_precedes_all_git_mutations():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/hourly_research_pipeline.yml").read_text(
        encoding="utf-8"
    )

    blocks = re.findall(
        r"(?ms)^      - name: Sync and Commit[^\n]*\n.*?(?=^      - name:|\Z)",
        workflow,
    )
    assert len(blocks) == 1
    block = blocks[0]

    guard = 'if [[ "${GITHUB_REF:-}" != "refs/heads/main" ]]; then'
    assert guard in block
    assert "exit 0" in block[block.index(guard):]

    mutations = [
        block.find("git reset --hard HEAD"),
        block.find("git pull"),
        block.find("git add "),
        block.find("git push "),
    ]
    mutations = [position for position in mutations if position >= 0]
    assert mutations, "Expected state-sync git operations for guard validation"
    assert block.index(guard) < min(mutations)


def test_hourly_auditor_schedule_is_six_hourly():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/hourly_research_pipeline.yml").read_text(
        encoding="utf-8"
    )

    assert re.search(r"(?m)^[ \t]*-[ \t]*cron:[ \t]*'10 \*/6 \* \* \*'[ \t]*$", workflow)
    assert not re.search(r"(?m)^[ \t]*-[ \t]*cron:[ \t]*'10 \* \* \* \*'[ \t]*$", workflow)


def test_git_add_and_artifact_upload_targets_exclude_research_outputs():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/hourly_research_pipeline.yml").read_text(
        encoding="utf-8"
    )

    # Inspect actual git-add targets, not just arbitrary workflow text.
    git_add_targets = re.findall(
        r"(?m)^[ \t]*git add[ \t]+(.+?)[ \t]*$",
        workflow,
    )
    assert git_add_targets, "Expected auxiliary state staging commands"
    assert all("research_outputs" not in target.replace("\\", "/")
               for target in git_add_targets)

    # Parse every artifact-upload step and inspect only its configured path target.
    steps = re.findall(
        r"(?ms)^      - name:.*?(?=^      - name:|\Z)",
        workflow,
    )
    upload_steps = [
        step for step in steps
        if re.search(r"(?m)^[ \t]*uses:[ \t]*actions/upload-artifact@", step)
    ]

    for step in upload_steps:
        lines = step.splitlines()
        path_targets = []

        for index, line in enumerate(lines):
            match = re.match(r"^([ \t]*)path:[ \t]*(.*)$", line)
            if not match:
                continue

            indent = len(match.group(1))
            inline_value = match.group(2).strip()

            if inline_value and inline_value != "|":
                path_targets.append(inline_value)
                continue

            for path_line in lines[index + 1:]:
                if path_line.strip():
                    path_indent = len(path_line) - len(path_line.lstrip(" \t"))
                    if path_indent <= indent:
                        break
                path_targets.append(path_line.strip())

        assert path_targets, "Artifact upload step has no inspectable path target"
        assert all("research_outputs" not in target.replace("\\", "/")
                   for target in path_targets)