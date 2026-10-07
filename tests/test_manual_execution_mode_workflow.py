from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "crypto_bot.yml"
)


def test_manual_execution_mode_workflow_wiring_is_fail_closed():
    text = WORKFLOW.read_text(encoding="utf-8-sig")

    dispatch_start = text.index("  workflow_dispatch:")
    dispatch_end = text.index(
        "  repository_dispatch:",
        dispatch_start,
    )

    dispatch = text[dispatch_start:dispatch_end]

    assert "execution_mode:" in dispatch
    assert "default: PREDICT_ONLY" in dispatch
    assert "- PREDICT_ONLY" in dispatch
    assert "- MANAGE_ONLY" in dispatch
    assert "- FULL" not in dispatch

    assert (
        text.count(
            "      - name: Download Model Artifact from GitHub Release"
        )
        == 1
    )

    scanner_start = text.index(
        "      - name: Run Trading Scanner"
    )
    scanner_end = text.index(
        "      - name: Validate scanner liveness",
        scanner_start,
    )

    scanner = text[scanner_start:scanner_end]

    assert "EXECUTION_MODE_OVERRIDE:" in scanner

    assert (
        "MANAGE_ONLY: skipping prediction model download."
        in text
    )

    assert (
        "MANAGE_ONLY: prediction scanner intentionally bypassed."
        in text
    )

    assert (
        "MANAGE_ONLY: skipping Historical Auditor trigger."
        in text
    )
